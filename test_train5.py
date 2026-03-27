import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2
from PIL import Image
from tqdm import tqdm
import zipfile

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = "cv_hw1_data"
MODEL_PATH = "best_model3.pth"
OUTPUT_CSV = "prediction.csv"
BATCH_SIZE = 64
NUM_WORKERS = 0
NUM_CLASSES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Must match train3.py definitions ─────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg_w = self.fc(x.mean(dim=(2, 3)))
        max_w = self.fc(x.amax(dim=(2, 3)))
        w = torch.sigmoid(avg_w + max_w)
        return x * w.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=11):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        w = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * w


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=11):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels // 2

        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1, bias=False)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, 1, bias=False)
        self.g = nn.Conv2d(in_channels, self.inter_channels, 1, bias=False)
        self.out = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        nn.init.zeros_(self.out[1].weight)

    def forward(self, x):
        b, c, h, w = x.shape
        theta = self.theta(x).view(b, self.inter_channels, -1)
        phi = self.phi(x).view(b, self.inter_channels, -1)
        g = self.g(x).view(b, self.inter_channels, -1)

        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = F.softmax(attn * (self.inter_channels ** -0.5), dim=-1)

        out = torch.bmm(g, attn.permute(0, 2, 1))
        out = out.view(b, self.inter_channels, h, w)
        return x + self.out(out)


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(2, 3)).pow(1.0 / self.p)


class ScaleAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


def inject_cbam_blocks(model):
    for _, module in model.named_modules():
        if isinstance(module, models.resnet.Bottleneck):
            channels = module.bn3.num_features
            cbam = CBAM(channels).to(next(module.parameters()).device)
            module.cbam = cbam

            def make_forward(mod):
                def forward(x):
                    identity = x
                    out = mod.conv1(x)
                    out = mod.bn1(out)
                    out = mod.relu(out)
                    out = mod.conv2(out)
                    out = mod.bn2(out)
                    out = mod.relu(out)
                    out = mod.conv3(out)
                    out = mod.bn3(out)
                    out = mod.cbam(out)
                    if mod.downsample is not None:
                        identity = mod.downsample(x)
                    out += identity
                    out = mod.relu(out)
                    return out
                return forward

            module.forward = make_forward(module)


class MultiScaleResNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()

        backbone = models.resnet152(weights=None)
        inject_cbam_blocks(backbone)

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.nonlocal3 = NonLocalBlock(in_channels=1024)
        self.nonlocal4 = NonLocalBlock(in_channels=2048, inter_channels=512)

        self.reduce2 = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.reduce3 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.reduce4 = nn.Sequential(
            nn.Conv2d(2048, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.pool2 = GeM(p=3.0)
        self.pool3 = GeM(p=3.0)
        self.pool4 = GeM(p=3.0)

        fused_dim = 256 + 512 + 512   # 1280
        self.scale_attn = ScaleAttention(fused_dim, reduction=4)

        self.head = nn.Sequential(
            nn.BatchNorm1d(fused_dim),
            nn.Dropout(p=0.3),
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)

        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f3 = self.nonlocal3(f3)
        f4 = self.layer4(f3)
        f4 = self.nonlocal4(f4)

        feat2 = self.pool2(self.reduce2(f2))
        feat3 = self.pool3(self.reduce3(f3))
        feat4 = self.pool4(self.reduce4(f4))

        fused = torch.cat([feat2, feat3, feat4], dim=1)
        fused = self.scale_attn(fused)
        return self.head(fused)


def build_model():
    return MultiScaleResNet(num_classes=NUM_CLASSES)


# ── Test dataset ─────────────────────────────────────────────────────────────
class TestDataset(Dataset):
    def __init__(self, test_dir, transform):
        self.test_dir = test_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(test_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img = Image.open(os.path.join(self.test_dir, filename)).convert("RGB")
        img = self.transform(img)
        name = os.path.splitext(filename)[0]
        return img, name


def main():
    test_transform = v2.Compose([
        v2.Resize(400, antialias=True),
        v2.CenterCrop(320),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = TestDataset(os.path.join(DATA_DIR, "test"), test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'))

    print(f"Test images: {len(test_dataset)}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = build_model().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: int(k) for k, v in class_to_idx.items()}

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from {MODEL_PATH} (val acc: {checkpoint['val_acc']:.2f}%)")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_names = []
    all_preds = []

    with torch.no_grad():
        for images, names in tqdm(test_loader, desc="Predicting"):
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_names.extend(names)
            all_preds.extend(predicted.cpu().tolist())

    # ── Write CSV ─────────────────────────────────────────────────────────────
    pred_labels = [idx_to_class[p] for p in all_preds]
    df = pd.DataFrame({"image_name": all_names, "pred_label": pred_labels})
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved {len(df)} predictions to {OUTPUT_CSV}")

    # ── Compress CSV to ZIP ───────────────────────────────────────────────────
    zip_name = "solution3.zip"
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(OUTPUT_CSV)

    print(f"Compressed CSV into {zip_name}")


if __name__ == "__main__":
    main()
