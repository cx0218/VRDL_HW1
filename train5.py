import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import v2
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
torch.backends.cudnn.benchmark = True


DATA_DIR = "cv_hw1_data"
BATCH_SIZE = 32
NUM_WORKERS = 8
EPOCHS = 125
NUM_CLASSES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "best_model3.pth"

# ── Attention modules ────────────────────────────────────────────────────────

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


# ── NEW: Scale Attention for weighted fusion ─────────────────────────────────

class ScaleAttention(nn.Module):
    """
    SE-style attention on the fused multi-scale vector.
    Learns to reweight which scale (layer2/3/4) matters more per sample.
    """
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


# ── CBAM injection ───────────────────────────────────────────────────────────

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

# ── Multi-scale feature fusion model (enhanced) ─────────────────────────────

class MultiScaleResNet(nn.Module):
    """
    ResNet50 + CBAM + NonLocal (layer3 & layer4) + Multi-scale Fusion
    + Scale Attention.

    Enhancements over train2.py:
      1. NonLocal after layer4 (inter_channels=512): global self-attention
         at the highest semantic level. 7x7=49 tokens, very cheap.
      2. ScaleAttention on fused vector: learns per-sample weighting
         of which scale matters most.

    layer2: 512ch  28x28 → 256-dim
    layer3: 1024ch 14x14 → 512-dim  (with non-local attention)
    layer4: 2048ch  7x7  → 512-dim  (with non-local attention)
    concat: 1280-dim → scale attention → classification head
    """
    def __init__(self, num_classes=100):
        super().__init__()

        backbone = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
        inject_cbam_blocks(backbone)

        # Backbone stages
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        )
        self.layer1 = backbone.layer1      # 256ch, 56x56
        self.layer2 = backbone.layer2      # 512ch, 28x28
        self.layer3 = backbone.layer3      # 1024ch, 14x14
        self.layer4 = backbone.layer4      # 2048ch, 7x7

        # Non-local block after layer3 (same as train2)
        self.nonlocal3 = NonLocalBlock(in_channels=1024)

        # NEW: Non-local block after layer4, with reduced inter_channels
        # 7x7 = 49 tokens → full self-attention is very cheap here
        self.nonlocal4 = NonLocalBlock(in_channels=2048, inter_channels=512)

        # Multi-scale branch: 1x1 conv to reduce channels + GeM pool
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

        # Scale attention on fused features
        fused_dim = 256 + 512 + 512   # 1280
        self.scale_attn = ScaleAttention(fused_dim, reduction=4)

        # Classification head
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

        f2 = self.layer2(x)          # (B, 512, 28, 28)
        f3 = self.layer3(f2)         # (B, 1024, 14, 14)
        f3 = self.nonlocal3(f3)      # non-local self-attention
        f4 = self.layer4(f3)         # (B, 2048, 7, 7)
        f4 = self.nonlocal4(f4)      # non-local at highest level

        # Multi-scale pool (layer2/3/4)
        feat2 = self.pool2(self.reduce2(f2))   # (B, 256)
        feat3 = self.pool3(self.reduce3(f3))   # (B, 512)
        feat4 = self.pool4(self.reduce4(f4))   # (B, 512)

        fused = torch.cat([feat2, feat3, feat4], dim=1)  # (B, 1280)
        fused = self.scale_attn(fused)
        return self.head(fused)


def build_model():
    model = MultiScaleResNet(num_classes=NUM_CLASSES)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M (limit: 100M)")
    return model


def main():

    train_transform = v2.Compose([
        v2.RandomResizedCrop(320, scale=(0.5, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandAugment(num_ops=2, magnitude=9),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.RandomErasing(p=0.25),
    ])

    val_transform = v2.Compose([
        v2.Resize(400, antialias=True),
        v2.CenterCrop(320),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    mixup_cutmix = v2.RandomChoice([
        v2.MixUp(alpha=0.8, num_classes=NUM_CLASSES),
        v2.CutMix(alpha=1.0, num_classes=NUM_CLASSES),
    ])

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model().to(DEVICE)

    # ── Optimizer (3-tier learning rate) ──────────────────────────────────────
    backbone_early = []  # stem, layer1, layer2
    backbone_late = []   # layer3, layer4, nonlocal3/4
    head_params = []     # reduce*, pool*, head, cbam, scale_attn

    for name, param in model.named_parameters():
        if any(k in name for k in ['head.', 'reduce', 'pool', '.cbam.', 'scale_attn']):
            head_params.append(param)
        elif any(k in name for k in ['layer3', 'layer4', 'nonlocal']):
            backbone_late.append(param)
        else:
            backbone_early.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_early, 'lr': 2e-5},
        {'params': backbone_late, 'lr': 1e-4},
        {'params': head_params, 'lr': 5e-4},
    ], weight_decay=1e-4)

    # ── Loss & Scheduler ──────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    warmup_epochs = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_epochs])

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch[{epoch}/{EPOCHS}] Train", leave=False)

        for images, labels in train_pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            images, labels = mixup_cutmix(images, labels)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch}/{EPOCHS}] Val  ", leave=False)

        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "class_to_idx": train_dataset.class_to_idx,
            }, SAVE_PATH)
            print(f"  -> Saved best model (val acc: {val_acc:.2f}%)")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.2f}%")

    epochs_range = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs_range, history["train_loss"], label="Train Loss")
    ax1.plot(epochs_range, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, history["val_acc"], label="Val Accuracy", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Saved training curves to training_curves.png")

if __name__ == '__main__':
    main()
