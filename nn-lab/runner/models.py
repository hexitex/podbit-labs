"""
Fixed-menu model architectures.

Each builder takes (depth, width, input_shape, num_classes, modifications, **kwargs)
and returns a torch.nn.Module. No arbitrary definitions.
"""

import torch
import torch.nn as nn
import math


# =============================================================================
# MODIFICATIONS
# =============================================================================

def _apply_modifications(mods: list | None) -> dict:
    """Parse modification list into a config dict."""
    cfg = {
        "dropout": 0.0,
        "batch_norm": False,
        "layer_norm": False,
        "bottleneck": False,
        "weight_decay": 0.0,  # handled at optimizer level, not model level
    }
    if not mods:
        return cfg
    for mod in mods:
        if isinstance(mod, str):
            name, params = mod, {}
        else:
            name, params = mod.get("name", mod), mod
        if name == "dropout":
            cfg["dropout"] = params.get("rate", 0.5) if isinstance(params, dict) else 0.5
        elif name == "batch_norm":
            cfg["batch_norm"] = True
        elif name == "layer_norm":
            cfg["layer_norm"] = True
        elif name == "bottleneck":
            cfg["bottleneck"] = True
        elif name == "weight_decay":
            cfg["weight_decay"] = params.get("lambda", 1e-4) if isinstance(params, dict) else 1e-4
    return cfg


# =============================================================================
# MLP
# =============================================================================

class MLP(nn.Module):
    def __init__(self, depth: int, width: int, input_dim: int, num_classes: int,
                 residual: bool = False, mods: dict | None = None):
        super().__init__()
        m = mods or {}
        layers: list[nn.Module] = []

        # Input layer
        layers.append(nn.Linear(input_dim, width))
        if m.get("batch_norm"):
            layers.append(nn.BatchNorm1d(width))
        if m.get("layer_norm"):
            layers.append(nn.LayerNorm(width))
        layers.append(nn.ReLU())
        if m.get("dropout", 0) > 0:
            layers.append(nn.Dropout(m["dropout"]))

        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            if m.get("batch_norm"):
                layers.append(nn.BatchNorm1d(width))
            if m.get("layer_norm"):
                layers.append(nn.LayerNorm(width))
            layers.append(nn.ReLU())
            if m.get("dropout", 0) > 0:
                layers.append(nn.Dropout(m["dropout"]))

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(width, num_classes)
        self.residual = residual and depth > 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        if self.residual:
            h = self.features[0](x)  # first linear
            start = 1
            # Find ReLU boundaries for skip connections
            for i in range(start, len(self.features)):
                h_in = h
                h = self.features[i](h)
                if isinstance(self.features[i], nn.Linear) and h.shape == h_in.shape:
                    h = h + h_in
        else:
            h = self.features(x)
        return self.classifier(h)


# =============================================================================
# CNN
# =============================================================================

class CNN(nn.Module):
    def __init__(self, depth: int, width: int, input_channels: int, spatial_size: int,
                 num_classes: int, residual: bool = False, mods: dict | None = None):
        super().__init__()
        m = mods or {}
        conv_layers: list[nn.Module] = []
        in_ch = input_channels
        out_ch = width
        current_size = spatial_size

        for i in range(depth):
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            if m.get("batch_norm"):
                conv_layers.append(nn.BatchNorm2d(out_ch))
            conv_layers.append(nn.ReLU())
            if i < depth - 1 and current_size > 4:
                conv_layers.append(nn.MaxPool2d(2))
                current_size //= 2
            if m.get("dropout", 0) > 0:
                conv_layers.append(nn.Dropout2d(m["dropout"]))
            in_ch = out_ch
            if i > 0 and i % 2 == 0:
                out_ch = min(out_ch * 2, 512)

        self.features = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_ch, num_classes)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.classifier(h)


# =============================================================================
# CNN DEPTHWISE SEPARABLE
# =============================================================================

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class CNNDepthwiseSeparable(nn.Module):
    def __init__(self, depth: int, width: int, input_channels: int, spatial_size: int,
                 num_classes: int, residual: bool = False, mods: dict | None = None):
        super().__init__()
        m = mods or {}
        conv_layers: list[nn.Module] = []
        in_ch = input_channels
        out_ch = width
        current_size = spatial_size

        for i in range(depth):
            if i == 0:
                conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            else:
                conv_layers.append(DepthwiseSeparableConv(in_ch, out_ch))
            if m.get("batch_norm"):
                conv_layers.append(nn.BatchNorm2d(out_ch))
            conv_layers.append(nn.ReLU())
            if i < depth - 1 and current_size > 4:
                conv_layers.append(nn.MaxPool2d(2))
                current_size //= 2
            in_ch = out_ch
            if i > 0 and i % 2 == 0:
                out_ch = min(out_ch * 2, 512)

        self.features = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.classifier(h)


# =============================================================================
# RESNET (small-scale)
# =============================================================================

class ResBlock(nn.Module):
    def __init__(self, channels: int, bottleneck: bool = False, bn: bool = True):
        super().__init__()
        if bottleneck:
            mid = max(channels // 4, 8)
            self.block = nn.Sequential(
                nn.Conv2d(channels, mid, 1), nn.BatchNorm2d(mid) if bn else nn.Identity(), nn.ReLU(),
                nn.Conv2d(mid, mid, 3, padding=1), nn.BatchNorm2d(mid) if bn else nn.Identity(), nn.ReLU(),
                nn.Conv2d(mid, channels, 1), nn.BatchNorm2d(channels) if bn else nn.Identity(),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels) if bn else nn.Identity(),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels) if bn else nn.Identity(),
            )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + x)


class ResNet(nn.Module):
    def __init__(self, depth: int, width: int, input_channels: int, spatial_size: int,
                 num_classes: int, mods: dict | None = None):
        super().__init__()
        m = mods or {}
        bn = m.get("batch_norm", True)
        bottleneck = m.get("bottleneck", False)

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, 3, padding=1),
            nn.BatchNorm2d(width) if bn else nn.Identity(),
            nn.ReLU(),
        )

        blocks: list[nn.Module] = []
        for _ in range(depth):
            blocks.append(ResBlock(width, bottleneck=bottleneck, bn=bn))
        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.blocks(h)
        h = self.pool(h).flatten(1)
        return self.classifier(h)


# =============================================================================
# TRANSFORMER TINY
# =============================================================================

class TransformerTiny(nn.Module):
    """Tiny vision transformer: patch embed → transformer encoder → classify."""

    def __init__(self, depth: int, width: int, input_channels: int, spatial_size: int,
                 num_classes: int, mods: dict | None = None):
        super().__init__()
        m = mods or {}
        patch_size = 4 if spatial_size >= 16 else 2
        num_patches = (spatial_size // patch_size) ** 2
        patch_dim = input_channels * patch_size * patch_size

        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_dim, width)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, width) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, width) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=max(1, width // 32),
            dim_feedforward=width * 4, dropout=m.get("dropout", 0.0),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        p = self.patch_size
        # Patchify: (B, C, H, W) → (B, num_patches, patch_dim)
        patches = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H', W', p, p)
        patches = patches.contiguous().view(B, -1, patches.size(1) * p * p)

        h = self.patch_embed(patches)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.pos_embed[:, :h.size(1)]

        h = self.encoder(h)
        h = self.norm(h[:, 0])
        return self.classifier(h)


# =============================================================================
# BUILDER
# =============================================================================

def _get_input_shape(dataset_name: str, dataset_params: dict | None = None) -> tuple:
    """Return (channels, spatial_size, input_dim, num_classes) for a dataset."""
    ds = dataset_name
    if ds == "mnist" or ds == "fashion_mnist":
        return (1, 28, 784, 10)
    elif ds == "cifar10":
        return (3, 32, 3072, 10)
    elif ds == "synthetic_quadratic":
        dim = (dataset_params or {}).get("dim", 10)
        return (0, 0, dim, 1)  # regression
    elif ds == "synthetic_regression":
        dim = (dataset_params or {}).get("dim", 10)
        return (0, 0, dim, 1)  # regression
    else:
        raise ValueError(f"Unknown dataset: {ds}")


def build_model(spec: dict) -> nn.Module:
    """Build a model from the architecture section of the spec."""
    arch = spec["architecture"]
    model_type = arch["model_type"]
    depth = arch["depth"]
    width = arch["width"]
    residual = arch.get("residual_connections", False)
    mods = _apply_modifications(arch.get("modifications"))

    # Determine input shape from dataset
    training = spec.get("training", {})
    ds = training.get("dataset", "mnist")
    ds_name = ds if isinstance(ds, str) else ds.get("name", "mnist")
    ds_params = ds if isinstance(ds, dict) else {}
    channels, spatial, input_dim, num_classes = _get_input_shape(ds_name, ds_params)

    is_image = channels > 0

    if model_type == "mlp":
        return MLP(depth, width, input_dim, num_classes, residual=residual, mods=mods)

    elif model_type == "cnn":
        if not is_image:
            raise ValueError("CNN requires image dataset (mnist, cifar10, fashion_mnist)")
        return CNN(depth, width, channels, spatial, num_classes, residual=residual, mods=mods)

    elif model_type == "cnn_depthwise_separable":
        if not is_image:
            raise ValueError("CNN depthwise separable requires image dataset")
        return CNNDepthwiseSeparable(depth, width, channels, spatial, num_classes,
                                     residual=residual, mods=mods)

    elif model_type == "resnet":
        if not is_image:
            raise ValueError("ResNet requires image dataset")
        return ResNet(depth, width, channels, spatial, num_classes, mods=mods)

    elif model_type == "transformer_tiny":
        if not is_image:
            raise ValueError("Transformer requires image dataset")
        return TransformerTiny(depth, width, channels, spatial, num_classes, mods=mods)

    else:
        raise ValueError(f"not supported: model_type '{model_type}'")
