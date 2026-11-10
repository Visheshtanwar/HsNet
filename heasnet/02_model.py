"""HEASNet component: HEASNet architecture and attention blocks.

Extracted from the supplied final experiment notebook; execution order is defined in run_experiment.py.
"""

# ======================================== MODELS ========================================
class SqueezeExcitation(nn.Module):
    def __init__(self, c, reduction=16):
        super().__init__(); h = max(c // reduction, 8); self.fc1 = nn.Linear(c, h); self.fc2 = nn.Linear(h, c)
    def forward(self, x):
        z = torch.sigmoid(self.fc2(F.relu(self.fc1(x.mean(dim=(2, 3))))))
        return x * z[:, :, None, None]

class WindowAttention(nn.Module):
    def __init__(self, dim, heads=8, window=4):
        super().__init__(); self.window = window
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True); self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        b, c, h, w_ = x.shape; w = self.window
        ph, pw = (-h) % w, (-w_) % w
        if ph or pw: x = F.pad(x, (0, pw, 0, ph))
        hp, wp = h + ph, w_ + pw
        t = x.permute(0, 2, 3, 1).reshape(b, hp // w, w, wp // w, w, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, w * w, c)
        t = self.norm(t + self.attn(t, t, t, need_weights=False)[0])
        x = t.reshape(b, hp // w, wp // w, w, w, c).permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, c).permute(0, 3, 1, 2)
        return x[:, :, :h, :w_]

class AxialAttentionHeight(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__(); self.attn = nn.MultiheadAttention(dim, heads, batch_first=True); self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        b, c, h, w = x.shape
        t = x.permute(0, 3, 2, 1).reshape(b * w, h, c)
        t = self.norm(t + self.attn(t, t, t, need_weights=False)[0])
        return t.reshape(b, w, h, c).permute(0, 3, 2, 1)

class HEASNet(nn.Module):
    """EfficientNetV2-S + 1x1 projection + SE + window attention + height-axis attention + GAP + dropout + linear.
    use_* flags only switch modules off for the ablation study (all True = proposed model)."""
    def __init__(self, num_classes, pretrained=True, use_se=True, use_window=True, use_axial=True):
        super().__init__()
        w = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_v2_s(weights=w).features
        self.projection = nn.Conv2d(1280, EMBED_DIM, 1, bias=False)
        self.se = SqueezeExcitation(EMBED_DIM) if use_se else nn.Identity()
        self.window_attention = WindowAttention(EMBED_DIM, NUM_HEADS, WINDOW_SIZE) if use_window else nn.Identity()
        self.axial_attention = AxialAttentionHeight(EMBED_DIM, NUM_HEADS) if use_axial else nn.Identity()
        self.dropout = nn.Dropout(DROPOUT); self.classifier = nn.Linear(EMBED_DIM, num_classes)
    def forward(self, x):
        x = self.projection(self.backbone(x)); x = self.se(x); x = self.window_attention(x); x = self.axial_attention(x)
        return self.classifier(self.dropout(x.mean(dim=(2, 3))))
    def backbone_parameters(self): return list(self.backbone.parameters())
    def head_parameters(self):
        return [p for m in [self.projection, self.se, self.window_attention, self.axial_attention, self.classifier] for p in m.parameters()]

class BaselineNet(nn.Module):
    """Standard torchvision classifiers for the retrained baselines (same protocol: frozen body, then fine-tuning)."""
    def __init__(self, name, num_classes, pretrained=True):
        super().__init__(); self.name = name
        W = lambda e: (e.DEFAULT if pretrained else None)
        if name == "ResNet50":
            self.net = models.resnet50(weights=W(models.ResNet50_Weights)); self.net.fc = nn.Linear(self.net.fc.in_features, num_classes); self.head = self.net.fc
        elif name == "EfficientNetV2-S":
            self.net = models.efficientnet_v2_s(weights=W(models.EfficientNet_V2_S_Weights))
            self.net.classifier[-1] = nn.Linear(self.net.classifier[-1].in_features, num_classes); self.head = self.net.classifier
        elif name == "ConvNeXt-T":
            self.net = models.convnext_tiny(weights=W(models.ConvNeXt_Tiny_Weights))
            self.net.classifier[-1] = nn.Linear(self.net.classifier[-1].in_features, num_classes); self.head = self.net.classifier
        elif name == "ViT-B/16":
            self.net = models.vit_b_16(weights=W(models.ViT_B_16_Weights))
            self.net.heads.head = nn.Linear(self.net.heads.head.in_features, num_classes); self.head = self.net.heads
        else: raise ValueError(name)
    def forward(self, x): return self.net(x)
    def head_parameters(self): return list(self.head.parameters())
    def backbone_parameters(self):
        hid = {id(p) for p in self.head.parameters()}; return [p for p in self.net.parameters() if id(p) not in hid]
