import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# Skenario A: LoRA Modules (Linear & Conv1d)
# ==========================================

class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=8, alpha=16):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # Freezing base weights
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        # LoRA weights
        self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # Base forward
        base_out = self.base_layer(x)
        # LoRA path
        lora_out = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return base_out + lora_out

class LoRAConv1d(nn.Module):
    def __init__(self, base_layer, rank=8, alpha=16):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_channels = base_layer.in_channels
        out_channels = base_layer.out_channels
        kernel_size = base_layer.kernel_size[0]
        stride = base_layer.stride[0]
        padding = base_layer.padding[0]
        
        # Freezing base weights
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        # LoRA weights (applied to convolution weight)
        self.lora_A = nn.Parameter(torch.zeros((rank, in_channels * kernel_size)))
        self.lora_B = nn.Parameter(torch.zeros((out_channels, rank)))
        
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def forward(self, x):
        # Base forward
        base_out = self.base_layer(x)
        
        # LoRA path using 1d conv operations
        # x shape: [B, C_in, T]
        # We can compute LoRA weight delta: W_delta = B @ A -> shape [C_out, C_in * K] -> view [C_out, C_in, K]
        # and do F.conv1d
        lora_weight = (self.lora_B @ self.lora_A).view(self.out_channels, self.in_channels, self.kernel_size) * self.scaling
        lora_out = F.conv1d(x, lora_weight, bias=None, stride=self.stride, padding=self.padding)
        
        return base_out + lora_out


# ==========================================
# Skenario B: Style Laten Adapter (Residual MLP)
# ==========================================

class StyleAdapter(nn.Module):
    def __init__(self, style_dim=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, style_dim)
        )
        # Initialize identity mapping (weights near zero so it starts as identity)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        
    def forward(self, s):
        # s shape: [B, style_dim]
        # Add residual connection
        return s + self.net(s)


# ==========================================
# Skenario C: Prefix Tuning Wrapper
# ==========================================

class PrefixTuningWrapper(nn.Module):
    def __init__(self, prefix_len=10, feature_dim=512):
        super().__init__()
        self.prefix_len = prefix_len
        self.prefix_embedding = nn.Parameter(torch.zeros(1, prefix_len, feature_dim))
        nn.init.normal_(self.prefix_embedding, std=0.02)
        
    def forward(self, x):
        # x shape: [B, T, feature_dim]
        batch_size = x.size(0)
        prefix = self.prefix_embedding.expand(batch_size, -1, -1)
        return torch.cat([prefix, x], dim=1)


# ==========================================
# Skenario D: Duration Alignment Adapter
# ==========================================

class DurationAlignmentAdapter(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.conv1 = nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm1d(feature_dim, affine=True)
        self.conv2 = nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1)
        
        # Initialize as close to identity as possible
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
    def forward(self, x):
        # x shape: [B, feature_dim, T] (from text encoder)
        res = x
        out = F.gelu(self.norm(self.conv1(x)))
        out = self.conv2(out)
        return res + out
