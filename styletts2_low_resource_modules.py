#coding:utf-8
"""
Arsitektur Tambahan untuk StyleTTS2 Cross-Lingual Adaptation (Indo-Jawa)
Implementasi: LPEP, PPIM, Weight Cloning, & 2-Stage PEFT Controller
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LPEP(nn.Module):
    """
    Language Phoneme Embedding Processor (LPEP)
    Menggunakan eksplisit lang_id tanpa fitur Phoible untuk efisiensi parameter.
    """
    def __init__(self, n_symbols, hidden_dim, n_langs=2, lang_emb_dim=16, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.phone_emb = nn.Embedding(n_symbols, hidden_dim)
        self.lang_emb = nn.Embedding(n_langs, lang_emb_dim)

        in_dim = hidden_dim + lang_emb_dim

        # Sirkuit adaptasi penentu deviasi fonem
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Gerbang penentu aktivasi jalur Jawa (Sigmoid)
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def _normalize_lang_id(self, tokens, lang_id):
        batch_size = tokens.size(0)
        device = tokens.device
        if lang_id is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if not torch.is_tensor(lang_id):
            lang_id = torch.tensor(lang_id, dtype=torch.long, device=device)
        else:
            lang_id = lang_id.to(device=device, dtype=torch.long)
        if lang_id.dim() == 0:
            lang_id = lang_id.expand(batch_size)
        return lang_id.view(batch_size)

    def forward(self, tokens, lang_id=None):
        phone = self.phone_emb(tokens) # [B, T, C]
        lang_id = self._normalize_lang_id(tokens, lang_id)
        lang = self.lang_emb(lang_id).unsqueeze(1).expand(-1, tokens.size(1), -1) # [B, T, C_lang]

        z = torch.cat([phone, lang], dim=-1) # Konkatenasi eksplisit
        delta = self.proj(z)
        gate = self.gate(z)
        LPEP.last_gate_val = gate.mean().item()
        
        # Jalur residual terkontrol
        return self.out_norm(phone + gate * delta)


class PPIM(nn.Module):
    """
    Phoneme Prosody Interaction Module (PPIM)
    Menggunakan Cross-Attention untuk menginfusi vektor gaya prosodi s_p ke teks.
    """
    def __init__(self, hidden_dim, style_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.style_proj = nn.Linear(style_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        
        # Zero-Gate Initialization untuk PPIM
        for layer in self.gate:
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.weight, 0.0)
                nn.init.constant_(layer.bias, -4.0)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, text_hidden, style, mask=None):
        # text_hidden shape: [B, C, T] -> transpose ke [B, T, C] untuk modul Attention
        h = text_hidden.transpose(1, 2)
        time_steps = h.size(1)

        style_token = self.style_proj(style).unsqueeze(1).expand(-1, time_steps, -1)
        attn_out, _ = self.attn(
            query=h,
            key=style_token,
            value=style_token,
            key_padding_mask=mask if mask is not None else None,
            need_weights=False,
        )

        gate = self.gate(torch.cat([h, attn_out], dim=-1))
        PPIM.last_gate_val = gate.mean().item()
        h = self.norm1(h + self.dropout(gate * attn_out))
        h = self.norm2(h + self.dropout(self.ffn(h)))

        if mask is not None:
            h = h.masked_fill(mask.unsqueeze(-1), 0.0)

        return h.transpose(1, 2) # Kembalikan ke [B, C, T]


# =====================================================================
# WEIGHT CLONING & INITIALIZATION LOGIC
# =====================================================================

def inject_pretrained_embeddings_to_lpep(base_model_indo, lpep_module):
    """
    Menyalin bobot embedding dari checkpoint bahasa Indonesia lama ke LPEP baru,
    sekaligus menerapkan skenario Zero-Gate initialization.
    """
    print(">>> Memulai Proses Penyalinan Bobot (Weight Cloning)...")
    
    # Ambil data bobot dari model standar lama
    try:
        old_weight = base_model_indo.text_encoder.embedding.weight.data
    except AttributeError:
        # Menangani pembungkus modul DDP
        old_weight = base_model_indo.text_encoder.module.embedding.weight.data
        
    # Salin paksa ke phone_emb di LPEP
    lpep_module.phone_emb.weight.data.copy_(old_weight)
    print(f"    [SUKSES] Menyalin parameter embedding berdimensi: {old_weight.shape}")
    
    # Zero-Gate Setup: Set bias gerbang Sigmoid ke nilai negatif besar (-4.0)
    # Ini memastikan gerbang tertutup rapat di awal latih (Identity Function)
    for layer in lpep_module.gate:
        if isinstance(layer, nn.Linear):
            nn.init.constant_(layer.weight, 0.0)
            nn.init.constant_(layer.bias, -4.0)
    print("    [SUKSES] Gerbang LPEP Berhasil Dikunci pada Nilai ~0 (Mencegah Amnesia).")
    
    return lpep_module


# =====================================================================
# 2-STAGE PEFT PARAMETER CONTROLLER
# =====================================================================

def configure_peft_stage(nets, stage=1):
    """
    Mengontrol pembekuan (freezing) parameter model sesuai target tahap latihan.
    """
    # 1. Bekukan seluruh parameter di awal
    for module_name, module in nets.items():
        if isinstance(module, nn.Module):
            for param in module.parameters():
                param.requires_grad = False
                
    # 2. Buka gerbang gradient secara lokal sesuai skenario tahap latihan
    if stage == 1:
        print(">>> [PEFT CONFIG] Mengaktifkan Parameter STAGE 1 (Acoustic Alignment)...")
        # Unfreeze LPEP
        text_encoder = getattr(nets, "text_encoder", None)
        if text_encoder is not None:
            text_encoder_module = getattr(text_encoder, "module", text_encoder)
            if hasattr(text_encoder_module, "embedding"):
                for param in text_encoder_module.embedding.parameters():
                    param.requires_grad = True
                    
        # Unfreeze Decoder & Discriminators
        for param in nets.decoder.parameters():
            param.requires_grad = True
        for param in nets.mpd.parameters():
            param.requires_grad = True
        for param in nets.msd.parameters():
            param.requires_grad = True
        if hasattr(nets, "wd"):
            for param in nets.wd.parameters():
                param.requires_grad = True
                
        # Set mode
        nets.text_encoder.train()
        nets.decoder.train()
        nets.mpd.train()
        nets.msd.train()
        if hasattr(nets, "wd"):
            nets.wd.train()
            
    elif stage == 2:
        print(">>> [PEFT CONFIG] Mengaktifkan Parameter STAGE 2 (Prosody Adaptation)...")
        # Unfreeze PPIM
        ppim = getattr(nets, "ppim", None)
        if ppim is not None:
            ppim_module = getattr(ppim, "module", ppim)
            for param in ppim_module.parameters():
                param.requires_grad = True
                
        # Unfreeze Prosody Predictor & Style Diffusion
        for param in nets.predictor.parameters():
            param.requires_grad = True
        for param in nets.diffusion.parameters():
            param.requires_grad = True
            
        # Pastikan sirkuit akustik terkunci rapat (Freeze Stage 1 output)
        nets.text_encoder.eval()
        nets.decoder.eval()
        
        # Set mode latihan untuk komponen aktif
        if ppim is not None:
            ppim.train()
        nets.predictor.train()
        nets.diffusion.train()
        
    else:
        raise ValueError(f"Skenario Tahap {stage} tidak terdefinisi!")
