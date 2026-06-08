import torch
import torch.nn as nn
from peft_modules import LoRALinear, LoRAConv1d, StyleAdapter, PrefixTuningWrapper, DurationAlignmentAdapter

def get_inner_module(module):
    """
    Helper to unwrap DataParallel / MyDataParallel models.
    """
    if hasattr(module, 'module'):
        return module.module
    return module

def wrap_module_with_lora(module, rank=8):
    """
    Recursively replaces all nn.Linear and nn.Conv1d layers in a module with their LoRA counterparts.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if child.__class__.__name__ == 'LinearNorm':
                child.linear_layer = LoRALinear(child.linear_layer, rank=rank)
            else:
                setattr(module, name, LoRALinear(child, rank=rank))
        elif isinstance(child, nn.Conv1d):
            setattr(module, name, LoRAConv1d(child, rank=rank))
        else:
            wrap_module_with_lora(child, rank=rank)

# Wrapper Classes for Non-Invasive Adapter Injection

class StyleEncoderPEFTWrapper(nn.Module):
    def __init__(self, original_encoder, style_adapter):
        super().__init__()
        self.original_encoder = original_encoder
        self.style_adapter = style_adapter
        
    def forward(self, x):
        s = self.original_encoder(x)
        return self.style_adapter(s)

class BertEncoderPEFTWrapper(nn.Module):
    def __init__(self, original_bert_encoder, duration_adapter):
        super().__init__()
        self.original_bert_encoder = original_bert_encoder
        self.duration_adapter = duration_adapter
        
    def forward(self, x):
        out = self.original_bert_encoder(x)
        out_transposed = out.transpose(-1, -2)
        adapted = self.duration_adapter(out_transposed)
        return adapted.transpose(-1, -2)

class DurationEncoderPEFTWrapper(nn.Module):
    def __init__(self, original_duration_encoder, prefix_wrapper):
        super().__init__()
        self.original_duration_encoder = original_duration_encoder
        self.prefix_wrapper = prefix_wrapper
        
    def forward(self, x, style, text_lengths, m):
        x_t = x.transpose(-1, -2)
        x_prefixed = self.prefix_wrapper(x_t)
        x_final = x_prefixed.transpose(-1, -2)
        
        prefix_len = self.prefix_wrapper.prefix_len
        m_extended = torch.cat([torch.zeros(m.size(0), prefix_len, dtype=torch.bool, device=m.device), m], dim=1)
        text_lengths_extended = text_lengths + prefix_len
        
        out = self.original_duration_encoder(x_final, style, text_lengths_extended, m_extended)
        return out[:, prefix_len:, :]

class PEFTStyleTTS2:
    def __init__(self, model_dict, mode='A', rank=8):
        self.model_dict = model_dict
        self.mode = mode
        self.rank = rank
        self.inject_adapters()
        
    def inject_adapters(self):
        # 1. Scenario A: Selective LoRA on Prosody Predictor
        if self.mode == 'A':
            print("PEFT: Injecting Selective LoRA (Scenario A) into Prosody Predictor...")
            predictor_inner = get_inner_module(self.model_dict.predictor)
            wrap_module_with_lora(predictor_inner, rank=self.rank)
            
        # 2. Scenario B: Style-Adapter (Residual MLP) on Style Encoders
        elif self.mode == 'B':
            print("PEFT: Injecting Style Adapter (Scenario B)...")
            self.style_adapter_acoustic = StyleAdapter(style_dim=128)
            self.style_adapter_prosodic = StyleAdapter(style_dim=128)
            
            style_encoder_inner = get_inner_module(self.model_dict.style_encoder)
            predictor_encoder_inner = get_inner_module(self.model_dict.predictor_encoder)
            
            if hasattr(self.model_dict.style_encoder, 'module'):
                self.model_dict.style_encoder.module = StyleEncoderPEFTWrapper(
                    style_encoder_inner, self.style_adapter_acoustic
                )
            else:
                self.model_dict.style_encoder = StyleEncoderPEFTWrapper(
                    style_encoder_inner, self.style_adapter_acoustic
                )
                
            if hasattr(self.model_dict.predictor_encoder, 'module'):
                self.model_dict.predictor_encoder.module = StyleEncoderPEFTWrapper(
                    predictor_encoder_inner, self.style_adapter_prosodic
                )
            else:
                self.model_dict.predictor_encoder = StyleEncoderPEFTWrapper(
                    predictor_encoder_inner, self.style_adapter_prosodic
                )
            
        # 3. Scenario C: Prefix Tuning on Duration Encoder
        elif self.mode == 'C':
            print("PEFT: Injecting Prefix Tuning (Scenario C)...")
            bert_encoder_inner = get_inner_module(self.model_dict.bert_encoder)
            feature_dim = bert_encoder_inner.out_features
            self.prefix_wrapper = PrefixTuningWrapper(prefix_len=10, feature_dim=feature_dim)
            
            predictor_inner = get_inner_module(self.model_dict.predictor)
            predictor_inner.text_encoder = DurationEncoderPEFTWrapper(
                predictor_inner.text_encoder, self.prefix_wrapper
            )
            
        # 4. Scenario D: Duration Alignment Adapter on Bert Encoder
        elif self.mode == 'D':
            print("PEFT: Injecting Duration Alignment Adapter (Scenario D)...")
            bert_encoder_inner = get_inner_module(self.model_dict.bert_encoder)
            feature_dim = bert_encoder_inner.out_features
            self.duration_adapter = DurationAlignmentAdapter(feature_dim=feature_dim)
            
            if hasattr(self.model_dict.bert_encoder, 'module'):
                self.model_dict.bert_encoder.module = BertEncoderPEFTWrapper(
                    bert_encoder_inner, self.duration_adapter
                )
            else:
                self.model_dict.bert_encoder = BertEncoderPEFTWrapper(
                    bert_encoder_inner, self.duration_adapter
                )
            
        # 5. Scenario E: Combination (Duration Adapter + Selective LoRA)
        elif self.mode == 'E':
            print("PEFT: Injecting Combination (Scenario E) - Duration Adapter + Selective LoRA...")
            # Inject Duration Adapter
            bert_encoder_inner = get_inner_module(self.model_dict.bert_encoder)
            feature_dim = bert_encoder_inner.out_features
            self.duration_adapter = DurationAlignmentAdapter(feature_dim=feature_dim)
            
            if hasattr(self.model_dict.bert_encoder, 'module'):
                self.model_dict.bert_encoder.module = BertEncoderPEFTWrapper(
                    bert_encoder_inner, self.duration_adapter
                )
            else:
                self.model_dict.bert_encoder = BertEncoderPEFTWrapper(
                    bert_encoder_inner, self.duration_adapter
                )
                
            # Inject LoRA
            predictor_inner = get_inner_module(self.model_dict.predictor)
            wrap_module_with_lora(predictor_inner, rank=self.rank)
            
        else:
            raise ValueError(f"Unknown PEFT mode: {self.mode}")
            
        # Freeze base models
        self.freeze_base_model()

    def freeze_base_model(self):
        print("PEFT: Freezing base model parameters...")
        for key in self.model_dict:
            module = self.model_dict[key]
            if not isinstance(module, nn.Module):
                continue
                
            for name, param in module.named_parameters():
                if 'lora_' in name or 'style_adapter' in name or 'prefix_embedding' in name or 'duration_adapter' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
        # Double check if any parameter in added classes requires grad
        if self.mode == 'B':
            for param in self.style_adapter_acoustic.parameters():
                param.requires_grad = True
            for param in self.style_adapter_prosodic.parameters():
                param.requires_grad = True
        elif self.mode == 'C':
            for param in self.prefix_wrapper.parameters():
                param.requires_grad = True
        elif self.mode == 'D':
            for param in self.duration_adapter.parameters():
                param.requires_grad = True
        elif self.mode == 'E':
            for param in self.duration_adapter.parameters():
                param.requires_grad = True

    def get_trainable_parameters(self):
        trainable = []
        for key in self.model_dict:
            module = self.model_dict[key]
            if isinstance(module, nn.Module):
                for name, param in module.named_parameters():
                    if param.requires_grad:
                        trainable.append((f"{key}.{name}", param))
        return trainable
