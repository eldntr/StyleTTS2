#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest import mock
from munch import Munch, recursive_munch

# 1. Mock transformers.AutoModel to avoid downloading WavLM (which can be 400MB+ or fail if offline)
class MockWavLM(torch.nn.Module):
    def __init__(self, hidden_dim=768, n_layers=13):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_values, output_hidden_states=True):
        batch_size, seq_len = input_values.shape
        # WavLM downsamples audio roughly by a factor of 320
        downsampled_len = max(1, seq_len // 320)
        hidden_states = []
        for _ in range(self.n_layers):
            # Create dummy features
            states = torch.zeros(batch_size, downsampled_len, self.hidden_dim, device=input_values.device)
            # Add dummy_param to hook parameters to gradients
            states = states + self.dummy_param
            hidden_states.append(states)
        
        class Output:
            pass
        out = Output()
        out.hidden_states = tuple(hidden_states)
        return out

mock_auto_model = mock.patch('transformers.AutoModel.from_pretrained', return_value=MockWavLM())
mock_auto_model.start()

# Now import project modules
from models import build_model, load_ASR_models, load_F0_models
from Utils.PLBERT.util import load_plbert, CustomAlbert
from transformers import AlbertConfig
from losses import MultiResolutionSTFTLoss, GeneratorLoss, DiscriminatorLoss, WavLMLoss
from optimizers import build_optimizer
from utils import length_to_mask, mask_from_lens, maximum_path, log_norm

# Override loaders to be robust to missing checkpoints
def robust_load_plbert(BERT_path):
    try:
        return load_plbert(BERT_path)
    except Exception as e:
        print(f"[*] Fallback: Loading random Albert model (error: {e})")
        config_path = os.path.join(BERT_path, "config.yml")
        plbert_config = yaml.safe_load(open(config_path))
        albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
        bert = CustomAlbert(albert_base_configuration)
        return bert

def robust_load_F0_models(path):
    F0_model = JDCNet(num_class=1, seq_len=192)
    try:
        params = torch.load(path, map_location='cpu')['net']
        F0_model.load_state_dict(params)
        print("[*] JDCNet loaded from checkpoint successfully.")
    except Exception as e:
        print(f"[*] Fallback: Initializing random JDCNet (error: {e})")
    F0_model.train()
    return F0_model

def robust_load_ASR_models(ASR_MODEL_PATH, ASR_MODEL_CONFIG):
    with open(ASR_MODEL_CONFIG) as f:
        config = yaml.safe_load(f)
    model_config = config['model_params']
    model = ASRCNN(**model_config)
    try:
        params = torch.load(model_path, map_location='cpu')['model']
        model.load_state_dict(params)
        print("[*] ASR model loaded from checkpoint successfully.")
    except Exception as e:
        print(f"[*] Fallback: Initializing random ASR model (error: {e})")
    model.train()
    return model


def check_tensor_nan_inf(tensor, name):
    if torch.isnan(tensor).any():
        return f"{name} contains NaN"
    if torch.isinf(tensor).any():
        return f"{name} contains Inf"
    return None

def test_batch_size(batch_size, device='cpu', config_path='Configs/config.yml'):
    print(f"\n==========================================")
    print(f"Testing Batch Size: {batch_size} on Device: {device}")
    print(f"==========================================")
    
    # 2. Load configurations
    config = yaml.safe_load(open(config_path))
    model_params = recursive_munch(config['model_params'])
    loss_params = Munch(config['loss_params'])
    sr = config['preprocess_params'].get('sr', 24000)
    max_len = config.get('max_len', 200)
    multispeaker = model_params.multispeaker
    n_token = model_params.n_token

    # Initialize models
    print("[*] Initializing models...")
    text_aligner = robust_load_ASR_models(config.get('ASR_path'), config.get('ASR_config'))
    pitch_extractor = robust_load_F0_models(config.get('F0_path'))
    plbert = robust_load_plbert(config.get('PLBERT_dir'))
    
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    for k in model:
        model[k] = model[k].to(device)
        model[k].train()
        
    try:
        n_down = model.text_aligner.n_down
    except:
        n_down = model.text_aligner.module.n_down
        
    # Setup losses
    stft_loss = MultiResolutionSTFTLoss().to(device)
    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = WavLMLoss(model_params.slm.model, model.wd, sr, model_params.slm.sr).to(device)

    # Setup optimizers
    scheduler_params = {
        "max_lr": float(config['optimizer_params'].get('lr', 1e-4)),
        "pct_start": float(config['optimizer_params'].get('pct_start', 0.0)),
        "epochs": 1,
        "steps_per_epoch": 10,
    }
    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                scheduler_params_dict={key: scheduler_params.copy() for key in model},
                                lr=float(config['optimizer_params'].get('lr', 1e-4)))

    # 3. Create dummy data matching collated output format
    print("[*] Creating dummy data batch...")
    max_text_len = 50
    # mel lengths must be even and large enough (e.g. 192)
    max_mel_len = 192 
    
    # texts: [B, T_text]
    texts = torch.randint(1, n_token, (batch_size, max_text_len), dtype=torch.long, device=device)
    input_lengths = torch.randint(20, max_text_len, (batch_size,), dtype=torch.long, device=device)
    # Ensure they are sorted descending for packing if necessary (or text_encoder requires it)
    input_lengths, perm_index = input_lengths.sort(descending=True)
    texts = texts[perm_index]
    
    # mels: [B, n_mels, T_mel]
    n_mels = model_params.n_mels
    mels = torch.randn(batch_size, n_mels, max_mel_len, device=device)
    mel_input_length = torch.randint(120, max_mel_len, (batch_size,), dtype=torch.long, device=device)
    mel_input_length = (mel_input_length // 2) * 2 # make even
    
    # waves: list of numpy arrays representing raw wave
    waves = []
    for bib in range(batch_size):
        length = int(mel_input_length[bib].item()) * 300 # hop length is 300
        waves.append(np.random.randn(length).astype(np.float32))

    # 4. Perform Forward and Backward Steps
    print("[*] Starting check iteration...")
    start_time = time.time()
    
    try:
        # Reset memory tracking if CUDA
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            
        optimizer.zero_grad()
        
        # Prepare masks
        mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        # 1. Text aligner forward
        ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)

        s2s_attn = s2s_attn.transpose(-1, -2)
        s2s_attn = s2s_attn[..., 1:]
        s2s_attn = s2s_attn.transpose(-1, -2)

        attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
        attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
        attn_mask = (attn_mask < 1)

        s2s_attn.masked_fill_(attn_mask, 0.0)
                    
        mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

        # 2. Text Encoder forward
        t_en = model.text_encoder(texts, input_lengths, text_mask)

        # 50% of chance of using monotonic version
        if bool(random.getrandbits(1)):
            asr = (t_en @ s2s_attn)
        else:
            asr = (t_en @ s2s_attn_mono)

        # Clip processing
        mel_len = min([int(mel_input_length.min().item() / 2 - 1), max_len // 2])
        mel_len_st = int(mel_input_length.min().item() / 2 - 1)
    
        en = []
        gt = []
        wav = []
        st = []
        
        for bib in range(batch_size):
            mel_length = int(mel_input_length[bib].item() / 2)
            random_start = np.random.randint(0, mel_length - mel_len)
            en.append(asr[bib, :, random_start:random_start+mel_len])
            gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])

            y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
            wav.append(torch.from_numpy(y).to(device))
            
            random_start = np.random.randint(0, mel_length - mel_len_st)
            st.append(mels[bib, :, (random_start * 2):((random_start+mel_len_st) * 2)])

        en = torch.stack(en)
        gt = torch.stack(gt).detach()
        st = torch.stack(st).detach()
        wav = torch.stack(wav).float().detach()

        # F0 and Style Encoding
        real_norm = log_norm(gt.unsqueeze(1)).squeeze(1).detach()
        F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
        s = model.style_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))
        
        # Decoder reconstruction
        y_rec = model.decoder(en, F0_real, real_norm, s)

        # 5. Check outputs and losses for NaN
        errors = []
        nan_check = check_tensor_nan_inf(y_rec, "decoder output (y_rec)")
        if nan_check: errors.append(nan_check)

        # Test discriminators loss
        d_loss = dl(wav.detach().unsqueeze(1).float(), y_rec.detach()).mean()
        nan_check = check_tensor_nan_inf(d_loss, "discriminator loss (d_loss)")
        if nan_check: errors.append(nan_check)

        # Test generator loss
        loss_mel = stft_loss(y_rec.squeeze(), wav.detach())
        loss_s2s = 0
        for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
            loss_s2s += F.cross_entropy(_s2s_pred[:_text_length], _text_input[:_text_length])
        loss_s2s /= texts.size(0)
        loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10
        loss_gen_all = gl(wav.detach().unsqueeze(1).float(), y_rec).mean()
        loss_slm = wl(wav.detach(), y_rec).mean()

        g_loss = loss_params.lambda_mel * loss_mel + \
                 loss_params.lambda_mono * loss_mono + \
                 loss_params.lambda_s2s * loss_s2s + \
                 loss_params.lambda_gen * loss_gen_all + \
                 loss_params.lambda_slm * loss_slm

        nan_check = check_tensor_nan_inf(g_loss, "generator loss (g_loss)")
        if nan_check: errors.append(nan_check)

        # Backward passes
        print("[*] Running backward pass for Discriminator...")
        d_loss.backward(retain_graph=True)
        print("[*] Running backward pass for Generator...")
        g_loss.backward()

        # Check gradients for NaN/Inf
        print("[*] Checking gradients for NaNs/Infs...")
        for name, p in model.decoder.named_parameters():
            if p.grad is not None:
                grad_check = check_tensor_nan_inf(p.grad, f"Decoder parameter gradient '{name}'")
                if grad_check:
                    errors.append(grad_check)
                    break
        
        duration = time.time() - start_time
        print(f"[+] Iteration completed in {duration:.3f} seconds.")
        
        if device == 'cuda':
            max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"[+] Peak CUDA Memory allocated: {max_mem:.2f} MB")
        else:
            max_mem = 0
            
        if errors:
            print("[!] Check Failed: Numerical issues detected!")
            for err in errors:
                print(f"    - {err}")
            return False, errors, max_mem
        else:
            print("[+] Check Passed: No OOM, No NaNs, No Infs!")
            return True, [], max_mem

    except torch.cuda.OutOfMemoryError as oom:
        print(f"[!] Check Failed: Out of Memory (OOM) error occurred!")
        return False, ["CUDA Out of Memory"], 0
    except Exception as e:
        print(f"[!] Check Failed: Unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False, [str(e)], 0

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test multiple batch sizes
    batch_sizes = [2, 4, 8, 16, 32]
    results = {}
    
    for bs in batch_sizes:
        success, errors, mem = test_batch_size(bs, device=device)
        results[bs] = {
            "success": success,
            "errors": errors,
            "memory_mb": mem
        }
        # Clear cache between runs
        if device == 'cuda':
            torch.cuda.empty_cache()
            
    print("\n" + "="*50)
    print("SUMMARY REPORT")
    print("="*50)
    for bs, res in results.items():
        status = "PASSED" if res["success"] else "FAILED"
        mem_info = f" ({res['memory_mb']:.1f} MB)" if device == 'cuda' and res["success"] else ""
        err_info = f" -> Errors: {res['errors']}" if res['errors'] else ""
        print(f"Batch Size {bs:2d}: {status}{mem_info}{err_info}")
    print("="*50)
