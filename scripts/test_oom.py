import os
import torch
import yaml
import time
from munch import Munch
import numpy as np

# Suppress warnings
import warnings
warnings.simplefilter('ignore')

from accelerate import Accelerator
from meldataset import build_dataloader
from utils import get_data_path_list, length_to_mask, maximum_path, mask_from_lens, log_norm, recursive_munch
from models import build_model, load_ASR_models, load_F0_models
from Utils.PLBERT.util import load_plbert
from losses import MultiResolutionSTFTLoss, GeneratorLoss, DiscriminatorLoss, WavLMLoss
from optimizers import build_optimizer
import torch.nn.functional as F
from torch import nn

from Modules.slmadv import SLMAdversarialLoss
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def get_memory_info():
    mem_alloc = torch.cuda.max_memory_allocated() / (1024**2)
    torch.cuda.reset_peak_memory_stats()
    return f"{mem_alloc:.1f} MB"

def test_phase(phase_name, model, optimizer, batch, device, sr, n_down, stft_loss, gl, dl, wl, multispeaker, max_len, config, phase_flags):
    print(f"--- Menguji Fase: {phase_name} ---")
    torch.cuda.empty_cache()
    
    tma_active = phase_flags.get('tma', False)
    diff_active = phase_flags.get('diff', False)
    joint_active = phase_flags.get('joint', False)
    is_stage2 = phase_flags.get('stage2', False)

    _ = [model[key].train() for key in model]

    # Unpack batch
    waves = batch[0]
    batch_t = [b.to(device) for b in batch[1:]]
    texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch_t

    try:
        if not is_stage2:
            # ================= STAGE 1 LOGIC =================
            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** n_down)).to('cuda')
                text_mask = length_to_mask(input_lengths).to(texts.device)

            ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)
            s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)

            with torch.no_grad():
                attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                attn_mask = (attn_mask < 1)
            s2s_attn.masked_fill_(attn_mask, 0.0)
                        
            with torch.no_grad():
                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            t_en = model.text_encoder(texts, input_lengths, text_mask)
            import random
            asr = (t_en @ s2s_attn) if bool(random.getrandbits(1)) else (t_en @ s2s_attn_mono)
            
            mel_len = min([int(mel_input_length.min().item() / 2 - 1), max_len // 2])
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)
        
            en, gt, wav, st = [], [], [], []
            for bib in range(len(mel_input_length)):
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

            with torch.no_grad():    
                real_norm = log_norm(gt.unsqueeze(1)).squeeze(1).detach()
                F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
                
            s = model.style_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))
            y_rec = model.decoder(en, F0_real, real_norm, s)
            
            if tma_active:
                optimizer.zero_grad()
                d_loss = dl(wav.detach().unsqueeze(1).float(), y_rec.detach()).mean()
                d_loss.backward()
                optimizer.step('msd')
                optimizer.step('mpd')

            optimizer.zero_grad()
            loss_mel = stft_loss(y_rec.squeeze(1), wav.detach())
            
            if tma_active:
                loss_s2s = 0
                for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
                    loss_s2s += F.cross_entropy(_s2s_pred[:_text_length], _text_input[:_text_length])
                loss_s2s /= texts.size(0)
                loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10
                loss_gen_all = gl(wav.detach().unsqueeze(1).float(), y_rec).mean()
                loss_slm = wl(wav.detach(), y_rec).mean()
                g_loss = loss_mel + loss_mono + loss_s2s + loss_gen_all + loss_slm
            else:
                g_loss = loss_mel
            
            g_loss.backward()
            optimizer.step('text_encoder')
            optimizer.step('style_encoder')
            optimizer.step('decoder')
            if tma_active: 
                optimizer.step('text_aligner')
                optimizer.step('pitch_extractor')
                
        else:
            # ================= STAGE 2 LOGIC =================
            model.predictor_encoder.load_state_dict(model.style_encoder.state_dict())
            model.predictor_encoder.train()
            
            sampler = DiffusionSampler(model.diffusion.diffusion, sampler=ADPM2Sampler(), sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), clamp=False)
            
            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                mel_mask = length_to_mask(mel_input_length).to(device)
                text_mask = length_to_mask(input_lengths).to(texts.device)

                _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)

                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
                t_en = model.text_encoder(texts, input_lengths, text_mask)
                asr = (t_en @ s2s_attn_mono)
                d_gt = s2s_attn_mono.sum(axis=-1).detach()
                
                if multispeaker and diff_active:
                    ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                    ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                    ref = torch.cat([ref_ss, ref_sp], dim=1)

            ss, gs = [], []
            for bib in range(len(mel_input_length)):
                mel = mels[bib, :, :mel_input_length[bib]]
                ss.append(model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1)))
                gs.append(model.style_encoder(mel.unsqueeze(0).unsqueeze(1)))

            s_dur = torch.stack(ss).squeeze()
            gs = torch.stack(gs).squeeze() 
            s_trg = torch.cat([gs, s_dur], dim=-1).detach()

            bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 
            
            if diff_active:
                num_steps = np.random.randint(3, 5)
                if multispeaker:
                    s_preds = sampler(noise=torch.randn_like(s_trg).unsqueeze(1).to(device), embedding=bert_dur, embedding_scale=1, features=ref, embedding_mask_proba=0.1, num_steps=num_steps).squeeze(1)
                    loss_diff = model.diffusion(s_trg.unsqueeze(1), embedding=bert_dur, features=ref).mean() 
                    loss_sty = F.l1_loss(s_preds, s_trg.detach()) 
                else:
                    s_preds = sampler(noise=torch.randn_like(s_trg).unsqueeze(1).to(device), embedding=bert_dur, embedding_scale=1, embedding_mask_proba=0.1, num_steps=num_steps).squeeze(1)                    
                    loss_diff = model.diffusion.module.diffusion(s_trg.unsqueeze(1), embedding=bert_dur).mean() 
                    loss_sty = F.l1_loss(s_preds, s_trg.detach()) 
            else:
                loss_sty = loss_diff = 0

            d, p = model.predictor(d_en, s_dur, input_lengths, s2s_attn_mono, text_mask)
            
            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)
            en, gt, st, p_en, wav = [], [], [], [], []

            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)
                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[bib, :, random_start:random_start+mel_len])
                p_en.append(p[bib, :, random_start:random_start+mel_len])
                gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
                y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y).to(device))
                random_start = np.random.randint(0, mel_length - mel_len_st)
                st.append(mels[bib, :, (random_start * 2):((random_start+mel_len_st) * 2)])
                
            wav = torch.stack(wav).float().detach()
            en = torch.stack(en)
            p_en = torch.stack(p_en)
            gt = torch.stack(gt).detach()
            st = torch.stack(st).detach()

            s_dur = model.predictor_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))
            s = model.style_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))
            
            with torch.no_grad():
                F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                F0 = F0.reshape(F0.shape[0], F0.shape[1] * 2, F0.shape[2], 1).squeeze()
                N_real = log_norm(gt.unsqueeze(1)).squeeze(1)
                y_rec_gt_pred = model.decoder(en, F0_real, N_real, s)
                wav = wav.unsqueeze(1) if joint_active else y_rec_gt_pred

            F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)
            y_rec = model.decoder(en, F0_fake, N_fake, s)

            loss_F0_rec =  (F.smooth_l1_loss(F0_real, F0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)

            if diff_active:
                optimizer.zero_grad()
                d_loss = dl(wav.detach(), y_rec.detach()).mean()
                d_loss.backward()
                optimizer.step('msd')
                optimizer.step('mpd')

            optimizer.zero_grad()
            loss_mel = stft_loss(y_rec, wav)
            loss_gen_all = gl(wav, y_rec).mean() if diff_active else 0
            loss_lm = wl(wav.detach().squeeze(1), y_rec.squeeze(1)).mean()

            loss_ce = loss_dur = 0
            for _s2s_pred, _text_input, _text_length in zip(d, d_gt, input_lengths):
                _s2s_pred = _s2s_pred[:_text_length, :]
                _text_input = _text_input[:_text_length].long()
                _s2s_trg = torch.zeros_like(_s2s_pred)
                for p_idx in range(_s2s_trg.shape[0]):
                    _s2s_trg[p_idx, :_text_input[p_idx]] = 1
                _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                loss_dur += F.l1_loss(_dur_pred[1:_text_length-1], _text_input[1:_text_length-1])
                loss_ce += F.binary_cross_entropy_with_logits(_s2s_pred.flatten(), _s2s_trg.flatten())

            loss_ce /= texts.size(0)
            loss_dur /= texts.size(0)

            g_loss = loss_mel + loss_F0_rec + loss_ce + loss_norm_rec + loss_dur + loss_gen_all + loss_lm + loss_sty + loss_diff
            g_loss.backward()

            optimizer.step('bert_encoder')
            optimizer.step('bert')
            optimizer.step('predictor')
            optimizer.step('predictor_encoder')
            
            if diff_active: optimizer.step('diffusion')
            if joint_active:
                optimizer.step('style_encoder')
                optimizer.step('decoder')

        print(f"[SUCCESS] {phase_name} berjalan tanpa OOM! Peak Memory: {get_memory_info()}")
        return True

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"[FAIL OOM] {phase_name} gagal karena Out of Memory. ({get_memory_info()})")
        else:
            print(f"[FAIL ERROR] {phase_name} gagal karena error lain: {e}")
        return False


def main():
    config_path = 'Configs/config.yml'
    config = yaml.safe_load(open(config_path))
    device = 'cuda'
    
    batch_size = config.get('batch_size', 2)
    max_len = config.get('max_len', 400)
    
    print(f"Menginisialisasi Dataloader dengan batch_size: {batch_size}, max_len: {max_len}...")
    
    data_params = config.get('data_params', None)
    train_list, val_list = get_data_path_list(data_params['train_data'], data_params['val_data'])

    train_dataloader = build_dataloader(train_list,
                                        data_params['root_path'],
                                        OOD_data=data_params['OOD_data'],
                                        min_length=data_params['min_length'],
                                        batch_size=batch_size,
                                        num_workers=0,
                                        dataset_config={},
                                        device=device)

    # Get one batch
    batch = next(iter(train_dataloader))
    
    print("Memuat Model (Ini mungkin memakan sedikit VRAM)...")
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    
    _ = [model[key].to(device) for key in model]
    
    # Init optimizer dummy
    scheduler_params = {
        "max_lr": 1e-4, "pct_start": 0.0, "epochs": 200, "steps_per_epoch": 10,
    }
    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                  scheduler_params_dict={key: scheduler_params.copy() for key in model},
                               lr=1e-4)

    n_down = model.text_aligner.n_down
    stft_loss = MultiResolutionSTFTLoss().to(device)
    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = WavLMLoss(model_params.slm.model, model.wd, 24000, model_params.slm.sr).to(device)

    print(f"Base Memory Terpakai: {get_memory_info()}")

    # Test Phase 1: Pre-TMA
    test_phase("1. Pre-TMA", model, optimizer, batch, device, 24000, n_down, stft_loss, gl, dl, wl, multispeaker, max_len, config, {'tma': False})
    
    # Test Phase 2: TMA
    test_phase("2. TMA Active", model, optimizer, batch, device, 24000, n_down, stft_loss, gl, dl, wl, multispeaker, max_len, config, {'tma': True})

    # Add DP wrapper required for stage 2
    model.predictor_encoder.load_state_dict(model.style_encoder.state_dict())
    model.predictor_encoder.to(device)
    for key in model:
        if key != "mpd" and key != "msd" and key != "wd":
            model[key] = MyDataParallel(model[key])
    gl = MyDataParallel(gl)
    dl = MyDataParallel(dl)
    wl = MyDataParallel(wl)
    
    # Reinit optimizer with DP models
    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                  scheduler_params_dict={key: scheduler_params.copy() for key in model},
                               lr=1e-4)

    print("\n[Pindah ke Model Arsitektur Tahap 2]")
    
    # Test Phase 3: Pre-Diff
    test_phase("3. Pre-Diff (Stage 2)", model, optimizer, batch, device, 24000, n_down, stft_loss, gl, dl, wl, multispeaker, max_len, config, {'stage2': True})

    # Test Phase 4: Diff Active
    test_phase("4. Diff Active (Stage 2)", model, optimizer, batch, device, 24000, n_down, stft_loss, gl, dl, wl, multispeaker, max_len, config, {'stage2': True, 'diff': True})

    # Test Phase 5: Joint Active
    test_phase("5. Joint Active (Stage 2)", model, optimizer, batch, device, 24000, n_down, stft_loss, gl, dl, wl, multispeaker, max_len, config, {'stage2': True, 'diff': True, 'joint': True})

if __name__ == "__main__":
    main()
