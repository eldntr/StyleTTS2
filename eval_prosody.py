import os
import sys
import torch
import yaml
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
from scipy.stats import pearsonr

# Add local path to import models and utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import build_model, load_ASR_models, load_F0_models
from utils import recursive_munch
from Utils.PLBERT.util import load_plbert
from text_utils import TextCleaner

def hz_to_semitones(f0, fmin=16.35):
    f0_semi = np.zeros_like(f0)
    idx = f0 > 0
    f0_semi[idx] = 12 * np.log2(f0[idx] / fmin)
    return f0_semi

def get_f0(wav_path, sr=24000):
    wav, _ = librosa.load(wav_path, sr=sr)
    # Extract pitch using librosa YIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        wav, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7'), 
        sr=sr,
        frame_length=2048,
        hop_length=300
    )
    # Fill NaNs with 0
    f0 = np.nan_to_num(f0)
    return f0

def evaluate_models(config_path, checkpoint_path, test_list_path, limit=20):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading config {config_path} on {device}...")
    config = yaml.safe_load(open(config_path))
    
    # Load model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    
    # Load checkpoint
    print(f"Loading checkpoint {checkpoint_path}...")
    params_whole = torch.load(checkpoint_path, map_location='cpu')
    params = params_whole['net']
    for key in model:
        if key in params:
            try:
                model[key].load_state_dict(params[key])
            except Exception as e:
                print(f"Loading non-strict for {key}: {e}")
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                model[key].load_state_dict(new_state_dict, strict=False)
                
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]
    
    textcleaner = TextCleaner()
    
    # Parse test list
    test_samples = []
    with open(test_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                test_samples.append((parts[0], parts[1]))
                
    print(f"Loaded {len(test_samples)} test samples. Evaluating on first {min(limit, len(test_samples))} samples...")
    test_samples = test_samples[:limit]
    
    rmse_semitones_list = []
    f0_corr_list = []
    vuv_error_list = []
    
    wav_root_path = config['data_params']['root_path']
    os.makedirs("eval_outputs", exist_ok=True)
    
    for i, (wav_name, text) in enumerate(tqdm(test_samples)):
        gt_wav_path = os.path.join(wav_root_path, wav_name)
        if not os.path.exists(gt_wav_path):
            print(f"GT wav not found: {gt_wav_path}")
            continue
            
        # Get style reference from gt
        wave, sr = librosa.load(gt_wav_path, sr=24000)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        
        # Style encoding
        wave_tensor = torch.from_numpy(audio).float().to(device)
        to_mel = torchaudio_transforms_mel(device)
        mel_tensor = to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - (-4)) / 4
        
        with torch.no_grad():
            ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))
            style = torch.cat([ref_s, ref_p], dim=1)
            
            # Synthesize
            tokens = textcleaner(text)
            tokens.insert(0, 0)
            tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
            
            text_mask = length_to_mask(input_lengths).to(device)
            t_en = model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 
            
            s = style[:, 128:]
            ref = style[:, :128]
            
            d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
            x, _ = model.predictor.lstm(d)
            duration = model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            
            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for k in range(pred_aln_trg.size(0)):
                pred_aln_trg[k, c_frame:c_frame + int(pred_dur[k].data)] = 1
                c_frame += int(pred_dur[k].data)
            
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
            
            # Breathy stops pitch conditioning
            use_breathy = model_params.get('use_breathy', False)
            breathy_mask_frames = None
            if use_breathy:
                breathy_chars = ['b', 'd', 'g', 'ɖ', 'ɟ']
                breathy_token_ids = [textcleaner.word_index_dictionary[char] for char in breathy_chars if char in textcleaner.word_index_dictionary]
                idx_d = textcleaner.word_index_dictionary.get('d', -1)
                idx_h = textcleaner.word_index_dictionary.get('h', -1)
                
                # We need get_breathy_mask_tensor logic
                mask = torch.zeros_like(tokens, dtype=torch.float32)
                for token_id in breathy_token_ids:
                    mask = mask + (tokens == token_id).float()
                if idx_d != -1 and idx_h != -1 and tokens.shape[-1] > 1:
                    is_d = (tokens == idx_d)
                    is_h = (tokens == idx_h)
                    is_dh = is_d[:, :-1] & is_h[:, 1:]
                    is_dh_padded = F.pad(is_dh, (0, 1), value=False)
                    mask = mask + is_dh_padded.float()
                breathy_mask_phon = mask.unsqueeze(1).to(device)
                breathy_mask_frames = (breathy_mask_phon @ pred_aln_trg.unsqueeze(0).to(device))
                
            F0_pred, N_pred = model.predictor.F0Ntrain(en, s, breathy_mask=breathy_mask_frames)
            out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
            
        gen_wav = out.squeeze().cpu().numpy()[..., :-50]
        gen_wav_path = f"eval_outputs/gen_{wav_name}"
        sf.write(gen_wav_path, gen_wav, 24000)
        
        # Calculate F0 metrics
        gt_f0 = get_f0(gt_wav_path)
        gen_f0 = get_f0(gen_wav_path)
        
        # Resample or pad/crop F0 arrays to align lengths
        min_len = min(len(gt_f0), len(gen_f0))
        if min_len < 10:
            continue
        gt_f0 = gt_f0[:min_len]
        gen_f0 = gen_f0[:min_len]
        
        # semitone converting
        gt_semi = hz_to_semitones(gt_f0)
        gen_semi = hz_to_semitones(gen_f0)
        
        # RMSE Semitones (only on voiced frames of both)
        voiced_idx = (gt_f0 > 0) & (gen_f0 > 0)
        if np.sum(voiced_idx) > 0:
            rmse_semi = np.sqrt(np.mean((gt_semi[voiced_idx] - gen_semi[voiced_idx])**2))
            rmse_semitones_list.append(rmse_semi)
            
            # Correlation
            if np.std(gt_semi[voiced_idx]) > 0 and np.std(gen_semi[voiced_idx]) > 0:
                corr, _ = pearsonr(gt_semi[voiced_idx], gen_semi[voiced_idx])
                if not np.isnan(corr):
                    f0_corr_list.append(corr)
                    
        # V/UV Error Rate
        gt_voiced = gt_f0 > 0
        gen_voiced = gen_f0 > 0
        vuv_error = np.mean(gt_voiced != gen_voiced)
        vuv_error_list.append(vuv_error)

    # Average metrics
    avg_rmse_semi = np.mean(rmse_semitones_list) if rmse_semitones_list else float('nan')
    avg_f0_corr = np.mean(f0_corr_list) if f0_corr_list else float('nan')
    avg_vuv_error = np.mean(vuv_error_list) if vuv_error_list else float('nan')
    
    print("\n" + "="*40)
    print("PROSODY EVALUATION METRICS REPORT")
    print("="*40)
    print(f"Model Checkpoint: {checkpoint_path}")
    print(f"F0 RMSE (Semitones): {avg_rmse_semi:.4f} (Lower is better)")
    print(f"F0 Correlation (r):  {avg_f0_corr:.4f} (Higher is better)")
    print(f"V/UV Error Rate:      {avg_vuv_error:.4f} (Lower is better)")
    print("="*40 + "\n")
    
    return {
        'F0_RMSE_Semitones': avg_rmse_semi,
        'F0_Correlation': avg_f0_corr,
        'VUV_Error_Rate': avg_vuv_error
    }

def torchaudio_transforms_mel(device):
    import torchaudio
    return torchaudio.transforms.MelSpectrogram(
        n_mels=80, 
        n_fft=2048, 
        win_length=1200, 
        hop_length=300
    ).to(device)

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Models/jv30/config_ft.yml')
    parser.add_argument('--checkpoint', type=str, default='Models/jv30/epoch_2nd_00024.pth')
    parser.add_argument('--test_list', type=str, default='../local/jv/phonemized_lists/test_list_phon.txt')
    parser.add_argument('--limit', type=int, default=20)
    args = parser.parse_args()
    
    # Adjust test list path relative to StyleTTS2 dir if necessary
    test_path = args.test_list
    if not os.path.exists(test_path) and os.path.exists(os.path.join('/workspace/StyleTTS2', test_path)):
        test_path = os.path.join('/workspace/StyleTTS2', test_path)
    
    evaluate_models(args.config, args.checkpoint, test_path, args.limit)
