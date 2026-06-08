import os
import sys
import argparse
import torch
import yaml
import librosa
import numpy as np
import torchaudio
import soundfile as sf
import pandas as pd
from tqdm import tqdm

# Add root directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models import build_model, load_ASR_models, load_F0_models
from utils import recursive_munch
from Utils.PLBERT.util import load_plbert
from text_utils import TextCleaner
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from metrics import compute_mcd_and_f0

def get_args():
    parser = argparse.ArgumentParser(description='Batch synthesis and evaluation on OOD/Test set')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--test_list', type=str, default='/workspace/local/jv/phonemized_lists/test_list_phon.txt', help='Path to test_list_phon.txt')
    parser.add_argument('--wav_dir', type=str, default='/workspace/local/jv/wavs', help='Path to GT wavs')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name (e.g., baseline_id or ft_jv30)')
    parser.add_argument('--alpha', type=float, default=0.3, help='Timbre preservation factor')
    parser.add_argument('--beta', type=float, default=0.7, help='Prosody preservation factor')
    parser.add_argument('--steps', type=int, default=5, help='Diffusion steps')
    return parser.parse_args()

def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create output directories
    output_dir = f"/workspace/StyleTTS2/experiments/results/{args.exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Config
    config = yaml.safe_load(open(args.config))
    
    # Load Models
    print("Loading models...")
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    print(f"Loading checkpoint {args.checkpoint}...")
    params_whole = torch.load(args.checkpoint, map_location='cpu')
    params = params_whole['net']
    
    # Auto-detect and apply PEFT wrapper if it's a PEFT checkpoint
    peft_mode = params_whole.get('peft_mode', 'none')
    if peft_mode != 'none':
        print(f"PEFT: Detected PEFT mode '{peft_mode}' in checkpoint. Wrapping model...")
        from peft_models import PEFTStyleTTS2
        peft_wrapper = PEFTStyleTTS2(model, mode=peft_mode)
        _ = [model[key].to(device) for key in model]
        
    for key in model:
        if key in params:
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                model[key].load_state_dict(new_state_dict, strict=False)
    _ = [model[key].eval() for key in model]

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )

    to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
    mean, std = -4, 4

    def length_to_mask(lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def preprocess(wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
        return mel_tensor

    def compute_style(path):
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = preprocess(audio).to(device)

        with torch.no_grad():
            ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)

    # Read test items
    with open(args.test_list, 'r', encoding='utf-8') as f:
        lines = [line.strip().split('|') for line in f if line.strip()]
        
    results = []
    textcleaner = TextCleaner()
    
    print(f"Starting evaluation of {len(lines)} samples...")
    for idx, parts in enumerate(tqdm(lines)):
        if len(parts) < 2:
            continue
        wav_name, phonemes = parts[0], parts[1]
        gt_wav_path = os.path.join(args.wav_dir, wav_name)
        pred_wav_path = os.path.join(output_dir, wav_name)
        
        if not os.path.exists(gt_wav_path):
            print(f"Warning: GT file not found: {gt_wav_path}")
            continue
            
        # 1. Compute style from GT audio (to ensure speaker identity is matched)
        ref_s = compute_style(gt_wav_path)
        
        # 2. Preprocess phonemes
        tokens = textcleaner(phonemes)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
        
        # 3. Synthesize
        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
            text_mask = length_to_mask(input_lengths).to(device)

            t_en = model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 

            s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device), 
                                              embedding=bert_dur,
                                              embedding_scale=1,
                                               features=ref_s,
                                                num_steps=args.steps).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = args.alpha * ref + (1 - args.alpha)  * ref_s[:, :128]
            s = args.beta * s + (1 - args.beta)  * ref_s[:, 128:]

            d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = model.predictor.lstm(d)
            duration = model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
            if model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
            if model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        wav = out.squeeze().cpu().numpy()[..., :-50]
        sf.write(pred_wav_path, wav, 24000)
        
        # 4. Compute Metrics
        try:
            mcd, f0_rmse = compute_mcd_and_f0(gt_wav_path, pred_wav_path)
            results.append({
                'filename': wav_name,
                'mcd': mcd,
                'f0_rmse': f0_rmse
            })
        except Exception as e:
            print(f"Error computing metrics for {wav_name}: {e}")
            
    # Save results to csv
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'metrics.csv')
    df.to_csv(csv_path, index=False)
    
    print("\n--- Evaluation Summary ---")
    print(f"Experiment: {args.exp_name}")
    print(f"Average MCD: {df['mcd'].mean():.4f}")
    print(f"Average F0 RMSE: {df['f0_rmse'].mean():.4f}")
    print(f"Results saved to {csv_path}")

if __name__ == '__main__':
    main()
