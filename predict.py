import os
import argparse
import torch
import yaml
import librosa
import numpy as np
import torchaudio
import soundfile as sf
import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

from models import build_model, load_ASR_models, load_F0_models
from utils import recursive_munch
from Utils.PLBERT.util import load_plbert
from text_utils import TextCleaner
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

def main():
    parser = argparse.ArgumentParser(description='Predict using trained StyleTTS2')
    parser.add_argument('--text', type=str, required=True, help='Text or phonemes to synthesize')
    parser.add_argument('--ref_audio', type=str, required=True, help='Path to reference audio for voice cloning')
    parser.add_argument('--output', type=str, default='output.wav', help='Output wav path')
    parser.add_argument('--config', type=str, default='Models/jv30/config_ft.yml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='Models/jv30/epoch_2nd_00024.pth', help='Path to checkpoint')
    parser.add_argument('--phonemized', action='store_true', help='Set this if text is already phonemized')
    parser.add_argument('--alpha', type=float, default=0.3, help='Timbre preservation factor (0 to 1)')
    parser.add_argument('--beta', type=float, default=0.7, help='Prosody preservation factor (0 to 1)')
    parser.add_argument('--steps', type=int, default=5, help='Diffusion steps')
    parser.add_argument('--lang_id', type=int, default=None, help='Language ID (0 for Indo, 1 for Javanese)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

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

    print("Computing style...")
    ref_s = compute_style(args.ref_audio)

    textcleaner = TextCleaner()
    text = args.text.strip()
    
    if not args.phonemized:
        try:
            import phonemizer
            # Use indonesian or javanese fallback
            global_phonemizer = phonemizer.backend.EspeakBackend(language='id', preserve_punctuation=True, with_stress=True)
            text = global_phonemizer.phonemize([text])[0]
        except Exception as e:
            print(f"Warning: phonemization failed: {e}. If text is already phonemized, pass --phonemized.")

    ps = word_tokenize(text)
    ps = ' '.join(ps)
    tokens = textcleaner(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    print("Synthesizing audio...")
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        if getattr(model_params, 'use_lpep', False):
            lang_id_val = args.lang_id if args.lang_id is not None else config.get('lang_id', 1)
            t_en = model.text_encoder(tokens, input_lengths, text_mask, lang_id=lang_id_val)
        else:
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

        if getattr(model_params, 'use_ppim', False) and hasattr(model, 'ppim'):
            t_en = model.ppim(t_en, s, text_mask)

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
    
    sf.write(args.output, wav, 24000)
    print(f"Successfully saved generated audio to {args.output}")

if __name__ == '__main__':
    main()
