import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

try:
    import librosa
except ImportError:
    print("Error: 'librosa' tidak ditemukan. Jalankan: pip install librosa")
    exit(1)

try:
    import jiwer
    HAS_JIWER = True
except ImportError:
    print("Warning: 'jiwer' tidak ditemukan. (pip install jiwer)")
    HAS_JIWER = False

try:
    from transformers import pipeline, Wav2Vec2Processor, HubertModel
    import torch
    import torchaudio
    HAS_TRANSFORMERS = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading ASR Whisper (for WER & CER)...")
    asr_model = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device)
    
    print("Loading Phoneme Recognizer (for PER)...")
    phoneme_model = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-xlsr-53-espeak-cv-ft", device=device)
    
    print("Loading HuBERT model (for Cosine Similarity)...")
    hubert_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to(device)
    hubert_model.eval()

except ImportError:
    print("Warning: 'transformers' / 'torch' / 'torchaudio' tidak ditemukan. Metrik berbasis ML akan dilewati.")
    HAS_TRANSFORMERS = False

try:
    from speechbrain.inference.speaker import EncoderClassifier
    print("Loading ECAPA-TDNN Speaker Model (for SEC)...")
    device_opts = {"device": "cuda"} if torch.cuda.is_available() else {"device": "cpu"}
    speaker_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_speechbrain", run_opts=device_opts)
    HAS_SPEECHBRAIN = True
except ImportError:
    print("Warning: 'speechbrain' tidak ditemukan. SEC akan dilewati.")
    HAS_SPEECHBRAIN = False

def compute_mcd(ref_wav, synth_wav):
    try:
        y_ref, sr = librosa.load(ref_wav, sr=24000)
        y_syn, _ = librosa.load(synth_wav, sr=24000)
        
        y_ref, _ = librosa.effects.trim(y_ref, top_db=30)
        y_syn, _ = librosa.effects.trim(y_syn, top_db=30)
        
        mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr, n_mfcc=35, hop_length=256)[1:, :]
        mfcc_syn = librosa.feature.mfcc(y=y_syn, sr=sr, n_mfcc=35, hop_length=256)[1:, :]
        
        mfcc_ref = (mfcc_ref - np.mean(mfcc_ref, axis=1, keepdims=True)) / (np.std(mfcc_ref, axis=1, keepdims=True) + 1e-8)
        mfcc_syn = (mfcc_syn - np.mean(mfcc_syn, axis=1, keepdims=True)) / (np.std(mfcc_syn, axis=1, keepdims=True) + 1e-8)
        
        D, wp = librosa.sequence.dtw(mfcc_ref, mfcc_syn, metric='euclidean')
        
        dist = 0
        for i, j in wp:
            diff = mfcc_ref[:, i] - mfcc_syn[:, j]
            dist += np.sqrt(np.sum(diff**2))
            
        return dist / len(wp)
    except Exception as e:
        return np.nan

def compute_f0_rmse(ref_wav, synth_wav):
    try:
        y_ref, sr = librosa.load(ref_wav, sr=24000)
        y_syn, _ = librosa.load(synth_wav, sr=24000)
        
        y_ref, _ = librosa.effects.trim(y_ref, top_db=30)
        y_syn, _ = librosa.effects.trim(y_syn, top_db=30)
        
        f0_ref, _, _ = librosa.pyin(y_ref, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_syn, _, _ = librosa.pyin(y_syn, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        
        f0_ref = np.nan_to_num(f0_ref)
        f0_syn = np.nan_to_num(f0_syn)
        
        mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr, n_mfcc=13, hop_length=512)
        mfcc_syn = librosa.feature.mfcc(y=y_syn, sr=sr, n_mfcc=13, hop_length=512)
        
        mfcc_ref = (mfcc_ref - np.mean(mfcc_ref, axis=1, keepdims=True)) / (np.std(mfcc_ref, axis=1, keepdims=True) + 1e-8)
        mfcc_syn = (mfcc_syn - np.mean(mfcc_syn, axis=1, keepdims=True)) / (np.std(mfcc_syn, axis=1, keepdims=True) + 1e-8)
        
        _, wp = librosa.sequence.dtw(mfcc_ref, mfcc_syn, metric='euclidean')
        
        sq_err = 0
        valid_frames = 0
        for i, j in wp:
            if f0_ref[i] > 0 and f0_syn[j] > 0:
                sq_err += (np.log2(f0_ref[i]) - np.log2(f0_syn[j]))**2
                valid_frames += 1
                
        if valid_frames == 0:
            return np.nan
            
        return np.sqrt(sq_err / valid_frames)
    except Exception as e:
        return np.nan

def compute_gpe_vde(ref_wav, synth_wav):
    try:
        y_ref, sr = librosa.load(ref_wav, sr=24000)
        y_syn, _ = librosa.load(synth_wav, sr=24000)
        
        y_ref, _ = librosa.effects.trim(y_ref, top_db=30)
        y_syn, _ = librosa.effects.trim(y_syn, top_db=30)
        
        f0_ref, voiced_ref, _ = librosa.pyin(y_ref, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_syn, voiced_syn, _ = librosa.pyin(y_syn, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        
        mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr, n_mfcc=13, hop_length=512)
        mfcc_syn = librosa.feature.mfcc(y=y_syn, sr=sr, n_mfcc=13, hop_length=512)
        
        mfcc_ref = (mfcc_ref - np.mean(mfcc_ref, axis=1, keepdims=True)) / (np.std(mfcc_ref, axis=1, keepdims=True) + 1e-8)
        mfcc_syn = (mfcc_syn - np.mean(mfcc_syn, axis=1, keepdims=True)) / (np.std(mfcc_syn, axis=1, keepdims=True) + 1e-8)
        
        _, wp = librosa.sequence.dtw(mfcc_ref, mfcc_syn, metric='euclidean')
        
        vde_errors = 0
        gpe_errors = 0
        voiced_frames_total = 0
        total_frames = len(wp)
        
        for i, j in wp:
            v_ref = voiced_ref[i]
            v_syn = voiced_syn[j]
            
            if v_ref != v_syn:
                vde_errors += 1
                
            if v_ref and v_syn:
                voiced_frames_total += 1
                f0_r = f0_ref[i]
                f0_s = f0_syn[j]
                if abs(f0_r - f0_s) / f0_r > 0.2:
                    gpe_errors += 1
                    
        vde = vde_errors / total_frames if total_frames > 0 else 0
        gpe = gpe_errors / voiced_frames_total if voiced_frames_total > 0 else 0
        
        return vde, gpe
    except Exception as e:
        return np.nan, np.nan

def compute_dur_diff(ref_wav, synth_wav):
    try:
        dur_ref = librosa.get_duration(path=ref_wav)
        dur_syn = librosa.get_duration(path=synth_wav)
        return abs(dur_ref - dur_syn)
    except Exception as e:
        return np.nan

def compute_wer_cer(ground_truth_text, synth_wav):
    if not HAS_TRANSFORMERS or not HAS_JIWER:
        return np.nan, np.nan, "", ""
    try:
        transcription = asr_model(synth_wav)["text"].lower()
        gt = jiwer.RemovePunctuation()(ground_truth_text.lower())
        pred = jiwer.RemovePunctuation()(transcription)
        wer = jiwer.wer(gt, pred)
        cer = jiwer.cer(gt, pred)
        return wer, cer, gt, pred
    except Exception as e:
        return np.nan, np.nan, "", ""

def compute_per(gt_wav, synth_wav):
    if not HAS_TRANSFORMERS or not HAS_JIWER:
        return np.nan, "", ""
    try:
        gt_phonemes = phoneme_model(gt_wav)["text"]
        syn_phonemes = phoneme_model(synth_wav)["text"]
        
        gt_phonemes_clean = gt_phonemes.replace(" ", "")
        syn_phonemes_clean = syn_phonemes.replace(" ", "")
        
        if len(gt_phonemes_clean) == 0:
            return np.nan, gt_phonemes_clean, syn_phonemes_clean
            
        gt_phonemes_spaced = " ".join(list(gt_phonemes_clean))
        syn_phonemes_spaced = " ".join(list(syn_phonemes_clean))
        
        per = jiwer.wer(gt_phonemes_spaced, syn_phonemes_spaced)
        return per, gt_phonemes_clean, syn_phonemes_clean
    except Exception as e:
        return np.nan, "", ""

def compute_hubert_similarity(ref_wav, synth_wav):
    if not HAS_TRANSFORMERS:
        return np.nan
    try:
        import torch.nn.functional as F
        
        y_ref, _ = librosa.load(ref_wav, sr=16000)
        y_syn, _ = librosa.load(synth_wav, sr=16000)
        
        inputs_ref = hubert_processor(y_ref, sampling_rate=16000, return_tensors="pt").to(device)
        inputs_syn = hubert_processor(y_syn, sampling_rate=16000, return_tensors="pt").to(device)
        
        with torch.no_grad():
            emb_ref = hubert_model(**inputs_ref).last_hidden_state.mean(dim=1)
            emb_syn = hubert_model(**inputs_syn).last_hidden_state.mean(dim=1)
            
        sim = F.cosine_similarity(emb_ref, emb_syn, dim=1).item()
        return sim
    except Exception as e:
        print(f"HuBERT Error: {e}")
        return np.nan

def compute_sec(ref_wav, synth_wav):
    if not HAS_SPEECHBRAIN:
        return np.nan
    try:
        import torch.nn.functional as F
        import librosa
        import torch
        
        y_ref, _ = librosa.load(ref_wav, sr=16000)
        y_syn, _ = librosa.load(synth_wav, sr=16000)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        signal_ref = torch.from_numpy(y_ref).to(device)
        signal_syn = torch.from_numpy(y_syn).to(device)
        
        emb_ref = speaker_model.encode_batch(signal_ref)
        emb_syn = speaker_model.encode_batch(signal_syn)
        
        sim = F.cosine_similarity(emb_ref.squeeze(), emb_syn.squeeze(), dim=0).item()
        return sim
    except Exception as e:
        print(f"SEC Error: {e}")
        return np.nan

def main():
    base_dir = "inference"
    gt_dir = os.path.join(base_dir, "ground_truth")
    
    val_basenames = set()
    val_list_path = "../local/jv/phonemized_lists/val_list_phon.txt"
    if os.path.exists(val_list_path):
        with open(val_list_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 1:
                    basename = parts[0].replace(".wav", "")
                    val_basenames.add(basename)
                    
    metadata_path = "../local/jv/metadata.csv"
    raw_texts = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    basename = parts[0].replace(".wav", "")
                    if basename in val_basenames:
                        raw_texts[basename] = parts[1]
    
    models = ["id_base", "id_zeroshot", "jv_baseline", "exp01", "exp02", "exp03"]
    
    if not os.path.exists(gt_dir):
        print(f"Folder Ground Truth tidak ditemukan: {gt_dir}")
        return

    results = []
    
    print("\nMemulai evaluasi metrik objektif ekstensif (MCD, F0 RMSE, WER, CER, PER, HuBERT Sim)...")
    
    gt_files = glob.glob(os.path.join(gt_dir, "*.wav"))
    
    for gt_wav in tqdm(gt_files, desc="Evaluasi File Audio"):
        basename = os.path.splitext(os.path.basename(gt_wav))[0]
        gt_text = raw_texts.get(basename, "")
        
        for model in models:
            synth_wav = os.path.join(base_dir, model, f"{basename}.wav")
            if not os.path.exists(synth_wav):
                continue
                
            mcd = compute_mcd(gt_wav, synth_wav)
            f0_rmse = compute_f0_rmse(gt_wav, synth_wav)
            vde, gpe = compute_gpe_vde(gt_wav, synth_wav)
            dur_diff = compute_dur_diff(gt_wav, synth_wav)
            wer, cer, gt_str, pred_str = compute_wer_cer(gt_text, synth_wav) if gt_text else (np.nan, np.nan, "", "")
            per, gt_phon, pred_phon = compute_per(gt_wav, synth_wav)
            hubert_sim = compute_hubert_similarity(gt_wav, synth_wav)
            sec = compute_sec(gt_wav, synth_wav)
            
            model_label = {
                "exp01": "LPEP",
                "exp02": "PPIM",
                "exp03": "LPEP_PPIM"
            }.get(model, model)
            
            results.append({
                "File": basename,
                "Model": model_label,
                "GT_Text": gt_str,
                "Pred_Text": pred_str,
                "GT_Phonemes": gt_phon,
                "Pred_Phonemes": pred_phon,
                "MCD": round(mcd, 3),
                "F0_RMSE_Log": round(f0_rmse, 3),
                "VDE": round(vde, 3),
                "GPE": round(gpe, 3),
                "Dur_Diff_s": round(dur_diff, 3),
                "WER": round(wer, 3),
                "CER": round(cer, 3),
                "PER": round(per, 3),
                "HuBERT_Sim": round(hubert_sim, 3) if not np.isnan(hubert_sim) else np.nan,
                "SEC": round(sec, 3) if not np.isnan(sec) else np.nan
            })

    if len(results) == 0:
        print("Tidak ada audio yang dievaluasi.")
        return
        
    df = pd.DataFrame(results)
    
    print("\n================ HASIL EVALUASI RATA-RATA ================")
    summary_df = df.groupby("Model").mean(numeric_only=True).round(3)
    print(summary_df)
    print("===========================================================\n")
    
    out_csv = "evaluation_metrics_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Hasil perhitungan detail per-file berhasil disimpan ke '{out_csv}'")
    print("Tips: \n- CER & PER yang lebih rendah = Cacat pelafalan (artifak di ujung kata) lebih sedikit.")
    print("- HuBERT_Sim yang lebih tinggi = Karakteristik suara dan logat lebih identik dengan aslinya.")

if __name__ == "__main__":
    main()
