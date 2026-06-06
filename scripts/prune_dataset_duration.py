import os
import glob
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm

def prune_dataset(lang_dir, min_dur=3.0, max_dur=6.0):
    final_dataset_dir = lang_dir
    wavs_dir = os.path.join(final_dataset_dir, "wavs")
    
    if not os.path.exists(wavs_dir):
        print(f"Error: Folder wavs tidak ditemukan di {wavs_dir}")
        return
        
    print(f"\n==================================================")
    # Get language code from directory name
    lang_name = os.path.basename(lang_dir)
    print(f"Memulai Pruning Dataset [{lang_name.upper()}] (Durasi: {min_dur}s - {max_dur}s)")
    print(f"==================================================")
    
    # 1. Scan semua file wav dan periksa durasinya
    wav_files = glob.glob(os.path.join(wavs_dir, "*.wav"))
    print(f"Ditemukan {len(wav_files)} file wav. Menganalisis durasi...")
    
    valid_wavs_dur = {}
    deleted_count = 0
    valid_durations = []
    total_duration_sec = 0.0
    
    for wav_path in tqdm(wav_files, desc="Checking audio durations"):
        filename = os.path.basename(wav_path)
        try:
            info = sf.info(wav_path)
            duration = info.duration
            
            if min_dur <= duration <= max_dur:
                valid_wavs_dur[filename] = duration
                valid_durations.append(duration)
                total_duration_sec += duration
            else:
                # Durasi di luar batas, hapus file wav
                os.remove(wav_path)
                deleted_count += 1
        except Exception as e:
            print(f"Gagal memproses file {filename}: {e}")
            
    # Saring speaker untuk bahasa Jawa (jv)
    if lang_name == "jv":
        print("\nBahasa Jawa terdeteksi. Menyaring speaker untuk mengurangi variasi...")
        speaker_wavs = {}
        for wav_path in wav_files:
            filename = os.path.basename(wav_path)
            if filename not in valid_wavs_dur:
                continue
            parts = filename.split('_')
            if len(parts) >= 2:
                spk = parts[1]
                if spk not in speaker_wavs:
                    speaker_wavs[spk] = []
                speaker_wavs[spk].append((wav_path, valid_wavs_dur[filename]))
        
        # Hitung total durasi per speaker dan urutkan
        speaker_totals = []
        for spk, files in speaker_wavs.items():
            tot = sum(d for _, d in files)
            speaker_totals.append((spk, tot, files))
        
        # Urutkan berdasarkan durasi terbanyak
        speaker_totals.sort(key=lambda x: x[1], reverse=True)
        
        # Pilih top 3 speaker
        top_speakers = speaker_totals[:3]
        top_spk_ids = [x[0] for x in top_speakers]
        print(f"  - Speaker terpilih: {', '.join(top_spk_ids)}")
        for spk, tot, _ in top_speakers:
            print(f"    * Speaker {spk}: {tot/60:.2f} menit")
            
        top_wavs = set()
        for _, _, files in top_speakers:
            for filepath, _ in files:
                top_wavs.add(os.path.basename(filepath))
                
        # Hapus file yang tidak berasal dari top 3 speaker
        for filename in list(valid_wavs_dur.keys()):
            if filename not in top_wavs:
                wav_path = os.path.join(wavs_dir, filename)
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                del valid_wavs_dur[filename]
                deleted_count += 1
                
        # Rekalkulasi durasi
        valid_durations = list(valid_wavs_dur.values())
        total_duration_sec = sum(valid_durations)
        
    # Memeriksa panjang fonem untuk memastikan tidak melebihi 512 token
    print("\nMemeriksa panjang fonem untuk memastikan tidak melebihi 512 token...")
    import sys
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
        
    try:
        from text_utils import TextCleaner
        cleaner = TextCleaner()
        
        phon_files = []
        phon_dir = os.path.join(final_dataset_dir, "phonemized_lists")
        if os.path.exists(phon_dir):
            for file in os.listdir(phon_dir):
                if file.endswith(".txt") and "phon" in file:
                    phon_files.append(os.path.join(phon_dir, file))
                    
        too_long_wavs = set()
        for filepath in phon_files:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line_stripped = line.strip()
                        if not line_stripped:
                            continue
                        parts = line_stripped.split('|')
                        if len(parts) >= 2:
                            wav_name = parts[0]
                            if wav_name in valid_wavs_dur:
                                tokens = cleaner(parts[1])
                                # Tambahkan 2 token blank (di awal & akhir) seperti dataloader
                                if len(tokens) + 2 > 512:
                                    too_long_wavs.add(wav_name)
                                    
        if too_long_wavs:
            print(f"  - Ditemukan {len(too_long_wavs)} file audio dengan token fonem > 512. Menghapus...")
            for wav_name in too_long_wavs:
                wav_path = os.path.join(wavs_dir, wav_name)
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                if wav_name in valid_wavs_dur:
                    del valid_wavs_dur[wav_name]
                deleted_count += 1
                
            # Rekalkulasi durasi
            valid_durations = list(valid_wavs_dur.values())
            total_duration_sec = sum(valid_durations)
        else:
            print("  - Semua file memenuhi syarat (<= 512 token).")
    except Exception as e:
        print(f"Peringatan: Gagal memvalidasi panjang fonem: {e}")
        
    valid_wavs = set(valid_wavs_dur.keys())
                
    print(f"\nHasil Pemfilteran Audio:")
    print(f"  - File Valid (Disimpan) : {len(valid_wavs)}")
    print(f"  - File Dibuang (Dihapus): {deleted_count}")
    
    # 2. Filter metadata.csv, lists, dan phonemized_lists
    print("\nMenyaring metadata dan file list...")
    txt_and_csv_files = []
    
    # Scan semua file .csv dan .txt di final_dataset secara rekursif
    for root, _, files in os.walk(final_dataset_dir):
        for file in files:
            if file.endswith('.txt') or file.endswith('.csv'):
                txt_and_csv_files.append(os.path.join(root, file))
                
    for filepath in txt_and_csv_files:
        # Jangan edit total_duration.txt atau speaker_map / speaker_mapping
        filename_only = os.path.basename(filepath)
        if filename_only in ["total_duration.txt", "speaker_map.txt", "speaker_mapping.txt", "judul_map.txt"]:
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        filtered_lines = []
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            parts = line_stripped.split('|')
            # Kolom pertama harus berupa filename.wav yang ada di set valid_wavs
            if parts and parts[0] in valid_wavs:
                filtered_lines.append(line)
                
        # Tulis kembali file dengan data yang sudah difilter
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(filtered_lines)
        print(f"  - Diperbarui: {os.path.relpath(filepath, final_dataset_dir)} ({len(lines)} -> {len(filtered_lines)} baris)")
        
    # 3. Perbarui file total_duration.txt
    hours = int(total_duration_sec // 3600)
    minutes = int((total_duration_sec % 3600) // 60)
    seconds = total_duration_sec % 60
    
    duration_path = os.path.join(final_dataset_dir, "total_duration.txt")
    duration_text = f"Total Durasi Dataset Final: {hours} jam {minutes} menit {seconds:.2f} detik\nTotal detik: {total_duration_sec:.2f}\n"
    with open(duration_path, "w", encoding="utf-8") as f:
        f.write(duration_text)
    print(f"  - Diperbarui: total_duration.txt ({hours}j {minutes}m {seconds:.2f}s)")
    
    # 4. Perbarui duration_distribution.png
    if valid_durations:
        plt.figure(figsize=(10, 6))
        plt.hist(valid_durations, bins=50, color='skyblue', edgecolor='black')
        plt.title(f'Distribusi Durasi Audio (Final Dataset - {lang_name.upper()})')
        plt.xlabel('Durasi (detik)')
        plt.ylabel('Jumlah File')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        graph_path = os.path.join(final_dataset_dir, "duration_distribution.png")
        plt.savefig(graph_path)
        plt.close()
        print(f"  - Diperbarui: duration_distribution.png")
        
    print(f"\nPruning untuk [{lang_name.upper()}] selesai!")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Jalankan pruning untuk dataset Indonesia (id) dan Jawa (jv)
    local_dir = os.path.join(script_dir, "..", "..", "local")
    for lang in ["id", "jv"]:
        lang_path = os.path.join(local_dir, lang)
        if os.path.exists(lang_path):
            prune_dataset(lang_path, min_dur=3.0, max_dur=6.0)
        else:
            print(f"Direktori tidak ditemukan: {lang_path}")