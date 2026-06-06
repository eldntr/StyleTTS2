import os
import glob
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm

def prune_dataset(lang_dir, min_dur=2.0, max_dur=7.0):
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
    
    valid_wavs = set()
    deleted_count = 0
    valid_durations = []
    total_duration_sec = 0.0
    
    for wav_path in tqdm(wav_files, desc="Checking audio durations"):
        filename = os.path.basename(wav_path)
        try:
            info = sf.info(wav_path)
            duration = info.duration
            
            if min_dur <= duration <= max_dur:
                valid_wavs.add(filename)
                valid_durations.append(duration)
                total_duration_sec += duration
            else:
                # Durasi di luar batas, hapus file wav
                os.remove(wav_path)
                deleted_count += 1
        except Exception as e:
            print(f"Gagal memproses file {filename}: {e}")
                
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
            prune_dataset(lang_path, min_dur=2.0, max_dur=7.0)
        else:
            print(f"Direktori tidak ditemukan: {lang_path}")