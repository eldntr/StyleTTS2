import os
import glob
import random
import argparse

def generate_subsampled_files(lang_dir, seed=42):
    lang_name = os.path.basename(lang_dir)
    phon_dir = os.path.join(lang_dir, "phonemized_lists")
    
    if not os.path.exists(phon_dir):
        print(f"Error: Folder {phon_dir} tidak ditemukan.")
        return
        
    print(f"\n--- Membuat File Subsample untuk [{lang_name.upper()}] (2 Speaker, Minimal 30 Audio) ---")
    
    train_file = os.path.join(phon_dir, "train_list_phon.txt")
    val_file = os.path.join(phon_dir, "val_list_phon.txt")
    test_file = os.path.join(phon_dir, "test_list_phon.txt")
    
    if not os.path.exists(train_file): return
    
    # Group by speaker from train
    with open(train_file, 'r', encoding='utf-8') as f:
        train_lines = f.readlines()
        
    from collections import defaultdict
    spk_to_lines = defaultdict(list)
    for line in train_lines:
        line_s = line.strip()
        if not line_s: continue
        parts = line_s.split('|')
        if len(parts) >= 3:
            spk = parts[-1]
            spk_to_lines[spk].append(line)
            
    # Pick 2 speakers that have at least 15 audios in train
    valid_spks = [spk for spk, lines in spk_to_lines.items() if len(lines) >= 15]
    random.seed(seed)
    if len(valid_spks) < 2:
        print("Error: Tidak cukup speaker dengan 15+ audio.")
        return
        
    selected_spks = random.sample(valid_spks, 2)
    print(f"  - Terpilih 2 speaker: {selected_spks}")
    
    # Process train, val, test
    for filepath in [train_file, val_file, test_file]:
        if not os.path.exists(filepath): continue
        filename = os.path.basename(filepath)
        sub_filename = filename.replace(".txt", "_sub.txt")
        sub_filepath = os.path.join(phon_dir, sub_filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        sampled_lines = []
        for spk in selected_spks:
            spk_lines = [l for l in lines if l.strip() and l.strip().split('|')[-1] == spk]
            n_take = 15 if "train" in filename else 5
            sampled_lines.extend(spk_lines[:n_take])
            
        sampled_lines.sort()
        with open(sub_filepath, 'w', encoding='utf-8') as f:
            f.writelines(sampled_lines)
            
        print(f"  - Generated: {sub_filename} (Total: {len(sampled_lines)} baris)")

def update_configs(configs_dir, use_sub=True):
    # Cari semua file .yml di folder Configs
    config_files = []
    for root, _, files in os.walk(configs_dir):
        for file in files:
            if file.endswith('.yml') or file.endswith('.yaml'):
                config_files.append(os.path.join(root, file))
                
    print(f"\n--- Mengubah Konfigurasi di {configs_dir} ---")
    
    replaced_count = 0
    for filepath in config_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        new_content = content
        if use_sub:
            # Ganti dari full ke subsampled
            if "train_list_phon_sub.txt" not in new_content:
                new_content = new_content.replace("train_list_phon.txt", "train_list_phon_sub.txt")
            if "val_list_phon_sub.txt" not in new_content:
                new_content = new_content.replace("val_list_phon.txt", "val_list_phon_sub.txt")
            if "test_list_phon_sub.txt" not in new_content:
                new_content = new_content.replace("test_list_phon.txt", "test_list_phon_sub.txt")
                
            # Coba replace yang non-phon jika ada
            if "train_list_sub.txt" not in new_content:
                new_content = new_content.replace("train_list.txt", "train_list_sub.txt")
            if "val_list_sub.txt" not in new_content:
                new_content = new_content.replace("val_list.txt", "val_list_sub.txt")
            if "test_list_sub.txt" not in new_content:
                new_content = new_content.replace("test_list.txt", "test_list_sub.txt")
        else:
            # Kembalikan ke full
            new_content = new_content.replace("train_list_phon_sub.txt", "train_list_phon.txt")
            new_content = new_content.replace("val_list_phon_sub.txt", "val_list_phon.txt")
            new_content = new_content.replace("test_list_phon_sub.txt", "test_list_phon.txt")
            
            new_content = new_content.replace("train_list_sub.txt", "train_list.txt")
            new_content = new_content.replace("val_list_sub.txt", "val_list.txt")
            new_content = new_content.replace("test_list_sub.txt", "test_list.txt")
            
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"  - Diperbarui: {os.path.relpath(filepath, configs_dir)}")
            replaced_count += 1
            
    print(f"Total {replaced_count} file konfigurasi diperbarui.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample StyleTTS2 lists and toggle config files.")
    parser.add_argument("--restore", action="store_true", help="Kembalikan file konfigurasi ke 100% full data")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    configs_dir = os.path.join(project_root, "Configs")
    local_dir = os.path.join(project_root, "..", "local")
    
    if args.restore:
        print("==================================================")
        print("Menghubungkan Ulang Konfigurasi ke 100% Full Data")
        print("==================================================")
        update_configs(configs_dir, use_sub=False)
    else:
        print("==================================================")
        print(f"Membuat File Subsample & Update Konfigurasi")
        print("==================================================")
        
        # 1. Generate file list terpisah (*_phon_sub.txt)
        for lang in ["id", "jv"]:
            lang_path = os.path.join(local_dir, lang)
            if os.path.exists(lang_path):
                generate_subsampled_files(lang_path)
                
        # 2. Update config files untuk menunjuk ke file *_phon_sub.txt tersebut
        update_configs(configs_dir, use_sub=True)
        
    print("\nSelesai!")