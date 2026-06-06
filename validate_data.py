import os
import yaml
from text_utils import TextCleaner

def main():
    print("Mencari data dengan teks yang terlalu pendek (<= 2 token)...")
    
    # Load config untuk mendapatkan lokasi data
    config_path = 'Configs/config.yml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    data_params = config['data_params']
    lists_to_check = [
        data_params.get('train_data', ''),
        data_params.get('val_data', ''),
        data_params.get('OOD_data', '')
    ]
    
    # Inisialisasi Text Cleaner
    textcleaner = TextCleaner()
    
    found_error = False
    
    for list_path in lists_to_check:
        if not list_path or not os.path.exists(list_path):
            continue
            
        print(f"\nMemeriksa file: {list_path}")
        with open(list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines, 1):
            parts = line.strip().split('|')
            if len(parts) < 2:
                continue
                
            audio_path = parts[0]
            text = parts[1]
            
            # Bersihkan dan konversi ke token
            tokens = textcleaner(text)
            
            # Periksa panjang token
            if len(tokens) <= 2:
                print(f"  [ERROR] Baris {line_num} -> Panjang Token: {len(tokens)}")
                print(f"    - File Audio: {audio_path}")
                print(f"    - Teks Fonem: '{text}'")
                print(f"    - Hasil Token: {tokens}")
                found_error = True
                
    if not found_error:
        print("\nSELAMAT! Tidak ditemukan data teks yang terlalu pendek di dalam dataset Anda.")
    else:
        print("\nSARAN: Hapus baris-baris data di atas dari file list Anda, karena terlalu pendek untuk dipelajari oleh model dan akan menyebabkan perhitungan durasi menghasilkan tensor kosong (NaN).")

if __name__ == "__main__":
    main()
