#!/bin/bash
set -e
export HF_TOKEN="${HF_TOKEN:-}"

echo "Membuat direktori..."
mkdir -p /workspace/local/id
mkdir -p /workspace/local/jv

cat << 'EOF' > download_temp.py
import os
import shutil
import zipfile
from huggingface_hub import hf_hub_download

# print("Mendownload final_dataset.zip (ID)...")
# id_zip = hf_hub_download(
#     repo_id="eldntr/final_dataset",
#     filename="final_dataset.zip",
#     repo_type="dataset"
# )

# print("Mengekstrak final_dataset.zip ke /workspace/local/id...")
# with zipfile.ZipFile(id_zip, 'r') as zip_ref:
#     zip_ref.extractall('/workspace/local/id')

# print("Mendownload final_dataset_jv.zip (JV)...")
# jv_zip = hf_hub_download(
#     repo_id="eldntr/final_dataset",
#     filename="final_dataset_jv.zip",
#     repo_type="dataset"
# )

# print("Mengekstrak final_dataset_jv.zip ke /workspace/local/jv...")
# with zipfile.ZipFile(jv_zip, 'r') as zip_ref:
#     zip_ref.extractall('/workspace/local/jv')

print("Mendownload Model StyleTTS2-LibriTTS (epochs_2nd_00020.pth)...")
weights_path = hf_hub_download(
    repo_id="yl4579/StyleTTS2-LibriTTS", 
    filename="Models/LibriTTS/epochs_2nd_00020.pth"
)

print("Mendownload Config StyleTTS2-LibriTTS...")
config_path = hf_hub_download(
    repo_id="yl4579/StyleTTS2-LibriTTS", 
    filename="Models/LibriTTS/config.yml"
)

print("Menyalin model dan config ke /workspace/local/...")
os.makedirs('/workspace/local', exist_ok=True)
shutil.copy(weights_path, '/workspace/local/epochs_2nd_00020.pth')
shutil.copy(config_path, '/workspace/local/config.yml')

print("Semua proses selesai!")
EOF

echo "Menjalankan skrip Python untuk download..."
/workspace/StyleTTS2/.venv/bin/python download_temp.py

echo "Merapikan struktur folder..."
if [ -d "/workspace/local/id/final_dataset" ]; then
    mv /workspace/local/id/final_dataset/* /workspace/local/id/ 2>/dev/null || true
    rm -rf /workspace/local/id/final_dataset
fi

if [ -d "/workspace/local/jv/create-dataset" ]; then
    mv /workspace/local/jv/create-dataset/jv/final_dataset_jv/* /workspace/local/jv/ 2>/dev/null || true
    rm -rf /workspace/local/jv/create-dataset
fi

echo "Membersihkan file sementara..."
rm download_temp.py

echo "Selesai!"
