#!/bin/bash
set -e

echo "=================================================="
echo "    MEMULAI INISIALISASI STYLE-TTS 2 WORKSPACE"
echo "=================================================="

# 1. Setup Virtual Environment
echo ""
echo "[1/4] Menyiapkan Python Virtual Environment (.venv)..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Virtual Environment berhasil dibuat."
else
    echo "Virtual Environment sudah ada, melewatinya."
fi

# Aktifkan venv
source .venv/bin/activate

# 2. Instalasi Dependensi
echo ""
echo "[2/4] Menginstal dependensi dari requirements.txt..."
# Pastikan pip diperbarui
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "PERINGATAN: requirements.txt tidak ditemukan!"
fi
# Pastikan huggingface_hub terinstal untuk skrip download
pip install huggingface_hub zipfile36

# Beri izin eksekusi untuk semua skrip di dalam folder scripts
chmod +x scripts/*.sh

# 3. Download Data & Model
echo ""
echo "[3/4] Mengunduh Dataset dan Model Pre-Trained..."
if [ -f "scripts/download_data.sh" ]; then
    bash scripts/download_data.sh
else
    echo "GAGAL: scripts/download_data.sh tidak ditemukan!"
fi

# 4. Prune dan Subsample Dataset
echo ""
echo "[4/4] Melakukan Pruning Dataset..."
if [ -f "scripts/prune_dataset.sh" ]; then
    bash scripts/prune_dataset.sh
else
    echo "GAGAL: scripts/prune_dataset.sh tidak ditemukan!"
fi



echo ""
echo "=================================================="
echo " INISIALISASI SELESAI! LINGKUNGAN SIAP DIGUNAKAN "
echo "=================================================="
echo "Catatan: Jangan lupa jalankan 'source .venv/bin/activate' jika Anda membuka terminal baru."
