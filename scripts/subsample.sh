#!/bin/bash

# Dapatkan lokasi absolut direktori tempat script ini berada
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Cari interpreter Python yang tepat
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif [ -f "../.venv/bin/python" ]; then
    PYTHON_BIN="../.venv/bin/python"
else
    PYTHON_BIN="python"
fi

# Parsing argumen bash
ACTION="subsample"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --restore) ACTION="restore" ;;
        *) echo "Argumen tidak dikenal: $1"; exit 1 ;;
    esac
    shift
done

if [ "$ACTION" == "restore" ]; then
    echo "========================================================="
    echo "  Mengembalikan Semua Konfigurasi ke 100% Full Data"
    echo "========================================================="
    $PYTHON_BIN subsample_dataset.py --restore
else
    echo "========================================================="
    echo "  Membuat Subsample List Data & Mengubah Konfigurasi"
    echo "========================================================="
    $PYTHON_BIN subsample_dataset.py
fi

echo "========================================================="
echo "  Proses Selesai!"
echo "========================================================="