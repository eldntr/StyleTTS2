#!/bin/bash

# Dapatkan lokasi absolut direktori tempat script ini berada
SCRIPT_DIR="$( cd "$( dirname "$0" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR"

echo "========================================================="
echo "  StyleTTS2 Dataset Pruner (Filter Durasi 3-6 Detik)"
echo "========================================================="

# Cari interpreter Python yang tepat
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif [ -f "../.venv/bin/python" ]; then
    PYTHON_BIN="../.venv/bin/python"
else
    PYTHON_BIN="python"
fi

# Jalankan script pruning
$PYTHON_BIN prune_dataset_duration.py

echo "========================================================="
echo "  Pruning selesai!"
echo "========================================================="