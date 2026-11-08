# Configuración global del proyecto YOHO

from pathlib import Path

# --- Rutas del dataset (disco externo F:) ---
DATASET_ROOT = Path("/mnt/f/DATASETS/DataSED")
AUDIO_DIR = DATASET_ROOT / "SED_wav"
CSV_MONO = DATASET_ROOT / "SED_ground_truth" / "Monophonic_sound_detection.csv"
CSV_POLY = DATASET_ROOT / "SED_ground_truth" / "Polyphonic_sound_detection.csv"

# --- Rutas de salida (en el disco externo también, para no llenar WSL) ---
OUTPUT_ROOT = DATASET_ROOT / "yolo_dataset"
SPECTROGRAM_DIR = OUTPUT_ROOT / "images"
LABELS_DIR = OUTPUT_ROOT / "labels"

# --- Parámetros de audio ---
SAMPLE_RATE = 44100
N_FFT = 4096
HOP_LENGTH = 512
N_MELS = 128

# --- Parámetros de YOLO ---
YOLO_MODEL = "yolov8n.pt"
EPOCHS = 100
BATCH_SIZE = 8
IMG_SIZE_YOLO = 1024

# --- Clases (se llenarán automáticamente desde el CSV) ---
# Esto es solo un placeholder; el script de conversión generará el data.yaml
