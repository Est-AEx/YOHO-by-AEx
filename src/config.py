# config.py
# Configuracion global del proyecto YOHO
# Version RGB con 3 canales para YOLO26n

from pathlib import Path

# ============================================================
# RUTAS DEL DATASET ORIGINAL
# ============================================================
DATASET_ROOT = Path("/mnt/f/DATASETS/DataSED")
AUDIO_DIR = DATASET_ROOT / "SED_wav"
CSV_MONO = DATASET_ROOT / "SED_ground_truth" / "Monophonic_sound_detection.csv"
CSV_POLY = DATASET_ROOT / "SED_ground_truth" / "Polyphonic_sound_detection.csv"

# ============================================================
# RUTAS DE SALIDA (RGB, 3 canales)
# ============================================================
OUTPUT_ROOT = DATASET_ROOT / "yolo_dataset"
SPECTROGRAM_DIR = OUTPUT_ROOT / "images"
LABELS_DIR = OUTPUT_ROOT / "labels"
DATA_YAML = OUTPUT_ROOT / "data.yaml"

# ============================================================
# PARAMETROS DE AUDIO Y VENTANAS
# ============================================================
SAMPLE_RATE = 44100
VENTANA_SEG = 24
SOLAPAMIENTO_SEG = 5
HOP_SEG = VENTANA_SEG - SOLAPAMIENTO_SEG  # 19 s

N_FFT = 2048
HOP_LENGTH = 1024

# ============================================================
# PARAMETROS DE LA ESCALA LOGARITMICA (canales G y B)
# ============================================================
FREQ_MIN = 20.0
FREQ_MAX = 22050.0  # Nyquist para SR=44100
N_FILAS = 1024

# CQT
CQT_BINS_POR_OCTAVA = 12
CQT_N_BINS = 120
CQT_FMIN = 20.0

# ============================================================
# PARAMETROS DE CADA CANAL (vmin, vmax)
# ============================================================
R_VMIN = -70
R_VMAX = -6
R_INVERTIR_X = False
R_INVERTIR_Y = False

G_VMIN = -50
G_VMAX = -3
G_INVERTIR_X = False
G_INVERTIR_Y = False

B_VMIN = -50
B_VMAX = -6
B_INVERTIR_X = False
B_INVERTIR_Y = False

# ============================================================
# PARAMETROS DE YOLO26n (ESTADO DEL ARTE)
# ============================================================
YOLO_MODEL = "yolo26n.pt"
EPOCHS = 128
BATCH_SIZE = -1          # AutoBatch
IMG_SIZE_YOLO = 1024
PATIENCE = 30
WORKERS = 4
DEVICE = 0

# Nombre del experimento (carpeta de runs)
RUN_NAME_TRAIN = "yoho_rgb_train"
RUN_NAME_TEST = "yoho_rgb_test"