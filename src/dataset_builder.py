"""
dataset_builder_rgb.py
Construye el dataset YOLO RGB (3 canales, 8 bits) para YOHO.

Reemplaza el dataset anterior de 1 canal en OUTPUT_ROOT.
Cada imagen tiene 3 canales:
- R: STFT lineal en dB (20 Hz - Nyquist)
- G: STFT log en dB (20 Hz - Nyquist)
- B: CQT interpolado a rejilla log (20 Hz - Nyquist)

Ventanas de 24 s con 5 s de solape (hop = 19 s). Se generan TODAS las ventanas
necesarias para cubrir el dataset completo (~3541).
Imagenes casi cuadradas: 1024 filas x ~1032 columnas.

Las rutas se importan desde config.py.
"""

import sys
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from PIL import Image
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    AUDIO_DIR,
    CSV_MONO,
    OUTPUT_ROOT,
    SPECTROGRAM_DIR,
    LABELS_DIR,
)

# ============================================================
# PARAMETROS DE AUDIO
# ============================================================
SR = 44100
N_FFT = 2048
HOP_LENGTH = 1024

# Rango de frecuencias comun (20 Hz a Nyquist)
NYQUIST = SR / 2
FREQ_MIN = 20.0
FREQ_MAX = NYQUIST
N_FILAS = 1024

# CQT
CQT_BINS_POR_OCTAVA = 12
CQT_N_BINS = 120
CQT_FMIN = 20.0

# ============================================================
# PARAMETROS DE VENTANAS
# ============================================================
VENTANA_SEG = 24
SOLAPAMIENTO_SEG = 6
HOP_SEG = VENTANA_SEG - SOLAPAMIENTO_SEG  # 19 s

# ============================================================
# PARAMETROS DE CADA CANAL
# ============================================================
# Canal R
R_VMIN = -70
R_VMAX = -6
R_INVERTIR_X = False
R_INVERTIR_Y = False

# Canal G
G_VMIN = -50
G_VMAX = -3
G_INVERTIR_X = False
G_INVERTIR_Y = False

# Canal B
B_VMIN = -50
B_VMAX = -6
B_INVERTIR_X = False
B_INVERTIR_Y = False

# ============================================================
# RUTAS DE SALIDA (usan OUTPUT_ROOT directamente)
# ============================================================
DATA_YAML = OUTPUT_ROOT / "data.yaml"


# ============================================================
# LINEA DE TIEMPO GLOBAL
# ============================================================
def construir_linea_tiempo(audio_dir):
    audios = sorted([f.name for f in Path(audio_dir).glob("*.wav")])
    registros = []
    offset = 0.0
    for audio in audios:
        info = sf.info(Path(audio_dir) / audio)
        dur = info.duration
        registros.append({'sound_name': audio, 'offset_global': offset, 'duracion': dur})
        offset += dur
    return pd.DataFrame(registros), offset


# ============================================================
# VENTANAS (sin limite, se generan todas)
# ============================================================
def generar_ventanas(duracion_total, ventana_seg, hop_seg):
    ventanas = []
    inicio = 0.0
    while inicio < duracion_total:
        ventanas.append((inicio, inicio + ventana_seg))
        inicio += hop_seg
    return ventanas


def extraer_audio_ventana(inicio, fin, df_timeline, audio_dir, sr):
    longitud = int((fin - inicio) * sr)
    buffer = np.zeros(longitud, dtype=np.float32)
    for _, row in df_timeline.iterrows():
        a_ini = row['offset_global']
        a_fin = a_ini + row['duracion']
        s_ini = max(inicio, a_ini)
        s_fin = min(fin, a_fin)
        if s_fin <= s_ini:
            continue
        t_ini = s_ini - a_ini
        t_fin = s_fin - a_ini
        y, _ = librosa.load(
            Path(audio_dir) / row['sound_name'],
            sr=sr, mono=True, offset=t_ini, duration=t_fin - t_ini
        )
        pos = int((s_ini - inicio) * sr)
        buffer[pos:pos + len(y)] = y
    return buffer


# ============================================================
# CALCULO DE CADA CANAL (matriz 2D en uint8)
# ============================================================
def calcular_canal_r(y, sr, n_fft, hop_length, n_filas,
                     freq_min, freq_max, vmin, vmax,
                     invertir_x, invertir_y):
    """Canal R: STFT lineal en dB, rejilla lineal de freq_min a freq_max."""
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    freqs_stft = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    mask = (freqs_stft >= freq_min) & (freqs_stft <= freq_max)
    freqs_r = freqs_stft[mask]
    mag_r = mag_db[mask, :]
    n_frames = mag_r.shape[1]

    freqs_lineal = np.linspace(freq_min, freq_max, n_filas)

    canal = np.zeros((n_filas, n_frames), dtype=np.float32)
    for i in range(n_frames):
        canal[:, i] = np.interp(freqs_lineal, freqs_r, mag_r[:, i])

    canal_norm = np.clip((canal - vmin) / (vmax - vmin), 0, 1)
    canal_u8 = (canal_norm * 255).astype(np.uint8)

    if invertir_x:
        canal_u8 = np.fliplr(canal_u8)
    if invertir_y:
        canal_u8 = np.flipud(canal_u8)

    return canal_u8


def calcular_canal_g(y, sr, n_fft, hop_length, n_filas,
                     freq_min, freq_max, vmin, vmax,
                     invertir_x, invertir_y):
    """Canal G: STFT log en dB, rejilla logaritmica de freq_min a freq_max."""
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    freqs_stft = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    mask = (freqs_stft >= freq_min) & (freqs_stft <= freq_max)
    freqs_r = freqs_stft[mask]
    mag_r = mag_db[mask, :]
    n_frames = mag_r.shape[1]

    freqs_log = np.logspace(np.log10(freq_min), np.log10(freq_max), n_filas)

    canal = np.zeros((n_filas, n_frames), dtype=np.float32)
    for i in range(n_frames):
        canal[:, i] = np.interp(freqs_log, freqs_r, mag_r[:, i])

    canal_norm = np.clip((canal - vmin) / (vmax - vmin), 0, 1)
    canal_u8 = (canal_norm * 255).astype(np.uint8)

    if invertir_x:
        canal_u8 = np.fliplr(canal_u8)
    if invertir_y:
        canal_u8 = np.flipud(canal_u8)

    return canal_u8


def calcular_canal_b(y, sr, hop_length, n_filas,
                     freq_min, freq_max, cqt_fmin, cqt_n_bins, cqt_bpo,
                     vmin, vmax, invertir_x, invertir_y):
    """Canal B: CQT interpolado a rejilla logaritmica de freq_min a freq_max."""
    cqt = np.abs(librosa.cqt(
        y, sr=sr, hop_length=hop_length,
        fmin=cqt_fmin, n_bins=cqt_n_bins, bins_per_octave=cqt_bpo
    ))
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
    freqs_cqt = librosa.cqt_frequencies(
        n_bins=cqt_n_bins, fmin=cqt_fmin, bins_per_octave=cqt_bpo
    )
    n_frames = cqt_db.shape[1]

    freqs_log = np.logspace(np.log10(freq_min), np.log10(freq_max), n_filas)

    canal = np.zeros((n_filas, n_frames), dtype=np.float32)
    for i in range(n_frames):
        canal[:, i] = np.interp(freqs_log, freqs_cqt, cqt_db[:, i])

    canal_norm = np.clip((canal - vmin) / (vmax - vmin), 0, 1)
    canal_u8 = (canal_norm * 255).astype(np.uint8)

    if invertir_x:
        canal_u8 = np.fliplr(canal_u8)
    if invertir_y:
        canal_u8 = np.flipud(canal_u8)

    return canal_u8


# ============================================================
# CONSTRUCCION DE LA IMAGEN RGB
# ============================================================
def generar_imagen_rgb(y, sr, out_path):
    """Genera una imagen RGB de 8 bits con 3 canales y la guarda."""
    canal_r = calcular_canal_r(
        y, sr, N_FFT, HOP_LENGTH, N_FILAS,
        FREQ_MIN, FREQ_MAX, R_VMIN, R_VMAX,
        R_INVERTIR_X, R_INVERTIR_Y
    )
    canal_g = calcular_canal_g(
        y, sr, N_FFT, HOP_LENGTH, N_FILAS,
        FREQ_MIN, FREQ_MAX, G_VMIN, G_VMAX,
        G_INVERTIR_X, G_INVERTIR_Y
    )
    canal_b = calcular_canal_b(
        y, sr, HOP_LENGTH, N_FILAS,
        FREQ_MIN, FREQ_MAX, CQT_FMIN, CQT_N_BINS, CQT_BINS_POR_OCTAVA,
        B_VMIN, B_VMAX, B_INVERTIR_X, B_INVERTIR_Y
    )

    # Igualar frames (por si el CQT tiene uno menos)
    min_frames = min(canal_r.shape[1], canal_g.shape[1], canal_b.shape[1])
    canal_r = canal_r[:, :min_frames]
    canal_g = canal_g[:, :min_frames]
    canal_b = canal_b[:, :min_frames]

    # Apilar en RGB
    rgb = np.stack([canal_r, canal_g, canal_b], axis=-1)

    # Invertir verticalmente para guardar con orientacion correcta
    rgb_guardar = np.flipud(rgb)

    img = Image.fromarray(rgb_guardar, mode='RGB')
    img.save(out_path, format='PNG')
    return img.width, img.height


# ============================================================
# CAJAS YOLO
# ============================================================
def calcular_cajas(ventana_ini, ventana_fin, df_eventos, df_timeline, clase_a_id):
    lineas = []
    dur = ventana_fin - ventana_ini
    for _, ev in df_eventos.iterrows():
        offset = df_timeline.loc[
            df_timeline['sound_name'] == ev['sound_name'], 'offset_global'
        ].values[0]
        t_ini_abs = offset + ev['start_time']
        t_fin_abs = offset + ev['end_time']
        t_ini_rec = max(t_ini_abs, ventana_ini)
        t_fin_rec = min(t_fin_abs, ventana_fin)
        if t_fin_rec <= t_ini_rec:
            continue
        x_ini = (t_ini_rec - ventana_ini) / dur
        x_fin = (t_fin_rec - ventana_ini) / dur
        x_center = (x_ini + x_fin) / 2
        w = x_fin - x_ini
        cid = clase_a_id[ev['class_name']]
        lineas.append(f"{cid} {x_center:.6f} 0.500000 {w:.6f} 1.000000")
    return lineas


# ============================================================
# DATA YAML (SIN channels: 1, porque es RGB)
# ============================================================
def generar_data_yaml(clases, out_path, output_root):
    with open(out_path, 'w') as f:
        f.write(f"path: {output_root}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write(f"\nnc: {len(clases)}\n")
        f.write("names:\n")
        for i, c in enumerate(clases):
            f.write(f"  {i}: {c}\n")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("CONSTRUYENDO DATASET YOLO RGB (3 CANALES)")
    print("=" * 60)
    print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
    print(f"SR: {SR} Hz")
    print(f"Nyquist: {NYQUIST} Hz")
    print(f"FREQ_MIN: {FREQ_MIN} Hz, FREQ_MAX: {FREQ_MAX} Hz")
    print(f"N_FILAS: {N_FILAS}")
    print(f"VENTANA_SEG: {VENTANA_SEG} s, SOLAPAMIENTO: {SOLAPAMIENTO_SEG} s")
    print(f"HOP_SEG: {HOP_SEG} s")
    print(f"R: vmin={R_VMIN}, vmax={R_VMAX}")
    print(f"G: vmin={G_VMIN}, vmax={G_VMAX}")
    print(f"B: vmin={B_VMIN}, vmax={B_VMAX}")

    SPECTROGRAM_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    df_eventos = pd.read_csv(CSV_MONO)
    print(f"\nEventos en CSV: {len(df_eventos)}")

    df_timeline, duracion_total = construir_linea_tiempo(AUDIO_DIR)
    print(f"Audios: {len(df_timeline)}")
    print(f"Duracion total: {duracion_total:.1f} s ({duracion_total/60:.1f} min)")

    ventanas = generar_ventanas(duracion_total, VENTANA_SEG, HOP_SEG)
    print(f"Ventanas a generar: {len(ventanas)}")

    clases = sorted(df_eventos['class_name'].unique().tolist())
    clase_a_id = {c: i for i, c in enumerate(clases)}
    print(f"Clases: {len(clases)}")

    generar_data_yaml(clases, DATA_YAML, OUTPUT_ROOT)
    print(f"data.yaml generado en: {DATA_YAML}")

    # Modo de prueba: descomenta para probar con 5 ventanas
    # ventanas = ventanas[:5]

    print("\nGenerando imagenes RGB...")
    anchos = set()
    alturas = set()
    for i, (v_ini, v_fin) in enumerate(tqdm(ventanas)):
        nombre = f"RGB-{i:04d}"
        y = extraer_audio_ventana(v_ini, v_fin, df_timeline, AUDIO_DIR, SR)
        img_path = SPECTROGRAM_DIR / f"{nombre}.png"
        ancho, alto = generar_imagen_rgb(y, SR, img_path)
        anchos.add(ancho)
        alturas.add(alto)

        lineas = calcular_cajas(v_ini, v_fin, df_eventos, df_timeline, clase_a_id)
        label_path = LABELS_DIR / f"{nombre}.txt"
        with open(label_path, 'w') as f:
            f.write("\n".join(lineas))

    print(f"\nDataset RGB generado en: {OUTPUT_ROOT}")
    print(f"   Imagenes: {len(list(SPECTROGRAM_DIR.glob('*.png')))}")
    print(f"   Labels:   {len(list(LABELS_DIR.glob('*.txt')))}")
    print(f"   Anchos unicos:  {sorted(anchos)}")
    print(f"   Alturas unicas: {sorted(alturas)}")


if __name__ == "__main__":
    main()