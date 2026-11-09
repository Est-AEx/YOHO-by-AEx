import sys
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Importar rutas desde config
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    DATASET_ROOT,
    AUDIO_DIR,
    CSV_MONO,
    OUTPUT_ROOT,
    SPECTROGRAM_DIR,
    LABELS_DIR,
)

# ============================================================
# CONFIGURACION LOCAL
# ============================================================
SR = 44100
VENTANA_SEG = 24
SOLAPAMIENTO_SEG = 5
HOP_SEG = VENTANA_SEG - SOLAPAMIENTO_SEG   # 19 s

N_FFT = 2048
HOP_LENGTH = 1024

CMAP = "gray"               # fondo negro (silencio), eventos blancos (energia)
DB_MIN, DB_MAX = -80, 0

DATA_YAML = OUTPUT_ROOT / "data.yaml"


# ============================================================
# PASO 1: LINEA DE TIEMPO GLOBAL
# ============================================================
def construir_linea_tiempo(audio_dir):
    audios = sorted([f.name for f in Path(audio_dir).glob("*.wav")])
    registros = []
    offset = 0.0
    for audio in audios:
        info = sf.info(Path(audio_dir) / audio)
        dur = info.duration
        registros.append({
            'sound_name': audio,
            'offset_global': offset,
            'duracion': dur
        })
        offset += dur
    return pd.DataFrame(registros), offset


# ============================================================
# PASO 2: GENERAR VENTANAS
# ============================================================
def generar_ventanas(duracion_total, ventana_seg, hop_seg):
    ventanas = []
    inicio = 0.0
    while inicio < duracion_total:
        fin = inicio + ventana_seg
        ventanas.append((inicio, fin))
        inicio += hop_seg
    return ventanas


# ============================================================
# PASO 3: EXTRAER AUDIO DE UNA VENTANA
# ============================================================
def extraer_audio_ventana(inicio, fin, df_timeline, audio_dir, sr):
    longitud_muestras = int((fin - inicio) * sr)
    buffer = np.zeros(longitud_muestras, dtype=np.float32)

    for _, row in df_timeline.iterrows():
        audio_ini = row['offset_global']
        audio_fin = audio_ini + row['duracion']

        solape_ini = max(inicio, audio_ini)
        solape_fin = min(fin, audio_fin)
        if solape_fin <= solape_ini:
            continue

        t_ini_local = solape_ini - audio_ini
        t_fin_local = solape_fin - audio_ini
        y, _ = librosa.load(
            Path(audio_dir) / row['sound_name'],
            sr=sr, mono=True,
            offset=t_ini_local, duration=t_fin_local - t_ini_local
        )

        pos_ini = int((solape_ini - inicio) * sr)
        pos_fin = pos_ini + len(y)
        buffer[pos_ini:pos_fin] = y

    return buffer


# ============================================================
# PASO 4: GENERAR ESPECTROGRAMA EN PNG DE 16 BITS (MODO I;16)
# ============================================================
def generar_espectrograma(y, sr, n_fft, hop_length, cmap, out_path):
    """
    Genera un espectrograma STFT en escala de grises y lo guarda como PNG
    de 16 bits (modo I;16, 1 canal, 16 bits por pixel).

    Convencion estandar:
    - Eje X: tiempo (izquierda = inicio, derecha = final).
    - Eje Y: frecuencia (abajo = 0 Hz, arriba = Nyquist).
    - Fondo negro (silencio), eventos blancos (energia) con cmap='gray'.
    """
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitud = np.abs(stft)
    magnitud_db = librosa.amplitude_to_db(magnitud, ref=np.max)

    # Normalizar a rango 0-65535 (16 bits)
    img_array = np.clip((magnitud_db - DB_MIN) / (DB_MAX - DB_MIN), 0, 1)
    img_uint16 = (img_array * 65535).astype(np.uint16)

    # cmap='gray': valores bajos -> negro (0), valores altos -> blanco (65535)
    # cmap='gray_r': invertir
    if cmap == "gray_r":
        img_uint16 = 65535 - img_uint16

    # Invertir eje vertical: en el array, fila 0 = 0 Hz.
    # Al guardar con Pillow, fila 0 queda arriba.
    # flipud pone las frecuencias bajas abajo (convencion estandar).
    img_uint16 = np.flipud(img_uint16)

    # Crear imagen SIN especificar mode (Pillow lo infiere del dtype)
    img = Image.fromarray(img_uint16)

    # Asegurar que el modo sea I;16 antes de guardar
    if img.mode != 'I;16':
        img = img.convert('I;16')

    img.save(out_path, format='PNG')

    return img.width, img.height


# ============================================================
# PASO 5: CALCULAR CAJAS YOLO PARA UNA VENTANA
# ============================================================
def calcular_cajas(ventana_ini, ventana_fin, df_eventos, df_timeline, clase_a_id):
    """
    Calcula las cajas YOLO de los eventos que caen en la ventana.
    Cajas de ancho completo en frecuencia: y_center=0.5, h=1.0.
    La inversion del eje Y no afecta a las cajas (cubren todo el alto).
    """
    lineas = []
    duracion_ventana = ventana_fin - ventana_ini

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

        x_ini = (t_ini_rec - ventana_ini) / duracion_ventana
        x_fin = (t_fin_rec - ventana_ini) / duracion_ventana

        x_center = (x_ini + x_fin) / 2
        w = x_fin - x_ini
        y_center = 0.5
        h = 1.0

        clase_id = clase_a_id[ev['class_name']]
        lineas.append(f"{clase_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    return lineas


# ============================================================
# PASO 6: GENERAR data.yaml CON channels: 1
# ============================================================
def generar_data_yaml(clases, out_path, output_root):
    with open(out_path, 'w') as f:
        f.write(f"path: {output_root}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write("channels: 1\n")
        f.write(f"\nnc: {len(clases)}\n")
        f.write("names:\n")
        for i, clase in enumerate(clases):
            f.write(f"  {i}: {clase}\n")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("CONSTRUYENDO DATASET YOLO PARA YOHO")
    print("=" * 60)

    SPECTROGRAM_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    df_eventos = pd.read_csv(CSV_MONO)
    print(f"Eventos en CSV: {len(df_eventos)}")

    print("\nConstruyendo linea de tiempo global...")
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

    # --- Modo de prueba: descomenta para probar con 5 ventanas ---
    # ventanas = ventanas[:5]

    print("\nGenerando ventanas...")
    anchos = set()
    alturas = set()
    for i, (v_ini, v_fin) in enumerate(tqdm(ventanas)):
        nombre = f"X-{i:04d}"

        y = extraer_audio_ventana(v_ini, v_fin, df_timeline, AUDIO_DIR, SR)

        img_path = SPECTROGRAM_DIR / f"{nombre}.png"
        ancho, alto = generar_espectrograma(y, SR, N_FFT, HOP_LENGTH, CMAP, img_path)
        anchos.add(ancho)
        alturas.add(alto)

        lineas = calcular_cajas(v_ini, v_fin, df_eventos, df_timeline, clase_a_id)

        label_path = LABELS_DIR / f"{nombre}.txt"
        with open(label_path, 'w') as f:
            f.write("\n".join(lineas))

    print(f"\nDataset generado en: {OUTPUT_ROOT}")
    print(f"   Imagenes: {len(list(SPECTROGRAM_DIR.glob('*.png')))}")
    print(f"   Labels:   {len(list(LABELS_DIR.glob('*.txt')))}")
    print(f"   Anchos de imagen unicos:  {sorted(anchos)}")
    print(f"   Alturas de imagen unicas: {sorted(alturas)}")


if __name__ == "__main__":
    main()