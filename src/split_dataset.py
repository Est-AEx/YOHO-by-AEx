"""
split_dataset.py
Divide el dataset YOLO en train/val/test manteniendo la estratificacion
por clase (aproximada por clase dominante de cada imagen).
"""

import sys
import shutil
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

sys.path.append(str(Path(__file__).resolve().parent))
from config import OUTPUT_ROOT, SPECTROGRAM_DIR, LABELS_DIR

# ============================================================
# CONFIGURACION
# ============================================================
PROP_TRAIN = 0.80
PROP_VAL = 0.10
PROP_TEST = 0.10
SEED = 42
MOVE_FILES = False   # True = mover, False = copiar (copiar es mas seguro)

DATA_YAML = OUTPUT_ROOT / "data.yaml"


def obtener_clase_dominante(label_path, clase_id_a_nombre):
    """
    Lee un .txt de label y devuelve la clase mas frecuente.
    Si el label esta vacio, devuelve -1.
    """
    with open(label_path, 'r') as f:
        lineas = [l.strip() for l in f if l.strip()]
    if not lineas:
        return -1
    clases = [int(l.split()[0]) for l in lineas]
    return Counter(clases).most_common(1)[0][0]


def main():
    print("=" * 60)
    print("SPLIT TRAIN / VAL / TEST ESTRATIFICADO")
    print("=" * 60)

    # Cargar nombres de clases
    with open(DATA_YAML, 'r') as f:
        data = yaml.safe_load(f)
    names = data.get('names', {})
    if isinstance(names, dict):
        clase_id_a_nombre = {i: names[i] for i in sorted(names.keys())}
    else:
        clase_id_a_nombre = {i: n for i, n in enumerate(names)}

    # Listar imagenes y labels
    imagenes = sorted(SPECTROGRAM_DIR.glob("*.png"))
    print(f"Imagenes: {len(imagenes)}")

    # Construir dataframe con clase dominante de cada imagen
    registros = []
    for img_path in imagenes:
        label_path = LABELS_DIR / (img_path.stem + ".txt")
        clase_dom = obtener_clase_dominante(label_path, clase_id_a_nombre)
        registros.append({
            'stem': img_path.stem,
            'img_path': img_path,
            'label_path': label_path,
            'clase_dominante': clase_dom,
        })
    df = pd.DataFrame(registros)

    print(f"\nDistribucion de clases dominantes:")
    for cid, cnt in df['clase_dominante'].value_counts().sort_index().items():
        nombre = clase_id_a_nombre.get(cid, 'vacio')
        print(f"  {nombre:<45} {cnt:>5}")

    # Separar imagenes vacias (clase_dominante = -1) para tratarlas aparte
    df_vacias = df[df['clase_dominante'] == -1].copy()
    df_con_clase = df[df['clase_dominante'] != -1].copy()
    print(f"\nImagenes sin cajas: {len(df_vacias)}")
    print(f"Imagenes con cajas: {len(df_con_clase)}")

    # Filtrar clases con muy pocas muestras (no se pueden estratificar con 1 miembro)
    # Se dejan en train siempre
    conteo = df_con_clase['clase_dominante'].value_counts()
    clases_raras = conteo[conteo < 2].index.tolist()
    df_raras = df_con_clase[df_con_clase['clase_dominante'].isin(clases_raras)].copy()
    df_con_clase = df_con_clase[~df_con_clase['clase_dominante'].isin(clases_raras)].copy()
    if len(df_raras) > 0:
        print(f"Clases con <2 muestras (van a train): {[clase_id_a_nombre.get(c) for c in clases_raras]}")

    # Split estratificado 1: train vs (val+test)
    train_df, temp_df = train_test_split(
        df_con_clase,
        test_size=(PROP_VAL + PROP_TEST),
        stratify=df_con_clase['clase_dominante'],
        random_state=SEED,
    )

    # Split 2: val vs test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=PROP_TEST / (PROP_VAL + PROP_TEST),
        stratify=temp_df['clase_dominante'],
        random_state=SEED,
    )

    # Anadir las clases raras y las vacias a train (o repartirlas si quieres)
    train_df = pd.concat([train_df, df_raras, df_vacias])

    print(f"\nSplit resultante:")
    print(f"  Train: {len(train_df)} ({100*len(train_df)/len(df):.1f}%)")
    print(f"  Val:   {len(val_df)} ({100*len(val_df)/len(df):.1f}%)")
    print(f"  Test:  {len(test_df)} ({100*len(test_df)/len(df):.1f}%)")

    # Crear carpetas
    for split in ['train', 'val', 'test']:
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copiar o mover archivos
    def mover_o_copiar(df_split, split_name):
        for _, row in df_split.iterrows():
            dst_img = OUTPUT_ROOT / "images" / split_name / row['img_path'].name
            dst_lbl = OUTPUT_ROOT / "labels" / split_name / row['label_path'].name
            if MOVE_FILES:
                shutil.move(str(row['img_path']), str(dst_img))
                shutil.move(str(row['label_path']), str(dst_lbl))
            else:
                shutil.copy2(str(row['img_path']), str(dst_img))
                shutil.copy2(str(row['label_path']), str(dst_lbl))

    print("\nCopiando archivos...")
    mover_o_copiar(train_df, 'train')
    mover_o_copiar(val_df, 'val')
    mover_o_copiar(test_df, 'test')

    # Verificar distribucion por clase en cada split
    print("\nDistribucion de clases por split (conteo de imagenes):")
    for split, df_split in [('train', train_df), ('val', val_df), ('test', test_df)]:
        print(f"\n  {split.upper()}:")
        conteo = df_split['clase_dominante'].value_counts().sort_index()
        for cid, cnt in conteo.items():
            nombre = clase_id_a_nombre.get(cid, 'vacio')
            print(f"    {nombre:<45} {cnt:>5}")

    # Generar data.yaml nuevo con train/val/test
    data_yaml_nuevo = OUTPUT_ROOT / "data.yaml"
    with open(data_yaml_nuevo, 'w') as f:
        f.write(f"path: {OUTPUT_ROOT}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("channels: 1\n")
        f.write(f"\nnc: {len(clase_id_a_nombre)}\n")
        f.write("names:\n")
        for i in sorted(clase_id_a_nombre.keys()):
            f.write(f"  {i}: {clase_id_a_nombre[i]}\n")
    print(f"\ndata.yaml actualizado en: {data_yaml_nuevo}")


if __name__ == "__main__":
    main()