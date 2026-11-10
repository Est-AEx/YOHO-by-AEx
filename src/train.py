"""
Entrena YOLO26n sobre el dataset RGB de 3 canales.
"""

from ultralytics import YOLO
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    OUTPUT_ROOT,
    DATA_YAML,
    YOLO_MODEL,
    EPOCHS,
    BATCH_SIZE,
    IMG_SIZE_YOLO,
    PATIENCE,
    WORKERS,
    DEVICE,
    RUN_NAME_TRAIN,
)


def main():
    print("=" * 60)
    print("ENTRENAMIENTO YOLO26n - DATASET RGB (3 CANALES)")
    print("=" * 60)
    print(f"Modelo: {YOLO_MODEL}")
    print(f"Data:   {DATA_YAML}")
    print(f"Existe: {DATA_YAML.exists()}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch:  {BATCH_SIZE}")
    print(f"Imgsz:  {IMG_SIZE_YOLO}")
    print(f"Device: {DEVICE}")

    model = YOLO(YOLO_MODEL)

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE_YOLO,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        workers=WORKERS,
        device=DEVICE,
        project=str(OUTPUT_ROOT / "runs"),
        name=RUN_NAME_TRAIN,
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )

    print("\nEntrenamiento finalizado.")
    print(f"Resultados en: {results.save_dir}")


if __name__ == "__main__":
    main()