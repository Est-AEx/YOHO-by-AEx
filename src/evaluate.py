"""
evaluate_rgb.py
Evalua YOLO26n sobre el conjunto de test del dataset RGB.
"""

from ultralytics import YOLO
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    OUTPUT_ROOT,
    DATA_YAML,
    IMG_SIZE_YOLO,
    DEVICE,
    RUN_NAME_TRAIN,
    RUN_NAME_TEST,
)


def main():
    best_pt = OUTPUT_ROOT / "runs" / RUN_NAME_TRAIN / "weights" / "best.pt"

    print("=" * 60)
    print("EVALUACION YOLO26n - DATASET RGB (TEST)")
    print("=" * 60)
    print(f"Modelo: {best_pt}")
    print(f"Existe: {best_pt.exists()}")
    print(f"Data:   {DATA_YAML}")
    print(f"Existe: {DATA_YAML.exists()}")

    if not best_pt.exists():
        print("Error: no se encontro best.pt. Ejecuta train_rgb.py primero.")
        return

    model = YOLO(str(best_pt))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMG_SIZE_YOLO,
        batch=8,
        device=DEVICE,
        project=str(OUTPUT_ROOT / "runs"),
        name=RUN_NAME_TEST,
        exist_ok=True,
    )

    print("\nMetricas sobre TEST (RGB, YOLO26n):")
    print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision:    {metrics.box.mp:.4f}")
    print(f"  Recall:       {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()