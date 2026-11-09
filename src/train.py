
from ultralytics import YOLO
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from config import OUTPUT_ROOT

def main():
    data_yaml = OUTPUT_ROOT / "data.yaml"
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=1024,
        batch=-1,
        patience=30,
        workers=4,
        device=0,
        project=str(OUTPUT_ROOT / "runs"),
        name="yoho_train",
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )
    print("Entrenamiento finalizado.")
    print(f"Resultados en: {results.save_dir}")

if __name__ == "__main__":
    main()
