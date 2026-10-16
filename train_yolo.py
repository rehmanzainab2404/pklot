from ultralytics import YOLO
import os
import shutil

def main():
    model = YOLO("yolov8n-cls.pt")
    
    results = model.train(
        data="dataset_patches",
        epochs=3,
        fraction=0.1,
        imgsz=128,
        batch=32,
        device="cpu",
        workers=2,
        project="smartpark_runs",
        name="exp"
    )
    
    best_weight_path = os.path.join("smartpark_runs", "exp", "weights", "best.pt")
    target_path = "smartpark_yolo.pt"
    
    if os.path.exists(best_weight_path):
        shutil.copy(best_weight_path, target_path)
        print(f"\nModel train ho gaya aur save ho gaya: {target_path}")

if __name__ == "__main__":
    main()
