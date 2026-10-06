from ultralytics import YOLO

# Load our trained wildlife model
model = YOLO("runs/detect/train/weights/best.pt")

# Run tracking on the lion video
results = model.track(
    source="videos/lion.mp4",
    tracker="bytetrack.yaml",
    save=True,
    conf=0.25,
    show=False
)