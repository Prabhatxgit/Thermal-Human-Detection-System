from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # or yolov8s.pt if resources allow
model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='thermal-human',
    device='cuda'  # use 'cpu' if no GPU
)