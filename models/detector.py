from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="models/weights/yolov8n.pt", device="cpu"):
        print("Loading model...")
        self.model = YOLO(model_path)
        self.device = device
        print("Model Loaded")
    def detect(self, frame):
        results = self.model.predict(frame, device=self.device,verbose=False)
        result = results[0]
        detections = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            })
        return detections, results
    def draw(self, frame, results):
        return results[0].plot()