import cv2
import numpy as np
from ultralytics import YOLO
from src.config import MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD, TARGET_CLASSES, COLOR_BBOX_DEFAULT, COLOR_PERSON


class YOLODetector:
    # constructor
    # Loads the YOLO model and prepares it for use.
    def __init__(self, model_path=MODEL_PATH):
        """
        Initializes the YOLOv8 model.
        If the weights file (.pt) is not found locally, Ultralytics will automatically
        download the pretrained weights (e.g., yolov8n.pt).
        """
        print(f"[*] Loading YOLO model from: {model_path}")
        try:
            self.model = YOLO(str(model_path))
            print("[+] YOLO model loaded successfully.")
        except Exception as e:
            print(f"[-] Error loading YOLO model: {e}")
            raise e

    # detection
    # Analyzes one frame and returns information about the detected objects.
    def detect(self, frame):
        """
        Runs object detection on a single video frame.
        
        Args:
            frame (np.ndarray): OpenCV BGR video frame.
            
        Returns:
            list of dict: Filtered detections containing bbox [x1, y1, x2, y2], 
                          confidence score, and class id.
        """
        # Run YOLO inference
        results = self.model(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )[0]
        
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()  # Coordinates (x1, y1, x2, y2)
            confidences = results.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = results.boxes.cls.cpu().numpy().astype(int)  # Class IDs
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                # Filter only target classes defined in config (person, chair, dining table)
                if cls_id in TARGET_CLASSES:
                    x1, y1, x2, y2 = map(int, box)
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(conf),
                        "class_id": int(cls_id),
                        "class_name": self.model.names[cls_id]
                    })
                    
        return detections

    # visualization
    # Draws the detection results on the original frame.
    def draw_detections(self, frame, detections):
        """
        Draws bounding boxes and labels for detected objects on the frame.
        
        Args:
            frame (np.ndarray): OpenCV BGR video frame.
            detections (list): List of detection dicts from detect().
            
        Returns:
            np.ndarray: Annotated video frame.
        """
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["class_name"]
            conf = det["confidence"]
            cls_id = det["class_id"]
            
            # Select color based on class
            if cls_id == 0:  # person
                color = COLOR_PERSON
            else:
                color = COLOR_BBOX_DEFAULT
                
            # Draw bounding box rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label = f"{cls_name}: {conf:.2f}"
            
            # Draw label background for better readability
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            
            # Draw white text over label background
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return frame


# ==========================================
# Standalone Unit Test Script for Phase 2(just for testing)
# ==========================================
if __name__ == "__main__":
    import sys
    print("Testing YOLODetector module standalone...")
    detector = YOLODetector()
    
    # Create a dummy test frame (black image 640x480)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8) #generate a black image
    dets = detector.detect(dummy_frame)
    print(f"Detections on empty frame: {len(dets)} (Expected: 0)")
    print("[+] Phase 2 detector module verified successfully.")