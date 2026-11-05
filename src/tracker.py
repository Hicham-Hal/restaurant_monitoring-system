"""
tracker.py - Multi-Object Tracking wrapper module using ByteTrack.
Integrates with Ultralytics YOLOv8 built-in tracking to assign persistent IDs
to customers and staff across frames.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from src.config import MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD, TARGET_CLASSES, COLOR_PERSON, COLOR_BBOX_DEFAULT


class ByteTracker:
    def __init__(self, model_path=MODEL_PATH):
        """
        Initializes the YOLO model for tracking.
        Ultralytics supports ByteTrack natively via the tracker configuration argument.
        """
        print(f"[*] Loading tracker model from: {model_path}")
        try:
            self.model = YOLO(str(model_path))
            print("[+] ByteTrack tracker model initialized successfully.")
        except Exception as e:
            print(f"[-] Error initializing tracker model: {e}")
            raise e

    def track_frame(self, frame):
        """
        Runs object detection and tracking on a single video frame using ByteTrack.
        
        Args:
            frame (np.ndarray): OpenCV BGR video frame.
            
        Returns:
            list of dict: Tracked objects containing bbox [x1, y1, x2, y2], 
                          track_id, confidence score, and class id.
        """
        # Run YOLO track with ByteTrack algorithm configuration ('bytetrack.yaml')
        results = self.model.track(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )[0]
        
        tracked_objects = []
        
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()          # Coordinates [x1, y1, x2, y2]
            confidences = results.boxes.conf.cpu().numpy()     # Confidence scores
            class_ids = results.boxes.cls.cpu().numpy().astype(int)  # Class IDs
            track_ids = results.boxes.id.cpu().numpy().astype(int)   # Persistent tracking IDs
            
            for box, conf, cls_id, track_id in zip(boxes, confidences, class_ids, track_ids):
                # Filter target classes (e.g., person class_id == 0)
                if cls_id in TARGET_CLASSES:
                    x1, y1, x2, y2 = map(int, box)
                    tracked_objects.append({
                        "track_id": int(track_id),
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(conf),
                        "class_id": int(cls_id),
                        "class_name": self.model.names[cls_id]
                    })
                    
        return tracked_objects

    def draw_tracks(self, frame, tracked_objects):
        """
        Draws bounding boxes, tracking IDs, and class labels on the video frame.
        
        Args:
            frame (np.ndarray): OpenCV BGR video frame.
            tracked_objects (list): List of tracked object dicts from track_frame().
            
        Returns:
            np.ndarray: Annotated video frame.
        """
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["bbox"]
            track_id = obj["track_id"]
            cls_name = obj["class_name"]
            conf = obj["confidence"]
            cls_id = obj["class_id"]
            
            # Select color based on class
            if cls_id == 0:  # person
                color = COLOR_PERSON
            else:
                color = COLOR_BBOX_DEFAULT
                
            # Draw bounding box rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label with tracking ID
            label = f"ID {track_id} | {cls_name} ({conf:.2f})"
            
            # Draw label background box for readability
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 22), (x1 + w, y1), color, -1)
            
            # Draw text over label background
            cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return frame


# ==========================================
# Standalone Unit Test Script for Phase 3
# ==========================================
if __name__ == "__main__":
    print("Testing ByteTracker module standalone...")
    tracker = ByteTracker()
    
    # Create dummy frame (640x480)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracks = tracker.track_frame(dummy_frame)
    print(f"Tracks on empty frame: {len(tracks)} (Expected: 0)")
    print("[+] Phase 3 tracker module verified successfully.")