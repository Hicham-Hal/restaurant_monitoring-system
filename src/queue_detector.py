"""
queue_detector.py - Queue detection and monitoring module.
Monitors the entrance or service queue ROI, counts waiting people, 
and calculates queue waiting durations.
"""

import cv2
import numpy as np
from src.config import QUEUE_ROI, CLASS_PERSON, COLOR_QUEUE, MAX_QUEUE_WAITING_TIME_SEC


class QueueDetector:
    def __init__(self, queue_roi=QUEUE_ROI):
        """
        Initializes the queue detector with a polygon ROI.
        
        Args:
            queue_roi (list of tuples): Polygon coordinates bounding the queue area.
        """
        self.queue_roi = queue_roi
        # Tracks arrival timestamp for each customer ID in the queue
        self.queue_entry_times = {}
        self.current_queue_ids = set()

    def _is_person_in_queue(self, person_bbox):
        """
        Determines if a person's position anchor is inside the queue polygon.
        """
        x1, y1, x2, y2 = person_bbox
        feet_point = (int((x1 + x2) / 2), y2)
        poly_pts = np.array(self.queue_roi, dtype=np.int32)
        
        inside_test = cv2.pointPolygonTest(poly_pts, feet_point, measureDist=False)
        return inside_test >= 0

    def update(self, tracked_objects, current_time=0.0):
        """
        Updates queue metrics based on current tracked objects.
        
        Args:
            tracked_objects (list of dict): List of tracked objects from ByteTracker.
            current_time (float): Current video timestamp in seconds.
            
        Returns:
            dict: Queue statistics including queue length, customer IDs, and waiting durations.
        """
        new_queue_ids = set()
        max_waiting_duration = 0.0
        
        for obj in tracked_objects:
            if obj["class_id"] == CLASS_PERSON:
                track_id = obj["track_id"]
                if self._is_person_in_queue(obj["bbox"]):
                    new_queue_ids.add(track_id)
                    
                    # If this person wasn't in the queue previously, record arrival time
                    if track_id not in self.queue_entry_times:
                        self.queue_entry_times[track_id] = current_time
                        
        # Clean up exit timestamps for customers who left the queue
        departed_ids = set(self.queue_entry_times.keys()) - new_queue_ids
        for dep_id in departed_ids:
            del self.queue_entry_times[dep_id]
            
        self.current_queue_ids = new_queue_ids
        queue_length = len(self.current_queue_ids)
        
        # Calculate maximum waiting duration currently in the queue
        waiting_durations = {}
        for q_id in self.current_queue_ids:
            duration = current_time - self.queue_entry_times[q_id]
            waiting_durations[q_id] = duration
            if duration > max_waiting_duration:
                max_waiting_duration = duration
                
        is_queue_delayed = max_waiting_duration > MAX_QUEUE_WAITING_TIME_SEC
        
        return {
            "queue_length": queue_length,
            "queue_customer_ids": list(self.current_queue_ids),
            "max_waiting_duration_sec": max_waiting_duration,
            "max_waiting_duration_min": max_waiting_duration / 60.0,
            "is_delayed": is_queue_delayed
        }

    def draw_queue(self, frame):
        """
        Draws the queue ROI polygon and current queue status on the video frame.
        """
        pts = np.array(self.queue_roi, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=COLOR_QUEUE, thickness=2)
        
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], COLOR_QUEUE)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        x, y = self.queue_roi[0]
        label = f"Queue Area | Waiting: {len(self.current_queue_ids)}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_QUEUE, 2)
        
        return frame


# ==========================================
# Standalone Unit Test Script for Phase 8
# ==========================================
if __name__ == "__main__":
    print("Testing QueueDetector module standalone...")
    detector = QueueDetector()
    
    # Simulate a person standing inside QUEUE_ROI [50, 50, 250, 180]
    mock_tracks = [
        {"track_id": 10, "class_id": CLASS_PERSON, "bbox": [80, 80, 120, 150]}
    ]
    
    stats = detector.update(mock_tracks, current_time=30.0)
    print(f"Queue Stats: {stats}")
    print("[+] Phase 8 queue detector module verified successfully.")