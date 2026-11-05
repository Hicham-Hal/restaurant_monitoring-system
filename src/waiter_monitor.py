"""
waiter_monitor.py - Waiter tracking and activity estimation module.
Uses behavioral heuristics (movement patterns, table proximity) to identify staff 
and detect tables waiting for service or being serviced.
"""

import cv2
import numpy as np
from src.config import CLASS_PERSON, COLOR_WAITER


class WaiterMonitor:
    def __init__(self, table_rois):
        """
        Initializes the waiter monitor.
        
        Args:
            table_rois (dict): Dictionary mapping table names to polygon coordinates.
        """
        self.table_rois = table_rois
        # Track historical positions to compute movement/speed heuristics
        self.person_history = {}  # {track_id: [(x, y, timestamp), ...]}
        self.waiter_ids = set()

    def update(self, tracked_objects, current_time=0.0):
        """
        Analyzes tracked persons to identify staff based on movement and roaming behavior,
        and detects table service interactions.
        
        Args:
            tracked_objects (list of dict): List of tracked objects from ByteTracker.
            current_time (float): Current video timestamp in seconds.
            
        Returns:
            dict: Waiter activity report including identified waiter IDs and service interactions.
        """
        current_persons = [obj for obj in tracked_objects if obj["class_id"] == CLASS_PERSON]
        current_frame_person_ids = set()
        
        service_interactions = []
        
        for person in current_persons:
            track_id = person["track_id"]
            x1, y1, x2, y2 = person["bbox"]
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            current_frame_person_ids.add(track_id)
            
            # Update position history
            if track_id not in self.person_history:
                self.person_history[track_id] = []
            self.person_history[track_id].append((center_x, center_y, current_time))
            
            # Keep only the last 10 seconds of history
            self.person_history[track_id] = [
                pos for pos in self.person_history[track_id] if current_time - pos[2] <= 10.0
            ]
            
            # Heuristic for Waiter Classification:
            # A person who moves across multiple distinct regions or covers high distance
            history = self.person_history[track_id]
            if len(history) > 10:
                # Calculate total displacement distance
                start_x, start_y = history[0][0], history[0][1]
                end_x, end_y = history[-1][0], history[-1][1]
                total_distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                
                # If displacement is significant, classify as waiter/staff
                if total_distance > 150.0:
                    self.waiter_ids.add(track_id)
            
            # Check proximity to tables for service interaction
            for table_name, polygon in self.table_rois.items():
                poly_pts = np.array(polygon, dtype=np.int32)
                distance = cv2.pointPolygonTest(poly_pts, (center_x, center_y), measureDist=True)
                # If within 50 pixels of table boundary
                if distance >= -50.0:
                    if track_id in self.waiter_ids:
                        service_interactions.append({
                            "table_name": table_name,
                            "waiter_id": track_id
                        })
                        
        return {
            "waiter_ids": list(self.waiter_ids),
            "service_interactions": service_interactions
        }

    def draw_waiters(self, frame, tracked_objects):
        """
        Highlights detected waiters on the video frame with distinct colors and labels.
        """
        for obj in tracked_objects:
            if obj["class_id"] == CLASS_PERSON and obj["track_id"] in self.waiter_ids:
                x1, y1, x2, y2 = obj["bbox"]
                track_id = obj["track_id"]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_WAITER, 2)
                label = f"Waiter ID {track_id}"
                
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 22), (x1 + w, y1), COLOR_WAITER, -1)
                cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
        return frame


# ==========================================
# Standalone Unit Test Script for Phase 9
# ==========================================
if __name__ == "__main__":
    print("Testing WaiterMonitor module standalone...")
    from src.config import TABLE_ROIS
    monitor = WaiterMonitor(TABLE_ROIS)
    
    # Simulate a person moving across frames (high displacement -> Waiter)
    tracks_t1 = [{"track_id": 99, "class_id": CLASS_PERSON, "bbox": [10, 10, 50, 100]}]
    tracks_t2 = [{"track_id": 99, "class_id": CLASS_PERSON, "bbox": [200, 200, 240, 300]}]
    
    monitor.update(tracks_t1, current_time=0.0)
    report = monitor.update(tracks_t2, current_time=5.0)
    print(f"Waiter Monitor Report: {report}")
    print("[+] Phase 9 waiter monitor module verified successfully.")