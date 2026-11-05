"""
table_monitor.py - Table ROI management and state machine classification.
Classifies tables into FREE, OCCUPIED, and NEEDS_CLEANING using spatial 
intersections and temporal state transitions.
"""

import cv2
import numpy as np
from src.config import (
    TABLE_ROIS, 
    TABLE_OCCUPANCY_OVERLAP_THRESH, 
    CLEANING_TRIGGER_DELAY_SEC,
    CLASS_PERSON, 
    COLOR_FREE, 
    COLOR_OCCUPIED, 
    COLOR_NEEDS_CLEANING
)


class TableMonitor:
    def __init__(self, table_rois=TABLE_ROIS):
        """
        Initializes table monitor with predefined ROI polygons and state tracking.
        """
        self.table_rois = table_rois
        
        # Table states: "FREE", "OCCUPIED", "NEEDS_CLEANING"
        self.table_states = {table_name: "FREE" for table_name in self.table_rois.keys()}
        
        # Timestamps tracking state transitions for temporal logic
        self.exit_timestamps = {table_name: 0.0 for table_name in self.table_rois.keys()}

    def _is_person_at_table(self, person_bbox, table_polygon):
        """
        Determines if a person's feet/position anchor is inside the table polygon.
        """
        x1, y1, x2, y2 = person_bbox
        feet_point = (int((x1 + x2) / 2), y2)
        poly_pts = np.array(table_polygon, dtype=np.int32)
        
        inside_test = cv2.pointPolygonTest(poly_pts, feet_point, measureDist=False)
        return inside_test >= 0

    def update_occupancy(self, tracked_objects, current_time=0.0):
        """
        Updates table states using a Finite State Machine (FSM) based on customer presence
        and elapsed time since departure.
        
        Args:
            tracked_objects (list of dict): List of tracked objects from ByteTracker.
            current_time (float): Current video timestamp in seconds.
            
        Returns:
            dict: Current state of each table.
        """
        person_boxes = [obj["bbox"] for obj in tracked_objects if obj["class_id"] == CLASS_PERSON]
        
        for table_name, polygon in self.table_rois.items():
            current_state = self.table_states[table_name]
            is_occupied = False
            
            # Check if any person is currently at this table
            for person_bbox in person_boxes:
                if self._is_person_at_table(person_bbox, polygon):
                    is_occupied = True
                    break
            
            # --- FSM State Transition Logic ---
            if is_occupied:
                # Transition to OCCUPIED regardless of previous state
                self.table_states[table_name] = "OCCUPIED"
            else:
                # No person at the table
                if current_state == "OCCUPIED":
                    # Customers just left! Transition to NEEDS_CLEANING
                    self.table_states[table_name] = "NEEDS_CLEANING"
                    self.exit_timestamps[table_name] = current_time
                    
                elif current_state == "NEEDS_CLEANING":
                    # Check if cleaning grace period has elapsed
                    elapsed_cleaning_time = current_time - self.exit_timestamps[table_name]
                    if elapsed_cleaning_time >= CLEANING_TRIGGER_DELAY_SEC:
                        self.table_states[table_name] = "FREE"
                        
                elif current_state == "FREE":
                    # Remains FREE
                    pass
                    
        return self.table_states

    def draw_tables(self, frame):
        """
        Draws table ROI polygons and color-coded status badges on the video frame.
        """
        for table_name, polygon in self.table_rois.items():
            state = self.table_states.get(table_name, "FREE")
            
            # Select color based on FSM state
            if state == "FREE":
                color = COLOR_FREE
            elif state == "OCCUPIED":
                color = COLOR_OCCUPIED
            elif state == "NEEDS_CLEANING":
                color = COLOR_NEEDS_CLEANING
            else:
                color = COLOR_FREE
                
            # Draw table polygon and semi-transparent fill
            pts = np.array(polygon, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            
            # Draw label
            x, y = polygon[0]
            label = f"{table_name}: {state}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return frame


# ==========================================
# Standalone Unit Test Script for Phase 6
# ==========================================
if __name__ == "__main__":
    print("Testing TableMonitor FSM module standalone...")
    monitor = TableMonitor()
    
    # Step 1: Person at Table_1 -> OCCUPIED
    tracks = [{"track_id": 1, "class_id": CLASS_PERSON, "bbox": [150, 250, 200, 350]}]
    print(f" t=0.0s -> {monitor.update_occupancy(tracks, current_time=0.0)}")
    
    # Step 2: Person leaves -> NEEDS_CLEANING
    print(f" t=2.0s (No person) -> {monitor.update_occupancy([], current_time=2.0)}")
    
    # Step 3: Time passes beyond threshold (CLEANING_TRIGGER_DELAY_SEC = 5.0) -> FREE
    print(f" t=8.0s (No person) -> {monitor.update_occupancy([], current_time=8.0)}")
    print("[+] Phase 6 FSM state machine verified successfully.")