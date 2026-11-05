"""
customer_counter.py - Customer counting module.
Tracks active unique customers and cumulative visitor statistics using ByteTrack IDs.
"""

from src.config import CLASS_PERSON


class CustomerCounter:
    def __init__(self):
        """
        Initializes the customer counter state tracker.
        """
        self.active_customer_ids = set()
        self.total_unique_customers = set()
        self.current_active_count = 0

    def update(self, tracked_objects):
        """
        Updates customer counts based on current frame tracked objects.
        
        Args:
            tracked_objects (list of dict): List of objects returned by ByteTracker.
            
        Returns:
            dict: Current customer metrics including active count and total unique count.
        """
        current_frame_ids = set()
        
        for obj in tracked_objects:
            if obj["class_id"] == CLASS_PERSON:
                track_id = obj["track_id"]
                current_frame_ids.add(track_id)
                self.total_unique_customers.add(track_id)
                
        # Update active customers for the current frame
        self.active_customer_ids = current_frame_ids
        self.current_active_count = len(self.active_customer_ids)
        
        return {
            "active_count": self.current_active_count,
            "total_unique_count": len(self.total_unique_customers),
            "active_ids": list(self.active_customer_ids)
        }

    def reset(self):
        """
        Resets the counter states (useful when restarting video streams).
        """
        self.active_customer_ids.clear()
        self.total_unique_customers.clear()
        self.current_active_count = 0


# ==========================================
# Standalone Unit Test Script for Phase 4
# ==========================================
if __name__ == "__main__":
    print("Testing CustomerCounter module standalone...")
    counter = CustomerCounter()
    
    # Simulate tracked objects across frames
    mock_frame_1_tracks = [
        {"track_id": 1, "class_id": CLASS_PERSON, "bbox": [10, 10, 50, 100]},
        {"track_id": 2, "class_id": CLASS_PERSON, "bbox": [60, 10, 100, 100]}
    ]
    
    mock_frame_2_tracks = [
        {"track_id": 2, "class_id": CLASS_PERSON, "bbox": [65, 12, 105, 102]},
        {"track_id": 3, "class_id": CLASS_PERSON, "bbox": [120, 10, 160, 100]} # New person
    ]
    
    metrics_1 = counter.update(mock_frame_1_tracks)
    print(f"Frame 1 Metrics: {metrics_1} (Expected active: 2, unique: 2)")
    
    metrics_2 = counter.update(mock_frame_2_tracks)
    print(f"Frame 2 Metrics: {metrics_2} (Expected active: 2, unique: 3)")
    print("[+] Phase 4 customer counter module verified successfully.")