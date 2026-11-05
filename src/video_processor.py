"""
video_processor.py - Main video processing pipeline.
Coordinates detection, tracking, counting, table monitoring, waiting time, 
queue detection, waiter monitoring, and database logging.
"""

import cv2
import time
from src.tracker import ByteTracker
from src.customer_counter import CustomerCounter
from src.table_monitor import TableMonitor
from src.waiting_time import WaitingTimeCalculator
from src.queue_detector import QueueDetector
from src.waiter_monitor import WaiterMonitor
from src.database import RestaurantDatabase
from src.config import TABLE_ROIS, QUEUE_ROI


class VideoProcessor:
    def __init__(self, video_source):
        """
        Initializes all pipeline modules and video capture source.
        
        Args:
            video_source (str or int): Path to video file or camera index (e.g., 0).
        """
        self.video_source = video_source
        self.tracker = ByteTracker()
        self.customer_counter = CustomerCounter()
        self.table_monitor = TableMonitor(TABLE_ROIS)
        self.waiting_time_calc = WaitingTimeCalculator()
        self.queue_detector = QueueDetector(QUEUE_ROI)
        self.waiter_monitor = WaiterMonitor(TABLE_ROIS)
        self.db = RestaurantDatabase()

    def process_stream(self, yield_frames=False):
        """
        Processes the video stream frame by frame, running analytics and yielding annotated frames.
        
        Args:
            yield_frames (bool): If True, yields annotated frames and metrics for Streamlit dashboard.
        """
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            raise FileNotFoundError(f"[-] Error: Could not open video source {self.video_source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Fallback default FPS

        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            current_time = frame_idx / fps
            frame_idx += 1

            # 1. Run ByteTracker
            tracks = self.tracker.track_frame(frame)

            # 2. Update Customer Counter
            customer_metrics = self.customer_counter.update(tracks)

            # 3. Update Table Occupancy & FSM States
            table_states = self.table_monitor.update_occupancy(tracks, current_time)

            # 4. Update Waiting Times
            waiting_report = self.waiting_time_calc.update(table_states, current_time)

            # 5. Update Queue Detection
            queue_stats = self.queue_detector.update(tracks, current_time)

            # 6. Update Waiter Monitoring & Service Interactions
            waiter_report = self.waiter_monitor.update(tracks, current_time)

            # 7. Count table state distributions
            free_count = sum(1 for s in table_states.values() if s == "FREE")
            occupied_count = sum(1 for s in table_states.values() if s == "OCCUPIED")
            cleaning_count = sum(1 for s in table_states.values() if s == "NEEDS_CLEANING")

            # 8. Log snapshot to database every 30 frames (~1 second)
            if frame_idx % 30 == 0:
                self.db.log_snapshot(
                    timestamp=current_time,
                    active_cust=customer_metrics["active_count"],
                    unique_cust=customer_metrics["total_unique_count"],
                    free_t=free_count,
                    occupied_t=occupied_count,
                    cleaning_t=cleaning_count,
                    queue_len=queue_stats["queue_length"]
                )

            # 9. Draw visual overlays on frame
            annotated_frame = frame.copy()
            annotated_frame = self.tracker.draw_tracks(annotated_frame, tracks)
            annotated_frame = self.table_monitor.draw_tables(annotated_frame)
            annotated_frame = self.queue_detector.draw_queue(annotated_frame)
            annotated_frame = self.waiter_monitor.draw_waiters(annotated_frame, tracks)

            # Package metrics for dashboard rendering
            metrics = {
                "timestamp": current_time,
                "active_customers": customer_metrics["active_count"],
                "total_unique_customers": customer_metrics["total_unique_count"],
                "free_tables": free_count,
                "occupied_tables": occupied_count,
                "cleaning_tables": cleaning_count,
                "queue_length": queue_stats["queue_length"],
                "table_states": table_states,
                "waiting_report": waiting_report,
                "waiter_report": waiter_report
            }

            if yield_frames:
                yield annotated_frame, metrics

        cap.release()