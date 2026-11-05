"""
app.py - Streamlit web dashboard for the AI-Powered Restaurant Monitoring System.
Provides live video feed visualization, operational metrics, and historical charts.
"""

import streamlit as st
import cv2
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.video_processor import VideoProcessor
from src.database import RestaurantDatabase
from src.config import DEFAULT_VIDEO_PATH, DATABASE_PATH


st.set_page_config(
    page_title="AI Restaurant Monitoring System",
    page_icon="🍽️",
    layout="wide"
)

# Sidebar Configuration
st.sidebar.title("🍽️ Restaurant Monitoring")
st.sidebar.markdown("---")

app_mode = st.sidebar.selectbox("Navigation", ["Live Monitoring Dashboard", "Historical Analytics"])
video_source_option = st.sidebar.radio("Video Source", ["Sample Video", "Upload Custom Video"])

video_path = DEFAULT_VIDEO_PATH
if video_source_option == "Upload Custom Video":
    uploaded_file = st.sidebar.file_uploader("Upload MP4 Video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name


# ==========================================
# PAGE 1: LIVE MONITORING DASHBOARD
# ==========================================
if app_mode == "Live Monitoring Dashboard":
    st.title("📊 AI-Powered Restaurant Monitoring Dashboard")
    st.markdown("Real-time CCTV analytics featuring YOLOv8 detection, ByteTrack tracking, and table state monitoring.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Live Video Stream")
        video_container = st.empty()
        start_button = st.button("Start Processing Stream", type="primary")

    with col2:
        st.subheader("📌 Operational Metrics")
        metric_active_cust = st.metric(label="Active Customers", value="0")
        metric_unique_cust = st.metric(label="Total Unique Visitors", value="0")
        metric_free_tables = st.metric(label="Free Tables", value="0")
        metric_occ_tables = st.metric(label="Occupied Tables", value="0")
        metric_clean_tables = st.metric(label="Tables Needing Cleaning", value="0")
        metric_queue_len = st.metric(label="People in Queue", value="0")

    if start_button:
        if not Path(video_path).exists():
            st.error(f"[-] Video file not found: {video_path}. Please place a sample video in `data/raw_videos/`.")
        else:
            processor = VideoProcessor(video_path)
            st.info("[*] Processing video stream. Watch live updates below...")

            for annotated_frame, metrics in processor.process_stream(yield_frames=True):
                # Convert BGR OpenCV frame to RGB for Streamlit
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_container.image(rgb_frame, channels="RGB", use_container_width=True)

                # Update metrics widgets
                metric_active_cust.metric(label="Active Customers", value=metrics["active_customers"])
                metric_unique_cust.metric(label="Total Unique Visitors", value=metrics["total_unique_customers"])
                metric_free_tables.metric(label="Free Tables", value=metrics["free_tables"])
                metric_occ_tables.metric(label="Occupied Tables", value=metrics["occupied_tables"])
                metric_clean_tables.metric(label="Tables Needing Cleaning", value=metrics["cleaning_tables"])
                metric_queue_len.metric(label="People in Queue", value=metrics["queue_length"])


# ==========================================
# PAGE 2: HISTORICAL ANALYTICS
# ==========================================
elif app_mode == "Historical Analytics":
    st.title("📈 Historical Restaurant Analytics")
    st.markdown("Review logged activity snapshots stored in the SQLite database.")

    db = RestaurantDatabase(DATABASE_PATH)
    df = db.fetch_logs()

    if df.empty:
        st.warning("No historical logs found yet. Run the video processor dashboard to generate data logs.")
    else:
        st.dataframe(df.tail(50), use_container_width=True)

        st.subheader("Customer Traffic Over Time")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["timestamp"], df["active_customers"], label="Active Customers", color="orange")
        ax.plot(df["timestamp"], df["unique_customers"], label="Cumulative Unique", color="blue")
        ax.set_xlabel("Video Timestamp (seconds)")
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Queue Length Over Time")
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(df["timestamp"], df["queue_length"], label="Queue Length", color="salmon")
        ax2.set_xlabel("Video Timestamp (seconds)")
        ax2.set_ylabel("People in Queue")
        st.pyplot(fig2)