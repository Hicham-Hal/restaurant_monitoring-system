
from pathlib import Path

# ==========================================
# 1. PATH CONFIGURATIONS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
OUTPUTS_DIR = DATA_DIR / "outputs"

MODELS_DIR = BASE_DIR / "models"
DEFAULT_YOLO_MODEL = "yolov8n.pt" 
MODEL_PATH = MODELS_DIR / DEFAULT_YOLO_MODEL

# Database path for historical statistics logging
DATABASE_PATH = OUTPUTS_DIR / "restaurant_stats.db"

# Default video file for testing
DEFAULT_VIDEO_PATH = RAW_VIDEOS_DIR / "sample_restaurant.mp4"


# ==========================================
# 2. MODEL & DETECTION PARAMETERS
# ==========================================
# Detection confidence threshold (0.0 - 1.0)
CONF_THRESHOLD = 0.40

# Non-Maximum Suppression (NMS) IoU threshold
IOU_THRESHOLD = 0.45

# COCO dataset class indices used by default YOLOv8 pretrained model
# 0: person, 56: chair, 60: dining table
CLASS_PERSON = 0
CLASS_CHAIR = 56
CLASS_DINING_TABLE = 60

# Filter detection to only relevant classes (saves computation time)
TARGET_CLASSES = [CLASS_PERSON, CLASS_CHAIR, CLASS_DINING_TABLE]


# ==========================================
# 3. SPATIAL REGIONS OF INTEREST (ROIs)
# Coordinates represented as [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] polygons
# Note: Adjust these pixel coordinates according to your specific input video resolution.
# ==========================================

# Table ROIs: Polygon coordinates bounding each physical table
TABLE_ROIS = {
    "Table_1": [
        (100, 200), (300, 200), (300, 400), (100, 400)
    ],
    "Table_2": [
        (400, 200), (600, 200), (600, 400), (400, 400)
    ],
    "Table_3": [
        (700, 200), (900, 200), (900, 400), (700, 400)
    ]
}

# Queue ROI: Polygon coordinates bounding the waiting/entrance area
QUEUE_ROI = [
    (50, 50), (250, 50), (250, 180), (50, 180)
]


# ==========================================
# 4. OPERATIONAL THRESHOLDS & TIMERS
# ==========================================
# Overlap threshold (IoU/Intersection ratio) to consider a person at a table
TABLE_OCCUPANCY_OVERLAP_THRESH = 0.15

# Time threshold (in seconds) after occupants leave before triggering 'NEEDS CLEANING'
CLEANING_TRIGGER_DELAY_SEC = 5.0

# Waiting time warning thresholds (in seconds)
MAX_WAITING_TIME_FOR_SERVICE_SEC = 300.0  # 5 minutes without waiter interaction
MAX_QUEUE_WAITING_TIME_SEC = 600.0        # 10 minutes in line


# ==========================================
# 5. VISUALIZATION COLORS (BGR Format for OpenCV)
# ==========================================
COLOR_PERSON = (255, 150, 0)       # Blue-Orange
COLOR_WAITER = (255, 0, 255)       # Magenta
COLOR_FREE = (0, 255, 0)           # Green
COLOR_OCCUPIED = (0, 0, 255)       # Red
COLOR_NEEDS_CLEANING = (0, 255, 255) # Yellow
COLOR_QUEUE = (250, 128, 114)      # Salmon Pink
COLOR_BBOX_DEFAULT = (200, 200, 200) # Gray