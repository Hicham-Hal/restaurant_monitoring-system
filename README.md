# 🍽️ AI-Powered Restaurant Monitoring System

A computer-vision system that watches a restaurant video (CCTV-style) and turns it into live numbers on a dashboard:

| Metric | Meaning |
|--------|---------|
| **Active Customers** | People visible in the current frame |
| **Total Unique Visitors** | Different people seen since the video started |
| **Free Tables** | Tables with nobody at them and no cleaning needed |
| **Occupied Tables** | Tables with at least one person at them |
| **Tables Needing Cleaning** | Tables whose customers just left |
| **People in Queue** | People standing inside the entrance/queue zone |

It also estimates **waiting times**, identifies likely **waiters**, and stores **history** in a database so you can view charts later.

**Technologies:** Python · YOLOv8 (Ultralytics) · ByteTrack · OpenCV · Streamlit · SQLite · Pandas · Matplotlib

---

## 📌 Table of Contents

 [Project Structure](#-project-structure)
 [How One Frame Flows Through the System](#-how-one-frame-flows-through-the-system)
 [File-by-File Explanation](#-file-by-file-explanation)
 [How Each Metric Is Calculated](#-how-each-metric-is-calculated)
 [Configuration Reference](#-configuration-reference)
 [Installation and Running](#-installation-and-running)
 [Known Limitations](#-known-limitations)

---

## 

A video is just a sequence of pictures (**frames**). For every frame the system does three things:

1. **Detect**: find the people in the picture (YOLOv8).
2. **Track**: give each person an ID that stays the same from frame to frame (ByteTrack). "Person #7" is the same person in frame 100 and frame 200.
3. **Reason**: use the IDs and the positions of people to answer questions like "is this table occupied?" or "how many people are in the queue?"

The reasoning step uses **ROIs** (Regions Of Interest). An ROI is a polygon drawn in pixel coordinates on the video, such as "this rectangle is Table 1" or "this rectangle is the queue area". The system checks whether people are standing inside these polygons.

---

## 📁 Project Structure

```
restaurant_monitoring-system/
├── app.py                    # Streamlit dashboard (the user interface)
├── requirements.txt          # Python dependencies
├── .gitignore
├── data/
│   ├── raw_videos/           # Put sample_restaurant.mp4 here
│   └── outputs/              # restaurant_stats.db is created here
├── models/                   # YOLO weights (yolov8n.pt)
└── src/
    ├── config.py             # All settings in one place
    ├── detector.py           # YOLO detection only (standalone/test module)
    ├── tracker.py            # YOLO + ByteTrack: detection AND persistent IDs
    ├── customer_counter.py   # Active customers + total unique visitors
    ├── table_monitor.py      # Table state machine: FREE / OCCUPIED / NEEDS_CLEANING
    ├── waiting_time.py       # How long each table has been in its current state
    ├── queue_detector.py     # Queue length and queue waiting time
    ├── waiter_monitor.py     # Guesses who is a waiter, detects table visits
    ├── database.py           # SQLite logging and history retrieval
    └── video_processor.py    # The pipeline that connects all modules
```

---

## 🔄 How One Frame Flows Through the System

```
                    video file (frame by frame)
                               │
                               ▼
                    ┌─────────────────────┐
                    │  tracker.py         │  YOLOv8 detects + ByteTrack gives IDs
                    └─────────┬───────────┘
                              │  list of tracked objects
                              │  (track_id, bbox, class, confidence)
   ┌──────────────┬───────────┼───────────────┬────────────────┐
   ▼              ▼           ▼               ▼                ▼
customer_      table_      queue_         waiter_         (all use the
counter.py     monitor.py  detector.py    monitor.py       same tracks)
   │              │                           
   │              ▼                           
   │         waiting_time.py  (uses table states)
   │              │
   └──────┬───────┴───────────┬───────────────┘
          ▼                   ▼
   metrics dictionary    drawing on the frame
          │                   │
          ├─► database.py     └─► annotated frame
          │   (every 30 frames)
          ▼
        app.py  →  shows video + the 6 metric boxes live
```

---

## 📄 File-by-File Explanation

### `app.py`: the dashboard

This is the file you run. It uses **Streamlit** to build the web page.

- **Sidebar:** lets the user pick a page (*Live Monitoring Dashboard* or *Historical Analytics*) and a video source (*Sample Video* or *Upload Custom Video*). An uploaded video is saved to a temporary file.
- **Live Monitoring page:**
  - Shows a video area on the left and the 6 metric boxes on the right.
  - When you click **Start Processing Stream**, it creates a `VideoProcessor` and loops over `process_stream()`. For every frame it receives, it converts the colors from BGR (OpenCV) to RGB (Streamlit), shows the annotated image, and updates the six numbers.
  - If the video file is missing, it shows an error telling you to put a video in `data/raw_videos/`.
- **Historical Analytics page:** reads all saved snapshots from the database with `fetch_logs()`, shows the last 50 rows in a table, and draws two charts: *Customer Traffic Over Time* (active vs. cumulative unique) and *Queue Length Over Time*.

---

### `src/config.py`: the control panel

Every setting lives here so you never need to hunt through the code. It contains:

- **Paths:** the model, the default video, and the database file.
- **Detection settings:** confidence threshold `0.40`, NMS IoU threshold `0.45`.
- **Class IDs:** `person = 0`, `chair = 56`, `dining table = 60` (the COCO dataset IDs used by the pretrained YOLOv8 model). `TARGET_CLASSES` keeps only these three.
- **ROIs:** polygons for `Table_1`, `Table_2`, `Table_3`, and the queue area. **These are pixel coordinates and must be adjusted to your video's resolution and camera angle.**
- **Timers:** cleaning delay (5 s), maximum service wait (300 s), maximum queue wait (600 s).
- **Colors** used when drawing.

---

### `src/detector.py`: `YOLODetector`

Loads the YOLOv8 model and detects objects in a **single frame**, with no tracking.

- `detect(frame)` runs the model, keeps only the target classes, and returns a list of dictionaries with `bbox` `[x1, y1, x2, y2]`, `confidence`, `class_id`, and `class_name`.
- `draw_detections(frame, detections)` draws a box and a label on each object.

> **Note:** the main pipeline does **not** use this file. `tracker.py` does detection and tracking in one step. `detector.py` is the earlier development phase (Phase 2) and is useful for testing detection alone.

---

### `src/tracker.py`: `ByteTracker`

The most important module. It gives people **persistent IDs**.

- Loads the YOLO model once.
- `track_frame(frame)` calls `model.track(..., tracker="bytetrack.yaml", persist=True)`. The key option is `persist=True`: it tells the tracker to remember the previous frames, so the same person keeps the same ID.
- It returns a list of tracked objects, each with `track_id`, `bbox`, `confidence`, `class_id`, and `class_name`.
- `draw_tracks(frame, objects)` draws boxes with labels like `ID 7 | person (0.86)`.

**Every other module receives this list as input.**

---

### `src/customer_counter.py`: `CustomerCounter`

Counts people using their track IDs. It keeps two things:

- `active_customer_ids`: IDs seen in the **current frame** only (reset every frame).
- `total_unique_customers`: a Python **set** of every ID ever seen. A set cannot contain duplicates, so adding the same ID again changes nothing.

`update(tracks)` returns `active_count`, `total_unique_count`, and `active_ids`. `reset()` clears everything.

---

### `src/table_monitor.py`: `TableMonitor`

Decides the state of each table using a **Finite State Machine (FSM)**, a system that is always in exactly one state and moves between states by fixed rules. The three states are `FREE`, `OCCUPIED`, and `NEEDS_CLEANING`. See [the table section below](#-free-tables-occupied-tables-and-tables-needing-cleaning) for the full logic.

- `_is_person_at_table()` checks whether a person's **feet point** is inside the table polygon.
- `update_occupancy()` applies the state rules to every table each frame and returns `{ "Table_1": "OCCUPIED", ... }`.
- `draw_tables()` colors each table polygon: green = free, red = occupied, yellow = needs cleaning.

---

### `src/waiting_time.py`: `WaitingTimeCalculator`

Measures **how long each table has been in its current state**.

- When a table's state changes (or the table is seen for the first time), it stores the current time as the start time.
- Duration = current time − start time.
- If a table has been `OCCUPIED` for more than 300 seconds, it sets `is_delayed = True`.
- Returns, per table: `state`, `duration_sec`, `duration_min`, and `is_delayed`.

---

### `src/queue_detector.py`: `QueueDetector`

Counts the people waiting in the queue zone and how long they have waited. See [the queue section below](#-people-in-queue).

- `update()` returns `queue_length`, `queue_customer_ids`, `max_waiting_duration_sec/min`, and `is_delayed` (true if someone has waited more than 600 s).
- `draw_queue()` draws the queue polygon with the label `Queue Area | Waiting: N`.

---

### `src/waiter_monitor.py`: `WaiterMonitor`

Tries to figure out **which tracked people are staff**, without needing a uniform detector. It uses a behavior rule:

1. It stores each person's position over the **last 10 seconds**.
2. If a person has more than 10 stored positions and moved more than **150 pixels** from their first to their latest position, that ID is marked as a **waiter**. Customers who sit still are not.
3. If a waiter is inside a table polygon or within 50 pixels of its edge, it records a **service interaction** (`table_name`, `waiter_id`).
4. `draw_waiters()` draws waiters in magenta with the label `Waiter ID N`.

---

### `src/database.py`: `RestaurantDatabase`

Stores history in **SQLite**, a simple database kept in one file (`data/outputs/restaurant_stats.db`).

- Creates a table `restaurant_logs` with columns: `timestamp`, `active_customers`, `unique_customers`, `free_tables`, `occupied_tables`, `cleaning_tables`, `queue_length`.
- `log_snapshot(...)` inserts one row.
- `fetch_logs()` reads everything into a Pandas DataFrame for the Historical Analytics page.

---

### `src/video_processor.py`: `VideoProcessor`

The **conductor** that connects all the modules. It creates one instance of each module, opens the video with OpenCV, and for every frame runs the following steps in order:

1. `tracker.track_frame()`: detect and track.
2. `customer_counter.update()`: active and unique counts.
3. `table_monitor.update_occupancy()`: table states.
4. `waiting_time_calc.update()`: durations and delay flags.
5. `queue_detector.update()`: queue length.
6. `waiter_monitor.update()`: waiters and service visits.
7. Count how many tables are `FREE`, `OCCUPIED`, and `NEEDS_CLEANING`.
8. Every 30 frames (about 1 second of video), save a snapshot to the database.
9. Draw all overlays (people boxes, tables, queue, waiters) on a copy of the frame.

Then it `yield`s the annotated frame and a `metrics` dictionary to `app.py`. Because it uses `yield`, the dashboard can display each frame as soon as it is ready instead of waiting for the whole video.

**Time is computed from the frame number:** `current_time = frame_idx / fps`. This is "video time", so all timers work correctly even if processing is slower than real time.

---

## 🧮 How Each Metric Is Calculated

### 👥 Active Customers
Each frame, collect the track IDs of every detected `person`. The number of IDs in this frame is the active count.

### 🧍 Total Unique Visitors
This is the key idea: **tracking IDs + a set**.

1. ByteTrack assigns an ID to each person (for example 1, 2, 3).
2. Each frame, every person's ID is added to a set.
3. The size of the set is the total unique visitors.

**Example:**

| Frame | People seen (IDs) | Active | Total unique |
|-------|-------------------|--------|--------------|
| 1 | 1, 2 | 2 | 2 |
| 2 | 2, 3 | 2 | 3 (ID 2 was already counted; ID 3 is new) |
| 3 | 3 | 1 | 3 |

Without tracking, counting would add the same person again in every frame.

### 🪑 Free Tables, Occupied Tables, and Tables Needing Cleaning

Table status comes from the **state machine** in `table_monitor.py`, plus the three counts in `video_processor.py`.

**Step 1: is someone at the table?**
For each person, take the **feet point**, the bottom-center of their bounding box `((x1 + x2) / 2, y2)`. Feet are used because a person's body can overlap a table in the image while they are actually standing behind or beside it, but their feet show where they really stand. If the feet point is inside the table polygon, someone is at that table.

**Step 2: apply the state rules**

```
                  person at table                       person at table
          ┌───────────────────────────┐        ┌───────────────────────────┐
          │                           ▼        │                           ▼
       ┌──────┐                  ┌──────────┐   │   ┌──────────────────┐
       │ FREE │                  │ OCCUPIED │───┴──►│ NEEDS_CLEANING   │
       └──────┘                  └──────────┘ person└──────────────────┘
          ▲                                    leaves         │
          └─────────────────────────────────────────────────────┘
                     5 seconds pass with nobody at the table
```

| Current state | Situation | New state |
|---------------|-----------|-----------|
| Any | A person is at the table | `OCCUPIED` |
| `OCCUPIED` | Nobody at the table anymore | `NEEDS_CLEANING` (the time is saved) |
| `NEEDS_CLEANING` | Nobody there and 5 s have passed (`CLEANING_TRIGGER_DELAY_SEC`) | `FREE` |
| `FREE` | Nobody at the table | stays `FREE` |

**Step 3: count the states**
`video_processor.py` counts how many tables are in each state:

```python
free_count     = sum(1 for s in table_states.values() if s == "FREE")
occupied_count = sum(1 for s in table_states.values() if s == "OCCUPIED")
cleaning_count = sum(1 for s in table_states.values() if s == "NEEDS_CLEANING")
```

**Example** with three tables: Table_1 has a person → `OCCUPIED`; Table_2 had a person who just left → `NEEDS_CLEANING`; Table_3 is empty → `FREE`. The dashboard shows Free = 1, Occupied = 1, Needs Cleaning = 1.

### 🚶 People in Queue
1. The queue area is a polygon (`QUEUE_ROI`) around the entrance or waiting area.
2. For every tracked person, check whether their feet point is inside this polygon.
3. The number of people inside is the **queue length**.
4. The system also remembers when each ID entered the zone, so it knows how long each person has waited. The longest wait is compared with 600 s to set the `is_delayed` flag.

---

## ⚙️ Configuration Reference

| Setting | Default | What it does |
|---------|---------|--------------|
| `CONF_THRESHOLD` | 0.40 | Ignore detections the model is less than 40% sure about |
| `IOU_THRESHOLD` | 0.45 | Removes duplicate overlapping boxes (NMS) |
| `TARGET_CLASSES` | person, chair, dining table | Which classes to keep |
| `TABLE_ROIS` | 3 rectangles | Table zones in pixels, **adjust for your video** |
| `QUEUE_ROI` | 1 rectangle | Queue zone in pixels, **adjust for your video** |
| `CLEANING_TRIGGER_DELAY_SEC` | 5.0 | Seconds a table stays `NEEDS_CLEANING` before becoming `FREE` |
| `MAX_WAITING_TIME_FOR_SERVICE_SEC` | 300 | Occupied longer than this sets the delay alert |
| `MAX_QUEUE_WAITING_TIME_SEC` | 600 | Waiting longer than this in the queue sets the delay alert |

---

## 🛠️ Installation and Running

```bash
git clone https://github.com/Hicham-Hal/restaurant_monitoring-system.git
cd restaurant_monitoring-system

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

1. Put your video at `data/raw_videos/sample_restaurant.mp4` (or upload one in the dashboard).
2. Put the model weights in `models/` (named as in `MODEL_PATH` in `config.py`).
3. Edit `TABLE_ROIS` and `QUEUE_ROI` in `src/config.py` to match your video.
4. Run:

```bash
streamlit run app.py
```

Each module also has a small self-test. For example, `python -m src.table_monitor` runs a quick demo of the table state machine.

---