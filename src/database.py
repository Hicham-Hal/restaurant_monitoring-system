"""
database.py - SQLite database module for historical restaurant analytics logging.
Stores timestamped snapshots of customer counts, table states, and queue lengths.
"""

import sqlite3
import pandas as pd
from src.config import DATABASE_PATH


class RestaurantDatabase:
    def __init__(self, db_path=DATABASE_PATH):
        """
        Initializes SQLite connection and creates tables if they do not exist.
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _create_tables(self):
        """
        Creates the historical metrics table.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS restaurant_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                active_customers INTEGER,
                unique_customers INTEGER,
                free_tables INTEGER,
                occupied_tables INTEGER,
                cleaning_tables INTEGER,
                queue_length INTEGER
            )
        """)
        conn.commit()
        conn.close()

    def log_snapshot(self, timestamp, active_cust, unique_cust, free_t, occupied_t, cleaning_t, queue_len):
        """
        Logs a single frame/interval metric snapshot to the database.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO restaurant_logs (timestamp, active_customers, unique_customers, free_tables, occupied_tables, cleaning_tables, queue_length)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, active_cust, unique_cust, free_t, occupied_t, cleaning_t, queue_len))
        conn.commit()
        conn.close()

    def fetch_logs(self):
        """
        Retrieves all historical logs as a pandas DataFrame for dashboard visualization.
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM restaurant_logs", conn)
        conn.close()
        return df


# ==========================================
# Standalone Unit Test Script for Database
# ==========================================
if __name__ == "__main__":
    print("Testing RestaurantDatabase module standalone...")
    db = RestaurantDatabase()
    db.log_snapshot(0.0, 5, 5, 2, 1, 0, 1)
    logs_df = db.fetch_logs()
    print(f"Logged records count: {len(logs_df)}")
    print("[+] Phase 10 database module verified successfully.")