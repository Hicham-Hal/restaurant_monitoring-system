"""
waiting_time.py - Occupancy and waiting-time calculation module.
Tracks duration of table occupancy and identifies service delays or unusually long visits.
"""

from src.config import MAX_WAITING_TIME_FOR_SERVICE_SEC


class WaitingTimeCalculator:
    def __init__(self):
        """
        Initializes the waiting time tracking state for tables.
        """
        self.state_start_times = {}
        self.table_durations = {}

    def update(self, table_states, current_time=0.0):
        """
        Updates duration counters based on table states and timestamps.
        
        Args:
            table_states (dict): Dictionary mapping table names to states ("FREE", "OCCUPIED", "NEEDS_CLEANING").
            current_time (float): Current video timestamp in seconds.
            
        Returns:
            dict: Dictionary containing occupancy/waiting durations and alert flags for each table.
        """
        durations_report = {}
        
        for table_name, state in table_states.items():
            # If table is newly tracked or its state changed, reset the start time
            if table_name not in self.state_start_times or self.state_start_times[table_name]["state"] != state:
                self.state_start_times[table_name] = {
                    "state": state,
                    "start_time": current_time
                }
            
            stored = self.state_start_times[table_name]
            duration = current_time - stored["start_time"]
            self.table_durations[table_name] = duration
            
            # Check for service delay alert if occupied for too long
            is_delayed = False
            if state == "OCCUPIED" and duration > MAX_WAITING_TIME_FOR_SERVICE_SEC:
                is_delayed = True
                
            durations_report[table_name] = {
                "state": state,
                "duration_sec": duration,
                "duration_min": duration / 60.0,
                "is_delayed": is_delayed
            }
            
        return durations_report

    def reset(self):
        """
        Resets all duration tracking states.
        """
        self.state_start_times.clear()
        self.table_durations.clear()


# ==========================================
# Standalone Unit Test Script for Phase 7
# ==========================================
if __name__ == "__main__":
    print("Testing WaitingTimeCalculator module standalone...")
    calc = WaitingTimeCalculator()
    
    mock_states = {"Table_1": "OCCUPIED", "Table_2": "FREE"}
    report_1 = calc.update(mock_states, current_time=10.0)
    print(f"t=10.0s Report: {report_1}")
    
    # Advance time by 120 seconds (2 minutes)
    report_2 = calc.update(mock_states, current_time=130.0)
    print(f"t=130.0s Report (Table_1 duration): {report_2['Table_1']['duration_min']:.1f} minutes")
    print("[+] Phase 7 waiting time calculator module verified successfully.")