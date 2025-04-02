import pandas as pd
import random
import re

# Define failure types
failure_types = [
    "Overheating-induced Shutdown",
    "Tool Wear Failure",
    "Bearing Failure",
    "Torque Instability",
    "Cooling System Malfunction"
]

entries = []

for _ in range(1000):
    log_lines = []
    # Random parameter ranges for this entry
    process_temp_min = round(random.uniform(308.5, 309.2), 1)
    process_temp_max = round(random.uniform(309.3, 310.0), 1)
    speed_min = random.randint(1250, 1350)
    speed_max = random.randint(1900, 2050)
    torque_min = round(random.uniform(27.0, 30.0), 1)
    torque_max = round(random.uniform(55.0, 60.0), 1)
    air_temp_min = round(random.uniform(296.5, 297.8), 1)
    air_temp_max = round(random.uniform(299.0, 300.5), 1)
    
    # Generate 50 iterations of log lines
    for _ in range(50):
        log_lines.append(f"Process temperature fluctuating near {round(random.uniform(process_temp_min, process_temp_max), 1)}K.")
        log_lines.append(f"High rotational speed detected: {random.randint(speed_max-50, speed_max)} rpm.")
        log_lines.append(f"Torque increasing to {round(random.uniform(torque_max-2, torque_max), 1)} Nm.")
        log_lines.append(f"Air temperature rising to {round(random.uniform(air_temp_max-0.5, air_temp_max), 1)}K.")
        log_lines.append(f"Low rotational speed: {random.randint(speed_min, speed_min+50)} rpm.")
        log_lines.append(f"Torque decreasing to {round(random.uniform(torque_min, torque_min+2), 1)} Nm.")
        log_lines.append(f"Air temperature dropping to {round(random.uniform(air_temp_min, air_temp_min+0.5), 1)}K.")
    
    # Append a failure event
    failure = random.choice(failure_types)
    log_lines.append(f"Failure detected: {failure}.")
    
    log_entry = "\n".join(log_lines)
    
    # Extract numerical values from the log entry using regex
    temps = re.findall(r'(\d+\.\d+)K', log_entry)
    speeds = re.findall(r'(\d+) rpm', log_entry)
    torques = re.findall(r'(\d+\.\d+) Nm', log_entry)
    
    process_temp_min_extracted = min(map(float, temps)) if temps else None
    process_temp_max_extracted = max(map(float, temps)) if temps else None
    speed_min_extracted = min(map(int, speeds)) if speeds else None
    speed_max_extracted = max(map(int, speeds)) if speeds else None
    torque_min_extracted = min(map(float, torques)) if torques else None
    torque_max_extracted = max(map(float, torques)) if torques else None
    
    # Define multiple summary templates for variety
    summary_templates = [
        f"Process temperatures ranged from {process_temp_min_extracted}K to {process_temp_max_extracted}K, with speeds between {speed_min_extracted} rpm and {speed_max_extracted} rpm. A {failure} occurred, requiring maintenance.",
        f"Recorded system data shows temperatures varying from {process_temp_min_extracted}K to {process_temp_max_extracted}K and torque fluctuating between {torque_min_extracted} Nm and {torque_max_extracted} Nm. {failure} was detected.",
        f"Throughout the observation period, speeds ranged from {speed_min_extracted} rpm to {speed_max_extracted} rpm, with process temperatures peaking at {process_temp_max_extracted}K. {failure} may indicate equipment wear.",
        f"Operational fluctuations were noted in temperature ({process_temp_min_extracted}K to {process_temp_max_extracted}K), speed ({speed_min_extracted} rpm to {speed_max_extracted} rpm), and torque ({torque_min_extracted} Nm to {torque_max_extracted} Nm). {failure} was flagged.",
        f"Process logs reveal significant variations, with temperatures reaching {process_temp_max_extracted}K and speed peaking at {speed_max_extracted} rpm. {failure} suggests potential system instability."
    ]
    summary = random.choice(summary_templates)
    
    entries.append({
        "log_entry": log_entry,
        "summary": summary
    })

# Create a DataFrame and save as CSV
df = pd.DataFrame(entries)
csv_filename_fixed = "assets/system_logs_dataset_fixed.csv"
df.to_csv(csv_filename_fixed, index=False)
csv_filename_fixed
