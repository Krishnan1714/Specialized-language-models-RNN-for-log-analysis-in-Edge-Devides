import pandas as pd
import random
import re

# --- Template Functions for Log Lines ---
def gen_process_temp():
    temp = round(random.uniform(309.0, 310.0), 1)
    return f"Process temperature exceeds threshold at {temp}K."

def gen_high_rot_speed():
    rpm = random.randint(1800, 2000)
    return f"High rotational speed detected: {rpm} rpm."

def gen_low_rot_speed():
    rpm = random.randint(1300, 1400)
    return f"Low rotational speed: {rpm} rpm."

def gen_torque_increasing():
    torque = round(random.uniform(45.0, 56.0), 1)
    return f"Torque increasing to {torque} Nm."

def gen_torque_decreasing():
    torque = round(random.uniform(25.0, 30.0), 1)
    return f"Torque decreasing to {torque} Nm."

def gen_air_temp_rising():
    temp = round(random.uniform(299.0, 300.0), 1)
    return f"Air temperature rising to {temp}K."

def gen_air_temp_dropping():
    temp = round(random.uniform(297.0, 298.0), 1)
    return f"Air temperature dropping to {temp}K."

def gen_normal():
    return "System operating within normal parameters."

def gen_failure():
    return "Failure detected: Tool Wear Failure."

# --- Group Generation Patterns ---
def pattern1():
    # Pattern 1: Process temp high, high rot speed, torque increasing, failure.
    logs = [gen_process_temp(), gen_high_rot_speed(), gen_torque_increasing(), gen_failure()]
    return logs

def pattern2():
    # Pattern 2: Low rot speed, process temp high, torque increasing, air temp rising.
    logs = [gen_low_rot_speed(), gen_process_temp(), gen_torque_increasing(), gen_air_temp_rising()]
    return logs

def pattern3():
    # Pattern 3: Air temp dropping, low rot speed, torque decreasing, normal operation.
    logs = [gen_air_temp_dropping(), gen_low_rot_speed(), gen_torque_decreasing(), gen_normal()]
    return logs

def generate_group():
    # Randomly choose one of the three patterns for each group
    pattern = random.choice([pattern1, pattern2, pattern3])
    return pattern()

# --- Helper Functions to Extract Numeric Values ---
def extract_numeric(log, field):
    # Use regex to extract numeric value after a field label (case-insensitive)
    pattern = rf"{field}:\s*([\d\.]+)"
    match = re.search(pattern, log, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except:
            return None
    return None

def compute_range(logs, field):
    values = [extract_numeric(log, field) for log in logs]
    values = [v for v in values if v is not None]
    if values:
        return min(values), max(values)
    return None, None

# --- Function to Generate Concise Summary with Ranges ---
def generate_concise_summary(logs):
    # Compute ranges for rotational speed and torque
    rot_min, rot_max = compute_range(logs, "rotational speed")
    torque_min, torque_max = compute_range(logs, "Torque")
    
    parts = []
    # Include process temperature info if any log mentions it
    if any("Process temperature exceeds threshold" in log for log in logs):
        parts.append("process temperature frequently exceeded the threshold")
    
    if rot_min is not None and rot_max is not None:
        parts.append(f"fluctuating rotational speeds ({int(rot_min)}–{int(rot_max)} rpm)")
        
    if torque_min is not None and torque_max is not None:
        parts.append(f"torque ranging from {torque_min} to {torque_max} Nm")
    
    # Determine failure message if any log line reports a failure
    failure_msgs = [log for log in logs if "Failure detected:" in log]
    failure_text = ""
    if failure_msgs:
        # Take the first failure occurrence
        match = re.search(r"Failure detected:\s*([^\.]+)", failure_msgs[0], re.IGNORECASE)
        if match:
            failure_text = f"A {match.group(1).strip()} was detected."
    
    # Build the summary
    if parts:
        summary = "The " + ", ".join(parts) + ". " + (failure_text if failure_text else "")
    else:
        summary = "System operating normally."
    
    return summary

# --- Generate the Dataset ---
num_groups = 100  # Adjust the number of groups as needed (each group is 4 log lines)
data = []
for _ in range(num_groups):
    logs = generate_group()
    log_block = "\n".join(logs)
    summary = generate_concise_summary(logs)
    data.append([log_block, summary])

df = pd.DataFrame(data, columns=["Logs", "Summary"])

# Show a sample of 3 groups
print(df.sample(3, random_state=42).to_string(index=False))

# Save the dataset as CSV
output_file = "assets/synthetic_timeseries_with_concise_summary.csv"
df.to_csv(output_file, index=False)
print(f"\nSynthetic dataset saved as '{output_file}'")
