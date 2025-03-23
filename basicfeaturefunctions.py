import pandas as pd
import re
import numpy as np
from modelsumm2 import generate_summary
# Load dataset
file_path = "cleaned_data.csv"  # Update with the correct path
df = pd.read_csv(file_path)

# Function to parse input
def parse_input(user_input):
    """
    Parses user input into a structured list of (feature, function).
    Example: "Torque: maximum, Air temperature: mean" -> [('Torque', 'maximum'), ('Air temperature', 'mean')]
    """
    pattern = r'([\w\s]+):\s*(\w+)'  # Matches "Feature: Function"
    matches = re.findall(pattern, user_input)
    return [(feature.strip(), function.strip().lower()) for feature, function in matches]

# Function to execute operations
def execute_function(parsed_data, df):
    results = {}

    for feature, function in parsed_data:
        if feature in df.columns:
            values = df[feature].dropna().values  # Remove NaN values
            
            # Perform requested operation
            if function == "maximum":
                results[feature] = np.max(values)
            elif function == "minimum":
                results[feature] = np.min(values)
            elif function in ["mean", "average"]:
                results[feature] = np.mean(values)
            elif function == "standard deviation":
                results[feature] = np.std(values)
            elif function == "summary":
                results[feature] = df.describe()
    
    return results

# Example input from chatbot
user_input = "Torque: maximum, Air temperature: minimum, Rotational speed: mean, Tool wear: std, Summary : summary"

# Process input and execute functions
parsed_data = parse_input(user_input)
output = execute_function(parsed_data, df)

# ✅ Convert NumPy floats to standard Python floats for clean output
output_cleaned = {key: float(value) for key, value in output.items()}

# Display cleaned results
print("Results:", output_cleaned)
