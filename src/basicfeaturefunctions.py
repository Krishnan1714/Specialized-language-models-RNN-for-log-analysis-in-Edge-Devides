import pandas as pd
import re
import numpy as np
#from modelsumm2 import generate_summary
# Load dataset
file_path = "assets/predictive_maintenance.csv"  # Update with the correct path
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
# Function to execute operations
def execute_function(parsed_data, df):
    results = {}

    # Convert column names to lowercase for case-insensitive matching
    df_columns_lower = {col.lower(): col for col in df.columns}

    for feature, function in parsed_data:
        feature_lower = feature.lower()  # Convert user input feature to lowercase
        print(feature_lower)
        for column in df_columns_lower:
            if feature_lower in column:  # Check against lowercased columns
                actual_feature = df_columns_lower[column]  # Get actual column name
                values = df[actual_feature].dropna().values  # Remove NaN values
                
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
                    # results[feature] = generate_summary(df)
                    #results[feature] =generate_summary(df)
                    print( df.describe().to_string())

    
    return ouput_str(results,parsed_data)

def ouput_str(output,parsed_data):
    output_cleaned = {key: round(float(value), 3) for key, value in output.items()}
    # Format the output into a readable sentence
    # Format the output into a readable sentence
    # Format the output into a readable sentence
    output_sentence = []
    for feature, function in parsed_data:
        if feature in output_cleaned:  # Ensure feature exists
            value = output_cleaned[feature]
            
            if function == "maximum":
                output_sentence.append(f"Maximum {feature.lower()} is {value}")
            elif function == "minimum":
                output_sentence.append(f"Minimum {feature.lower()} is {value}")
            elif function in ["mean", "average"]:
                output_sentence.append(f"Average {feature.lower()} is {value}")
            elif function == "standard deviation":
                output_sentence.append(f"Standard deviation of {feature.lower()} is {value}")

    # Join the phrases into a final sentence
    final_output = ", ".join(output_sentence)
    return final_output


if __name__ == "__main__":
    # Example input from chatbot
    user_input = "torque: maximum, Summary: summary"

    # Process input and execute functions
    parsed_data = parse_input(user_input)
    output = execute_function(parsed_data, df)

    # ✅ Convert NumPy floats to standard Python floats for clean output
    
    print(execute_function(parsed_data, df))


