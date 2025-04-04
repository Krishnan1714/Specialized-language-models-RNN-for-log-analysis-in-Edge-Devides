import pandas as pd
from datetime import datetime, timedelta

#this py file to be used in the predictive_maintenance.py file for converting it to a time series dataset from a single device or sensor

# Load the CSV file
file_path = 'assets/predictive_maintenance.csv'# Update with the correct path
df = pd.read_csv(file_path)

# Generate a continuously increasing timestamp column
start_time = datetime.now()
df["Timestamp"] = [start_time + timedelta(seconds=i) for i in range(len(df))]

# Drop 'uid' and 'productid' columns
df.drop(columns=["UDI", "Product ID","Type"], errors="ignore", inplace=True)

# Reorder columns to place 'Timestamp' at the beginning
df = df[["Timestamp"] + [col for col in df.columns if col != "Timestamp"]]

output_file_path="assets/predictive_maintenance_updated.csv"
df.to_csv(output_file_path, index=False)
print(f"Updated file saved as {output_file_path}")
