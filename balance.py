import pandas as pd
from sklearn.utils import resample

# Load the dataset
df = pd.read_csv("final_log_analysis_prompts_1.csv")  # Update with correct path

# Ensure category column is treated as a string
df["Category"] = df["Category"].astype(str)

# Find the maximum class count
max_count = df["Category"].value_counts().max()

# Perform oversampling
df_balanced = df.copy()
for category in df["Category"].unique():
    df_minority = df[df["Category"] == category]
    if len(df_minority) < max_count:
        df_minority_upsampled = resample(df_minority, replace=True, n_samples=max_count, random_state=42)
        df_balanced = pd.concat([df_balanced, df_minority_upsampled], ignore_index=True)

# Save to CSV
df_balanced.to_csv("oversampled_log_analysis.csv", index=False)

print("✅ Oversampled dataset saved as 'oversampled_log_analysis.csv'")
