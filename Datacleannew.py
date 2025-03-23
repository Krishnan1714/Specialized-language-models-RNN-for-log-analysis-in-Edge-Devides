import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt



# Show plots without blocking execution
plt.show(block=False)  # Alternative: plt.pause(1)


# Load Data
df = pd.read_csv('assets/predictive_maintenance.csv')

# Drop unnecessary columns
df.drop(['UDI', 'Product ID'], axis=1, inplace=True)

# Rename columns
df = df.rename(columns={
    'Air temperature [K]': 'Air temperature',
    'Process temperature [K]': 'Process temperature',
    'Rotational speed [rpm]': 'Rotational speed',
    'Torque [Nm]': 'Torque', 
    'Tool wear [min]': 'Tool wear'
})

print("Initial Dataset Info:")
print(df.info())

# Generate histograms
df[["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]].hist(figsize=(12, 8), bins=30)
plt.show()


# Remove misclassified instances
df = df[~((df['Target'] == 1) & (df['Failure Type'] == 'No Failure'))]
df = df[~((df['Target'] == 0) & (df['Failure Type'] == 'Random Failures'))]

# Reset index
df.reset_index(inplace=True, drop=True)

# Define features before encoding
features = ["Air temperature", "Process temperature", "Rotational speed", 
            "Torque", "Tool wear", "Target", "Type", "Failure Type"]

# Ordinal Encoding for categorical variables
ord_enc = OrdinalEncoder(categories=[['L', 'M', 'H'], df['Failure Type'].unique().tolist()])
df[['Type', 'Failure Type']] = ord_enc.fit_transform(df[['Type', 'Failure Type']])

# Drop duplicate rows if any
df = df.drop_duplicates()

# Handle missing values (Drop or Fill)
df = df.dropna()

# Define feature groups based on distribution
standard_features = ["Torque", "Process temperature", "Air temperature"]
robust_features = ["Rotational speed", "Tool wear"]

# Apply StandardScaler to Gaussian-like features
scaler_standard = StandardScaler()
df[standard_features] = scaler_standard.fit_transform(df[standard_features])

# Apply RobustScaler to skewed features
scaler_robust = RobustScaler()
df[robust_features] = scaler_robust.fit_transform(df[robust_features])

# Save the cleaned and scaled dataset
df.to_csv("cleaned_data_scaled.csv", index=False)
print("✅ Data preprocessing complete with appropriate scaling!")
