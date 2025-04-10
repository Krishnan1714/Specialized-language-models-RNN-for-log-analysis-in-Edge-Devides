import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import joblib

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Drop unnecessary columns
    df.drop(['Timestamp'], axis=1, inplace=True, errors='ignore')

    # Rename columns
    df = df.rename(columns={
        'Air temperature [K]': 'Air temperature',
        'Process temperature [K]': 'Process temperature',
        'Rotational speed [rpm]': 'Rotational speed',
        'Torque [Nm]': 'Torque', 
        'Tool wear [min]': 'Tool wear'
    })

    # Remove misclassified instances
    df = df[~((df['Target'] == 1) & (df['Failure Type'] == 'No Failure'))]
    df = df[~((df['Target'] == 0) & (df['Failure Type'] == 'Random Failures'))]

    # Reset index
    df.reset_index(inplace=True, drop=True)

    # Define features before encoding
    features = ["Air temperature", "Process temperature", "Rotational speed", 
                "Torque", "Tool wear", "Target", "Failure Type"]

    # Ordinal Encoding for categorical variables
    failure_type_order = ["No Failure", "Heat Dissipation Failure", "Power Failure", "Overstrain Failure", "Random Failures","Tool Wear Failure"]
    # ord_enc = OrdinalEncoder(categories=[failure_type_order])
    # df[["Failure Type"]] = ord_enc.fit_transform(df[["Failure Type"]])
    # df["Failure Type"] = df["Failure Type"].astype(int)  # Ensure integer encoding

    # Drop duplicate rows if any and handle missing values
    df = df.drop_duplicates().dropna()

    return df

def scale_features(df):
    # Define feature groups based on distribution
    standard_features = ["Torque", "Process temperature", "Air temperature"]
    robust_features = ["Rotational speed", "Tool wear"]

    # Apply StandardScaler to Gaussian-like features
    scaler_standard = StandardScaler()
    df[standard_features] = scaler_standard.fit_transform(df[standard_features])

    # Apply RobustScaler to skewed features
    scaler_robust = RobustScaler()
    df[robust_features] = scaler_robust.fit_transform(df[robust_features])
    return df

def save_data(df, file_path):
    df.to_csv(file_path, index=False)

def clean_and_scale_data(csv_file_path):
    df = load_data(csv_file_path)
    df = preprocess_data(df)
    df = scale_features(df)
    return df

def main():
    file_path = 'assets/predictive_maintenance_updated.csv'
    df = load_data(file_path)
    df = preprocess_data(df)
    df = scale_features(df)
    save_data(df, 'assets/cleaned_data_scaled.csv')

    print("Initial Dataset Info:")
    print(df.info())

    # Generate histograms
    df[["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]].hist(figsize=(12, 8), bins=30)
    plt.show()

    print("✅ Data preprocessing complete with appropriate scaling!")

if __name__ == "__main__":
    main()