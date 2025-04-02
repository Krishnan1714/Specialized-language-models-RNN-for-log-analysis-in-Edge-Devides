import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import torch
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler

# Define device (use GPU if available, otherwise fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Dataset (this file is already cleaned and scaled)
df = pd.read_csv("cleaned_data_scaled.csv")  # Replace with actual file
df.columns = df.columns.str.strip()  # Remove any extra spaces
print(df.columns)
print(df.head())

# Features for trend detection in desired order
features = ["Air temperature", "Process temperature", "Rotational speed", 
            "Torque", "Tool wear", "Target", "Type", "Failure Type"]

# Use the data as a numpy array
df_scaled = df[features].values

# Function to create LSTM sequences
def create_sequences(data, seq_length=10):
    """Creates sequences of data for LSTM training, ensuring dataset is large enough."""
    if len(data) < seq_length:
        raise ValueError("Dataset too small for the specified sequence length.")
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(df_scaled, seq_length=10)

# Time Series Cross-Validation: use the last split for testing
tscv = TimeSeriesSplit(n_splits=5)
train_index, test_index = list(tscv.split(X))[-1]
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

# LSTM Model
def build_model():
    """Builds and compiles an LSTM model."""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(len(features))
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

model = build_model()
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Predict Trends
predictions = model.predict(X_test)

# --- Inverse Scaling ---
# Our desired grouping based on features order:
#   Index 0: Air temperature       --> StandardScaler
#   Index 1: Process temperature   --> StandardScaler
#   Index 2: Rotational speed      --> RobustScaler
#   Index 3: Torque                --> StandardScaler
#   Index 4: Tool wear             --> RobustScaler
#   Index 5-7: Target, Type, Failure Type --> unscaled

# Extract groups from predictions
pred_standard = predictions[:, [0, 1, 3]]   # Standard-scaled: columns 0,1,3
pred_robust   = predictions[:, [2, 4]]      # Robust-scaled: columns 2,4
pred_unscaled = predictions[:, 5:]          # Unscaled: columns 5,6,7

# Fit scalers on the training targets (y_train) using the same grouping
scaler_standard = StandardScaler()
scaler_robust = RobustScaler()
scaler_standard.fit(y_train[:, [0, 1, 3]])
scaler_robust.fit(y_train[:, [2, 4]])

# Inverse transform each group
pred_standard_original = scaler_standard.inverse_transform(pred_standard)
pred_robust_original = scaler_robust.inverse_transform(pred_robust)

# Combine the groups in the original order:
# [Air temperature (std, col0), Process temperature (std, col1),
#  Rotational speed (robust, col0), Torque (std, col2), Tool wear (robust, col1),
#  Target, Type, Failure Type (unscaled)]
predictions_original = np.hstack((
    pred_standard_original[:, [0, 1]],    # Air temperature, Process temperature
    pred_robust_original[:, [0]],         # Rotational speed
    pred_standard_original[:, [2]],         # Torque
    pred_robust_original[:, [1]],         # Tool wear
    pred_unscaled                         # Target, Type, Failure Type
))

# Save the model after training
model.save("trained_model.h5")
print("Model saved successfully.")

# Generate a log-like text summary (example uses the last two predictions)
def generate_log_text(predicted_trends):
    """Generates a textual summary of predicted trends."""
    log_summary = "🔍 System Log Summary:\n"
    for i, feature in enumerate(features):
        rate_of_change = predicted_trends[-1][i] - predicted_trends[-2][i]
        if rate_of_change > 0:
            log_summary += f"📈 {feature.replace('_', ' ').title()} is increasing by {rate_of_change:.2f}.\n"
        elif rate_of_change < 0:
            log_summary += f"📉 {feature.replace('_', ' ').title()} is decreasing by {abs(rate_of_change):.2f}.\n"
        else:
            log_summary += f"✅ {feature.replace('_', ' ').title()} remains stable.\n"
    return log_summary

def generate_summary(df):
    # Print Statistical Summary
    print("📊 Statistical Summary:")
    print(df.describe())
    # Print Failure Summary
    print("⚠️ Failure Summary:")
    print(df["Failure Type"].value_counts())
    # LSTM-generated trend summary (using last 10 predictions from the raw predictions)
    log_summary = generate_log_text(predictions[-10:])
    print(log_summary)

generate_summary(df)

# Test Model Predictions and print them like the original CSV
def test_predictions():
    # Inverse transform the actual y_test in the same way:
    y_test_standard = scaler_standard.inverse_transform(y_test[:, [0, 1, 3]])
    y_test_robust = scaler_robust.inverse_transform(y_test[:, [2, 4]])
    y_test_unscaled = y_test[:, 5:]
    y_test_original = np.hstack((
        y_test_standard[:, [0, 1]],  # Air temperature, Process temperature
        y_test_robust[:, [0]],       # Rotational speed
        y_test_standard[:, [2]],       # Torque
        y_test_robust[:, [1]],       # Tool wear
        y_test_unscaled             # Target, Type, Failure Type
    ))
    
    columns_order = ["Air temperature", "Process temperature", "Rotational speed",
                     "Torque", "Tool wear", "Target", "Type", "Failure Type"]
    
    # Create DataFrames for a neat, CSV-like print
    df_predictions = pd.DataFrame(predictions_original, columns=columns_order)
    df_actual = pd.DataFrame(y_test_original, columns=columns_order)
    
    print("First 5 Predictions:")
    print(df_predictions.head())
    print("\nFirst 5 Actual Values:")
    print(df_actual.head())
    
    # Single prediction demonstration:
    single_input = np.expand_dims(X_test[0], axis=0)
    single_prediction = model.predict(single_input)
    single_standard = single_prediction[:, [0, 1, 3]]
    single_robust = single_prediction[:, [2, 4]]
    single_unscaled = single_prediction[:, 5:]
    single_standard_original = scaler_standard.inverse_transform(single_standard)
    single_robust_original = scaler_robust.inverse_transform(single_robust)
    single_prediction_original = np.hstack((
        single_standard_original[:, [0, 1]],
        single_robust_original[:, [0]],
        single_standard_original[:, [2]],
        single_robust_original[:, [1]],
        single_unscaled
    ))
    df_single = pd.DataFrame(single_prediction_original, columns=columns_order)
    print("\nSingle Prediction Output:")
    print(df_single)

test_predictions()
