import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import torch

# Define device (use GPU if available, otherwise fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Dataset
df = pd.read_csv("cleaned_data.csv")  # Replace with actual file

print(df.columns)
df.columns = df.columns.str.strip()  # Removes any extra spaces
print(df.head())

# Print Statistical Summary
statistical_summary = df.describe()
print("📊 Statistical Summary:")
print(statistical_summary)

# Print Failure Summary
failure_summary = df["Failure Type"].value_counts()
print("⚠️ Failure Summary:")
print(failure_summary)

# Features for trend detection
features = ["Air temperature","Process temperature","Rotational speed","Torque","Tool wear","Target","Type","Failure Type"]

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])

# Function to create LSTM sequences
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(df_scaled, seq_length=10)

# Split into Train/Test
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(len(features))  # Multi-output prediction
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Predict Trends
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Convert trend predictions into a log-like text format
def generate_log_text(predicted_trends):
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

# Example LSTM-generated summary
log_summary = generate_log_text(predictions[-10:])
print(log_summary)
