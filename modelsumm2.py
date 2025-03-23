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

# Define device (use GPU if available, otherwise fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Dataset
df = pd.read_csv("cleaned_data.csv")  # Replace with actual file

print(df.columns)
df.columns = df.columns.str.strip()  # Removes any extra spaces
print(df.head())



# Features for trend detection
features = ["Air temperature","Process temperature","Rotational speed","Torque","Tool wear","Target","Type","Failure Type"]

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])

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

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

# Get the last train-test split from TimeSeriesSplit
train_index, test_index = list(tscv.split(X))[-1]  # Use only the last split
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

# Train the model only once
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Predict Trends
predictions = model.predict(X_test)
predictions_original = scaler.inverse_transform(predictions)
y_test_original = scaler.inverse_transform(y_test)


# Convert trend predictions into a log-like text format
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
    statistical_summary = df.describe()
    print("📊 Statistical Summary:")
    print(statistical_summary)

    # Print Failure Summary
    failure_summary = df["Failure Type"].value_counts()
    print("⚠️ Failure Summary:")
    print(failure_summary)

    #trends
    # Example LSTM-generated summary
    log_summary = generate_log_text(predictions[-10:])
    print(log_summary)
generate_summary(df)
# Test Model Predictions
def test_predictions():
    """Tests model predictions against actual values."""
    print("First 5 Predictions:", predictions[:5])
    print("First 5 Actual Values:", scaler.inverse_transform(y_test[:5]))
    
    single_input = np.expand_dims(X_test[0], axis=0)
    single_prediction = model.predict(single_input)
    print("Single Prediction Output:", scaler.inverse_transform(single_prediction))

test_predictions()

