import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, TensorDataset
import joblib

# Load the StandardScaler used during preprocessing
scaler_standard = joblib.load("scaler_standard.pkl")


# Define device (use GPU if available, otherwise fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the cleaned and scaled dataset
df = pd.read_csv("cleaned_data_scaled.csv")

# Define features and target
target_column = "Target"
features = [col for col in df.columns if col != target_column]

# Function to create sequences for LSTM training
def create_sequences(data, target, seq_length=20):  
    """Creates sequences of data for LSTM training."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])  # Input sequence
        y.append(target[i + seq_length])  # Corresponding target
    return np.array(X), np.array(y)

# Create sequences
X, y = create_sequences(df[features].values, df[target_column].values, seq_length=20)

# Use TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)
train_index, test_index = list(tscv.split(X))[-1]  # Get last train-test split
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

# Convert to PyTorch tensors and move to device
X_train, X_test = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(X_test, dtype=torch.float32).to(device)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)

# Create DataLoader for memory-efficient mini-batch training
batch_size = 32  
train_dataset = TensorDataset(X_train, y_train.unsqueeze(1))  # Match LSTM expected shape
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define Optimized LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, output_size=1):  
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Initialize Model
model = LSTMModel(input_size=len(features)).to(device)

# Define Loss Function and Optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training Function with Mini-Batches and Early Stopping
def train_model(model, train_loader, epochs=20, patience=5):
    model.train()
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)  
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Train the Model
train_model(model, train_loader)

# Predict Next Log Using Last 20 Logs (RAG-Inspired)
def retrieve_last_n_logs(data, n=20):
    """Retrieve the last N logs from the dataset."""
    return data[-n:] if len(data) >= n else data

def predict_with_retrieval(model, full_data, seq_length=20):
    """Predicts the next value using retrieved last N logs."""
    retrieved_logs = retrieve_last_n_logs(full_data, seq_length)
    retrieved_logs = np.array(retrieved_logs).reshape(1, seq_length, -1)
    retrieved_logs = torch.tensor(retrieved_logs, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        prediction = model(retrieved_logs).cpu().numpy()
        # Assume 'predictions' is the LSTM output in scaled format
        real_prediction = scaler_standard.inverse_transform(prediction)

    
    return real_prediction

def rag_time_series_pipeline(model, df, seq_length=20):
    """RAG-like pipeline for LSTM-based time-series forecasting."""
    retrieved_logs = retrieve_last_n_logs(df[features].values, seq_length)
    next_prediction = predict_with_retrieval(model, retrieved_logs, seq_length)

    print("📊 Next-Step Forecast Based on Retrieved Logs:")
    print(next_prediction)
    
    return next_prediction

# Predict Next Log
predicted_next_log = rag_time_series_pipeline(model, df, seq_length=20)

# Trend Analysis Using Linear Regression
def trend_analysis(predictions_original, window=20):
    """Analyzes trends using the last 20 predictions."""
    if len(predictions_original) < window:
        window = len(predictions_original)  
    
    trend_summary = "🔍 System Trend Analysis:\n"
    
    for i, feature in enumerate(features):
        if i >= predictions_original.shape[1]:  
            continue  

        last_n_values = predictions_original[-window:, i]
        x = np.arange(len(last_n_values))
        y = last_n_values

        # Fit a simple linear trend line (Least Squares)
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]  

        if slope > 0:
            trend_summary += f"📈 {feature} is **increasing** (slope: {slope:.4f}).\n"
        elif slope < 0:
            trend_summary += f"📉 {feature} is **decreasing** (slope: {slope:.4f}).\n"
        else:
            trend_summary += f"✅ {feature} remains **stable**.\n"

    return trend_summary

# Generate and Print Trend Analysis
trend_summary = trend_analysis(df[features].values)
print(trend_summary)
