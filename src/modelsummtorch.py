import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

# Define device (use GPU if available, otherwise fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Dataset
df = pd.read_csv(".csv")  # Replace with actual file

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


# Function to create sequences for LSTM training
def create_sequences(data, seq_length=10):
    """Creates sequences of data for LSTM training, ensuring dataset is large enough."""
    if len(data) < seq_length:
        raise ValueError("Dataset too small for the specified sequence length.")
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])  # Input sequence
        y.append(data[i + seq_length])  # Corresponding target
    return np.array(X), np.array(y)

X, y = create_sequences(df_scaled, seq_length=10)

# Use TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Get the last train-test split from TimeSeriesSplit
train_index, test_index = list(tscv.split(X))[-1]  # Use only the last split
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

# Convert to PyTorch tensors and move to device
X_train, X_test = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(X_test, dtype=torch.float32).to(device)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=4, output_size=len(features)):
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

# Training Function with Early Stopping
def train_model(model, X_train, y_train, epochs=20, patience=10):
    model.train()
    best_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Check for early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0  # Reset patience
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Train the Model
train_model(model, X_train, y_train)

# Define function to make predictions
def predict(model, X_test):
    """Generates predictions using the trained model."""
    model.eval()
    with torch.no_grad():
        return model(X_test).cpu().numpy()

# Generate predictions
predictions = predict(model, X_test)

# Ensure correct inverse transformation before evaluation
y_test_original = scaler.inverse_transform(y_test.cpu().numpy())
predictions_original = scaler.inverse_transform(predictions)

# Clip negative values to zero for stability
y_test_original = np.clip(y_test_original, 0, None)
predictions_original = np.clip(predictions_original, 0, None)

# Evaluate Model
def evaluate_model(predictions, y_test_original):
    """Evaluates the model using MSE, R² Score, MAE, and Evans' method."""
    loss = criterion(torch.tensor(predictions, dtype=torch.float32), torch.tensor(y_test_original, dtype=torch.float32)).item()
    print(f"Test Loss (MSE): {loss:.4f}")

    r2 = r2_score(y_test_original, predictions)  # Compute R² Score
    r2 = max(0, r2)  # Prevent negative R²
    print(f"R² Score (Accuracy): {r2:.4f}")
    
    # Classify R² Score using Evans' Method
    strength = "Very Weak" if r2 < 0.20 else "Weak" if r2 < 0.40 else "Moderate" if r2 < 0.60 else "Strong" if r2 < 0.80 else "Very Strong"
    print(f"Evans' Classification: {strength}")

    mae = mean_absolute_error(y_test_original, predictions)  # Compute MAE
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

evaluate_model(predictions_original, y_test_original)

# Trend Analysis Function
def trend_analysis(predictions_original, y_test_original, features):
    """Analyzes trends for each feature by comparing predicted and actual values."""
    trend_summary = "🔍 System Trend Analysis:\n"
    for i, feature in enumerate(features):
        rate_of_change = predictions_original[-1][i] - predictions_original[-2][i]
        if rate_of_change > 0:
            trend_summary += f"📈 {feature} is increasing by {rate_of_change:.4f}.\n"
        elif rate_of_change < 0:
            trend_summary += f"📉 {feature} is decreasing by {abs(rate_of_change):.4f}.\n"
        else:
            trend_summary += f"✅ {feature} remains stable.\n"
    return trend_summary

# Generate and print trend analysis
trend_summary = trend_analysis(predictions_original, y_test_original, features)
print(trend_summary)
