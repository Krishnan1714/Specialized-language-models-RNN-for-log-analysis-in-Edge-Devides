import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load cleaned numerical dataset
df = pd.read_csv("cleaned_data.csv")

# Load TinyBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
bert_model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D").to(device)

# Function to get TinyBERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# Example log data (Replace with actual log column)
df["log_text"] = df.get("log_text", "Default log")

# Convert logs to embeddings
df["log_embedding"] = df["log_text"].apply(get_bert_embedding)

# Prepare features (numerical + embeddings)
X_num = df.drop(columns=["Target", "Failure Type", "log_text", "log_embedding"], errors='ignore').values
X_text = np.stack(df["log_embedding"].values)
X = np.hstack((X_num, X_text))

# Target (Fault Prediction)
y = df["Target"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM model for Fault Prediction
class FaultPredictionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1):
        super(FaultPredictionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take last timestep output
        return self.sigmoid(out)

# Initialize model
input_size = X_train.shape[1]
model = FaultPredictionLSTM(input_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# Train the model
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "fault_prediction_lstm.pth")
