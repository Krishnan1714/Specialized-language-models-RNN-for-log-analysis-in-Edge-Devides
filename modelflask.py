import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load TinyBERT model & tokenizer
tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
bert_model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D").to(device)

# Load dataset
df = pd.read_csv("cleaned_data.csv")

# Function to get TinyBERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# Define LSTM model for fault prediction
class FaultPredictionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1):
        super(FaultPredictionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# Load trained model
input_size = 318  # Adjust based on your dataset
model = FaultPredictionLSTM(input_size).to(device)
model.load_state_dict(torch.load("fault_prediction_lstm.pth"))
model.eval()

# Flask Chatbot API
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_fault():
    data = request.json
    log_text = data.get("log_text", "")
    
    # Select a sample from dataset for numerical data
    numerical_data = df.sample(1).drop(columns=["Target", "Failure Type", "log_text"], errors='ignore').values
    
    # Get text embedding
    text_embedding = get_bert_embedding(log_text).reshape(1, -1)
    
    # Combine numerical + text features
    input_data = np.hstack((numerical_data, text_embedding))
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Predict fault probability
    with torch.no_grad():
        fault_prob = model(input_tensor).item()
    
    return jsonify({"fault_probability": round(fault_prob, 2)})

@app.route("/summarize", methods=["POST"])
def summarize_logs():
    data = request.json
    logs = data.get("logs", [])
    
    # Concatenate logs and get embedding
    log_text = " ".join(logs)
    embedding = get_bert_embedding(log_text)
    
    return jsonify({"summary": "Log summary generated.", "embedding": embedding.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
