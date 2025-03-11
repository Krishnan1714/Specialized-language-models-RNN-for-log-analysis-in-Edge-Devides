import pandas as pd
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch


# Load dataset (assuming CSV file)
df = pd.read_csv("cleaned_data.csv")

# Statistical Summary
summary = df.describe().to_dict()

# Detect Anomalies (e.g., top 5% highest temperatures)
df["temp_anomaly"] = df["Process temperature"] > df["Process temperature"].quantile(0.95)

# Count failure occurrences
failure_summary = df["Failure Type"].value_counts().to_dict()

print("Statistical Summary:", summary)
print("\n")
print("Failure Summary:", failure_summary)




# Load TinyBERT model
tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = BertModel.from_pretrained("prajjwal1/bert-tiny")

# Generate input text based on statistical summary
text_input = f"""
The average air temperature was {summary['Air temperature']['mean']}°C,
with a peak of {summary['Air temperature']['max']}°C. 
Process temperatures reached a high of {summary['Process temperature']['max']}°C.
Rotational speed averaged {summary['Rotational speed']['mean']} RPM.
Torque values varied from {summary['Torque']['min']} Nm to {summary['Torque']['max']} Nm.
Tool wear increased over time, with {failure_summary.get('Tool Wear Failure', 0)} recorded failures.
"""

# Tokenize input
inputs = tokenizer(text_input, return_tensors="pt")

# Generate embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

print("TinyBERT Embeddings Generated!")

class LSTMFailurePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMFailurePredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Use last LSTM output
        return output

# Define LSTM model
input_dim = len(df.columns) - 1  # Exclude target column
hidden_dim = 128
output_dim = 1  # Predict failure (binary classification)

model = LSTMFailurePredictor(input_dim, hidden_dim, output_dim)

print("LSTM Model Initialized!")




# Prepare input: Select last 50 logs
num_logs = 50
last_50_logs = df.iloc[-num_logs:, :-1].values  # Exclude 'Target' column
sample_input = torch.tensor(last_50_logs, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# Predict failure probability
with torch.no_grad():
    prediction = torch.sigmoid(model(sample_input)).item()  # Apply sigmoid for probability

# Convert prediction to natural language
if prediction > 0.5:
    trend_summary = "The system is trending toward failure."
else:
    trend_summary = "The system is stable with normal operational trends."

print("📝 Natural Language Summary:", trend_summary)