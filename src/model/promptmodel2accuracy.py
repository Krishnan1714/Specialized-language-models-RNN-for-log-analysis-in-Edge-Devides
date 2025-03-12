import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

# ----- Step 1: Load the Saved Model and Tokenizer -----
model_path = "./tinybert_log_classifier"  # Update if necessary
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ----- Step 2: Load the Oversampled Dataset -----
df = pd.read_csv("oversampled_log_analysis.csv")  # Update with correct path
df["Category"] = df["Category"].astype(str)  # Ensure categories are strings

# ----- Step 3: Encode Labels -----
label_encoder = LabelEncoder()
df["Encoded_Category"] = label_encoder.fit_transform(df["Category"])

# ----- Step 4: Split Dataset (Stratified) -----
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Prompt"],
    df["Encoded_Category"],
    test_size=0.2,
    random_state=42,
    stratify=df["Encoded_Category"]
)

# ----- Step 5: Tokenize the Validation Set -----
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

# ----- Step 6: Create a PyTorch Dataset Class -----
class LogAnalysisDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = np.array(labels)  # Convert to NumPy array for proper indexing

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

val_dataset = LogAnalysisDataset(val_encodings, val_labels)

# ----- Step 7: Create a Trainer for Evaluation -----
trainer = Trainer(model=model)

# ----- Step 8: Get Predictions on the Validation Set -----
predictions = trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids  # True labels from the dataset

# ----- Step 9: Calculate Overall Accuracy and Detailed Classification Report -----
accuracy = accuracy_score(true_labels, preds)
report = classification_report(true_labels, preds, target_names=label_encoder.classes_, digits=4)

print(f"Overall Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)
