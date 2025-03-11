import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline

# Step 1: Load the Dataset
df = pd.read_csv("final_log_analysis_prompts_1.csv")  # Update with the correct path

# Step 2: Split into Train and Validation Sets
train_texts, val_texts, train_labels, val_labels = train_test_split(df["Prompt"], df["Category"], test_size=0.2, random_state=42)

# Step 3: Encode Labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
val_labels = label_encoder.transform(val_labels)

# Save label mapping for future use
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:", label_mapping)

# Step 4: Load TinyBERT Tokenizer and Tokenize Texts
tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

# Step 5: Create PyTorch Dataset
class LogAnalysisDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = LogAnalysisDataset(train_encodings, train_labels)
val_dataset = LogAnalysisDataset(val_encodings, val_labels)

# Step 6: Load TinyBERT Model for Classification
num_labels = len(label_mapping)
model = AutoModelForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", num_labels=num_labels)

# Step 7: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./tinybert_log_classifier",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
)

# Step 8: Create Trainer and Train Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# Step 9: Evaluate Model Performance
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Step 10: Save Model and Tokenizer
model.save_pretrained("./tinybert_log_classifier")
tokenizer.save_pretrained("./tinybert_log_classifier")

# Step 11: Test Prediction Pipeline
classifier = pipeline(
    "text-classification", 
    model=model, 
    tokenizer=tokenizer, 
    truncation=True, 
    padding=True, 
    max_length=128  # Adjust if needed
)

# Test Example
example_text = input("Enter Input: ")
prediction = classifier(example_text)
# Convert model's predicted label (e.g., "LABEL_19") to integer
predicted_label_id = int(prediction[0]["label"].replace("LABEL_", ""))

# Map integer ID back to the original category name
predicted_label = label_encoder.inverse_transform([predicted_label_id])

print(f"Predicted Category for '{example_text}':", predicted_label[0])

