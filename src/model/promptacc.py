import os
import ast
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from torch.utils.data import DataLoader

# -------------------------------
# 1. Set Device and Hard-Coded Label Mapping
# -------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set your preferred GPU index
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Hard-coded label mapping (must match training)
label2id = {"B-FUNC": 0, "B-PARAM": 1, "I-PARAM": 2, "O": 3}
id2label = {i: label for label, i in label2id.items()}
print("Hard-coded label mapping:", label2id)

# -------------------------------
# 2. Load the Cleaned Dataset
# -------------------------------
csv_path = "assets/log_analysis_ner_large_dataset_cleaned.csv"
df = pd.read_csv(csv_path)

# Convert the "tokens" and "ner_tags" columns from string to lists.
df["tokens"] = df["tokens"].apply(ast.literal_eval)
df["ner_tags"] = df["ner_tags"].apply(ast.literal_eval)

# Create a Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# -------------------------------
# 3. Tokenization and Label Alignment Function
# -------------------------------
MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Mask special tokens.
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[labels[word_idx]])
            else:
                # For sub-tokens, assign the same label.
                label_ids.append(label2id[labels[word_idx]])
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

# Map the function over the dataset.
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
# Remove original columns.
for col in ["tokens", "ner_tags", "__index_level_0__"]:
    if col in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.remove_columns(col)
tokenized_dataset.set_format("torch")

# For testing, we'll use the entire tokenized_dataset as the evaluation set.
eval_dataset = tokenized_dataset

# -------------------------------
# 4. Load the Trained Model
# -------------------------------
model = AutoModelForTokenClassification.from_pretrained(
    "./tinybert_ner_trained_model",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)
model.to(device)
model.eval()

# -------------------------------
# 5. Create DataLoader for Evaluation
# -------------------------------
data_collator = DataCollatorForTokenClassification(tokenizer)
eval_dataloader = DataLoader(eval_dataset, batch_size=16, collate_fn=data_collator)

# -------------------------------
# 6. Compute Token-wise Accuracy
# -------------------------------
total_correct = 0
total_tokens = 0

for batch in eval_dataloader:
    # Move batch to device.
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    predictions = outputs.logits.argmax(dim=2)  # (batch_size, seq_length)
    labels = batch["labels"]  # (batch_size, seq_length)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i, j].item() != -100:
                total_tokens += 1
                if predictions[i, j].item() == labels[i, j].item():
                    total_correct += 1

accuracy = total_correct / total_tokens if total_tokens > 0 else 0
print(f"Token-wise Accuracy: {accuracy:.4f}")
