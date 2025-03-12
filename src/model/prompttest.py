import os
import ast
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from torch.utils.data import DataLoader

# -------------------------------
# 1. Set Device and Hard-coded Label Mapping
# -------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change if you want a different GPU index
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Hard-coded label mapping (must match training)
label2id = {"B-FUNC": 0, "B-PARAM": 1, "I-PARAM": 2, "O": 3}
id2label = {i: label for label, i in label2id.items()}
print("Hard-coded label mapping:", label2id)

# -------------------------------
# 2. Load the Cleaned Dataset and Split into Test (20%)
# -------------------------------
csv_path = "assets/log_analysis_ner_large_dataset_cleaned.csv"
df = pd.read_csv(csv_path)

# Convert "tokens" and "ner_tags" columns from string to lists.
df["tokens"] = df["tokens"].apply(ast.literal_eval)
df["ner_tags"] = df["ner_tags"].apply(ast.literal_eval)

print(f"Total examples in dataset: {len(df)}")
# Split into train and test; we only use the test split (20%).
_, df_test = train_test_split(df, test_size=0.2, random_state=42)
print(f"Test examples: {len(df_test)}")

# Convert the test DataFrame into a Hugging Face Dataset.
test_dataset = Dataset.from_pandas(df_test)

# -------------------------------
# 3. Tokenization and Label Alignment for Test Data
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
                label_ids.append(-100)  # Special tokens (e.g. [CLS], [SEP]) get -100.
            elif word_idx != previous_word_idx:
                try:
                    label_ids.append(label2id[labels[word_idx]])
                except Exception as e:
                    print("Error with labels:", labels, "word_idx:", word_idx)
                    label_ids.append(-100)
            else:
                try:
                    label_ids.append(label2id[labels[word_idx]])
                except Exception as e:
                    label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)
# Remove original columns to free memory.
for col in ["tokens", "ner_tags", "__index_level_0__"]:
    if col in test_dataset.column_names:
        test_dataset = test_dataset.remove_columns(col)
test_dataset.set_format("torch")

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
# 5. Create DataLoader for the Test Set
# -------------------------------
data_collator = DataCollatorForTokenClassification(tokenizer)
eval_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=data_collator)

# -------------------------------
# 6. Compute Overall and Label-wise Token Accuracy
# -------------------------------
total_correct = 0
total_tokens = 0
label_correct = defaultdict(int)
label_total = defaultdict(int)

for batch in eval_dataloader:
    # Move batch to device.
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    predictions = outputs.logits.argmax(dim=2)  # (batch_size, seq_length)
    labels = batch["labels"]  # (batch_size, seq_length)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            true_val = labels[i, j].item()
            if true_val != -100:
                total_tokens += 1
                true_label = id2label[true_val]
                label_total[true_label] += 1
                pred_val = predictions[i, j].item()
                if pred_val == true_val:
                    total_correct += 1
                    label_correct[true_label] += 1

overall_accuracy = total_correct / total_tokens if total_tokens > 0 else 0
print(f"\nOverall Token-wise Accuracy on Test Set: {overall_accuracy:.4f}")

print("\nLabel-wise Accuracy on Test Set:")
for label in label2id.keys():
    if label_total[label] > 0:
        acc = label_correct[label] / label_total[label]
        print(f"  {label}: {acc:.4f} ({label_correct[label]}/{label_total[label]})")
    else:
        print(f"  {label}: No tokens found.")

print("\nTrue label distribution:", dict(label_total))
print("Predicted correct distribution:", dict(label_correct))

# -------------------------------
# 7. Diagnostic: Print a Few Examples from a Batch
# -------------------------------
print("\n=== Diagnostic Example from a Batch ===")
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    predictions = outputs.logits.argmax(dim=2)
    # Print up to 3 examples
    for i in range(min(3, batch["input_ids"].shape[0])):
        tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][i].tolist())
        true_labels = [id2label[label.item()] if label.item() != -100 else "-" for label in batch["labels"][i]]
        pred_labels = [id2label[pred.item()] for pred in predictions[i]]
        print("Tokens:         ", tokens)
        print("True Labels:    ", true_labels)
        print("Predicted Labels:", pred_labels)
        print("-" * 50)
    break  # Only print one batch for diagnostics
