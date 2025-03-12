import os
import torch
import json
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

# Set the GPU index if needed.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------------
# 1. Load and Process the Dataset
# -------------------------------
csv_path = "assets/log_analysis_ner_large_dataset_cleaned.csv"
df = pd.read_csv(csv_path)

def process_cell(cell):
    """
    Processes a cell containing a string representation of a list.
    Replaces single quotes with double quotes and then loads using json.loads.
    """
    try:
        cell_fixed = cell.replace("'", "\"")
        return json.loads(cell_fixed)
    except Exception as e:
        print("Error processing cell:", cell, e)
        # Fallback: split on whitespace
        return cell.split()

# Process both tokens and ner_tags columns.
df["tokens"] = df["tokens"].apply(process_cell)
df["ner_tags"] = df["ner_tags"].apply(process_cell)

print("Dataset sample:")
print(df.head())

# -------------------------------
# 2. Hard-Coded Label Mapping
# -------------------------------
# We assume the allowed tags are exactly these four.
label2id = {"B-FUNC": 0, "B-PARAM": 1, "I-PARAM": 2, "O": 3}
id2label = {i: label for label, i in label2id.items()}
print("Hard-coded label mapping:", label2id)

# Save the label mapping for later use.
with open("label_map.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f)

# -------------------------------
# 3. Convert DataFrame to Hugging Face Dataset
# -------------------------------
dataset = Dataset.from_pandas(df)

# -------------------------------
# 4. Tokenization and Label Alignment
# -------------------------------
MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(examples):
    # Tokenize the tokens using is_split_into_words=True.
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    
    all_labels = []
    # Align the ner_tags with the tokenized inputs.
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens get -100.
            elif word_idx != previous_word_idx:
                try:
                    label_ids.append(label2id[labels[word_idx]])
                except KeyError:
                    print("KeyError for label:", labels[word_idx])
                    label_ids.append(-100)
            else:
                try:
                    label_ids.append(label2id[labels[word_idx]])
                except KeyError:
                    print("KeyError for label:", labels[word_idx])
                    label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
# Remove original columns.
for col in ["tokens", "ner_tags", "__index_level_0__"]:
    if col in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.remove_columns(col)
tokenized_dataset.set_format("torch")

# -------------------------------
# 5. Split the Dataset for Training/Evaluation
# -------------------------------
split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# -------------------------------
# 6. Load the Model for Token Classification
# -------------------------------
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)
model.to(device)

# -------------------------------
# 7. Set Up Training Arguments and Trainer
# -------------------------------
data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir="./tinybert_ner_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=2,
    weight_decay=0.01,
    fp16=True if device == "cuda" else False,
    logging_dir="./logs",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# -------------------------------
# 8. Train and Save the Model
# -------------------------------
trainer.train()

model.save_pretrained("./tinybert_ner_trained_model")
tokenizer.save_pretrained("./tinybert_ner_trained_model")

print("✅ Training complete. Model saved at './tinybert_ner_trained_model'")
