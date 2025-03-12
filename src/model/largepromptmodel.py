import os
import torch
import json
import pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification, 
                          TrainingArguments, Trainer, DataCollatorForTokenClassification)

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------------
# 1. Load and Preprocess the Dataset
# -------------------------------
# CSV file path (adjust if needed)
csv_path = "assets/log_analysis_ner_large_dataset_cleaned.csv"
df = pd.read_csv(csv_path)

# Inspect a few rows (for debugging)
print("Dataset sample:")
print(df.head())

# Our CSV has two columns: "tokens" and "ner_tags"
# Convert the space-separated strings into lists.
df["tokens"] = df["tokens"].apply(lambda x: x.split())
df["ner_tags"] = df["ner_tags"].apply(lambda x: x.split())

# -------------------------------
# 2. Create Label Mappings
# -------------------------------
# We expect only these tags in a proper NER dataset:
ALLOWED_TAGS = {"B-FUNC", "B-PARAM", "I-PARAM", "O"}

# Optionally, you can filter out any tag not in ALLOWED_TAGS.
def filter_tags(tag_list):
    return [tag if tag in ALLOWED_TAGS else "O" for tag in tag_list]

df["ner_tags"] = df["ner_tags"].apply(filter_tags)

unique_labels = set()
for tags in df["ner_tags"]:
    unique_labels.update(tags)
unique_labels = sorted(list(unique_labels))
print("Unique labels:", unique_labels)

label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
print("Label mapping:", label2id)

# Save the label mapping for later use
with open("label_map.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f)

# Convert the DataFrame into a Hugging Face Dataset.
dataset = Dataset.from_pandas(df)

# -------------------------------
# 3. Tokenization and Label Alignment
# -------------------------------
# Use a TinyBERT model available on HF hub.
# (If needed, change the model name to one that exists in your environment.)
MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"

# Load the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(examples):
    # Tokenize the tokens using is_split_into_words=True (since our tokens are already split)
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    
    all_labels = []
    # Loop over each example in the batch
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens get a label of -100 (ignored in loss)
            elif word_idx != previous_word_idx:
                # Set the label for the first token of the word.
                label_ids.append(label2id[labels[word_idx]])
            else:
                # For sub-tokens of the same word, we can either repeat the label or set to -100.
                label_ids.append(label2id[labels[word_idx]])
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

# Map the tokenization function over the dataset (batched processing)
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Remove the original columns to free up space.
cols_to_remove = ["tokens", "ner_tags"]
for col in cols_to_remove:
    if col in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.remove_columns(col)

# Set the dataset format to PyTorch tensors.
tokenized_dataset.set_format("torch")

# -------------------------------
# 4. Split the Dataset
# -------------------------------
split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# -------------------------------
# 5. Load the Model for Token Classification
# -------------------------------
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)
model.to(device)

# -------------------------------
# 6. Set Up Training Arguments and Trainer
# -------------------------------
data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir="./tinybert_ner_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=10,
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
# 7. Train and Save the Model
# -------------------------------
trainer.train()

model.save_pretrained("./tinybert_ner_trained_model")
tokenizer.save_pretrained("./tinybert_ner_trained_model")

print("✅ Training complete. Model saved at ./tinybert_ner_trained_model")
