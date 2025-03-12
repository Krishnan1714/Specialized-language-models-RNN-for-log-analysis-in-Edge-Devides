import os
import ast
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import evaluate

# -------------------------------
# 1. Set Device and Load Label Mapping
# -------------------------------
# Set GPU index if needed
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Adjust if necessary
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Hard-coded label mapping (as used during training)
label2id = {"B-FUNC": 0, "B-PARAM": 1, "I-PARAM": 2, "O": 3}
id2label = {i: label for label, i in label2id.items()}
print("Hard-coded label mapping:", label2id)

# Optionally, load the label mapping from file:
# with open("label_map.json", "r") as f:
#     label_map = json.load(f)
# label2id = label_map["label2id"]
# id2label = {int(k): v for k, v in label_map["id2label"].items()}

# -------------------------------
# 2. Load the Saved Model and Tokenizer
# -------------------------------
MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
model_path = "./tinybert_ner_trained_model"
model = AutoModelForTokenClassification.from_pretrained(
    model_path,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.to(device)

# -------------------------------
# 3. Prepare the Test Dataset
# -------------------------------
def load_test_dataset(csv_path):
    # Load CSV containing a cleaned dataset.
    df = pd.read_csv(csv_path)
    # Assume that the CSV columns "tokens" and "ner_tags" are string representations of lists.
    # Convert them using ast.literal_eval.
    df["tokens"] = df["tokens"].apply(ast.literal_eval)
    df["ner_tags"] = df["ner_tags"].apply(ast.literal_eval)
    return df

test_csv_path = "assets/synthetic_log_analysis_ner.csv"
df_test = load_test_dataset(test_csv_path)
test_dataset = Dataset.from_pandas(df_test)

# Define tokenization and label alignment (same as during training)
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
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[labels[word_idx]])
            else:
                label_ids.append(label2id[labels[word_idx]])
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)
for col in ["tokens", "ner_tags", "__index_level_0__"]:
    if col in test_dataset.column_names:
        test_dataset = test_dataset.remove_columns(col)
test_dataset.set_format("torch")

# -------------------------------
# 4. Define Evaluation Metrics
# -------------------------------
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=2)
    true_predictions = [
        [id2label[pred] for (pred, lab) in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[lab] for (pred, lab) in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# -------------------------------
# 5. Set Up Trainer for Evaluation
# -------------------------------
training_args = TrainingArguments(
    output_dir="./results_eval",
    per_device_eval_batch_size=16,
    logging_dir="./logs_eval",
    do_train=False,
    do_eval=True,
)

data_collator = DataCollatorForTokenClassification(tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Evaluate the model
eval_metrics = trainer.evaluate()
print("Evaluation Metrics:")
for key, value in eval_metrics.items():
    print(f"{key}: {value}")

# -------------------------------
# 6. Interactive Testing
# -------------------------------
def interactive_test():
    print("\nEnter a sentence to test the NER model (type 'exit' to quit):")
    while True:
        sentence = input("Input: ")
        if sentence.strip().lower() == "exit":
            break
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predictions = logits.argmax(dim=2).squeeze().tolist()
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
        predicted_labels = [id2label.get(pred, "O") for pred in predictions]
        print("\n--- Results ---")
        print("Tokens:")
        print(tokens)
        print("\nPredicted Labels:")
        print(predicted_labels)
        print("----------------\n")

interactive_test()
