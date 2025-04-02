import torch
import pprint
import evaluate
import numpy as np

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

pp = pprint.PrettyPrinter()

# Load CSV file. Adjust the file path as needed.
dataset = load_dataset('csv', data_files='assets/logsummary.csv', split='train')

# Split the dataset into train and validation sets
full_dataset = dataset.train_test_split(test_size=0.2, shuffle=True)

dataset_train = full_dataset['train']
dataset_valid = full_dataset['test']

print(dataset_train)
print(dataset_valid)

def find_longest_length(dataset):
    """
    Find the longest Logs and Summary in the entire training set.
    """
    max_length = 0
    counter_4k = 0
    counter_2k = 0
    counter_1k = 0
    counter_500 = 0
    for text in dataset:
        corpus = text.split()
        if len(corpus) > max_length:
            max_length = len(corpus)
        if len(corpus) >= 500:
            counter_4k += 1
        elif len(corpus) >= 250:
            counter_2k += 1
        elif len(corpus) >= 100:
            counter_1k += 1
        elif len(corpus) >= 30:
            counter_500 += 1

    return max_length, counter_4k, counter_2k, counter_1k, counter_500

max_length, counter_4k, counter_2k, counter_1k, counter_500 = find_longest_length(dataset_train['Logs'])
print(f"Longest Logs length: {max_length}")
print(f"Logs with 500+ words: {counter_4k}")
print(f"Logs with 250+ words: {counter_2k}")
print(f"Logs with 100+ words: {counter_1k}")
print(f"Logs with 30+ words: {counter_500}")

max_length, counter_4k, counter_2k, counter_1k, counter_500 = find_longest_length(dataset_train['Summary'])
print(f"Longest Summary length: {max_length}")
print(f"Summaries with 500+ words: {counter_4k}")
print(f"Summaries with 250+ words: {counter_2k}")
print(f"Summaries with 100+ words: {counter_1k}")
print(f"Summaries with 30+ words: {counter_500}")

model_checkpoint = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

# Setting max source and target lengths after inspection from find_longest_length output
max_source_length = 1024
max_target_length = 256

def preprocess_data(examples):
    inputs = examples["Logs"]
    targets = examples["Summary"]
    # Use padding="max_length" to ensure sequences have the same length
    model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True, padding="max_length")

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset_train = dataset_train.map(
    preprocess_data,
    batched=True,
    remove_columns=["Logs", "Summary"]
)
tokenized_dataset_valid = dataset_valid.map(
    preprocess_data,
    batched=True,
    remove_columns=["Logs", "Summary"]
)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=10,
    learning_rate=2e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    logging_dir="resultst5",
    logging_steps=10,
    save_steps=500,
    num_train_epochs=10,
    push_to_hub=False,
)

metric = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Convert predictions into numpy array and ensure they are flattened properly
    if isinstance(predictions, tuple):
        predictions = predictions[0]  # Extract predictions if wrapped in a tuple

    # Convert nested lists into a proper format for batch decoding
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Ensure label padding values are properly handled
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions.tolist(), skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)

    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    # Extract relevant ROUGE scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Compute length of generated predictions
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_valid,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
from transformers import TrainerCallback, TrainerControl, TrainerState
from tabulate import tabulate

class TableLoggerCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return

        # Ensure the current step is in logs
        if "step" not in logs:
            logs["step"] = state.global_step

        # Prepare table data
        table_data = [[metric, value] for metric, value in sorted(logs.items())]
        table = tabulate(table_data, headers=["Metric", "Value"], tablefmt="pretty")
        
        print(f"Step {state.global_step}:\n{table}\n")

trainer.add_callback(TableLoggerCallback())

history =trainer.train()
print(history)
