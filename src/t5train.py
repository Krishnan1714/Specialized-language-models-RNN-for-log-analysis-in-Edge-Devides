import torch
import pandas as pd
import evaluate
from datasets import Dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
)

# ✅ Load Data
csv_path = "assets/system_logs_dataset_fixed.csv"
df = pd.read_csv(csv_path)

# ✅ Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df[['Logs', 'Summary']])
model_name = "t5-small"

# ✅ Initialize Tokenizer & Model
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")

# ✅ Preprocessing Function
max_input_length = 512
max_target_length = 150

def preprocess_function(examples):
    inputs = [f"Log: {doc} -> Summary:" for doc in examples["Logs"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="longest")
    labels = tokenizer(text_target=examples["Summary"], max_length=max_target_length, truncation=True, padding="longest")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# ✅ Training Arguments (Fixed Settings)
training_args = TrainingArguments(
    output_dir="./t5_summarization_v2",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    num_train_epochs=1,  # ✅ Only 1 epoch per iteration, loop externally
    weight_decay=0.01,
    save_total_limit=2,
    save_strategy="epoch",
    fp16=True,
    push_to_hub=False,
)

# ✅ Data Collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ✅ Dynamic Training Over Multiple Epochs
num_epochs = 10
for epoch in range(num_epochs):
    print(f"🔥 Training Epoch {epoch + 1}/{num_epochs} with a new dataset split...")

    # ✅ Vary Train-Test Split
    split_dataset = dataset.train_test_split(test_size=0.2 + (epoch * 0.01), seed=42)  # Slightly increase test size
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # ✅ Tokenize New Split
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

    # ✅ Initialize Trainer with New Data
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ✅ Train on New Split
    trainer.train()

    # ✅ Save Model Progress (Optional)
trainer.save_model(f"./t5_summarization_final")

print("✅ Training Complete!")
