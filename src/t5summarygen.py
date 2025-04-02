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
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# ✅ Initialize Tokenizer & Model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")

# ✅ Preprocessing Function (Improved)
max_input_length = 512
max_target_length = 150

def preprocess_function(examples):
    inputs = [f"Log: {doc} -> Summary:" for doc in examples["Logs"]]  # 🔥 Improved input structure
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="longest")

    labels = tokenizer(text_target=examples["Summary"], max_length=max_target_length, truncation=True, padding="longest")
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# ✅ Tokenize datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

# ✅ Data Collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ✅ Training Arguments (Fixed for CUDA & Longer Training)
training_args = TrainingArguments(
    output_dir="./t5_summarization_v2",
    evaluation_strategy="epoch",  # Evaluate after every epoch
    logging_strategy="epoch",
    learning_rate=3e-5,  # 🔥 Slightly lower learning rate for better stability
    per_device_train_batch_size=4,  # ✅ Reduced batch size for lower memory usage
    per_device_eval_batch_size=2,
    num_train_epochs=10,  # ✅ Increased for better generalization
    weight_decay=0.01,
    save_total_limit=2,
    save_strategy="epoch",
    fp16=True,  # ✅ Mixed precision for faster training on CUDA
    push_to_hub=False,
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ✅ Start Training
trainer.train()

# ✅ Save Model & Tokenizer
trainer.save_model("./t5_summarization_final_v3")
tokenizer.save_pretrained("./t5_summarization_final_v3")
