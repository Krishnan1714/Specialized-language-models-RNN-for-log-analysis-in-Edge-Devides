import torch
import pandas as pd
import evaluate
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load ROUGE metric
rouge = evaluate.load("rouge")

# Load model and tokenizer with FP16 for speed
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./t5_summarization_final_v3"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device).half()
model = torch.compile(model)  # Optimize model execution (PyTorch 2.0+)

# Load test dataset (20% of CSV)
csv_path = "assets/system_logs_dataset_fixed.csv"
test_df = pd.read_csv(csv_path).sample(frac=0.2, random_state=42)

# Define a function to generate a summary with beam search
@torch.inference_mode()  # Disables autograd for faster inference
def generate_summary(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
    output_ids = model.generate(
        inputs.input_ids,
        max_length=50,  # Lower max length for faster output
        min_length=20,
        num_beams=3,  # Reduce beams to speed up decoding
        repetition_penalty=1.2,
        early_stopping=True
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Batch inference for speed
logs = test_df["Logs"].tolist()
predictions = [generate_summary(text) for text in logs]

# Debug: Print sample outputs
for i in range(5):  
    print(f"\n🔹 **Log {i+1}**: {logs[i]}")
    print(f"✅ **Generated**: {predictions[i]}")
    print(f"📌 **Actual**: {test_df['Summary'].iloc[i]}")

# Compute ROUGE scores
rouge_scores = rouge.compute(predictions=predictions, references=test_df["Summary"].tolist(), use_stemmer=True)
rouge_scores = {key: float(value) * 100 for key, value in rouge_scores.items()}

print("\n🚀 **Final ROUGE Scores:**", rouge_scores)
