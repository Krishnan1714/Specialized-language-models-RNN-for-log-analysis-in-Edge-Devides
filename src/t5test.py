import torch
import pandas as pd
import evaluate
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load ROUGE metric
rouge = evaluate.load("rouge")

# Load the trained model and tokenizer
model_path = "./t5_summarization_final_v3"  # or your latest saved model directory
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")

# Load test dataset (using 20% of the CSV for testing)
csv_path = "assets/system_logs_dataset_fixed.csv"
test_df = pd.read_csv(csv_path).sample(frac=0.2, random_state=42)

# Define a function to generate a summary using beam search
def generate_summary_beam(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(model.device)
    output_ids = model.generate(
        inputs.input_ids,
        max_length=150,
        min_length=20,
        num_beams=5,
        repetition_penalty=1.2,
        early_stopping=True
    )
    # Debug: print raw token IDs (if needed)
    # print("Raw output IDs:", output_ids)
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded

# Define a function to generate a summary using greedy decoding (as an alternative)
def generate_summary_greedy(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(model.device)
    output_ids = model.generate(
        inputs.input_ids,
        max_length=150,
        min_length=20,
        do_sample=False  # Greedy decoding
    )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded

# Choose your decoding strategy: beam search or greedy
# Try beam search first:
def generate_summary(text):
    summary = generate_summary_beam(text)
    # If the summary is empty, try greedy decoding
    if not summary.strip():
        summary = generate_summary_greedy(text)
    return summary

# Generate predictions
predictions = [generate_summary(text) for text in test_df["Logs"]]

# Debug: Print sample outputs for comparison
for i in range(5):  # Print 5 sample outputs
    print(f"\n🔹 **Log {i+1}**: {test_df['Logs'].iloc[i]}")
    print(f"✅ **Generated**: {predictions[i]}")
    print(f"📌 **Actual**: {test_df['Summary'].iloc[i]}")

# Compute ROUGE scores
rouge_scores = rouge.compute(predictions=predictions, references=test_df["Summary"].tolist(), use_stemmer=True)
# If the returned values are scalar, simply multiply them by 100
rouge_scores = {key: float(value) * 100 for key, value in rouge_scores.items()}

print("\n🚀 **Final ROUGE Scores:**", rouge_scores)
