import onnxruntime as ort
import numpy as np
import pandas as pd
from transformers import T5Tokenizer

# ✅ Load the tokenizer
model_path = "./models/t5_summarization_final_v3"  # Change to your correct path
tokenizer = T5Tokenizer.from_pretrained(model_path)

# ✅ Load ONNX model
onnx_model_path = "t5_summarization.onnx"  # Change if your model is in a different location
session = ort.InferenceSession(onnx_model_path)

# ✅ Load test dataset
csv_path = "assets/system_logs_dataset_fixed.csv"  # Update path if needed
df = pd.read_csv(csv_path).sample(frac=0.2, random_state=42)  # Use 20% of data for testing

# ✅ Function to generate summary using ONNX
def generate_summary(text):
    inputs = tokenizer(text, return_tensors="np", truncation=True, padding="max_length", max_length=512)

    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)
    decoder_input_ids = np.ones((1, 1), dtype=np.int64) * tokenizer.pad_token_id  # Decoder starts with PAD token

    # Run ONNX inference
    outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask, "decoder_input_ids": decoder_input_ids})
    predicted_tokens = np.argmax(outputs[0], axis=-1)

    return tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)

# ✅ Generate summaries for sample logs
df["Generated_Summary"] = df["Logs"].apply(generate_summary)

# ✅ Print first 5 sample results
for i in range(5):
    print(f"\n🔹 **Log {i+1}**: {df['Logs'].iloc[i]}")
    print(f"✅ **Generated Summary**: {df['Generated_Summary'].iloc[i]}")
    print(f"📌 **Actual Summary**: {df['Summary'].iloc[i]}")

# ✅ Save results to a new CSV file
df.to_csv("assets/log_summary_predictions.csv", index=False)
print("\n🚀 Summaries saved to 'assets/log_summary_predictions.csv'!")
