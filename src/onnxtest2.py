import onnxruntime as ort
import numpy as np
import pandas as pd
from transformers import T5Tokenizer

# Load Tokenizer & ONNX Model
model_path = "./models/t5_summarization_final_v3"
onnx_model_path = "t5_summarization.onnx"
tokenizer = T5Tokenizer.from_pretrained(model_path)
session = ort.InferenceSession(onnx_model_path)

# Load Test Dataset
csv_path = "assets/system_logs_dataset_fixed.csv"
df = pd.read_csv(csv_path).sample(frac=0.2, random_state=42)

def generate_summary_iterative(text, max_iterations=50):
    # Tokenize the input text (increase max_length if logs are long)
    inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        padding="max_length",
        max_length=1024
    )
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)
    
    # Use T5's typical decoder start token (for T5-small, usually 0)
    decoder_start_token_id = 0
    decoder_input_ids = np.array([[decoder_start_token_id]], dtype=np.int64)
    
    for i in range(max_iterations):
        outputs = session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_input_ids": decoder_input_ids,
            }
        )
        # Extract logits for the last generated token (shape: (1, vocab_size))
        logits = outputs[0][:, -1, :]
        next_token = np.argmax(logits, axis=-1)  # shape: (1,)
        next_token = np.expand_dims(next_token, axis=0)  # shape: (1,1)
        
        # Append the predicted token to the decoder input sequence.
        decoder_input_ids = np.concatenate([decoder_input_ids, next_token], axis=1)
        
        # Debug: print current iteration and token id
        print(f"Iteration {i+1}: Generated token id {next_token[0][0]}")
        
        # Stop if EOS token is generated.
        if next_token[0][0] == tokenizer.eos_token_id:
            print("EOS token generated; stopping decoding.")
            break

    summary = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
    return summary

# Apply the iterative decoding function to each log in the dataset.
# df["Generated_Summary"] = df["Logs"].apply(generate_summary_iterative)
sample_df = df.sample(n=1, random_state=42)

# Apply the iterative decoding function to generate summaries for the sample entries
sample_df["Generated_Summary"] = sample_df["Logs"].apply(generate_summary_iterative)

# Print out the log, generated summary, and actual summary for each sample entry
for index, row in sample_df.iterrows():
    print(f"\n🔹 **Log:** {row['Logs']}")
    print(f"✅ **Generated Summary:** {row['Generated_Summary']}")
    print(f"📌 **Actual Summary:** {row['Summary']}")


# # Save results to a new CSV file.
# df.to_csv("assets/log_summary_predictions.csv", index=False)

# # Print sample results.
# for i in range(5):
#     print(f"\n🔹 **Log {i+1}**: {df['Logs'].iloc[i]}")
#     print(f"✅ **Generated Summary**: {df['Generated_Summary'].iloc[i]}")
#     print(f"📌 **Actual Summary**: {df['Summary'].iloc[i]}")

# print("\n🚀 Summaries saved to 'assets/log_summary_predictions.csv'!")
