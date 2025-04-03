import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load tokenizer and model
model_path = "./models/t5_summarization_final_v3"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to("cpu")  # Use CPU for ONNX export
model.eval()

# Example input text
example_text = "System logs indicate a memory leak issue causing slowdowns."
inputs = tokenizer(example_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

# Prepare decoder inputs (T5 requires explicit decoder_input_ids)
decoder_input_ids = torch.ones((1, 1), dtype=torch.long) * tokenizer.pad_token_id  # Start with PAD token

# Define ONNX export path
onnx_path = "t5_summarization.onnx"

# Export model with explicit decoder inputs
torch.onnx.export(
    model,
    (inputs.input_ids, inputs.attention_mask, decoder_input_ids),  # Explicitly pass decoder input
    onnx_path,
    input_names=["input_ids", "attention_mask", "decoder_input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "decoder_input_ids": {0: "batch_size", 1: "sequence_length"}
    },
    opset_version=13  # Ensure compatibility
)

print(f"✅ ONNX model saved to {onnx_path}")
