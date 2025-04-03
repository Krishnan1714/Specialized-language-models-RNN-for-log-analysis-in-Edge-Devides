import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the model & tokenizer
model_path = "./models/t5_summarization_final_v3"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to("cpu")  # ONNX prefers CPU export

model.eval()  # Set the model to evaluation mode

# Example input text
example_text = "System logs indicate a memory leak issue causing slowdowns."
inputs = tokenizer(example_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

# Export the model to ONNX format
onnx_path = "t5_summarization.onnx"
torch.onnx.export(
    model,
    (inputs.input_ids, inputs.attention_mask),  # ONNX needs explicit inputs
    onnx_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch_size"}, "attention_mask": {0: "batch_size"}},
    opset_version=12  # Ensure compatibility
)

print(f"✅ ONNX model saved to {onnx_path}")
