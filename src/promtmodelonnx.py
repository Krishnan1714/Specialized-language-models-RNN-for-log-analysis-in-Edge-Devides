import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load tokenizer and model from trained directory
model_path = "models/tinybert_ner_trained_model_input"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()

# Example input (max_length should match training setup)
text = "Please provide the input parameters and function name."
inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
# Export to ONNX
torch.onnx.export(
    model,
    (input_ids, attention_mask),
    "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {1: "sequence_length"},
        "attention_mask": {1: "sequence_length"},
        "logits": {1: "sequence_length"}
    },
    opset_version=14  # or higher to support latest ops like scaled_dot_product_attention
)


print("Model exported to tinybert_ner.onnx")
