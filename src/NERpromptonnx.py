import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Load tokenizer (lightweight)
model_path = "models/tinybert_ner_trained_model_input"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Label mappings
label2id = {"B-FUNC": 0, "B-PARAM": 1, "I-PARAM": 2, "B-SUMMARY": 3, "O": 4}
id2label = {v: k for k, v in label2id.items()}

# ONNX session
session = ort.InferenceSession("tinybert_ner.onnx")

def extract_entities(tokens, labels):
    filtered = [(tok, lab) for tok, lab in zip(tokens, labels) if tok not in {"[CLS]", "[SEP]", "[PAD]"}]
    n = len(filtered)

    param_groups = []  
    summaries = []
    i = 0

    while i < n:
        token, lab = filtered[i]

        if lab == "B-PARAM":
            start = i
            param_tokens = [token]
            i += 1
            while i < n and filtered[i][1] == "I-PARAM":
                param_tokens.append(filtered[i][0])
                i += 1
            param_groups.append((" ".join(param_tokens), start, i))

        elif lab == "B-SUMMARY":
            summary_tokens = [token]
            i += 1
            while i < n and filtered[i][1] == "B-SUMMARY":
                summary_tokens.append(filtered[i][0])
                i += 1
            summaries.append(" ".join(summary_tokens))

        else:
            i += 1

    entities = []
    last_index = 0

    for param_text, start, end in param_groups:
        funcs = [filtered[j][0] for j in range(last_index, start) if filtered[j][1] == "B-FUNC"]
        if funcs:
            for func in funcs:
                entities.append(f"{param_text}: {func}")
        else:
            entities.append(param_text)
        last_index = end

    for j in range(last_index, n):
        if filtered[j][1] == "B-FUNC":
            entities.append(filtered[j][0])

    if summaries:
        summary_text = " | ".join(summaries)
        entities.append(f"Summary: {summary_text}")

    return ", ".join(entities) if entities else "No valid entities extracted. Please rephrase your prompt."

def predict_text_onnx(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="np", padding='max_length', truncation=True, max_length=128)
    onnx_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }

    # Run ONNX inference
    outputs = session.run(None, onnx_inputs)
    logits = outputs[0]
    predictions = np.argmax(logits, axis=2)[0]

    # Get tokens and labels
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    predicted_labels = [id2label.get(idx, "O") for idx in predictions]

    structured_output = extract_entities(tokens, predicted_labels)
    return tokens, predicted_labels, structured_output

# CLI Entry
if __name__ == "__main__":
    print("Enter a sentence for NER (type 'exit' to quit):")
    while True:
        text = input("Input: ")
        if text.strip().lower() == "exit":
            break
        tokens, pred_labels, output_str = predict_text_onnx(text)
        print("\n--- Results (ONNX) ---")
        print("Tokens:")
        print(tokens)
        print("\nPredicted Labels:")
        print(pred_labels)
        print("\nStructured Output:")
        print(output_str)
        print("----------------\n")
