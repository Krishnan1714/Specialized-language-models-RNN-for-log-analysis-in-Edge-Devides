import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Updated label mapping
label2id = {"B-FUNC": 0, "B-PARAM": 1, "I-PARAM": 2, "B-SUMMARY": 3, "O": 4}
id2label = {i: label for label, i in label2id.items()}

# Load the trained model and tokenizer
MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
model_path = "models/tinybert_ner_trained_model_input"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(
    model_path,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)
model.to(device)
model.eval()

def extract_entities(tokens, labels):
    """
    Extracts structured output from tokens and predicted labels.
    - Extracts parameters and their corresponding function names.
    - Extracts summary phrases from B-SUMMARY labels.
    """
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
            while i < n and filtered[i][1] == "B-SUMMARY":  # Allow multi-word summaries
                summary_tokens.append(filtered[i][0])
                i += 1
            summaries.append(" ".join(summary_tokens))

        else:
            i += 1

    entities = []
    last_index = 0

    # Assign function names to parameters
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

def predict_text(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2).squeeze().tolist()
    
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    predicted_labels = [id2label.get(pred, "O") for pred in predictions]

    structured_output = extract_entities(tokens, predicted_labels)
    
    return tokens, predicted_labels, structured_output

if __name__ == "__main__":
    print("Enter a sentence for NER (type 'exit' to quit):")
    while True:
        text = input("Input: ")
        if text.strip().lower() == "exit":
            break
        tokens, pred_labels, output_str = predict_text(text)
        print("\n--- Results ---")
        print("Tokens:")
        print(tokens)
        print("\nPredicted Labels:")
        print(pred_labels)
        print("\nStructured Output:")
        print(output_str)
        print("----------------\n")
