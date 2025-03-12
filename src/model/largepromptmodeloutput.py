import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Set your GPU index if needed.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # change as needed
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Hard-coded label mapping (must match training)
label2id = {"B-FUNC": 0, "B-PARAM": 1, "I-PARAM": 2, "O": 3}
id2label = {i: label for label, i in label2id.items()}

# Load the trained model and tokenizer.
MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
model_path = "./tinybert_ner_trained_model"
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
    This function looks for parameter groups (B-PARAM followed by I-PARAM tokens)
    and associates them with preceding function tokens (B-FUNC). If no valid group
    is found, it returns an empty string.
    """
    # Filter out special tokens.
    filtered = [(tok, lab) for tok, lab in zip(tokens, labels) if tok not in {"[CLS]", "[SEP]", "[PAD]"}]
    n = len(filtered)
    param_groups = []  # Each element is (start_index, end_index, parameter_str)
    
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
            parameter = " ".join(param_tokens)
            param_groups.append((start, i, parameter))
        else:
            i += 1

    entities = []
    last_index = 0
    # For each parameter group, look for preceding B-FUNC tokens.
    for (start, end, parameter) in param_groups:
        funcs = []
        for j in range(last_index, start):
            if filtered[j][1] == "B-FUNC":
                funcs.append(filtered[j][0])
        if funcs:
            for func in funcs:
                entities.append(f"{parameter}: {func}")
        else:
            # If no function token is found before the parameter, output the parameter alone.
            entities.append(parameter)
        last_index = end
    # Any remaining B-FUNC tokens after the last parameter group are added as isolated entities.
    for j in range(last_index, n):
        if filtered[j][1] == "B-FUNC":
            entities.append(filtered[j][0])
    return ", ".join(entities)

def predict_text(input_text):
    # Tokenize input text.
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits  # (1, seq_length, num_labels)
    predictions = torch.argmax(logits, dim=2).squeeze().tolist()  # (seq_length,)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    predicted_labels = [id2label.get(pred, "O") for pred in predictions]
    structured_output = extract_entities(tokens, predicted_labels)
    # If no entities were extracted, provide a fallback message.
    if not structured_output.strip():
        structured_output = "No valid entities extracted. Please rephrase your prompt."
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
