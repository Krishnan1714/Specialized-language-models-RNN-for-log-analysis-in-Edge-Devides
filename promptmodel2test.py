import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# ----- Step 1: Load Your Fine-Tuned Model & Tokenizer -----
model_path = "./tinybert_log_classifier"  # Path to your saved fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_path)
# Load the encoder part of your fine-tuned model (with output_hidden_states enabled)
model = AutoModel.from_pretrained(model_path, output_hidden_states=True)
model.eval()  # set to evaluation mode

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ----- Step 2: Load Your Label Encoder -----
# This encoder was created during training and holds your training classes.
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Get the list of training label names.
label_texts = list(label_encoder.classes_)  # note: this list may not include unseen combinations

# ----- Step 3: Define a Function to Compute Sentence Embeddings -----
def get_embedding(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pool the last hidden state to get a single sentence embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

# ----- Step 4: Compute Label Embeddings Dynamically -----
# Compute embeddings for each training label (using the label text itself)
label_embeddings = {label: get_embedding(label) for label in label_texts}

# ----- Step 5: Define a Function for Fuzzy Matching Classification -----
def classify_with_fallback(user_input):
    # Compute the embedding for the user input
    user_embedding = get_embedding(user_input)
    
    # Compute cosine similarity with all stored label embeddings
    similarities = {
        label: cosine_similarity(user_embedding, emb.reshape(1, -1))[0][0]
        for label, emb in label_embeddings.items()
    }
    
    # Find the label with the highest similarity
    best_label = max(similarities, key=similarities.get)
    confidence = similarities[best_label]
    return best_label, confidence

# ----- Step 6: Terminal Loop to Classify User Input -----
print("Enter text to classify (type 'exit' to quit):")
while True:
    user_input = input(">> ")
    if user_input.strip().lower() == "exit":
        break
    predicted_label, confidence = classify_with_fallback(user_input)
    print(f"Predicted Category: {predicted_label} (Confidence: {confidence:.4f})")
