import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the trained model and tokenizer
model_path = "./tinybert_log_classifier"  # Update if saved elsewhere
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load label encoder (ensure it's the same used during training)
df = pd.read_csv("final_log_analysis_prompts_1.csv")  # Update path if needed
label_encoder = LabelEncoder()
label_encoder.fit(df["Category"])  # Fit on the original categories

# Initialize text classification pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Function to classify input text
def classify_text(text):
    prediction = classifier(text)[0]
    
    # Convert "LABEL_x" to an integer
    predicted_label_id = int(prediction["label"].replace("LABEL_", ""))
    
    # Convert integer ID back to original category
    predicted_label = label_encoder.inverse_transform([predicted_label_id])[0]
    
    print(f"\n📝 Input: {text}")
    print(f"📌 Predicted Category: {predicted_label} (Confidence: {prediction['score']:.4f})")

# Interactive terminal input
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter log prompt (or type 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            break
        classify_text(user_input)
