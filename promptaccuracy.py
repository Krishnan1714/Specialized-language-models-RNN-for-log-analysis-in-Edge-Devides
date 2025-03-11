import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load trained model and tokenizer
model_path = "./tinybert_log_classifier"  # Update if saved elsewhere
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load dataset
df = pd.read_csv("final_log_analysis_prompts_1.csv")  # Update path if needed

# Encode labels
label_encoder = LabelEncoder()
df["Category"] = label_encoder.fit_transform(df["Category"])

# Split dataset into training (80%) and testing (20%)
_, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Initialize classification pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Function to get model predictions
def get_predictions(texts):
    predictions = classifier(texts, truncation=True, padding=True)
    
    # Convert "LABEL_x" to integer IDs
    predicted_label_ids = [int(pred["label"].replace("LABEL_", "")) for pred in predictions]
    
    # Convert integer IDs to original category names
    predicted_labels = label_encoder.inverse_transform(predicted_label_ids)
    
    return predicted_labels

# Run evaluation
test_texts = test_data["Prompt"].tolist()
true_labels = label_encoder.inverse_transform(test_data["Category"].tolist())

predicted_labels = get_predictions(test_texts)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print("\n🔍 Model Accuracy: {:.2f}%".format(accuracy * 100))

# Print detailed classification report
# Get only the unique labels present in the test set
unique_labels = sorted(set(true_labels) | set(predicted_labels))

print("\n📊 Classification Report:\n", classification_report(true_labels, predicted_labels, labels=unique_labels, target_names=unique_labels))
