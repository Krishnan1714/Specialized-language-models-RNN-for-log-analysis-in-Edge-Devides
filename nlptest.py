from transformers import pipeline

# Create a zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define candidate intents
candidate_labels = ["log summary", "fault prediction", "general inquiry"]

def get_intent(user_input):
    result = classifier(user_input, candidate_labels)
    return result["labels"][0]  # return the top intent

# Example usage
user_input = "Could you please summarize the recent logs?"
predicted_intent = get_intent(user_input)
print("Predicted intent:", predicted_intent)

if predicted_intent == "log summary":
    response = generate_log_summary()  # your function for log summary
elif predicted_intent == "fault prediction":
    # Process fault prediction
    response = "Fault prediction logic here"
else:
    response = "I can provide log summaries or fault predictions. Please specify your request."
