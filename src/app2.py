# chatbot_backend.py
import pandas as pd
import NERpromptonnx as ner
import basicfeaturefunctions as bff
import numpy as np

class ChatbotBackend:
    def __init__(self):
        self.df = None
        self.file_paths = []

    def upload_file(self, filepath):
        try:
            self.df = pd.read_csv(filepath)
            self.file_paths.append(filepath)
            return f"✅ File uploaded: {filepath}"
        except Exception as e:
            return f"❌ Error processing file: {e}"

    def summarize_dataframe(self):
        if self.df is None:
            return "❗ No file uploaded."
        summary = []
        stats = self.df.describe()
        for column in self.df.select_dtypes(include=[np.number]).columns:
            col_stats = stats[column]
            summary.append(
                f"{column} has {int(col_stats['count'])} entries. "
                f"Avg: {col_stats['mean']:.3f}, Std: {col_stats['std']:.3f}, "
                f"Min: {col_stats['min']:.3f}, Max: {col_stats['max']:.3f}, "
                f"Median: {col_stats['50%']:.3f}."
            )
        return "\n".join(summary)

    def process_message(self, message):
        if self.df is None:
            return "❗ Please upload a CSV file first."
        tokens, pred_labels, output_str = ner.predict_text_onnx(message)
        parsed_data = bff.parse_input(output_str)
        result = bff.execute_function(parsed_data, self.df)
        if "summary" in output_str:
            result += "\n\n" + self.summarize_dataframe()
        return result or "🤖 Sorry, I couldn't understand that. Please rephrase."

def main():
    bot = ChatbotBackend()
    print("📁 Welcome to the Log File Chatbot (Terminal Mode)")
    print("Commands:\n 1. upload <path-to-csv>\n 2. ask <your-question>\n 3. exit")

    while True:
        user_input = input("\n>>> ").strip()
        
        if user_input.lower() == "exit":
            print("👋 Exiting chatbot.")
            break

        if user_input.startswith("upload "):
            path = user_input[len("upload "):].strip()
            result = bot.upload_file(path)
            print(result)

        elif user_input.startswith("ask "):
            message = user_input[len("ask "):].strip()
            response = bot.process_message(message)
            print("🤖", response)

        else:
            print("❓ Unknown command. Try: upload <path>, ask <message>, or exit.")

if __name__ == "__main__":
    main()