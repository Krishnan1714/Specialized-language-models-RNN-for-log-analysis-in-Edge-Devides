import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import os
import pandas as pd
import sys
from model import NERpromptmodel as ner
import basicfeaturefunctions as bff
import numpy as np
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

class ChatbotGUI:
    df = None

    def __init__(self, master):
        self.master = master
        master.title("File-Based Chatbot")

        # Initialize ChatterBot
        self.chatbot = ChatBot("LogBot")
        self.trainer = ChatterBotCorpusTrainer(self.chatbot)
        self.trainer.train("chatterbot.corpus.english")  # Train on basic English conversation

        self.chat_log = scrolledtext.ScrolledText(master, state='disabled', height=20, width=60)
        self.chat_log.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.user_input = tk.Entry(master, width=50)
        self.user_input.grid(row=1, column=0, padx=10, pady=5)

        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=5, pady=5)

        self.file_button = tk.Button(master, text="Upload File", command=self.upload_file)
        self.file_button.grid(row=2, column=0, columnspan=2, pady=5)

        self.file_paths = []  # Store uploaded file paths

    def display_message(self, sender, message):
        self.chat_log.config(state='normal')
        self.chat_log.insert(tk.END, f"{sender}: {message}\n")
        self.chat_log.config(state='disabled')
        self.chat_log.yview(tk.END)  # Scroll to the bottom

    def send_message(self):
        user_message = self.user_input.get()
        if user_message:
            self.display_message("You", user_message)
            self.user_input.delete(0, tk.END)
            self.process_message(user_message)

    def upload_file(self):
        filepath = filedialog.askopenfilename()
        if filepath:
            self.file_paths.append(filepath)
            filename = os.path.basename(filepath)
            self.display_message("System", f"File uploaded: {filename}")
            self.process_uploaded_file(filepath)

    def process_uploaded_file(self, filepath):
        try:
            self.df = pd.read_csv(filepath)
            print(self.df.head())
        except Exception as e:
            self.display_message("System", f"Error processing file: {e}")

    def summarize_dataframe(self):
        df = self.df
        summary = []
        stats = df.describe()

        for column in df.select_dtypes(include=[np.number]).columns:
            col_stats = stats[column]
            summary.append(
                f"{column} has {int(col_stats['count'])} entries. "
                f"The average value is {col_stats['mean']:.3f}, with a standard deviation of {col_stats['std']:.3f}. "
                f"The minimum value is {col_stats['min']:.3f}, while the maximum is {col_stats['max']:.3f}. "
                f"Median is {col_stats['50%']:.3f}, with 25% of the values below {col_stats['25%']:.3f} "
                f"and 75% below {col_stats['75%']:.3f}."
            )

        return " ".join(summary)

    def process_message(self, message):
        if "hello" in message.lower() or "hi" in message.lower():
            self.display_message("Bot", "Hello there! Would you like some help analyzing a log file?")
        elif "files" in message.lower():
            if self.file_paths:
                file_names = [os.path.basename(path) for path in self.file_paths]
                self.display_message("Bot", f"Uploaded files: {', '.join(file_names)}")
            else:
                self.display_message("Bot", "No files uploaded yet.")
        else:
            if self.df is None:
                self.display_message("Bot", "It seems like you have not uploaded a CSV log file yet. Please do that first.")
                return

            # Process with NER model
            tokens, pred_labels, output_str = ner.predict_text(message)
            print("\n--- Results ---")
            print("Tokens:", tokens)
            print("Predicted Labels:", pred_labels)
            print("Structured Output:", output_str)
            print("----------------\n")

            # Process parsed data
            parsed_data = bff.parse_input(output_str)
            output = bff.execute_function(parsed_data, self.df)

            if output:
                self.display_message("Bot", output)
            elif "summary" in output_str:
                self.display_message("Bot", self.summarize_dataframe())
            else:
                # If no log-related response, fall back to ChatterBot
                bot_response = self.chatbot.get_response(message)
                self.display_message("Bot", str(bot_response))

# Run the chatbot
root = tk.Tk()
chatbot = ChatbotGUI(root)
root.mainloop()
