import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import threading
import speech_recognition as sr
import qrcode
from PIL import ImageTk, Image
import os

class ChatbotGUI:
    def __init__(self, root, chatbot_logic):
        self.root = root
        self.root.title("Chatbot with Voice and QR")
        self.chatbot_logic = chatbot_logic
        self.recognizer = sr.Recognizer()  # Speech recognition

        # Chat History
        self.chat_history = scrolledtext.ScrolledText(root, state='disabled', height=15, width=50)
        self.chat_history.pack(padx=10, pady=10)

        # Input Frame
        input_frame = tk.Frame(root)
        input_frame.pack(padx=10, pady=5, fill=tk.X)

        self.input_entry = tk.Entry(input_frame, width=40)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.send_button = tk.Button(input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.LEFT)

        # Voice Note Button
        self.voice_button = tk.Button(input_frame, text="Voice", command=self.record_voice)
        self.voice_button.pack(side=tk.LEFT)

        # QR Code Button
        self.qr_button = tk.Button(input_frame, text="QR", command=self.generate_qr)
        self.qr_button.pack(side=tk.LEFT)

        # QR Code Display (initially hidden)
        self.qr_label = tk.Label(root)
        self.qr_label.pack(pady=5)
        self.qr_label.pack_forget() #hide initially

    def display_message(self, message, sender):
        self.chat_history.config(state='normal')
        self.chat_history.insert(tk.END, f"{sender}: {message}\n")
        self.chat_history.config(state='disabled')
        self.chat_history.yview(tk.END)

    def send_message(self):
        user_input = self.input_entry.get()
        if user_input:
            self.display_message(user_input, "You")
            self.input_entry.delete(0, tk.END)
            threading.Thread(target=self.get_chatbot_response, args=(user_input,)).start()

    def get_chatbot_response(self, user_input):
        chatbot_response = self.chatbot_logic(user_input)
        self.root.after(0, self.display_message, chatbot_response, "Chatbot")

    def record_voice(self):
        recognizer = sr.Recognizer()
        self.display_message("Listening...", "System")
        self.root.update_idletasks()  # Force the GUI to update immediately
        
        # Start listening in a separate daemon thread
        thread = threading.Thread(target=self._listen_and_process, args=(recognizer,))
        thread.daemon = True  # This ensures the thread won't block the app exit
        thread.start()

    def _listen_and_process(self, recognizer):
        try:
            with sr.Microphone() as source:
                # Set a short timeout and phrase_time_limit for quicker returns
                audio = recognizer.listen(source, timeout=2, phrase_time_limit=3)
                self._process_audio(recognizer, audio)
        except sr.WaitTimeoutError:
            self.root.after(0, self.display_message, "Timeout: No speech detected.", "System")
        except sr.UnknownValueError:
            self.root.after(0, self.display_message, "Could not understand audio.", "System")
        except sr.RequestError as e:
            self.root.after(0, self.display_message, f"Speech recognition error: {e}", "System")
        except Exception as e:
            self.root.after(0, self.display_message, f"Unexpected error: {e}", "System")

    def _process_audio(self, recognizer, audio):
        try:
            text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            try:
                text = recognizer.recognize_sphinx(audio)
            except sr.UnknownValueError:
                self.root.after(0, self.display_message, "Could not understand audio.", "System")
                return
        except sr.RequestError:
            self.root.after(0, self.display_message, "Speech recognition service is unavailable.", "System")
            return

        # Update GUI with recognized text and process the chatbot response
        self.root.after(0, self.display_message, text, "You (Voice)")
        threading.Thread(target=self.get_chatbot_response, args=(text,), daemon=True).start()




    def generate_qr(self):
        text_to_encode = self.input_entry.get()
        if not text_to_encode:
            messagebox.showerror("Error", "Please enter text to encode.")
            return

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(text_to_encode)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        img_tk = ImageTk.PhotoImage(img) #Convert to Tkinter PhotoImage

        self.qr_label.config(image=img_tk)
        self.qr_label.image = img_tk # Keep a reference!
        self.qr_label.pack() #show the label
        self.input_entry.delete(0, tk.END)

def simple_chatbot(user_input):
    user_input = user_input.lower()
    if "hello" in user_input or "hi" in user_input:
        return "Hello! How can I help you?"
    elif "how are you" in user_input:
        return "I'm doing well, thank you!"
    elif "bye" in user_input:
        return "Goodbye!"
    else:
        return "I don't understand. Could you please rephrase?"

if __name__ == "__main__":
    root = tk.Tk()
    chatbot_gui = ChatbotGUI(root, simple_chatbot)
    root.mainloop()