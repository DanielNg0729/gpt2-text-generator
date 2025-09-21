import tkinter as tk
from tkinter import scrolledtext
from transformers import pipeline
import torch

# Load model
device = 0 if torch.cuda.is_available() else -1
model = pipeline("text-generation", model="gpt2", device=device)

# Function to handle text generation
def generate_text():
    prompt = entry.get("1.0", tk.END).strip()
    if not prompt:
        return
    result = model(
        prompt,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        max_length=256,
        num_return_sequences=1
    )
    output_text.delete("1.0", tk.END)  # clear previous
    output_text.insert(tk.END, result[0]["generated_text"].strip())

#Simple GUI Setup
root = tk.Tk()
root.title("GPT2 Text Generator")
root.geometry("600x400")

#Prompt input
tk.Label(root, text="Enter your prompt:").pack(anchor="w", padx=10, pady=5)
entry = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=5)
entry.pack(fill="x", padx=10, pady=5)

#Generate button
tk.Button(root, text="Generate", command=generate_text).pack(pady=5)

#Output box
tk.Label(root, text="Generated text:").pack(anchor="w", padx=10, pady=5)
output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=10)
output_text.pack(fill="both", expand=True, padx=10, pady=5)


root.mainloop()

