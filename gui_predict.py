import tkinter as tk
from tkinter import scrolledtext, messagebox
from predict_model import load_models, predict_sentiment

# Load model components
vectorizer, lda, model = load_models()

# Handle prediction button click
def on_predict():
    review = input_text.get("1.0", tk.END).strip()
    if not review:
        messagebox.showwarning("Warning", "Enter a review first.")
        return
    sentiment = predict_sentiment(review, vectorizer, lda, model)
    result_label.config(text=f"🎯 Sentiment: {sentiment}")

# Setup GUI window
root = tk.Tk()
root.title("IMDB Predictor")
root.geometry("680x460")
root.configure(bg="#FBEAFF")

# GUI layout
tk.Label(root, text="🎬 Enter a Movie Review:", font=("Helvetica", 16, "bold"), bg="#FBEAFF", fg="#7B4B94").pack(pady=10)

input_text = scrolledtext.ScrolledText(root, width=70, height=12, font=("Helvetica", 13),
                                       bg="#FFF9FB", fg="#3C3C3C", insertbackground="black")
input_text.pack(padx=20)

tk.Button(root, text="🔮 Predict Sentiment", command=on_predict, font=("Helvetica", 14, "bold"),
          bg="#EBD8F6", fg="#4B3F72", relief=tk.FLAT).pack(pady=20)

result_label = tk.Label(root, text="", font=("Helvetica", 18, "bold"), fg="#5F8D4E", bg="#FBEAFF")
result_label.pack(pady=10)

root.mainloop()
