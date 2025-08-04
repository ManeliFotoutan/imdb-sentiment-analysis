import tkinter as tk
from tkinter import scrolledtext, messagebox, simpledialog
from train_model import train_and_save_models

# Start training process and show results
def start_training():
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, "📥 Loading and training...\n")
    root.update()
    try:
        report, acc, f1 = train_and_save_models(data_dir)
        output_text.insert(tk.END, f"✅ Done! Accuracy: {acc:.4f}, F1: {f1:.4f}\n\n")
        output_text.insert(tk.END, "Classification Report:\n")
        output_text.insert(tk.END, report)
        predict_btn.pack(pady=10)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Open prediction GUI window
def run_predict():
    import subprocess
    subprocess.run(["python3", "gui_predict.py"])

# Setup main training GUI window
root = tk.Tk()
root.title("IMDB Trainer")
root.geometry("820x780")
root.configure(bg="#FBEAFF")

# Ask for dataset path
data_dir = simpledialog.askstring("📁 Dataset Path", "Enter IMDB dataset path:", parent=root)
if not data_dir:
    messagebox.showerror("Error", "Path is required.")
    root.destroy()
    exit()

# GUI elements
tk.Label(root, text="🎬 IMDB Trainer", font=("Helvetica", 18, "bold"), bg="#FBEAFF", fg="#7B4B94").pack(pady=15)

train_btn = tk.Button(root, text="🚀 Start Training", command=start_training,
                      font=("Helvetica", 14, "bold"), bg="#EBD8F6", fg="#4B3F72", relief=tk.FLAT)
train_btn.pack(pady=10)

predict_btn = tk.Button(root, text="🧪 Go to Prediction", command=run_predict,
                        font=("Helvetica", 14, "bold"), bg="#D5B3E6", fg="#4B3F72", relief=tk.FLAT, padx=30)

output_text = scrolledtext.ScrolledText(root, width=90, height=25, font=("Consolas", 10),
                                        bg="#FFF9FB", fg="#3C3C3C", insertbackground="black")
output_text.pack(padx=12, pady=12)

root.mainloop()
