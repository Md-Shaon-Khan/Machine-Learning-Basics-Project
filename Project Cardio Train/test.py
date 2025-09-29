import tkinter as tk
from tkinter import ttk, messagebox
from threading import Thread
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def run_training(progress_bar, log_box, canvas_frame, predict_btn, model_container):
    feature_names = [f"F{i+1}" for i in range(20)]
    X, y = make_classification(n_samples=70000, n_features=20, n_informative=15, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1, tree_method="hist", subsample=0.8, colsample_bytree=0.8, eval_metric="logloss")
    total_iters = model.get_params().get("n_estimators", 200)

    class ProgressCallback(xgb.callback.TrainingCallback):
        def __init__(self):
            self.iter = 0
        def after_iteration(self, model, epoch, evals_log):
            self.iter += 1
            progress_bar["value"] = int(100 * self.iter / total_iters)
            if "validation_0" in evals_log and "logloss" in evals_log["validation_0"]:
                loss = evals_log["validation_0"]["logloss"][-1]
                log_box.insert(tk.END, f"Iteration {self.iter}/{total_iters} - logloss: {loss:.4f}\n")
            else:
                log_box.insert(tk.END, f"Iteration {self.iter}/{total_iters}\n")
            log_box.see(tk.END)
            return False

    log_box.insert(tk.END, "Training started...\n")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False, early_stopping_rounds=20, callbacks=[ProgressCallback()])
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    log_box.insert(tk.END, f"Training finished!\nTrain Accuracy: {train_acc*100:.2f}%\nTest Accuracy: {test_acc*100:.2f}%\n")
    model_container["model"] = model
    model_container["feature_names"] = feature_names
    def draw_plot():
        fig, ax = plt.subplots(figsize=(6, 4))
        xgb.plot_importance(model, ax=ax, importance_type="weight", title="Feature Importance")
        for widget in canvas_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    canvas_frame.after(0, draw_plot)

def start_training(progress_bar, log_box, canvas_frame, predict_btn, model_container):
    Thread(target=run_training, args=(progress_bar, log_box, canvas_frame, predict_btn, model_container), daemon=True).start()

def predict_input(model_container):
    model = model_container.get("model", None)
    feature_names = model_container.get("feature_names", [f"F{i+1}" for i in range(20)])
    if model is None:
        messagebox.showerror("Error", "Model not trained yet.")
        return
    input_window = tk.Toplevel()
    input_window.title("Enter Features for Prediction")
    entries = []
    for i, name in enumerate(feature_names):
        tk.Label(input_window, text=name).grid(row=i, column=0, padx=5, pady=2)
        entry = tk.Entry(input_window)
        entry.grid(row=i, column=1, padx=5, pady=2)
        entries.append(entry)
    def submit():
        try:
            values = [float(e.get()) for e in entries]
            X_new = np.array(values).reshape(1, -1)
            pred = model.predict(X_new)[0]
            prob = model.predict_proba(X_new)[0][pred]
            messagebox.showinfo("Prediction", f"Predicted Class: {pred}\nProbability: {prob:.4f}")
            input_window.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
    tk.Button(input_window, text="Predict", command=submit).grid(row=len(feature_names), column=0, columnspan=2, pady=10)

def main():
    root = tk.Tk()
    root.title("XGBoost Training GUI")
    root.geometry("950x650")
    model_container = {}
    progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
    progress_bar.pack(pady=10)
    log_box = tk.Text(root, height=12, width=100)
    log_box.pack(pady=10)
    canvas_frame = tk.Frame(root, width=800, height=300)
    canvas_frame.pack(fill="both", expand=True)
    predict_btn = tk.Button(root, text="Predict New Data", state="disabled", command=lambda: predict_input(model_container))
    predict_btn.pack(pady=5)
    start_btn = tk.Button(root, text="Start Training", command=lambda: start_training(progress_bar, log_box, canvas_frame, predict_btn, model_container))
    start_btn.pack(pady=5)
    root.mainloop()

if __name__ == "__main__":
    main()