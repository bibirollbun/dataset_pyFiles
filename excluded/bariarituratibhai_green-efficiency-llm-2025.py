# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ===============================================================
# ğŸŒ± GREEN AI PROJECT â€” ENERGY EFFICIENT TEXT CLASSIFICATION
# ===============================================================

# ================================
# ğŸ“¦ 1. Install & Import Libraries
# ================================
!pip install -q codecarbon tensorflow==2.17.0

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from codecarbon import EmissionsTracker

# ================================
# âš™ï¸� 2. Setup
# ================================
os.makedirs("outputs", exist_ok=True)
np.random.seed(42)

# ================================
# ğŸ“Š 3. Create Synthetic Dataset
# ================================
texts = [
    "AI saves energy",
    "Deep learning needs GPUs",
    "Green AI is efficient",
    "Optimize your ML model",
    "Data science is fun",
    "Carbon-aware execution helps",
    "Smaller models are better",
    "Use CPU when possible",
    "Avoid large datasets",
    "Efficiency is key"
]
labels = [1, 0, 1, 1, 0, 1, 1, 0, 0, 1]

df = pd.DataFrame({"text": texts, "label": labels})
df.to_csv("outputs/dataset.csv", index=False)

# ================================
# ğŸ“ˆ 4. Exploratory Data Analysis
# ================================
plt.figure(figsize=(6,4))
sns.countplot(x="label", data=df, palette="crest")
plt.title("Label Distribution")
plt.savefig("outputs/eda_analysis.png", bbox_inches="tight")
plt.close()

# ================================
# ğŸ§¹ 5. Tokenization & Padding
# ================================
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])
padded = pad_sequences(sequences, maxlen=10, padding="post")

X_train, X_test, y_train, y_test = train_test_split(
    padded, df["label"], test_size=0.2, random_state=42
)

# ================================
# ğŸ”‹ 6. Train Model with Emission Tracking
# ================================
tracker = EmissionsTracker(output_dir="outputs", output_file="emissions_log.csv")
tracker.start()

model = Sequential([
    Embedding(1000, 16),
    LSTM(16, return_sequences=False),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

checkpoint = ModelCheckpoint("outputs/best_model.h5", save_best_only=True, monitor="val_accuracy", mode="max")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=2, callbacks=[checkpoint], verbose=0)

tracker.stop()

# ================================
# ğŸ“Š 7. Plot Training History
# ================================
plt.figure(figsize=(6,4))
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.savefig("outputs/training_history.png", bbox_inches="tight")
plt.close()

# ================================
# ğŸ§ª 8. Evaluate Model
# ================================
pred_probs = model.predict(X_test)
preds = (pred_probs > 0.5).astype("int32")
acc = accuracy_score(y_test, preds)
cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png", bbox_inches="tight")
plt.close()

# ================================
# ğŸ§¾ 9. Save Reports
# ================================
eval_results = {
    "accuracy": float(acc),
    "classification_report": classification_report(y_test, preds, output_dict=True)
}
json.dump(eval_results, open("outputs/evaluation_results.json", "w"), indent=4)

final_report = {
    "project": "Green AI Text Classifier",
    "summary": "Energy-efficient model trained with emissions tracking.",
    "metrics": eval_results
}
json.dump(final_report, open("outputs/final_report.json", "w"), indent=4)

# ================================
# ğŸ“¤ 10. Create Submission File
# ================================
submission = pd.DataFrame({
    "Id": np.arange(1, len(preds) + 1),
    "Prediction": preds.flatten()
})
submission.to_csv("outputs/submission.csv", index=False)

print("âœ… submission.csv created successfully with 'Id' column!")

# ================================
# ğŸŒ± 11. Summary
# ================================
print("\nğŸŒ± GREEN AI SUMMARY ğŸŒ±")
print(f"Accuracy: {acc:.4f}")
print("All outputs saved under the 'outputs/' directory.")


