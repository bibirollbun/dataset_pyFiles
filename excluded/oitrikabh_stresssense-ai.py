# ============================
# SECTION 1 — IMPORT LIBRARIES
# ============================
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================================
# SECTION 2 — LOAD DATA
# ============================================================
# IMPORTANT:
# Make sure your dataset has columns: "text" and "label"
# Example labels: low, medium, high

df = pd.read_csv("/kaggle/input/stress-dataset/stress_dataset.csv")
df.head()

# ============================================================
# SECTION 3 — CLEANING THE TEXT
# ============================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)

# Convert labels to numbers
label_map = {"low": 0, "medium": 1, "high": 2}
df["label"] = df["label"].map(label_map)

df.head()


# ============================================================
# SECTION 4 — TF-IDF VECTORIZATION
# ============================================================
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

# ============================================================
# SECTION 5 — SPLIT DATA
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# SECTION 6 — TRAIN MODEL
# ============================================================
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))


# ============================================================
# SECTION 7 — EVALUATION
# ============================================================
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ============================================================
# SECTION 8 — WORD CLOUD (optional but nice for demo)
# ============================================================
for label, name in zip([0, 1, 2], ["Low Stress", "Medium Stress", "High Stress"]):
    text = " ".join(df[df["label"] == label]["clean_text"])
    wc = WordCloud(width=800, height=300, background_color="white").generate(text)
    
    plt.figure(figsize=(10, 4))
    plt.imshow(wc, interpolation="bilinear")
    plt.title(name)
    plt.axis("off")
    plt.show()


# ============================================================
# SECTION 9 — SIMPLE PREDICTION FUNCTION
# ============================================================
def predict_stress(message):
    clean = clean_text(message)
    feat = vectorizer.transform([clean])
    pred = model.predict(feat)[0]
    conf = round(np.max(model.predict_proba(feat)), 3)
    
    label_names = {0: "Low Stress", 1: "Medium Stress", 2: "High Stress"}
    return label_names[pred], conf

print(predict_stress("I feel overwhelmed with work today."))


# ============================================================
# SECTION 10 — MULTI-AGENT SYSTEM (Google requirement)
# ============================================================

# Agent 1 — Interpretation Agent (LLM-like)
def agent_1_interpret(text):
    return clean_text(text)

# Agent 2 — ML Prediction Agent
def agent_2_predict(cleaned_text):
    feat = vectorizer.transform([cleaned_text])
    label = model.predict(feat)[0]
    confidence = np.max(model.predict_proba(feat))
    return label, confidence

# Agent 3 — Feedback Agent
def agent_3_feedback(label, memory=[]):
    names = {0: "Low Stress", 1: "Medium Stress", 2: "High Stress"}
    
    response = f"Detected Stress Level: {names[label]}\n"
    
    if label == 2:
        response += "⚠ High stress! Take a break and breathe slowly.\n"
    elif label == 1:
        response += "⚠ Moderate stress. Pace your tasks and hydrate.\n"
    else:
        response += "✓ You seem calm. Keep it up!\n"

    if len(memory) >= 3:
        trend = "rising" if label > np.mean(memory[-3:]) else "improving"
        response += f"Trend Note: Your stress appears to be {trend}.\n"

    return response

# ============================================================
# SECTION 11 — FULL PIPELINE (Sequential agents + memory)
# ============================================================
memory = []  # Long-term session memory

def full_pipeline(user_input):
    cleaned = agent_1_interpret(user_input)
    label, conf = agent_2_predict(cleaned)
    memory.append(label)
    feedback = agent_3_feedback(label, memory)
    
    return label, conf, feedback

# Test the full system
label, conf, fb = full_pipeline("Deadlines are stressing me out so much.")
print("Confidence:", conf)
print(fb)

