import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")




# Drop rows with missing Category or Misconception
train = train.dropna(subset=["Category", "Misconception"]).copy()

# Create combined label: Category:Misconception
train["Label"] = train["Category"] + ":" + train["Misconception"]

# Combine QuestionText and StudentExplanation as features
train["text"] = train["QuestionText"].fillna("") + " " + train["StudentExplanation"].fillna("")
test["text"] = test["QuestionText"].fillna("") + " " + test["StudentExplanation"].fillna("")





# ================= TRAIN/VALIDATION SPLIT (no stratify) =================
X_train, X_val, y_train, y_val = train_test_split(
    train["text"], train["Label"], test_size=0.2, random_state=42, shuffle=True
)




#  MODEL PIPELINE 
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),  # unigrams + bigrams
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
])


# Train model
pipeline.fit(X_train, y_train)

#  VALIDATION CHECK 
val_probs = pipeline.predict_proba(X_val)
classes = pipeline.named_steps["clf"].classes_
val_preds_top3 = np.argsort(val_probs, axis=1)[:, -3:][:, ::-1]  # top-3 indices
val_preds_top3_labels = np.array(classes)[val_preds_top3]

# MAP@3 metric
def mapk(y_true, y_pred, k=3):
    score = 0.0
    for t, preds in zip(y_true, y_pred):
        for i, p in enumerate(preds[:k]):
            if p == t:
                score += 1.0 / (i+1)
                break
    return score / len(y_true)

val_score = mapk(y_val.tolist(), val_preds_top3_labels.tolist(), k=3)
print("Validation MAP@3:", val_score)

#  TRAIN ON FULL DATA & PREDICT TEST 
pipeline.fit(train["text"], train["Label"])

test_probs = pipeline.predict_proba(test["text"])
test_preds_top3 = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]
test_preds_top3_labels = np.array(classes)[test_preds_top3]

# space-separated top-3 predictions
submission = pd.DataFrame({
    "row_id": test["row_id"],
    "Category:Misconception": [" ".join(row) for row in test_preds_top3_labels]
})

submission.to_csv("submission.csv", index=False)

