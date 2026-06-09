import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.multioutput import MultiOutputClassifier


# ãƒ‡ãƒ¼ã‚¿èª­ã�¿è¾¼ã�¿
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


# ===============================
# ğŸ�¯ Encode target labels
# ===============================
le = LabelEncoder()
train["FertLabel"] = le.fit_transform(train["Fertilizer Name"])
target_classes = le.classes_

# ===============================
# ğŸ”¤ Encode categorical variables
# ===============================
cat_cols = ['Soil Type', 'Crop Type']
for col in cat_cols:
    train[col] = LabelEncoder().fit_transform(train[col])
    test[col] = LabelEncoder().fit_transform(test[col])  # Ensure same encoding


# ===============================
# ğŸ§ª Select features
# ===============================
features = [col for col in train.columns if col not in ['id', 'Fertilizer Name', 'FertLabel']]
X = train[features]
y = train["FertLabel"]
X_test = test[features]

# ===============================
# ğŸ“Š Train LightGBM model
# ===============================
model = lgb.LGBMClassifier(random_state=42)
model.fit(X, y)

# ===============================
# ğŸ“ˆ Predict class probabilities
# ===============================
probs = model.predict_proba(X_test)

# Get top 5 predicted classes (MAP@5)
top_5_preds = np.argsort(probs, axis=1)[:, -5:][:, ::-1]  # Reverse order (highest first)


preds_labels = [" ".join(le.inverse_transform(row)) for row in top_5_preds]
submission = pd.DataFrame({
    "id": test["id"],
    "Fertilizer Name": preds_labels
})

submission.to_csv("baseline_submission.csv", index=False)
print("âœ… baseline_submission.csv saved!")

