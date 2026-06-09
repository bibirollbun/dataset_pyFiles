# Fertilizer Prediction - MAP@3 Kaggle Competition Notebook

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.ensemble import RandomForestClassifier

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

# Encode categorical features
cat_cols = ['Soil Type', 'Crop Type']
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# Encode target
target_le = LabelEncoder()
train['Fertilizer Name'] = target_le.fit_transform(train['Fertilizer Name'])

# Features and targets
X = train.drop(columns=['id', 'Fertilizer Name'])
y = train['Fertilizer Name']
X_test = test.drop(columns=['id'])


# ðŸ“¦ Import Libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ðŸ§¼ Basic Preprocessing
features = ['Soil Type', 'Crop Type', 'Nitrogen', 'Phosphorous', 'Potassium']
target = 'Fertilizer Name'

# Encode categorical features
train = train.copy()
test = test.copy()

# Fill missing values if any
imputer = SimpleImputer(strategy='most_frequent')
train[features] = imputer.fit_transform(train[features])
test[features] = imputer.transform(test[features])

# Encode categorical variables
cat_cols = ['Soil Type', 'Crop Type']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    encoders[col] = le

# Encode Target
target_le = LabelEncoder()
train['label'] = target_le.fit_transform(train[target])

# ðŸ“Š Train-Test Split
X = train[features]
y = train['label']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ðŸŒ³ Model Training
model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# ðŸŽ¯ Evaluate
val_preds = model.predict(X_val)
acc = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {acc:.4f}")

# ðŸ”® Predict Top 3 Labels
probs = model.predict_proba(test[features])
top_3_idx = np.argsort(probs, axis=1)[:, -3:][:, ::-1]  # Get indices of top 3 probabilities
top_3_labels = target_le.inverse_transform(top_3_idx.ravel()).reshape(top_3_idx.shape)  # Convert indices to fertilizer names

# ðŸ“¤ Prepare Submission
submission = pd.DataFrame({
    "id": test["id"],
    "Fertilizer Name": [' '.join(preds) for preds in top_3_labels]  # Space-delimited top 3 fertilizer names
})

# Save to CSV
submission.to_csv("submission.csv", index=False)
print("âœ… Submission file saved as 'submission.csv'")


submission.head()




