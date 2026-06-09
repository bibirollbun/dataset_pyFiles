import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score


train_df = pd.read_csv('/kaggle/input/vector-borne-disease-classification-challenge/train.csv')
test_df = pd.read_csv('/kaggle/input/vector-borne-disease-classification-challenge/test.csv')
submission = pd.read_csv('/kaggle/input/vector-borne-disease-classification-challenge/sample_submission.csv')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print(train_df.head())


# Drop ID and Target
id_col = [col for col in train_df.columns if 'id' in col.lower()][0]
target_col = 'prognosis'

X = train_df.drop(columns=[id_col, target_col])
y = train_df[target_col]

# ⚠️ Make sure y is a list of lists
print(type(y.iloc[0]))  # should show <class 'list'>

# MultiLabel Binarization
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y_bin = mlb.fit_transform(y)

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Prepare test set
X_test = test_df.drop(columns=[id_col])
X_test_scaled = scaler.transform(X_test)



# Using Random Forest inside a One-vs-Rest for multi-label
clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
clf.fit(X_scaled, y_bin)


y_probs = clf.predict_proba(X_test_scaled)  # probabilities for each class
top_3 = np.argsort(y_probs, axis=1)[:, -3:][:, ::-1]  # top 3 indices per row

# Map back to label names
predictions = []
for row in top_3:
    predictions.append(' '.join([mlb.classes_[i] for i in row]))


submission['prognosis'] = predictions
submission.to_csv('vector_disease_submission.csv', index=False)
submission.head()


from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import label_ranking_average_precision_score
import numpy as np
import pandas as pd

# Model - you can replace this with LightGBM or any other
model = OneVsRestClassifier(RandomForestClassifier(n_estimators=200, random_state=42))
model.fit(X_scaled, y_bin)

# Predict probabilities
y_pred_proba = model.predict_proba(X_test_scaled)  # shape: (n_samples, n_classes)



# To calculate MAP@3 on training data (just for checking)
train_probs = model.predict_proba(X_scaled)
map3_score = label_ranking_average_precision_score(y_bin, train_probs)
print(f"Train MAP: {map3_score:.4f}")



# Get top 3 predictions per row
top3 = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]  # reverse for descending

# Convert indices back to labels
top3_labels = mlb.classes_[top3]

# Join top 3 as comma-separated string
predictions = [",".join(row) for row in top3_labels]

# Prepare submission
submission = pd.DataFrame({
    'patient_id': test_df[id_col],
    'prognosis': predictions
})

submission.to_csv("submission.csv", index=False)
print("✅ Submission file 'submission.csv' created.")
submission.head()



from sklearn.model_selection import train_test_split

# Split into train and val
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_bin, test_size=0.2, random_state=42)

# Train on training split
model.fit(X_train, y_train)

# Evaluate on validation split
val_probs = model.predict_proba(X_val)
val_map3 = label_ranking_average_precision_score(y_val, val_probs)
print(f"Validation MAP@3: {val_map3:.4f}")



# Rename the columns to match submission format
submission = submission.rename(columns={'patient_id': 'id', 'predicted': 'prognosis'})

# Save again
submission.to_csv("final_submission.csv", index=False)
print("✅ Submission file fixed and saved as 'submission.csv'")



# Load the submission to verify
import pandas as pd

submission = pd.read_csv("/kaggle/working/final_submission.csv")

# Check structure
print("✅ Columns:", submission.columns.tolist())
print("✅ Shape:", submission.shape)
print("✅ First few rows:")
print(submission.head())


