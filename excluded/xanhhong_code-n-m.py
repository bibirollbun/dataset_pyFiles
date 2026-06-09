import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sns.set()



train_df = pd.read_csv("/kaggle/input/mushroom-classification-btl/train.csv")
test_df  = pd.read_csv("/kaggle/input/mushroom-classification-btl/test.csv")

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print("\nClass distribution in train:")
print(train_df['class'].value_counts())

# keep original test ids for submission
test_ids = test_df['id'].copy()

train_df.head()



# Make copies to avoid modifying original DataFrames
train = train_df.copy()
test = test_df.copy()

# mark and make class column present in test for unified encoding
train['is_train'] = 1
test['is_train']  = 0
test['class'] = np.nan   # placeholder so columns align for concat

# concat
full = pd.concat([train, test], ignore_index=True, sort=False)

# columns to encode: all except id and is_train
cols_to_encode = [c for c in full.columns if c not in ('id','is_train')]

# create and store encoders per column
encoders = {}
for col in cols_to_encode:
    full[col] = full[col].fillna('NA').astype(str)   # safe string representation
    le = LabelEncoder()
    full[col] = le.fit_transform(full[col])
    encoders[col] = le

# split back
train_encoded = full[full['is_train'] == 1].drop('is_train', axis=1).reset_index(drop=True)
test_encoded  = full[full['is_train'] == 0].drop(['is_train','class'], axis=1).reset_index(drop=True)

# features / labels
X = train_encoded.drop(['id','class'], axis=1)
y = train_encoded['class'].astype(int)   # already encoded numeric
X_test = test_encoded.drop(['id'], axis=1)

print("X shape:", X.shape, "y shape:", y.shape, "X_test shape:", X_test.shape)

# quick check encoders for 'class'
print("Class encoder classes (original labels):", encoders['class'].classes_)



X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)
print(classification_report(y_val, y_val_pred))

# ROC AUC: get index of positive label (we'll treat 'p' as positive if encoders['class'] maps it so)
# find original label for poisonous 'p' in encoder
try:
    pos_label_encoded = encoders['class'].transform(['p'])[0]   # encoded number for 'p'
    pos_index = list(model.classes_).index(pos_label_encoded)
    y_val_proba = model.predict_proba(X_val)[:, pos_index]
    print("ROC AUC:", roc_auc_score(y_val, y_val_proba))
except Exception as e:
    print("Không thể tính ROC AUC tự động:", e)
    # fallback: print model.classes_
    print("Model classes (encoded):", model.classes_)



cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (validation)")
plt.show()



final_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
final_model.fit(X, y)

test_pred_encoded = final_model.predict(X_test)  # encoded numeric labels

# inverse transform to original 'e'/'p'
test_pred_labels = encoders['class'].inverse_transform(test_pred_encoded.astype(int))

# sanity
print("Unique predictions in submission:", np.unique(test_pred_labels))



submission = pd.DataFrame({
    "id": test_ids,
    "class": test_pred_labels
})
submission.to_csv("submission.csv", index=False)
print("Saved -> submission.csv (rows = {})".format(len(submission)))

# save model & encoders if you want
joblib.dump(final_model, "rf_final.joblib")
joblib.dump(encoders, "encoders.joblib")
print("Saved rf_final.joblib and encoders.joblib")
submission.head()


