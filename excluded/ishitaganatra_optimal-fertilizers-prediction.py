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


train_data = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


print("Train data shape : " , train_data.shape)
print("Test data shape : " , test_data.shape)


print("Train data head : \n" , train_data.head())
print("\n\nTest data head : \n" , test_data.head())


print("Train data info:")
print(train_data.info())
print("\n\nTest data info:")
print(test_data.info())


print("Train data null values sum for each column:")
print(train_data.isnull().sum())
print("\n\nTest data null values sum for each column:")
print(test_data.isnull().sum())


# Drop unnecessary columns
X = train_data.drop(['Fertilizer Name', 'id'], axis=1)
y = train_data['Fertilizer Name']
X_test = test_data.drop(['id'], axis=1)


from sklearn.preprocessing import LabelEncoder

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Encode categorical features
for col in ['Soil Type', 'Crop Type']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)


from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=4000,
    learning_rate=0.01,      
    max_depth=12,
    num_leaves=120,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_samples=15,
    reg_alpha=0.3,
    reg_lambda=0.4,
    random_state=42,
    device = 'GPU'
)

model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score

train_preds = model.predict(X_train)
train_acc = accuracy_score(y_train, train_preds)
print(f"Training Accuracy: {train_acc:.4f}")


val_preds = model.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {val_acc:.4f}")


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm_train = confusion_matrix(y_train, train_preds)
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=label_encoder.classes_)
plt.figure(figsize=(12, 10))
disp_train.plot(xticks_rotation=90)
plt.title("Confusion Matrix - Training Set")
plt.show()


cm_val = confusion_matrix(y_val, val_preds)
disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=label_encoder.classes_)
plt.figure(figsize=(12, 10))
disp_val.plot(xticks_rotation=90)
plt.title("Confusion Matrix - Validation Set")
plt.show()


val_probs = model.predict_proba(X_val)
val_top_3 = np.argsort(val_probs, axis=1)[:, -3:][:, ::-1]
val_top_3_labels = label_encoder.inverse_transform(val_top_3.flatten()).reshape(val_top_3.shape)


# Sample predictions
for i in range(5):
    actual = label_encoder.inverse_transform([y_val[i]])[0]
    preds = val_top_3_labels[i]
    print(f"Actual: {actual} | Predicted Top 3: {preds}")


def mapk(actual, predicted, k=3):
    score = 0.0
    for act, pred in zip(actual, predicted):
        if act in pred:
            rank = list(pred).index(act)
            score += 1.0 / (rank + 1)
    return score / len(actual)


actual_labels = label_encoder.inverse_transform(y_val)
map3_score = mapk(actual_labels, val_top_3_labels, k=3)
print(f"Validation MAP@3 Score: {map3_score:.4f}")


test_probs = model.predict_proba(X_test)
test_top_3 = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]
test_top_3_labels = label_encoder.inverse_transform(test_top_3.flatten()).reshape(test_top_3.shape)

# Convert top 3 to space-separated strings
submission_preds = [' '.join(row) for row in test_top_3_labels]

# Create and save submission DataFrame
submission_df = pd.DataFrame({
    'id': test_data['id'],
    'Fertilizer Name': submission_preds
})

submission_df.to_csv("submission.csv", index=False)
print("✅ Submission file created: submission.csv")

