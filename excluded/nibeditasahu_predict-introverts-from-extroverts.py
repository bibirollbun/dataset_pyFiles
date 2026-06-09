# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


train_data = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


train_data.head()


test_data.head()


train_data.shape


test_data.shape


train_data.info()


test_data.info()


train_data.isnull().sum()


test_data.isnull().sum()


missing_pctage = (train_data.isnull().sum() / len(train_data)) * 100
missing_pctage


numeric_features = train_data.select_dtypes(include=["int", "float"])
numeric_features.columns


for nf in numeric_features.columns:
    Q1 = numeric_features[nf].quantile(0.25)
    Q3 = numeric_features[nf].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = numeric_features[(numeric_features[nf] < lower_bound) | (numeric_features[nf] > upper_bound)]
    print(f'"{nf}": Number of outliers =', outliers.shape[0])


combined_data = pd.concat([train_data, test_data], axis = 0)
combined_data


combined_encoded_data = pd.get_dummies(combined_data, drop_first=True)

train_data_encoded  = combined_encoded_data.iloc[:len(train_data), :]
test_data_encoded  = combined_encoded_data.iloc[len(train_data):, :]


from sklearn.impute import SimpleImputer


imputer = SimpleImputer(strategy="median")
train_data_encoded[:] = imputer.fit_transform(train_data_encoded)
test_data_encoded[:] = imputer.transform(test_data_encoded)


train_data_encoded.columns.tolist()


X = train_data_encoded.drop("Personality_Introvert", axis=1)
y = train_data_encoded["Personality_Introvert"]


from sklearn.model_selection import train_test_split


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


from sklearn.ensemble import RandomForestClassifier


rf_model = RandomForestClassifier(random_state=42)


rf_model.fit(X_train, y_train)


y_val_pred = rf_model.predict(X_val)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


acc = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {acc:.3f}")


print("Classification Report:")
print(classification_report(y_val, y_val_pred))


print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))


importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
importances.sort_values(ascending=False).plot(kind="barh", color="g")
plt.title("Feature Importances")
plt.show()


test_preds = rf_model.predict(test_data_encoded.drop("Personality_Introvert", axis=1))
test_preds


submission = pd.DataFrame({
    "id": test_data["id"],
    "Personality_Introvert": test_preds
})


submission["Personality"] = submission["Personality_Introvert"].map({1.0: "Introvert", 0.0: "Extrovert"})


if "Personality_Introvert" in submission.columns:
    submission.drop("Personality_Introvert", axis=1, inplace=True)


submission.to_csv("submission.csv", index=False)
print("Submission Successful!")

