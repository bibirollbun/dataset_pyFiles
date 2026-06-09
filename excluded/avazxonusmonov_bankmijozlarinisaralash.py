import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

try:
    train_df = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/train.csv")
    test_df = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/test.csv")
except FileNotFoundError:
    print("Xatolik: Fayllar topilmadi. Iltimos, 'train.csv', 'test.csv' mavjudligini tekshiring.")
    exit()


test_ids = test_df['id']


def preprocess_data(df):    
    df = df.drop(['CustomerId', 'Surname'], axis=1)
    return df

train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)

X = train_df.drop(['id', 'Exited'], axis=1)
y = train_df['Exited']
X_test = test_df.drop('id', axis=1)

categorical_features = ['Geography', 'Gender']
numerical_features = X.select_dtypes(include=np.number).columns.tolist()


X_combined = pd.concat([X, X_test], ignore_index=True)
X_combined_encoded = pd.get_dummies(X_combined, columns=categorical_features, drop_first=True)

X_encoded = X_combined_encoded.iloc[:len(X)].copy()
X_test_encoded = X_combined_encoded.iloc[len(X):].copy()

scaler = StandardScaler()

X_encoded.loc[:, numerical_features] = scaler.fit_transform(X_encoded[numerical_features])
X_test_encoded.loc[:, numerical_features] = scaler.transform(X_test_encoded[numerical_features])

rf_model = RandomForestClassifier(
    n_estimators=500, 
    max_depth=10,     
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,        
    class_weight='balanced'
)

print("Random Forest modelini o\'rgitish...")
rf_model.fit(X_encoded, y)

y_pred_proba = rf_model.predict_proba(X_test_encoded)[:, 1]

submission_df = pd.DataFrame({
    'id': test_ids,
    'Exited': y_pred_proba
})

submission_file_name = 'submission_random_forest_fixed.csv'
submission_df.to_csv(submission_file_name, index=False)

print("\nNatija muvaffaqiyatli saqlandi!")
print(f"Fayl nomi: {submission_file_name}")
print("Taqdimot faylining dastlabki 5 qatori:")
print(submission_df.head())


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np


train_df1 = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/train.csv")

X_all = train_df1.drop(['id', 'CustomerId', 'Surname', 'Exited'], axis=1)
y_all = train_df1['Exited']

X_encoded = pd.get_dummies(X_all, columns=['Geography', 'Gender'], drop_first=True)

numerical_features = X_encoded.select_dtypes(include=np.number).columns.tolist()
scaler = StandardScaler()
X_encoded.loc[:, numerical_features] = scaler.fit_transform(X_encoded[numerical_features])

X_train, X_val, y_train, y_val = train_test_split(
    X_encoded, y_all, test_size=0.2, random_state=42, stratify=y_all
)

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

print("Random Forest modelini o'rgitish...")
rf_model.fit(X_train, y_train)

y_pred_hard = rf_model.predict(X_val)

y_pred_proba = rf_model.predict_proba(X_val)[:, 1]

accuracy = accuracy_score(y_val, y_pred_hard)
auc_score = roc_auc_score(y_val, y_pred_proba)

print("\nModel Baholash Natijalari")
print(f"Bashorat aniqlik foizi (Accuracy): {accuracy * 100:.2f}%")
print(f"Asosiy baholash mezoni (AUC Score): {auc_score:.4f}")

