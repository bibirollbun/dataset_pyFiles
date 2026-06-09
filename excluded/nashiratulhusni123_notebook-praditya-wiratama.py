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
!pip install optuna
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session!pip install optuna


import pandas as pd
import lightgbm as lgb
import joblib
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import numpy as np

# Load dataset
df = pd.read_csv("/kaggle/input/seleksi-academya-data-science-2025/train.csv")
df.head()

# Explanatory Data Analysis (EDA)
print("Dataset Overview:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

# Visualisasi distribusi target
sns.countplot(x=df['Result'])
plt.title("Distribution of Target Variable")
plt.show()

# Visualisasi korelasi fitur numerik
plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Visualisasi covariance matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include=['number']).cov(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Covariance Matrix")
plt.show()

# Pisahkan fitur dan target
X = df.drop(columns=["Result", "id"], errors="ignore")
y = df["Result"]

# Konversi target ke numerik
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)  # Mengubah 'Legitimate' & 'Phishing' ke angka 0 & 1

# Pisahkan fitur numerik dan kategorikal
num_cols = X.select_dtypes(include=['number']).columns.tolist()
cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()

# Imputasi nilai yang hilang
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

X[num_cols] = num_imputer.fit_transform(X[num_cols])
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

# Pastikan semua data numerik
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le


# Feature Scaling
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Mengatasi class imbalance dengan SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)


# Cek apakah missing values telah ditangani
print("\nMissing Values after preprocessing:")
print(pd.DataFrame(X, columns=num_cols + cat_cols).isnull().sum())

# Visualisasi outlier setelah preprocessing
plt.figure(figsize=(12, 6))
sns.boxplot(data=pd.DataFrame(X, columns=num_cols))
plt.xticks(rotation=90)
plt.title("Outlier Detection After Preprocessing")
plt.show()

# Split data menjadi train dan validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Simpan preprocessing tools
joblib.dump(num_imputer, "num_imputer.pkl")
joblib.dump(cat_imputer, "cat_imputer.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_dict, "label_encoders.pkl")


# Function untuk Hyperparameter Tuning dengan Optuna
def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'accuracy',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0)
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='logloss', callbacks=[lgb.early_stopping(100)])

    preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)

    return accuracy

# Optimasi Hyperparameter
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Gunakan parameter terbaik
best_params = study.best_params
print("Best Parameters:", best_params)

# Train model dengan parameter terbaik
best_model = lgb.LGBMClassifier(**best_params)
best_model.fit(X_train, y_train)



# Evaluasi model
preds = best_model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, preds))
print(classification_report(y_val, preds))


# Confusion Matrix
cm = confusion_matrix(y_val, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ROC Curve dan AUC
y_probs = best_model.predict_proba(X_val)[:, 1]
fpr, tpr, _ = roc_curve(y_val, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# Simpan model
joblib.dump(best_model, "best_lightgbm_model.pkl")

# Load test dataset
test_df = pd.read_csv("/kaggle/input/seleksi-academya-data-science-2025/test.csv")
df.head()
test_X = test_df.drop(columns=["id"], errors="ignore")

# Terapkan preprocessing ke data uji
test_X[num_cols] = num_imputer.transform(test_X[num_cols])
test_X[cat_cols] = cat_imputer.transform(test_X[cat_cols])
for col in cat_cols:
    if col in le_dict:
        le = le_dict[col]
        test_X[col] = test_X[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
test_X[num_cols] = scaler.transform(test_X[num_cols])

# Prediksi data uji
test_preds = best_model.predict(test_X)
submission = pd.DataFrame({"id": test_df["id"], "Result": test_preds})
submission.to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv")


# Visualisasi distribusi hasil prediksi
sns.countplot(x=test_preds)
plt.title("Distribusi Prediksi pada Data Uji")
plt.show()


