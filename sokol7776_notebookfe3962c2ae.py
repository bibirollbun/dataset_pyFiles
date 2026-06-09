# --- Ğ£Ñ�Ñ‚Ğ°Ğ½Ğ¾Ğ²ĞºĞ° Ğ´Ğ¾Ğ¿Ğ¾Ğ»Ğ½Ğ¸Ñ‚ĞµĞ»ÑŒĞ½Ñ‹Ñ… Ğ±Ğ¸Ğ±Ğ»Ğ¸Ğ¾Ñ‚ĞµĞº ---
# Ğ—Ğ°Ğ¿ÑƒÑ�Ñ‚Ğ¸Ñ‚Ğµ Ñ�Ñ‚Ñƒ Ñ�Ñ‡ĞµĞ¹ĞºÑƒ, ĞµÑ�Ğ»Ğ¸ Ñƒ Ğ²Ğ°Ñ� Ğ½Ğµ ÑƒÑ�Ñ‚Ğ°Ğ½Ğ¾Ğ²Ğ»ĞµĞ½Ñ‹ hyperopt Ğ¸ optuna
!pip install hyperopt
!pip install optuna

# --- Ğ˜Ğ¼Ğ¿Ğ¾Ñ€Ñ‚ Ğ²Ñ�ĞµÑ… Ğ½ĞµĞ¾Ğ±Ñ…Ğ¾Ğ´Ğ¸Ğ¼Ñ‹Ñ… Ğ¸Ğ½Ñ�Ñ‚Ñ€ÑƒĞ¼ĞµĞ½Ñ‚Ğ¾Ğ² ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ĞœĞ¾Ğ´ĞµĞ»Ğ¸
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Ğ˜Ğ½Ñ�Ñ‚Ñ€ÑƒĞ¼ĞµĞ½Ñ‚Ñ‹ Ğ´Ğ»Ñ� Ğ¾Ñ†ĞµĞ½ĞºĞ¸ Ğ¸ Ñ€Ğ°Ğ·Ğ´ĞµĞ»ĞµĞ½Ğ¸Ñ� Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score

# Ğ˜Ğ½Ñ�Ñ‚Ñ€ÑƒĞ¼ĞµĞ½Ñ‚Ñ‹ Ğ´Ğ»Ñ� Ğ¿Ğ¾Ğ´Ğ±Ğ¾Ñ€Ğ° Ğ³Ğ¸Ğ¿ĞµÑ€Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¾Ğ²
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import optuna

# Ğ�Ğ°Ñ�Ñ‚Ñ€Ğ¾Ğ¹ĞºĞ¸ Ğ´Ğ»Ñ� ĞºÑ€Ğ°Ñ�Ğ¸Ğ²Ğ¾Ğ³Ğ¾ Ğ¾Ñ‚Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ñ�
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("âœ… Ğ‘Ğ¸Ğ±Ğ»Ğ¸Ğ¾Ñ‚ĞµĞºĞ¸ ÑƒÑ�Ñ‚Ğ°Ğ½Ğ¾Ğ²Ğ»ĞµĞ½Ñ‹ Ğ¸ Ğ¸Ğ¼Ğ¿Ğ¾Ñ€Ñ‚Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ñ‹!")

# --- Ğ—Ğ°Ğ³Ñ€ÑƒĞ·ĞºĞ° Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ… ---
try:
    # Ğ•Ñ�Ğ»Ğ¸ Ğ²Ñ‹ Ñ€Ğ°Ğ±Ğ¾Ñ‚Ğ°ĞµÑ‚Ğµ Ğ² Google Colab, Ğ·Ğ°Ğ³Ñ€ÑƒĞ·Ğ¸Ñ‚Ğµ Ñ„Ğ°Ğ¹Ğ» Ğ² Ñ�ĞµÑ�Ñ�Ğ¸Ğ¾Ğ½Ğ½Ğ¾Ğµ Ñ…Ñ€Ğ°Ğ½Ğ¸Ğ»Ğ¸Ñ‰Ğµ
    df = pd.read_csv('/kaggle/input/bioresponse/train.csv')
    print("âœ… Ğ”Ğ°Ğ½Ğ½Ñ‹Ğµ ÑƒÑ�Ğ¿ĞµÑˆĞ½Ğ¾ Ğ·Ğ°Ğ³Ñ€ÑƒĞ¶ĞµĞ½Ñ‹!")
    print(f"Ğ Ğ°Ğ·Ğ¼ĞµÑ€ Ğ´Ğ°Ñ‚Ğ°Ñ�ĞµÑ‚Ğ°: {df.shape[0]} Ñ�Ñ‚Ñ€Ğ¾Ğº, {df.shape[1]} Ñ�Ñ‚Ğ¾Ğ»Ğ±Ñ†Ğ¾Ğ²")
except FileNotFoundError:
    print("â�—ï¸� Ğ�ÑˆĞ¸Ğ±ĞºĞ°: Ğ¤Ğ°Ğ¹Ğ» 'train.csv' Ğ½Ğµ Ğ½Ğ°Ğ¹Ğ´ĞµĞ½. Ğ£Ğ±ĞµĞ´Ğ¸Ñ‚ĞµÑ�ÑŒ, Ñ‡Ñ‚Ğ¾ Ğ¾Ğ½ Ğ»ĞµĞ¶Ğ¸Ñ‚ Ğ² Ñ‚Ğ¾Ğ¹ Ğ¶Ğµ Ğ¿Ğ°Ğ¿ĞºĞµ, Ñ‡Ñ‚Ğ¾ Ğ¸ Ğ²Ğ°Ñˆ Ğ±Ğ»Ğ¾ĞºĞ½Ğ¾Ñ‚, Ğ¸Ğ»Ğ¸ ÑƒĞºĞ°Ğ¶Ğ¸Ñ‚Ğµ Ğ¿Ñ€Ğ°Ğ²Ğ¸Ğ»ÑŒĞ½Ñ‹Ğ¹ Ğ¿ÑƒÑ‚ÑŒ.")



# ĞŸĞ¾Ñ�Ğ¼Ğ¾Ñ‚Ñ€Ğ¸Ğ¼ Ğ½Ğ° Ğ¿ĞµÑ€Ğ²Ñ‹Ğµ 5 Ñ�Ñ‚Ñ€Ğ¾Ğº
print("ĞŸĞµÑ€Ğ²Ñ‹Ğµ 5 Ñ�Ñ‚Ñ€Ğ¾Ğº Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…:")
display(df.head())

# ĞŸÑ€Ğ¾Ğ²ĞµÑ€Ğ¸Ğ¼, ĞµÑ�Ñ‚ÑŒ Ğ»Ğ¸ Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¸ (Ñ�Ğ¿Ğ¾Ğ¹Ğ»ĞµÑ€: Ğ½ĞµÑ‚, ĞºĞ°Ğº Ğ¸ Ğ¾Ğ±ĞµÑ‰Ğ°Ğ»Ğ¸)
print(f"\nĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ‰ĞµĞ½Ğ½Ñ‹Ñ… Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğ¹: {df.isnull().sum().sum()}")

# Ğ¡Ğ°Ğ¼Ğ¾Ğµ Ğ²Ğ°Ğ¶Ğ½Ğ¾Ğµ: Ğ¿Ğ¾Ñ�Ğ¼Ğ¾Ñ‚Ñ€Ğ¸Ğ¼ Ğ½Ğ° Ğ±Ğ°Ğ»Ğ°Ğ½Ñ� ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ² Ğ² Ğ½Ğ°ÑˆĞµĞ¹ Ñ†ĞµĞ»Ğ¸ (Activity)
# Ğ­Ñ‚Ğ¾ Ğ¿Ğ¾ĞºĞ°Ğ¶ĞµÑ‚, Ñ�ĞºĞ¾Ğ»ÑŒĞºĞ¾ Ñƒ Ğ½Ğ°Ñ� Ğ¼Ğ¾Ğ»ĞµĞºÑƒĞ» Ñ� Ğ¾Ñ‚Ğ²ĞµÑ‚Ğ¾Ğ¼ "1" Ğ¸ "0"
activity_balance = df['Activity'].value_counts(normalize=True) * 100
print(f"\nĞ‘Ğ°Ğ»Ğ°Ğ½Ñ� ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ²:")
print(f"ĞšĞ»Ğ°Ñ�Ñ� 1 (Ğ°ĞºÑ‚Ğ¸Ğ²Ğ½Ñ‹Ğ¹ Ğ¾Ñ‚Ğ²ĞµÑ‚): {activity_balance[1]:.2f}%")
print(f"ĞšĞ»Ğ°Ñ�Ñ� 0 (Ğ½ĞµĞ°ĞºÑ‚Ğ¸Ğ²Ğ½Ñ‹Ğ¹ Ğ¾Ñ‚Ğ²ĞµÑ‚): {activity_balance[0]:.2f}%")

# Ğ Ğ°Ğ·Ğ´ĞµĞ»Ğ¸Ğ¼ Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ Ğ½Ğ° Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸ (X) Ğ¸ Ñ†ĞµĞ»ÑŒ (y)
X = df.drop('Activity', axis=1)
y = df['Activity']

# Ğ Ğ°Ğ·Ğ´ĞµĞ»Ğ¸Ğ¼ Ğ½Ğ° Ğ¾Ğ±ÑƒÑ‡Ğ°Ñ�Ñ‰ÑƒÑ� Ğ¸ Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²ÑƒÑ� Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞ¸
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\nĞ”Ğ°Ğ½Ğ½Ñ‹Ğµ Ñ€Ğ°Ğ·Ğ´ĞµĞ»ĞµĞ½Ñ‹ Ğ½Ğ° Ğ¾Ğ±ÑƒÑ‡Ğ°Ñ�Ñ‰ÑƒÑ� Ğ¸ Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²ÑƒÑ� Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞ¸.")


# --- Ğ›Ğ¾Ğ³Ğ¸Ñ�Ñ‚Ğ¸Ñ‡ĞµÑ�ĞºĞ°Ñ� Ñ€ĞµĞ³Ñ€ĞµÑ�Ñ�Ğ¸Ñ� "Ğ¸Ğ· ĞºĞ¾Ñ€Ğ¾Ğ±ĞºĞ¸" ---
lr_base = LogisticRegression(max_iter=1000, random_state=42)
lr_base.fit(X_train, y_train)
y_pred_lr = lr_base.predict(X_test)
f1_lr_base = f1_score(y_test, y_pred_lr)
print(f"Ğ‘Ğ°Ğ·Ğ¾Ğ²Ñ‹Ğ¹ F1-score Ğ´Ğ»Ñ� Ğ›Ğ¾Ğ³Ğ¸Ñ�Ñ‚Ğ¸Ñ‡ĞµÑ�ĞºĞ¾Ğ¹ Ñ€ĞµĞ³Ñ€ĞµÑ�Ñ�Ğ¸Ğ¸: {f1_lr_base:.4f}")

# --- Ğ¡Ğ»ÑƒÑ‡Ğ°Ğ¹Ğ½Ñ‹Ğ¹ Ğ»ĞµÑ� "Ğ¸Ğ· ĞºĞ¾Ñ€Ğ¾Ğ±ĞºĞ¸" ---
rf_base = RandomForestClassifier(random_state=42)
rf_base.fit(X_train, y_train)
y_pred_rf = rf_base.predict(X_test)
f1_rf_base = f1_score(y_test, y_pred_rf)
print(f"Ğ‘Ğ°Ğ·Ğ¾Ğ²Ñ‹Ğ¹ F1-score Ğ´Ğ»Ñ� Ğ¡Ğ»ÑƒÑ‡Ğ°Ğ¹Ğ½Ğ¾Ğ³Ğ¾ Ğ»ĞµÑ�Ğ°: {f1_rf_base:.4f}")

# Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ğ¸Ğ¼ Ñ€ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ñ‹ Ğ´Ğ»Ñ� Ñ„Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½Ğ¾Ğ¹ Ñ‚Ğ°Ğ±Ğ»Ğ¸Ñ†Ñ‹
results = {
    'Ğ›Ğ¾Ğ³Ğ¸Ñ�Ñ‚Ğ¸Ñ‡ĞµÑ�ĞºĞ°Ñ� Ñ€ĞµĞ³Ñ€ĞµÑ�Ñ�Ğ¸Ñ� (Ğ±Ğ°Ğ·Ğ°)': f1_lr_base,
    'Ğ¡Ğ»ÑƒÑ‡Ğ°Ğ¹Ğ½Ñ‹Ğ¹ Ğ»ĞµÑ� (Ğ±Ğ°Ğ·Ğ°)': f1_rf_base
}


import time
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ĞŸÑ€ĞµĞ´Ğ¿Ğ¾Ğ»Ğ°Ğ³Ğ°ĞµĞ¼, Ñ‡Ñ‚Ğ¾ X_train, y_train Ğ¸ results ÑƒĞ¶Ğµ Ñ�ÑƒÑ‰ĞµÑ�Ñ‚Ğ²ÑƒÑ�Ñ‚

# --- 1. GridSearchCV Ğ´Ğ»Ñ� Ğ›Ğ¾Ğ³Ğ¸Ñ�Ñ‚Ğ¸Ñ‡ĞµÑ�ĞºĞ¾Ğ¹ Ñ€ĞµĞ³Ñ€ĞµÑ�Ñ�Ğ¸Ğ¸ ("ĞŸĞ�Ğ¡Ğ›Ğ•Ğ”Ğ�Ğ˜Ğ™ Ğ”Ğ�Ğ’Ğ�Ğ”") ---
print("\n--- 1. GridSearchCV Ğ´Ğ»Ñ� Ğ›Ğ¾Ğ³Ğ¸Ñ�Ñ‚Ğ¸Ñ‡ĞµÑ�ĞºĞ¾Ğ¹ Ñ€ĞµĞ³Ñ€ĞµÑ�Ñ�Ğ¸Ğ¸ (Ğ³Ğ°Ñ€Ğ°Ğ½Ñ‚Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ½Ğ¾Ğµ Ñ€ĞµÑˆĞµĞ½Ğ¸Ğµ) ---")

# âœ¨ Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ñ‚Ğ¾Ñ‚ Ğ¶Ğµ Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42))
])

# âœ¨ Ğ�Ğ´Ğ°Ğ¿Ñ‚Ğ¸Ñ€ÑƒĞµĞ¼ Ñ�ĞµÑ‚ĞºÑƒ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¾Ğ²:
param_grid_lr_final = {
    'model__C': [0.1, 1, 10, 20],
    'model__solver': ['liblinear'],  # <--- ĞœĞ•Ğ�Ğ¯Ğ•Ğœ SOLVER! Ğ­Ñ‚Ğ¾ ĞºĞ»Ñ�Ñ‡ĞµĞ²Ğ¾Ğµ Ğ¸Ğ·Ğ¼ĞµĞ½ĞµĞ½Ğ¸Ğµ.
    'model__penalty': ['l1', 'l2'],  # <--- liblinear Ğ¾Ñ‚Ğ»Ğ¸Ñ‡Ğ½Ğ¾ Ñ€Ğ°Ğ±Ğ¾Ñ‚Ğ°ĞµÑ‚ Ñ� Ğ¾Ğ±Ğ¾Ğ¸Ğ¼Ğ¸.
    'model__max_iter': [5000]        # <--- Ğ£Ğ²ĞµĞ»Ğ¸Ñ‡Ğ¸Ğ²Ğ°ĞµĞ¼ Ñ� Ğ¾Ğ³Ñ€Ğ¾Ğ¼Ğ½Ñ‹Ğ¼ Ğ·Ğ°Ğ¿Ğ°Ñ�Ğ¾Ğ¼.
}
# ĞšĞ¾Ğ¼Ğ±Ğ¸Ğ½Ğ°Ñ†Ğ¸Ğ¹ Ğ²Ñ�Ñ‘ ĞµÑ‰Ñ‘ 8, Ğ½Ğ¾ Ğ¾Ğ½Ğ¸ Ğ±ÑƒĞ´ÑƒÑ‚ Ñ�Ñ‡Ğ¸Ñ‚Ğ°Ñ‚ÑŒÑ�Ñ� Ğ±Ñ‹Ñ�Ñ‚Ñ€ĞµĞµ.

# âœ¨ Ğ—Ğ°Ğ¿ÑƒÑ�ĞºĞ°ĞµĞ¼ GridSearchCV Ñ� Ğ½Ğ¾Ğ²Ğ¾Ğ¹, ÑƒĞ±Ğ¾Ğ¹Ğ½Ğ¾Ğ¹ ĞºĞ¾Ğ½Ñ„Ğ¸Ğ³ÑƒÑ€Ğ°Ñ†Ğ¸ĞµĞ¹
grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid_lr_final,
    scoring='f1',
    cv=3,
    n_jobs=-1,
    verbose=1
)

print("Ğ—Ğ°Ğ¿ÑƒÑ�ĞºĞ°ĞµĞ¼ Ñ„Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¹ Ğ¿Ğ¾Ğ¸Ñ�Ğº Ñ� Ñ�Ğ¾Ğ»Ğ²ĞµÑ€Ğ¾Ğ¼ 'liblinear'. Ğ�Ğ½ Ğ´Ğ¾Ğ»Ğ¶ĞµĞ½ Ğ±Ñ‹Ñ‚ÑŒ Ğ±Ñ‹Ñ�Ñ‚Ñ€ĞµĞµ Ğ¸ Ğ±ĞµĞ· Ğ¿Ñ€ĞµĞ´ÑƒĞ¿Ñ€ĞµĞ¶Ğ´ĞµĞ½Ğ¸Ğ¹...")
start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()

print(f"\nĞŸĞ¾Ğ¸Ñ�Ğº Ğ·Ğ°Ğ²ĞµÑ€ÑˆĞµĞ½ Ğ·Ğ° {(end_time - start_time):.1f} Ñ�ĞµĞºÑƒĞ½Ğ´.")
print(f"Ğ›ÑƒÑ‡ÑˆĞ¸Ğµ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹: {grid_search.best_params_}")
print(f"Ğ›ÑƒÑ‡ÑˆĞ¸Ğ¹ F1-score Ğ½Ğ° ĞºÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¸: {grid_search.best_score_:.4f}")

results['Ğ›Ğ¾Ğ³. Ñ€ĞµĞ³Ñ€ĞµÑ�Ñ�Ğ¸Ñ� (liblinear + Pipeline)'] = grid_search.best_score_
print("\nâœ… Ğ ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½.")


print("\n--- 2. RandomizedSearchCV Ğ´Ğ»Ñ� Ğ¡Ğ»ÑƒÑ‡Ğ°Ğ¹Ğ½Ğ¾Ğ³Ğ¾ Ğ»ĞµÑ�Ğ° ---")

# Ğ—Ğ°Ğ´Ğ°ĞµĞ¼ Ğ´Ğ¸Ğ°Ğ¿Ğ°Ğ·Ğ¾Ğ½Ñ‹, Ğ¸Ğ· ĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ñ… Ğ±ÑƒĞ´ĞµĞ¼ Ñ�Ğ»ÑƒÑ‡Ğ°Ğ¹Ğ½Ğ¾ Ğ²Ñ‹Ğ±Ğ¸Ñ€Ğ°Ñ‚ÑŒ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹
param_dist_rf = {
    'n_estimators': [100, 200, 300],         # ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ´ĞµÑ€ĞµĞ²ÑŒĞµĞ²
    'max_depth': [10, 20, 30, None],         # ĞœĞ°ĞºÑ�. Ğ³Ğ»ÑƒĞ±Ğ¸Ğ½Ğ° Ğ´ĞµÑ€ĞµĞ²Ğ°
    'min_samples_leaf': [1, 2, 4],           # ĞœĞ¸Ğ½. Ñ‡Ğ¸Ñ�Ğ»Ğ¾ Ğ¾Ğ±Ñ€Ğ°Ğ·Ñ†Ğ¾Ğ² Ğ² Ğ»Ğ¸Ñ�Ñ‚Ğµ
    'min_samples_split': [2, 5, 10]          # ĞœĞ¸Ğ½. Ñ‡Ğ¸Ñ�Ğ»Ğ¾ Ğ¾Ğ±Ñ€Ğ°Ğ·Ñ†Ğ¾Ğ² Ğ´Ğ»Ñ� Ñ€Ğ°Ğ·Ğ´ĞµĞ»ĞµĞ½Ğ¸Ñ� ÑƒĞ·Ğ»Ğ°
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist_rf,
    n_iter=50,  # <-- Ğ’Ğ¾Ñ‚ Ğ½Ğ°ÑˆĞµ Ğ¾Ğ³Ñ€Ğ°Ğ½Ğ¸Ñ‡ĞµĞ½Ğ¸Ğµ Ğ² 50 Ğ¸Ñ‚ĞµÑ€Ğ°Ñ†Ğ¸Ğ¹
    scoring='f1',
    cv=3,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

print(f"Ğ›ÑƒÑ‡ÑˆĞ¸Ğµ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹: {random_search.best_params_}")
print(f"Ğ›ÑƒÑ‡ÑˆĞ¸Ğ¹ F1-score Ğ½Ğ° ĞºÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¸: {random_search.best_score_:.4f}")
results['Ğ¡Ğ»ÑƒÑ‡Ğ°Ğ¹Ğ½Ñ‹Ğ¹ Ğ»ĞµÑ� (RandomizedSearchCV)'] = random_search.best_score_


print("\n--- 3. Hyperopt Ğ´Ğ»Ñ� Ğ¡Ğ»ÑƒÑ‡Ğ°Ğ¹Ğ½Ğ¾Ğ³Ğ¾ Ğ»ĞµÑ�Ğ° ---")

# 1. Ğ—Ğ°Ğ´Ğ°ĞµĞ¼ Ğ¿Ñ€Ğ¾Ñ�Ñ‚Ñ€Ğ°Ğ½Ñ�Ñ‚Ğ²Ğ¾ Ğ¿Ğ¾Ğ¸Ñ�ĞºĞ° (Ğ¿Ğ¾Ñ…Ğ¾Ğ¶Ğµ Ğ½Ğ° Ğ¼ĞµĞ½Ñ�, Ğ½Ğ¾ Ñ� Ñ€Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ñ�Ğ¼Ğ¸)
space_rf = {
    'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400]),
    'max_depth': hp.quniform('max_depth', 5, 30, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1)
}

# 2. Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ "Ñ†ĞµĞ»ĞµĞ²ÑƒÑ� Ñ„ÑƒĞ½ĞºÑ†Ğ¸Ñ�", ĞºĞ¾Ñ‚Ğ¾Ñ€ÑƒÑ� Hyperopt Ğ±ÑƒĞ´ĞµÑ‚ Ğ¼Ğ¸Ğ½Ğ¸Ğ¼Ğ¸Ğ·Ğ¸Ñ€Ğ¾Ğ²Ğ°Ñ‚ÑŒ
# Ğ�Ğ½Ğ° Ğ¾Ğ±ÑƒÑ‡Ğ°ĞµÑ‚ Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ñ� Ğ·Ğ°Ğ´Ğ°Ğ½Ğ½Ñ‹Ğ¼Ğ¸ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ°Ğ¼Ğ¸ Ğ¸ Ğ²Ğ¾Ğ·Ğ²Ñ€Ğ°Ñ‰Ğ°ĞµÑ‚ Ğ¾Ñ†ĞµĞ½ĞºÑƒ
def objective_rf(params):
    # Hyperopt Ğ²Ñ‹Ğ´Ğ°ĞµÑ‚ float, Ğ° Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸ Ğ½ÑƒĞ¶Ğ½Ñ‹ int Ğ´Ğ»Ñ� Ğ½ĞµĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ñ… Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¾Ğ²
    params['max_depth'] = int(params['max_depth'])
    params['min_samples_leaf'] = int(params['min_samples_leaf'])

    model = RandomForestClassifier(**params, random_state=42)

    # Ğ˜Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·ÑƒĞµĞ¼ ĞºÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ñ� Ğ´Ğ»Ñ� Ğ½Ğ°Ğ´ĞµĞ¶Ğ½Ğ¾Ğ¹ Ğ¾Ñ†ĞµĞ½ĞºĞ¸
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='f1', n_jobs=-1).mean()

    # Hyperopt Ğ¼Ğ¸Ğ½Ğ¸Ğ¼Ğ¸Ğ·Ğ¸Ñ€ÑƒĞµÑ‚, Ğ° Ğ½Ğ°Ğ¼ F1 Ğ½ÑƒĞ¶Ğ½Ğ¾ Ğ¼Ğ°ĞºÑ�Ğ¸Ğ¼Ğ¸Ğ·Ğ¸Ñ€Ğ¾Ğ²Ğ°Ñ‚ÑŒ, Ğ¿Ğ¾Ñ�Ñ‚Ğ¾Ğ¼Ñƒ Ğ²Ğ¾Ğ·Ğ²Ñ€Ğ°Ñ‰Ğ°ĞµĞ¼ -score
    return {'loss': -score, 'status': STATUS_OK}

# 3. Ğ—Ğ°Ğ¿ÑƒÑ�ĞºĞ°ĞµĞ¼ Ğ¿Ğ¾Ğ¸Ñ�Ğº!
trials = Trials()
best_params_hyperopt = fmin(
    fn=objective_rf,
    space=space_rf,
    algo=tpe.suggest, # "Ğ£Ğ¼Ğ½Ñ‹Ğ¹" Ğ°Ğ»Ğ³Ğ¾Ñ€Ğ¸Ñ‚Ğ¼ Ğ¿Ğ¾Ğ¸Ñ�ĞºĞ°
    max_evals=50,    # <-- Ğ�Ğ°ÑˆĞµ Ğ¾Ğ³Ñ€Ğ°Ğ½Ğ¸Ñ‡ĞµĞ½Ğ¸Ğµ Ğ² 50 Ğ¸Ñ‚ĞµÑ€Ğ°Ñ†Ğ¸Ğ¹
    trials=trials
)

# Hyperopt Ğ²Ğ¾Ğ·Ğ²Ñ€Ğ°Ñ‰Ğ°ĞµÑ‚ Ğ¸Ğ½Ğ´ĞµĞºÑ�Ñ‹, Ğ° Ğ½Ğµ Ñ�Ğ°Ğ¼Ğ¸ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ñ� Ğ´Ğ»Ñ� hp.choice, Ğ¿Ñ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·ÑƒĞµĞ¼ Ğ¾Ğ±Ñ€Ğ°Ñ‚Ğ½Ğ¾
best_n_estimators = [100, 200, 300, 400][best_params_hyperopt['n_estimators']]
print(f"Ğ›ÑƒÑ‡ÑˆĞ¸Ğµ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹: n_estimators={best_n_estimators}, max_depth={int(best_params_hyperopt['max_depth'])}, min_samples_leaf={int(best_params_hyperopt['min_samples_leaf'])}")
best_score_hyperopt = -min(trials.losses())
print(f"Ğ›ÑƒÑ‡ÑˆĞ¸Ğ¹ F1-score Ğ½Ğ° ĞºÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¸: {best_score_hyperopt:.4f}")
results['Ğ¡Ğ»ÑƒÑ‡Ğ°Ğ¹Ğ½Ñ‹Ğ¹ Ğ»ĞµÑ� (Hyperopt)'] = best_score_hyperopt


import optuna
import time
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ĞŸÑ€ĞµĞ´Ğ¿Ğ¾Ğ»Ğ°Ğ³Ğ°ĞµĞ¼, Ñ‡Ñ‚Ğ¾ X_train, y_train Ğ¸ results ÑƒĞ¶Ğµ Ñ�ÑƒÑ‰ĞµÑ�Ñ‚Ğ²ÑƒÑ�Ñ‚

print("\n--- 4. Optuna Ğ´Ğ»Ñ� Ğ›Ğ¾Ğ³Ğ¸Ñ�Ñ‚Ğ¸Ñ‡ĞµÑ�ĞºĞ¾Ğ¹ Ñ€ĞµĞ³Ñ€ĞµÑ�Ñ�Ğ¸Ğ¸ (Ğ¡Ğ�ĞœĞ�Ğ¯ Ğ�Ğ‘Ğ©Ğ˜Ğ¢Ğ•Ğ›Ğ¬Ğ�Ğ�Ğ¯ Ğ’Ğ•Ğ Ğ¡Ğ˜Ğ¯) ---")

def objective_lr_optuna(trial):
    # âœ¨ Ğ�Ğ�Ğ’Ğ�Ğ•: Ğ¡Ğ¾Ğ¾Ğ±Ñ‰Ğ°ĞµĞ¼ Ğ¾ Ğ½Ğ°Ñ‡Ğ°Ğ»Ğµ Ñ€Ğ°Ğ±Ğ¾Ñ‚Ñ‹
    print(f"ğŸš€ Ğ�Ğ°Ñ‡Ğ¸Ğ½Ğ°Ñ� Trial Ğ½Ğ¾Ğ¼ĞµÑ€ {trial.number}...")
    
    params = {
        'model__C': trial.suggest_float('C', 1e-3, 1e2, log=True),
        'model__solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
        'model__penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'model__max_iter': trial.suggest_categorical('max_iter', [3000])
    }

    # ĞŸÑ€Ğ¾Ğ²ĞµÑ€ĞºĞ° Ñ�Ğ¾Ğ²Ğ¼ĞµÑ�Ñ‚Ğ¸Ğ¼Ğ¾Ñ�Ñ‚Ğ¸
    solver = params['model__solver']
    penalty = params['model__penalty']
    if solver == 'liblinear' and penalty not in ['l1', 'l2']:
        raise optuna.exceptions.TrialPruned()
    if solver == 'saga' and penalty not in ['l1', 'l2']:
        raise optuna.exceptions.TrialPruned()
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42))
    ])
    pipe.set_params(**params)

    # Ğ—Ğ°Ğ¿ÑƒÑ�ĞºĞ°ĞµĞ¼ Ğ¾Ñ†ĞµĞ½ĞºÑƒ (Ğ½Ğ° Ğ¾Ğ´Ğ½Ğ¾Ğ¼ Ñ�Ğ´Ñ€Ğµ, Ğ½Ğ¾ Ñ‚ĞµĞ¿ĞµÑ€ÑŒ Ğ¼Ñ‹ Ğ±ÑƒĞ´ĞµĞ¼ Ğ²Ğ¸Ğ´ĞµÑ‚ÑŒ Ğ¿Ñ€Ğ¾Ğ³Ñ€ĞµÑ�Ñ�)
    start_time = time.time()
    score = cross_val_score(pipe, X_train, y_train, cv=3, scoring='f1').mean()
    end_time = time.time()
    
    # âœ¨ Ğ�Ğ�Ğ’Ğ�Ğ•: Ğ¡Ğ¾Ğ¾Ğ±Ñ‰Ğ°ĞµĞ¼ Ğ¾ Ñ€ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ğµ Ğ¸ Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ¸
    print(f"ğŸ�� Trial {trial.number} Ğ·Ğ°Ğ²ĞµÑ€ÑˆĞµĞ½ Ğ·Ğ° {(end_time - start_time):.1f} Ñ�ĞµĞº. Ğ ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚ (F1): {score:.4f}")
    
    return score

# Ğ—Ğ°Ğ¿ÑƒÑ�ĞºĞ°ĞµĞ¼ Ğ¸Ñ�Ñ�Ğ»ĞµĞ´Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ
print("\nĞ�Ğ°Ñ‡Ğ¸Ğ½Ğ°ĞµĞ¼ Ğ¸Ñ�Ñ�Ğ»ĞµĞ´Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ Optuna... ĞŸĞ¾Ğ¶Ğ°Ğ»ÑƒĞ¹Ñ�Ñ‚Ğ°, Ğ±ÑƒĞ´ÑŒÑ‚Ğµ Ñ‚ĞµÑ€Ğ¿ĞµĞ»Ğ¸Ğ²Ñ‹, Ğ¿ĞµÑ€Ğ²Ğ°Ñ� Ğ¿Ğ¾Ğ¿Ñ‹Ñ‚ĞºĞ° Ğ¼Ğ¾Ğ¶ĞµÑ‚ Ğ·Ğ°Ğ½Ñ�Ñ‚ÑŒ Ğ½ĞµĞºĞ¾Ñ‚Ğ¾Ñ€Ğ¾Ğµ Ğ²Ñ€ĞµĞ¼Ñ�.")
study = optuna.create_study(direction='maximize')

# Ğ”Ğ»Ñ� Ñ‚ĞµÑ�Ñ‚Ğ° Ğ¼Ğ¾Ğ¶Ğ½Ğ¾ Ğ½Ğ°Ñ‡Ğ°Ñ‚ÑŒ Ñ� Ğ¼ĞµĞ½ÑŒÑˆĞµĞ³Ğ¾ Ñ‡Ğ¸Ñ�Ğ»Ğ° Ğ¿Ğ¾Ğ¿Ñ‹Ñ‚Ğ¾Ğº, Ğ½Ğ°Ğ¿Ñ€Ğ¸Ğ¼ĞµÑ€, 10
study.optimize(objective_lr_optuna, n_trials=10, show_progress_bar=True) 
# ĞšĞ¾Ğ³Ğ´Ğ° ÑƒĞ±ĞµĞ´Ğ¸ÑˆÑŒÑ�Ñ�, Ñ‡Ñ‚Ğ¾ Ñ€Ğ°Ğ±Ğ¾Ñ‚Ğ°ĞµÑ‚, Ğ¼Ğ¾Ğ¶ĞµÑˆÑŒ Ğ¿Ğ¾Ğ¼ĞµĞ½Ñ�Ñ‚ÑŒ 10 Ğ¾Ğ±Ñ€Ğ°Ñ‚Ğ½Ğ¾ Ğ½Ğ° 50

print("\nĞ˜Ñ�Ñ�Ğ»ĞµĞ´Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ Ğ·Ğ°Ğ²ĞµÑ€ÑˆĞµĞ½Ğ¾!")
print(f"Ğ›ÑƒÑ‡ÑˆĞ¸Ğµ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹: {study.best_params}")
print(f"Ğ›ÑƒÑ‡ÑˆĞ¸Ğ¹ F1-score Ğ½Ğ° ĞºÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¸: {study.best_value:.4f}")

# results['Ğ›Ğ¾Ğ³. Ñ€ĞµĞ³Ñ€ĞµÑ�Ñ�Ğ¸Ñ� (Optuna + Pipeline)'] = study.best_value
# print("âœ… Ğ ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½!")


# ĞŸÑ€ĞµĞ²Ñ€Ğ°Ñ‚Ğ¸Ğ¼ Ğ½Ğ°Ñˆ Ñ�Ğ»Ğ¾Ğ²Ğ°Ñ€ÑŒ Ñ� Ñ€ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ğ°Ğ¼Ğ¸ Ğ² ĞºÑ€Ğ°Ñ�Ğ¸Ğ²ÑƒÑ� Ñ‚Ğ°Ğ±Ğ»Ğ¸Ñ†Ñƒ (DataFrame)
results_df = pd.DataFrame(
    list(results.items()),
    columns=['ĞœĞµÑ‚Ğ¾Ğ´', 'F1-score']
).sort_values('F1-score', ascending=False)

print("\n\n--- ğŸ�† Ğ˜Ñ‚Ğ¾Ğ³Ğ¾Ğ²Ğ°Ñ� Ñ‚Ğ°Ğ±Ğ»Ğ¸Ñ†Ğ° Ñ€ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ğ¾Ğ² ---")
print(results_df)

# Ğ� Ñ‚ĞµĞ¿ĞµÑ€ÑŒ Ğ²Ğ¸Ğ·ÑƒĞ°Ğ»Ğ¸Ğ·Ğ¸Ñ€ÑƒĞµĞ¼ Ğ´Ğ»Ñ� Ğ½Ğ°Ğ³Ğ»Ñ�Ğ´Ğ½Ğ¾Ñ�Ñ‚Ğ¸
plt.figure(figsize=(12, 8))
ax = sns.barplot(
    x='F1-score',
    y='ĞœĞµÑ‚Ğ¾Ğ´',
    data=results_df,
    palette='viridis'
)
ax.set_title('Ğ¡Ñ€Ğ°Ğ²Ğ½ĞµĞ½Ğ¸Ğµ F1-score Ğ´Ğ»Ñ� Ñ€Ğ°Ğ·Ğ½Ñ‹Ñ… Ğ¼Ğ¾Ğ´ĞµĞ»ĞµĞ¹ Ğ¸ Ğ¼ĞµÑ‚Ğ¾Ğ´Ğ¾Ğ² Ğ¾Ğ¿Ñ‚Ğ¸Ğ¼Ğ¸Ğ·Ğ°Ñ†Ğ¸Ğ¸', fontsize=16)
ax.set_xlabel('F1-score', fontsize=12)
ax.set_ylabel('ĞœĞµÑ‚Ğ¾Ğ´', fontsize=12)

# Ğ”Ğ¾Ğ±Ğ°Ğ²Ğ¸Ğ¼ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ñ� Ğ½Ğ° Ğ³Ñ€Ğ°Ñ„Ğ¸Ğº
for p in ax.patches:
    width = p.get_width()
    ax.text(width + 0.001, p.get_y() + p.get_height()/2,
            f'{width:.4f}',
            va='center')

plt.xlim(0.7, max(results_df['F1-score']) * 1.05) # Ğ£Ñ�Ñ‚Ğ°Ğ½Ğ¾Ğ²Ğ¸Ğ¼ Ğ¿Ñ€ĞµĞ´ĞµĞ»Ñ‹ Ğ¾Ñ�Ğ¸ X Ğ´Ğ»Ñ� Ğ½Ğ°Ğ³Ğ»Ñ�Ğ´Ğ½Ğ¾Ñ�Ñ‚Ğ¸
plt.show()



import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
import joblib

print("--- ğŸš€ Ğ¤Ğ¸Ğ½Ğ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ� Ğ»ÑƒÑ‡ÑˆĞµĞ¹ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸ ---")

# 1. Ğ—Ğ°Ğ´Ğ°ĞµĞ¼ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹ Ğ½Ğ°ÑˆĞµĞ³Ğ¾ Ñ‡ĞµĞ¼Ğ¿Ğ¸Ğ¾Ğ½Ğ° (Ğ¸Ğ· RandomizedSearchCV)
best_rf_params = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'random_state': 42,
    'n_jobs': -1
}

# 2. Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ¸ Ğ¾Ğ±ÑƒÑ‡Ğ°ĞµĞ¼ Ñ„Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½ÑƒÑ� Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ğ½Ğ° Ğ’Ğ¡Ğ•Ğ¥ Ñ‚Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½Ñ‹Ñ… Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
print("\nĞ�Ğ±ÑƒÑ‡Ğ°ĞµĞ¼ Ñ„Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½ÑƒÑ� Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ğ½Ğ° Ğ²Ñ�ĞµĞ¹ Ğ¾Ğ±ÑƒÑ‡Ğ°Ñ�Ñ‰ĞµĞ¹ Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞµ...")
final_model = RandomForestClassifier(**best_rf_params)
final_model.fit(X_train, y_train)
print("âœ… ĞœĞ¾Ğ´ĞµĞ»ÑŒ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ°!")

# 3. Ğ¤Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½Ğ°Ñ� Ğ¿Ñ€Ğ¾Ğ²ĞµÑ€ĞºĞ° Ğ½Ğ° Ğ¾Ñ‚Ğ»Ğ¾Ğ¶ĞµĞ½Ğ½Ğ¾Ğ¹ Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ğ¾Ğ¹ Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞµ (X_test)
# Ğ­Ñ‚Ğ¾ Ñ�Ğ°Ğ¼Ğ°Ñ� Ñ‡ĞµÑ�Ñ‚Ğ½Ğ°Ñ� Ğ¾Ñ†ĞµĞ½ĞºĞ° ĞºĞ°Ñ‡ĞµÑ�Ñ‚Ğ²Ğ° Ğ½Ğ°ÑˆĞµĞ¹ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸
y_pred_final = final_model.predict(X_test)
final_f1_score = f1_score(y_test, y_pred_final)

print(f"\nğŸ�† Ğ¤Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¹ F1-score Ğ½Ğ° Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ñ‹Ñ… Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…: {final_f1_score:.4f}")
print("\nĞŸĞ¾Ğ´Ñ€Ğ¾Ğ±Ğ½Ñ‹Ğ¹ Ğ¾Ñ‚Ñ‡ĞµÑ‚ Ğ¿Ğ¾ ĞºĞ»Ğ°Ñ�Ñ�Ğ°Ğ¼:")
print(classification_report(y_test, y_pred_final))

# 4. Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ Ğ³Ğ¾Ñ‚Ğ¾Ğ²ÑƒÑ� Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ğ² Ñ„Ğ°Ğ¹Ğ»
model_filename = 'final_random_forest.joblib'
joblib.dump(final_model, model_filename)
print(f"\nğŸ’¾ ĞœĞ¾Ğ´ĞµĞ»ÑŒ ÑƒÑ�Ğ¿ĞµÑˆĞ½Ğ¾ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ° Ğ² Ñ„Ğ°Ğ¹Ğ»: '{model_filename}'")
print("\nğŸ�‰ ĞŸĞ¾Ğ·Ğ´Ñ€Ğ°Ğ²Ğ»Ñ�Ñ�! ĞŸÑ€Ğ¾ĞµĞºÑ‚ Ğ¿Ğ¾ Ğ²Ñ‹Ğ±Ğ¾Ñ€Ñƒ Ğ¸ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸ Ğ·Ğ°Ğ²ĞµÑ€ÑˆĞµĞ½!")


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
import joblib

print("--- ğŸš€ Ğ¤Ğ¸Ğ½Ğ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ� ĞœĞ�Ğ”Ğ•Ğ›Ğ˜ Ğ¡ Ğ‘Ğ�Ğ›Ğ�Ğ�Ğ¡Ğ˜Ğ Ğ�Ğ’ĞšĞ�Ğ™ Ğ’Ğ•Ğ¡Ğ�Ğ’ ---")

# 1. Ğ—Ğ°Ğ´Ğ°ĞµĞ¼ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹, Ğ´Ğ¾Ğ±Ğ°Ğ²Ğ¸Ğ² Ğ±Ğ°Ğ»Ğ°Ğ½Ñ�Ğ¸Ñ€Ğ¾Ğ²ĞºÑƒ
best_rf_params_balanced = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'class_weight': 'balanced', # <--- âœ¨ Ğ’Ğ�Ğ¢ Ğ�Ğ�Ğ�, Ğ“Ğ›Ğ�Ğ’Ğ�Ğ�Ğ• Ğ˜Ğ—ĞœĞ•Ğ�Ğ•Ğ�Ğ˜Ğ•!
    'random_state': 42,
    'n_jobs': -1
}

# 2. Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ¸ Ğ¾Ğ±ÑƒÑ‡Ğ°ĞµĞ¼ Ğ½Ğ¾Ğ²ÑƒÑ�, Ñ�Ğ±Ğ°Ğ»Ğ°Ğ½Ñ�Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ½ÑƒÑ� Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ
print("\nĞ�Ğ±ÑƒÑ‡Ğ°ĞµĞ¼ Ñ�Ğ±Ğ°Ğ»Ğ°Ğ½Ñ�Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ½ÑƒÑ� Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ğ½Ğ° Ğ²Ñ�ĞµĞ¹ Ğ¾Ğ±ÑƒÑ‡Ğ°Ñ�Ñ‰ĞµĞ¹ Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞµ...")
final_model_balanced = RandomForestClassifier(**best_rf_params_balanced)
final_model_balanced.fit(X_train, y_train)
print("âœ… ĞœĞ¾Ğ´ĞµĞ»ÑŒ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ°!")

# 3. Ğ¤Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½Ğ°Ñ� Ğ¿Ñ€Ğ¾Ğ²ĞµÑ€ĞºĞ° Ğ½Ğ° Ğ¾Ñ‚Ğ»Ğ¾Ğ¶ĞµĞ½Ğ½Ğ¾Ğ¹ Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ğ¾Ğ¹ Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞµ (X_test)
y_pred_balanced = final_model_balanced.predict(X_test)
final_f1_score_balanced = f1_score(y_test, y_pred_balanced)

print(f"\nğŸ�† Ğ¤Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¹ F1-score Ğ½Ğ° Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ñ‹Ñ… Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ… (Ñ� Ğ±Ğ°Ğ»Ğ°Ğ½Ñ�Ğ¸Ñ€Ğ¾Ğ²ĞºĞ¾Ğ¹): {final_f1_score_balanced:.4f}")
print("\nĞŸĞ¾Ğ´Ñ€Ğ¾Ğ±Ğ½Ñ‹Ğ¹ Ğ¾Ñ‚Ñ‡ĞµÑ‚ Ğ¿Ğ¾ ĞºĞ»Ğ°Ñ�Ñ�Ğ°Ğ¼ (Ñ� Ğ±Ğ°Ğ»Ğ°Ğ½Ñ�Ğ¸Ñ€Ğ¾Ğ²ĞºĞ¾Ğ¹):")
print(classification_report(y_test, y_pred_balanced))

# 4. Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ Ğ½Ğ¾Ğ²ÑƒÑ�, ÑƒĞ»ÑƒÑ‡ÑˆĞµĞ½Ğ½ÑƒÑ� Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ğ² Ñ„Ğ°Ğ¹Ğ»
model_filename = 'final_balanced_random_forest.joblib'
joblib.dump(final_model_balanced, model_filename)
print(f"\nğŸ’¾ Ğ¡Ğ±Ğ°Ğ»Ğ°Ğ½Ñ�Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ½Ğ°Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ ÑƒÑ�Ğ¿ĞµÑˆĞ½Ğ¾ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ° Ğ² Ñ„Ğ°Ğ¹Ğ»: '{model_filename}'")

