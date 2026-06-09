import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Ğ—Ğ°Ğ³Ñ€ÑƒĞ¶Ğ°ĞµĞ¼ Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ
data = pd.read_csv("../input/playground-series-s4e11/train.csv")

# Ğ’Ñ‹Ğ²Ğ¾Ğ´ Ğ¿ĞµÑ€Ğ²Ñ‹Ñ… Ñ�Ñ‚Ñ€Ğ¾Ğº
print("ĞŸĞµÑ€Ğ²Ñ‹Ğµ Ñ�Ñ‚Ñ€Ğ¾ĞºĞ¸ Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…:")
print(data.head())

# Ğ�Ñ�Ğ½Ğ¾Ğ²Ğ½Ğ°Ñ� Ğ¸Ğ½Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ†Ğ¸Ñ� Ğ¾ Ğ´Ğ°Ñ‚Ğ°Ñ�ĞµÑ‚Ğµ
print("\nĞ�Ğ±Ñ‰Ğ°Ñ� Ğ¸Ğ½Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ†Ğ¸Ñ� Ğ¾ Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…:")
print(data.info())

# Ğ¡Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸Ñ‡ĞµÑ�ĞºĞ¾Ğµ Ğ¾Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ Ñ‡Ğ¸Ñ�Ğ»Ğ¾Ğ²Ñ‹Ñ… Ğ¿ĞµÑ€ĞµĞ¼ĞµĞ½Ğ½Ñ‹Ñ…
print("\nĞ¡Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸ĞºĞ° Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…:")
print(data.describe())

# Ğ£ĞºĞ°Ğ·Ñ‹Ğ²Ğ°ĞµĞ¼ Ñ†ĞµĞ»ĞµĞ²Ğ¾Ğ¹ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°Ğº (Ğ·Ğ°Ğ¼ĞµĞ½Ğ¸Ñ‚Ğµ Ğ½Ğ° ĞºĞ¾Ñ€Ñ€ĞµĞºÑ‚Ğ½Ğ¾Ğµ Ğ½Ğ°Ğ·Ğ²Ğ°Ğ½Ğ¸Ğµ)
target_column = "Work Pressure"
if target_column not in data.columns:
    raise KeyError(f"Ğ¦ĞµĞ»ĞµĞ²Ğ¾Ğ¹ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°Ğº '{target_column}' Ğ¾Ñ‚Ñ�ÑƒÑ‚Ñ�Ñ‚Ğ²ÑƒĞµÑ‚ Ğ² Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…!")

# ĞŸÑ€Ğ¾Ğ²ĞµÑ€Ñ�ĞµĞ¼ Ñ‚Ğ¸Ğ¿ Ñ†ĞµĞ»ĞµĞ²Ğ¾Ğ³Ğ¾ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ°
if data[target_column].dtype == "object":
    task_type = "classification"
    print("\nâš¡ï¸� Ğ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ° Ğ·Ğ°Ğ´Ğ°Ñ‡Ğ° ĞºĞ»Ğ°Ñ�Ñ�Ğ¸Ñ„Ğ¸ĞºĞ°Ñ†Ğ¸Ğ¸.")
else:
    task_type = "regression"
    print("\nâš¡ï¸� Ğ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ° Ğ·Ğ°Ğ´Ğ°Ñ‡Ğ° Ñ€ĞµĞ³Ñ€ĞµÑ�Ñ�Ğ¸Ğ¸.")

# Ğ Ğ°Ğ·Ğ´ĞµĞ»Ñ�ĞµĞ¼ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸ Ğ¸ Ñ†ĞµĞ»ĞµĞ²ÑƒÑ� Ğ¿ĞµÑ€ĞµĞ¼ĞµĞ½Ğ½ÑƒÑ�
X = data.drop(columns=[target_column])
y = data[target_column]

# ĞŸÑ€Ğ¾Ğ²ĞµÑ€ĞºĞ° Ğ½Ğ° Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ‰ĞµĞ½Ğ½Ñ‹Ğµ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ñ�
print("\nĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ‰ĞµĞ½Ğ½Ñ‹Ñ… Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğ¹ Ğ² ĞºĞ°Ğ¶Ğ´Ğ¾Ğ¼ Ñ�Ñ‚Ğ¾Ğ»Ğ±Ñ†Ğµ:")
print(data.isna().sum())

# Ğ—Ğ°Ğ¿Ğ¾Ğ»Ğ½Ñ�ĞµĞ¼ Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ‰ĞµĞ½Ğ½Ñ‹Ğµ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ñ� Ğ¼ĞµĞ´Ğ¸Ğ°Ğ½Ğ¾Ğ¹ (Ğ´Ğ»Ñ� Ñ‡Ğ¸Ñ�Ğ»Ğ¾Ğ²Ñ‹Ñ… Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…)
X = X.select_dtypes(include=[np.number])
X.fillna(X.median(), inplace=True)
y.fillna(y.median() if task_type == "regression" else y.mode()[0], inplace=True)

# ĞŸÑ€Ğ¾Ğ²ĞµÑ€Ñ�ĞµĞ¼ Ğ±ĞµÑ�ĞºĞ¾Ğ½ĞµÑ‡Ğ½Ñ‹Ğµ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ñ� Ğ¸ Ğ·Ğ°Ğ¼ĞµĞ½Ñ�ĞµĞ¼ Ğ¸Ñ…
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

# Ğ’Ğ¸Ğ·ÑƒĞ°Ğ»Ğ¸Ğ·Ğ¸Ñ€ÑƒĞµĞ¼ Ñ€Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²
plt.figure(figsize=(10, 6))
sns.boxplot(data=X)
plt.xticks(rotation=90)
plt.title("Ğ Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ… (Ğ¿Ğ¾Ğ¸Ñ�Ğº Ğ²Ñ‹Ğ±Ñ€Ğ¾Ñ�Ğ¾Ğ²)")
plt.show()

# Ğ£Ğ´Ğ°Ğ»ĞµĞ½Ğ¸Ğµ Ğ²Ñ‹Ğ±Ñ€Ğ¾Ñ�Ğ¾Ğ² Ğ½Ğ° Ğ¾Ñ�Ğ½Ğ¾Ğ²Ğµ Ğ¼ĞµĞ¶ĞºĞ²Ğ°Ñ€Ñ‚Ğ¸Ğ»ÑŒĞ½Ğ¾Ğ³Ğ¾ Ñ€Ğ°Ğ·Ğ¼Ğ°Ñ…Ğ°
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
X, y = X[mask], y[mask]

# Ğ Ğ°Ğ·Ğ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ğ½Ğ° Ğ¾Ğ±ÑƒÑ‡Ğ°Ñ�Ñ‰ÑƒÑ� Ğ¸ Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²ÑƒÑ� Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞ¸
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ĞœĞ°Ñ�ÑˆÑ‚Ğ°Ğ±Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ğ’Ñ‹Ğ±Ğ¸Ñ€Ğ°ĞµĞ¼ Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ğ² Ğ·Ğ°Ğ²Ğ¸Ñ�Ğ¸Ğ¼Ğ¾Ñ�Ñ‚Ğ¸ Ğ¾Ñ‚ Ğ·Ğ°Ğ´Ğ°Ñ‡Ğ¸
if task_type == "regression":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nğŸ“Š Mean Squared Error: {mse:.2f}")
    print(f"ğŸ“ˆ RÂ² Score: {r2:.3f}")
else:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nâœ… Accuracy: {accuracy:.3f}")
    print("\nğŸ”� Classification Report:\n", classification_report(y_test, y_pred))

# ĞšÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ñ� (Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ Ğ´Ğ¾Ğ»Ğ¶Ğ½Ñ‹ Ğ±Ñ‹Ñ‚ÑŒ Ñ�Ñ‚Ğ°Ğ½Ğ´Ğ°Ñ€Ñ‚Ğ¸Ğ·Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ñ‹)
X_scaled = scaler.fit_transform(X)
scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2" if task_type == "regression" else "accuracy")

print("\nğŸ“Š Ğ¡Ñ€ĞµĞ´Ğ½Ñ�Ñ� Ğ¾Ñ†ĞµĞ½ĞºĞ° Ğ¿Ğ¾ ĞºÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¸:")
print(f"Ğ¡Ñ€ĞµĞ´Ğ½ĞµĞµ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğµ: {scores.mean():.3f}, Ğ¡Ñ‚Ğ°Ğ½Ğ´Ğ°Ñ€Ñ‚Ğ½Ğ¾Ğµ Ğ¾Ñ‚ĞºĞ»Ğ¾Ğ½ĞµĞ½Ğ¸Ğµ: {scores.std():.3f}")

# Ğ’Ñ‹Ğ²Ğ¾Ğ´Ñ‹
print("\nğŸ“¢ Ğ’Ñ‹Ğ²Ğ¾Ğ´Ñ‹:")
if task_type == "regression":
    print(f"ğŸ”¹ ĞœĞ¾Ğ´ĞµĞ»ÑŒ Ğ¿Ğ¾ĞºĞ°Ğ·Ğ°Ğ»Ğ° RÂ² = {r2:.3f}, Ñ‡Ñ‚Ğ¾ {'Ñ…Ğ¾Ñ€Ğ¾ÑˆĞ¾' if r2 > 0.7 else 'Ğ½ĞµÑƒĞ´Ğ¾Ğ²Ğ»ĞµÑ‚Ğ²Ğ¾Ñ€Ğ¸Ñ‚ĞµĞ»ÑŒĞ½Ğ¾'}.")
    print(f"ğŸ”¹ Ğ¡Ñ€ĞµĞ´Ğ½Ñ�Ñ� Ğ¾ÑˆĞ¸Ğ±ĞºĞ° MSE = {mse:.2f}, Ñ‡Ñ‚Ğ¾ {'Ğ½Ğ¸Ğ·ĞºĞ¾Ğµ' if mse < 10 else 'Ğ²Ñ‹Ñ�Ğ¾ĞºĞ¾Ğµ'} Ñ€Ğ°Ñ�Ñ…Ğ¾Ğ¶Ğ´ĞµĞ½Ğ¸Ğµ.")
else:
    print(f"ğŸ”¹ Ğ¢Ğ¾Ñ‡Ğ½Ğ¾Ñ�Ñ‚ÑŒ ĞºĞ»Ğ°Ñ�Ñ�Ğ¸Ñ„Ğ¸ĞºĞ°Ñ†Ğ¸Ğ¸ (Accuracy) = {accuracy:.3f}, Ñ‡Ñ‚Ğ¾ {'Ñ…Ğ¾Ñ€Ğ¾ÑˆĞ¸Ğ¹' if accuracy > 0.8 else 'Ñ�Ñ€ĞµĞ´Ğ½Ğ¸Ğ¹'} Ñ€ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚.")
    print(f"ğŸ”¹ Ğ¡Ñ€ĞµĞ´Ğ½Ñ�Ñ� Ğ¾Ñ†ĞµĞ½ĞºĞ° ĞºÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¸ = {scores.mean():.3f}, Ñ‡Ñ‚Ğ¾ ÑƒĞºĞ°Ğ·Ñ‹Ğ²Ğ°ĞµÑ‚ Ğ½Ğ° Ñ�Ñ‚Ğ°Ğ±Ğ¸Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚ÑŒ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸.")

print("ğŸ�¯ Ğ�Ğ¿Ñ‚Ğ¸Ğ¼Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ� Ğ²Ğ¾Ğ·Ğ¼Ğ¾Ğ¶Ğ½Ğ° Ğ¿ÑƒÑ‚ĞµĞ¼ Ğ¿Ğ¾Ğ´Ğ±Ğ¾Ñ€Ğ° Ğ³Ğ¸Ğ¿ĞµÑ€Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¾Ğ², Ğ²Ñ‹Ğ±Ğ¾Ñ€Ğ° Ğ´Ñ€ÑƒĞ³Ğ¾Ğ¹ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸ Ğ¸Ğ»Ğ¸ ÑƒĞ»ÑƒÑ‡ÑˆĞµĞ½Ğ¸Ñ� Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ….")


