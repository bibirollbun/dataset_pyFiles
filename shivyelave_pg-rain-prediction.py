# Basic Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import ADASYN

# Models
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

# Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Hyperparameter Tuning
import optuna

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")



# Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')



# Check Data
print(train.head())
print(train.info())
print("Train Shape:", train.shape, "Test Shape:", test.shape)


train.describe()


# Get unique values for each column
unique_values = {col: train[col].unique() for col in train.columns}

# Print unique values for each column
for col, values in unique_values.items():
    print(f"{col}: {values}")


# Define the number of days in each month (for a non-leap year)
month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Create 'month' and 'day_of_month' columns
train['month'] = 0
train['day_of_month'] = 0

day_count = 0
for month, days in enumerate(month_days, start=1):
    mask = (train['day'] > day_count) & (train['day'] <= day_count + days)
    train.loc[mask, 'month'] = month
    train.loc[mask, 'day_of_month'] = train.loc[mask, 'day'] - day_count
    day_count += days

# Display the transformed DataFrame
print(train[['day', 'month', 'day_of_month']])


# Define the number of days in each month (for a non-leap year)
month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Create 'month' and 'day_of_month' columns
test['month'] = 0
test['day_of_month'] = 0

day_count = 0
for month, days in enumerate(month_days, start=1):
    mask = (test['day'] > day_count) & (test['day'] <= day_count + days)
    test.loc[mask, 'month'] = month
    test.loc[mask, 'day_of_month'] = test.loc[mask, 'day'] - day_count
    day_count += days

# Display the transformed DataFrame
print(test[['day', 'month', 'day_of_month']])


# Check Missing Values
print(train.isnull().sum())

# Check Class Distribution
print(train["rainfall"].value_counts())

print(test.isnull().sum())




plt.figure(figsize=(10, 5))

# Histogram (Bar Graph)
sns.histplot(test['winddirection'], bins=30, kde=True, color='skyblue', edgecolor='black')

# Line Graph (KDE only)
sns.kdeplot(test['winddirection'], color='red', linewidth=2)

# Labels
plt.xlabel('Wind Direction')
plt.ylabel('Frequency')
plt.title('Wind Direction Distribution')

plt.show()


test = test.fillna(test.mode().iloc[0])
print(test.isnull().sum())


rainfall_counts = train['rainfall'].value_counts()
print(rainfall_counts)

# Plot the distribution
plt.figure(figsize=(6, 4))
sns.barplot(x=rainfall_counts.index, y=rainfall_counts.values)
plt.title("Rainfall Class Distribution")
plt.xlabel("Rainfall (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()


# Boxplots for numerical features
plt.figure(figsize=(12, 8))
for i, col in enumerate(train.columns[1:]):
    plt.subplot(4, 4, i + 1)
    sns.boxplot(y=train[col])
    plt.title(col)
plt.tight_layout()
plt.show()



# Feature Creation
train["temp_diff"] = train["maxtemp"] - train["mintemp"]
train["humidity_ratio"] = train["dewpoint"] / train["temparature"]
train["wind_effect"] = train["windspeed"] * train["cloud"]

test["temp_diff"] = test["maxtemp"] - test["mintemp"]
test["humidity_ratio"] = test["dewpoint"] / test["temparature"]
test["wind_effect"] = test["windspeed"] * test["cloud"]

train["wind_humidity_effect"] = train["windspeed"] * train["humidity"]
test["wind_humidity_effect"] = test["windspeed"] * test["humidity"]

train["temp_drop"] = (train["temp_diff"] < 5).astype(int)  # If temp drop > 5 degrees, mark as 1
test["temp_drop"] = (test["temp_diff"] < 5).astype(int)  # If temp drop > 5 degrees, mark as 1





corr_matrix = train.corr()

# Step 3: Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))  # Set the size of the figure
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Display the plot
plt.title('Correlation Matrix')
plt.show()


# Compute correlation of all columns with Rainfall
rainfall_corr = train.corr()['rainfall'].sort_values(ascending=False)

# Display correlation values
print(rainfall_corr)


train


test


# Drop ID Column
train.drop(columns=["id","day","maxtemp","mintemp"], inplace=True)
test_ids = test["id"]
test.drop(columns=["id","day","maxtemp","mintemp"], inplace=True)


# Compute correlation of all columns with Rainfall
rainfall_corr = train.corr()['rainfall'].sort_values(ascending=False)

# Display correlation values
print(rainfall_corr)




# Group by month and count rainy days (assuming 'rain' is binary: 1 = Rain, 0 = No Rain)
rain_count_by_month = train.groupby("month")["rainfall"].sum()

# Find the month with the highest rain count
max_rain_month = rain_count_by_month.idxmax()
max_rain_count = rain_count_by_month.max()

# Print results
print("Rainy Days Count per Month:\n", rain_count_by_month)
print(f"ğŸŒ§ï¸� Month with Most Rainy Days: {max_rain_month} ({max_rain_count} days)")

# Plot the distribution
import matplotlib.pyplot as plt
rain_count_by_month.plot(kind="bar", color="blue", figsize=(8, 5))
plt.xlabel("Month")
plt.ylabel("Number of Rainy Days")
plt.title("Rainy Days Count per Month")
plt.xticks(rotation=0)
plt.show()




# Define Features & Target
X = train.drop(columns=["rainfall"])
y = train["rainfall"]



X


test


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Apply ADASYN
adasyn = ADASYN(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)

# Check New Class Distribution
print(pd.Series(y_train_resampled).value_counts())



scaler = RobustScaler()

# Fit & Transform
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)
test_scaled = scaler.transform(test)



rf_params = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}

rf = RandomForestClassifier(random_state=42)

rf_random = RandomizedSearchCV(rf, rf_params, n_iter=20, cv=5, scoring="accuracy", random_state=42, n_jobs=-1)
rf_random.fit(X_train_resampled, y_train_resampled)
rf_best = rf_random.best_estimator_
print(rf_best)



xgb_params = {
    "n_estimators": [100, 300, 500, 700],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 9],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}

xgb = XGBClassifier(random_state=42)

xgb_random = RandomizedSearchCV(xgb, xgb_params, n_iter=20, cv=5, scoring="accuracy", random_state=42, n_jobs=-1)
xgb_random.fit(X_train_resampled, y_train_resampled)
xgb_best = xgb_random.best_estimator_
print(xgb_best)



def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1, 10)
    }

    model = CatBoostClassifier(**params, verbose=0, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

catboost_best = CatBoostClassifier(**study.best_params, verbose=0, random_state=42)
catboost_best.fit(X_train_resampled, y_train_resampled)



base_models = [
    ("RandomForest", rf_best),
    ("XGBoost", xgb_best),
    ("CatBoost", catboost_best)
]

from sklearn.svm import SVC
meta_model = RandomForestClassifier(n_estimators=100)

stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
stacking_model.fit(X_train_resampled, y_train_resampled)

y_pred_stack = stacking_model.predict(X_test)
y_pred_proba_stack = stacking_model.predict_proba(X_test)[:, 1]  # Probabilities for ROC AUC




from sklearn.metrics import roc_auc_score

# Get predictions and probabilities for each base model
y_pred_rf = rf_best.predict(X_test)
y_pred_xgb = xgb_best.predict(X_test)
y_pred_cat = catboost_best.predict(X_test)

y_proba_rf = rf_best.predict_proba(X_test)[:, 1]
y_proba_xgb = xgb_best.predict_proba(X_test)[:, 1]
y_proba_cat = catboost_best.predict_proba(X_test)[:, 1]

# Compute ROC AUC for each model
roc_rf = roc_auc_score(y_test, y_proba_rf)
roc_xgb = roc_auc_score(y_test, y_proba_xgb)
roc_cat = roc_auc_score(y_test, y_proba_cat)
roc_stack = roc_auc_score(y_test, y_pred_proba_stack)  # Stacking model

# Print results
print(f"ROC AUC Scores:")
print(f"ğŸ”¹ Random Forest: {roc_rf:.4f}")
print(f"ğŸ”¹ XGBoost: {roc_xgb:.4f}")
print(f"ğŸ”¹ CatBoost: {roc_cat:.4f}")
print(f"â­� Stacking Model: {roc_stack:.4f} (Final Model)")




# **Evaluation Metrics**
print("Stacking Model Accuracy:", accuracy_score(y_test, y_pred_stack))
print("Precision:", precision_score(y_test, y_pred_stack))
print("Recall:", recall_score(y_test, y_pred_stack))
print("F1 Score:", f1_score(y_test, y_pred_stack))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba_stack))
print("\nClassification Report:\n", classification_report(y_test, y_pred_stack))


final_predictions = stacking_model.predict_proba(test_scaled)[:,1]

submission = pd.DataFrame({"id": test_ids, "rainfall":final_predictions})
submission.to_csv("submission.csv", index=False)

print("Kaggle Submission File Created! ğŸš€")



submission.shape


submission

