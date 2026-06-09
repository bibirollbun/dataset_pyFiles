import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


train.columns # train data columns


train.head() # first 5 row of data


train.describe() # train data probabilistic values of columns


from sklearn.preprocessing import LabelEncoder

train_knn = train.copy(deep=True)
test_knn = test.copy(deep=True)


binary_cols = ["Stage_fear", "Drained_after_socializing"]

le = LabelEncoder()
for col in binary_cols:
    train_knn[col] = le.fit_transform(train_knn[col])
    test_knn[col] = le.transform(test_knn[col])

train_knn["Personality"] = train_knn["Personality"].map({
    "Introvert": 0,
    "Extrovert": 1
})


train_knn["Drained_after_socializing"].value_counts()


train_knn["Stage_fear"].value_counts()


train_knn.loc[train_knn["Drained_after_socializing"] == 2, "Drained_after_socializing"] = 1
test_knn.loc[test_knn["Drained_after_socializing"] == 2, "Drained_after_socializing"] = 1

train_knn.loc[train_knn["Stage_fear"] == 2, "Stage_fear"] = 1
test_knn.loc[test_knn["Stage_fear"] == 2, "Stage_fear"] = 1


train_knn["Drained_after_socializing"].value_counts()


train_knn["Stage_fear"].value_counts()


train_knn.isnull().sum()


#train_knn.dropna(inplace = True)
#test_knn.dropna(inplace = True)



train_knn.interpolate(limit_direction="both",inplace=True)
test_knn.interpolate(limit_direction="both",inplace=True)



train_knn.isnull().sum()


# new feature
#train_knn["Social_to_Alone_Ratio"] = train_knn["Social_event_attendance"] / (train_knn["Time_spent_Alone"] +1)
#test_knn["Social_to_Alone_Ratio"] = test_knn["Social_event_attendance"] / (test_knn["Time_spent_Alone"] +1)


y = train["Personality"]
E,I = y.value_counts()
sns.countplot(x = y, palette = "pastel")  # "Introvert": 0,   "Extrovert": 1
print("Extrovert: ",E)
print("Introvert: ",I)


y.value_counts()


X = train_knn.drop(columns=["id", "Personality"])
X["Personality"] = train_knn["Personality"]
df_melted = X.melt(id_vars="Personality", var_name="Feature", value_name="value")

sns.violinplot(x="Feature", y="value", hue="Personality", data=df_melted, split=True, palette="pastel")
plt.title("Features Introvert / Extrovert Distribution Introvert: 0,   Extrovert : 1")
plt.xticks(rotation=45)
plt.tight_layout() 
plt.show()


sns.boxplot(x="Feature", y="value", hue="Personality", data=df_melted)
plt.xticks(rotation=45)
plt.title("Features Introvert / Extrovert Distribution Introvert: 0,   Extrovert : 1")
plt.tight_layout()
plt.show()


# correlation map
plt.figure(figsize=(12, 8))
sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": 0.75})
plt.title("Correlation Matrix Chart")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Features and target variable
X = train_knn.drop(columns=["id", "Personality"])
y = train_knn["Personality"]

# Training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)

# Random Forest Classifier Model
#rf_model = RandomForestClassifier(random_state=42) #0.9763
#rf_model = RandomForestClassifier(max_depth=2, random_state=0) #0.9767
rf_model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42 )  # 0.9776

rf_model.fit(X_train, y_train)

# Prediction
y_pred = rf_model.predict(X_val)

# Evaluation
accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("\nClassification Report:\n", classification_report(y_val, y_pred))


X_test = test_knn.drop(columns=["id"])

# Predict
test_preds = rf_model.predict(X_test)

# Predict labels
pred_labels = ["Introvert" if p == 0 else "Extrovert" for p in test_preds]

# Submission
submission = pd.DataFrame({
    "id": test_knn["id"],
    "Personality": pred_labels
})

# to CSV
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv ready.")  # Score: 0.974898


from xgboost import XGBClassifier
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold


# Objective function: F1-score maximize
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'use_label_encoder' :'False',
        'eval_metric':'logloss',
        'random_state' :42
        
    }

    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1 = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1)
    return np.mean(f1)


# Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best parameter
print("Best F1 Score:", study.best_value)
print("Best Parameters:", study.best_params)


# Train
X = train_knn.drop(columns=["id", "Personality"])
y = train_knn["Personality"]
X_test = test_knn.drop(columns=["id"])

# Training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Optuna best parameters
best_params = {
    'n_estimators':259,
    'learning_rate':0.04965988176295548,
    'max_depth':3,
    'subsample':0.5913587203902693,
    'colsample_bytree':0.6829263432170996,
    'reg_alpha':4.411680773003437,
    'reg_lambda':3.5359893795529533,
    'use_label_encoder':'False',
    'eval_metric':'logloss',
    'random_state':42
}

# Model
final_model = XGBClassifier(**best_params)
final_model.fit(X, y)

# Predict
test_preds = final_model.predict(X_test)

# Predict labels
pred_labels = ["Introvert" if p == 0 else "Extrovert" for p in test_preds]

# Submission
submission = pd.DataFrame({
    "id": test_knn["id"],
    "Personality": pred_labels
})

# to CSV
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv ready.")




