import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train =  pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test =  pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


train.head(5)



train.tail(5)


train.shape
train.info()



train.describe()


train.isnull().sum()


train['diagnosed_diabetes'].value_counts()



test.head(2)


#train.drop('id', axis=1, inplace=True)
#test.drop('id', axis=1, inplace=True)



X = train.drop('diagnosed_diabetes', axis=1)
y = train['diagnosed_diabetes']



cat_cols = [
    'gender','ethnicity' ,	'education_level',	'income_level',	'smoking_status',	'employment_status'
  
]



combined = pd.concat([X, test], axis=0)

combined_encoded = pd.get_dummies(
    combined,
    columns=cat_cols,
    drop_first=True
)



#X = pd.get_dummies(X, drop_first=True)
#X_test = pd.get_dummies(X_test, drop_first=True)

# Align columns (IMPORTANT)
#X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)



X_encoded = combined_encoded.iloc[:len(X)]
test_encoded = combined_encoded.iloc[len(X):]



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




X_train, X_val, y_train, y_val = train_test_split(
    X_encoded, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_scaled = scaler.transform(test_encoded)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

log_preds = log_model.predict(X_val_scaled)
print("Logistic Accuracy:", accuracy_score(y_val, log_preds))


from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train, y_train)

gb_preds = gb_model.predict(X_val)

from sklearn.metrics import accuracy_score
print("Gradient Boosting Accuracy:", accuracy_score(y_val, gb_preds))



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)



rf_preds = rf_model.predict(X_val)

print("Random Forest Accuracy:", accuracy_score(y_val, rf_preds))
#print(classification_report(y_val, rf_preds))


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




model_comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Gradient Boosting", "Random Forest"],
    
    "Accuracy": [
        accuracy_score(y_val, log_preds),
        accuracy_score(y_val, gb_preds),
        accuracy_score(y_val, rf_preds)
    ],
    
    "Precision": [
        precision_score(y_val, log_preds),
        precision_score(y_val, gb_preds),
        precision_score(y_val, rf_preds)
    ],
    
    "Recall": [
        recall_score(y_val, log_preds),
        recall_score(y_val, gb_preds),
        recall_score(y_val, rf_preds)
    ],
    
    "F1 Score": [
        f1_score(y_val, log_preds),
        f1_score(y_val, gb_preds),
        f1_score(y_val, rf_preds)
    ]
})

model_comparison





models = {
    "Logistic Regression": log_preds,
    "Gradient Boosting": gb_preds,
    "Random Forest": rf_preds
}

rows = []

for model, preds in models.items():
    rows.append({
        "Model": model,
        "Accuracy": accuracy_score(y_val, preds),
        "Precision": precision_score(y_val, preds),
        "Recall": recall_score(y_val, preds),
        "F1": f1_score(y_val, preds)
    })

df = pd.DataFrame(rows)

# ðŸ‘‡ FINAL DECISION (Accuracy-based)
best_model = df.sort_values(by="Accuracy", ascending=False).iloc[0]

df, best_model["Model"]



best_model = gb_model
best_model.fit(X_encoded, y)



X_test = test.drop(columns=["id"])
X_test = pd.get_dummies(X_test, drop_first=True)

# Align test columns with training columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)



test_predictions = best_model.predict(X_test)
test_predictions



submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": test_predictions
})
submission.to_csv("submission.csv", index=False)
print("Submission Saved âœ…")


