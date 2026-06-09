import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report


#loading data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


# Drop 'id' column
train_df.drop(columns=["id"], inplace=True)
test_ids = test_df["id"]
test_df.drop(columns=["id"], inplace=True)


#check missing values
print(train_df.isnull().sum())
print(test_df.isnull().sum())

#filling missing values in numeric columns with median 
numeric_cols = [
    "Time_spent_Alone",
    "Social_event_attendance",
    "Going_outside",
    "Friends_circle_size",
    "Post_frequency"
]

for col in numeric_cols:
    train_df[col].fillna(train_df[col].median(), inplace=True)
    test_df[col].fillna(test_df[col].median(), inplace=True)

#filling missing values in categorical columns with mode
cat_cols = [
    "Stage_fear",
    "Drained_after_socializing"
]

for col in cat_cols:
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)
    test_df[col].fillna(test_df[col].mode()[0], inplace=True)


# Encode categorical columns
le_dict = {}
for col in cat_cols + ["Personality"]:
    le = LabelEncoder()
    if col == "Personality":
        train_df[col] = le.fit_transform(train_df[col])
        target_le = le  # save for inverse_transform later
    else:
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
    le_dict[col] = le


# Define Features & Target

X = train_df.drop(columns=["Personality"])
y = train_df["Personality"]

# Set Up RandomizedSearchCV

xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

#  Fit the Model
random_search.fit(X, y)
best_model = random_search.best_estimator_

print("Best Hyperparameters:")
print(random_search.best_params_)


# Predict on Test Data

test_preds = best_model.predict(test_df)
test_preds_labels = target_le.inverse_transform(test_preds)



from sklearn.metrics import accuracy_score

# Accuracy from cross-validation (mean of all folds)
print(f"\nCross-Validated Accuracy (Best): {random_search.best_score_:.4f}")



#saving output to csv file

output = pd.DataFrame({
    "id": test_ids,
    "Personality": test_preds_labels
})
output.to_csv("predictions.csv", index=False)



