import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv") # Load the training data
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")   # Load the test data
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv") # Load the sample submission file


train = train.dropna()


X, y = train.drop(columns=["Personality"]), train["Personality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, y_train.shape


le = LabelEncoder()
y_train = le.fit_transform(y_train)


cat_cols = X_train.select_dtypes(include="object").columns.tolist()
encoder = OrdinalEncoder()
X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])


X_train = X_train.iloc[:,1:]


X_train.head()


xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',  # Or 'multi:softmax' for multi-class, 'reg:squarederror' for regression
    n_estimators=100,             # Number of boosting rounds
    learning_rate=0.1,            # Step size shrinkage
    max_depth=3,                  # Maximum depth of a tree
    subsample=0.8,                # Subsample ratio of the training instance
    colsample_bytree=0.8,         # Subsample ratio of columns when constructing each tree
    use_label_encoder=False,      # Suppress warning for older versions
    eval_metric='logloss',        # Evaluation metric for cross-validation
    random_state=42               # For reproducibility
)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_val_score(xgb_model, X_train, y_train, cv=kf, scoring='accuracy', n_jobs=-1)

print(f"\nCross-validation Accuracy Scores: {cv_results}")
print(f"Mean CV Accuracy: {np.mean(cv_results):.4f}")
print(f"Standard Deviation of CV Accuracy: {np.std(cv_results):.4f}")


print("\nTraining final XGBoost model on the entire training data...")
xgb_model.fit(X_train, y_train)


y_test = le.transform(y_test)
X_test[cat_cols] = encoder.transform(X_test[cat_cols])

print("\nEvaluating the model on the test set...")
y_pred = xgb_model.predict(X_test.drop(columns=["id"]))

# Calculate various metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted') # Use 'weighted' for multi-class
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Test Set Accuracy: {accuracy:.4f}")
print(f"Test Set Precision: {precision:.4f}")
print(f"Test Set Recall: {recall:.4f}")
print(f"Test Set F1-Score: {f1:.4f}")


train["Drained_after_socializing"].mode()


cat_cols


test["Stage_fear"].fillna("No", inplace=True)
test["Drained_after_socializing"].fillna("No", inplace=True)

test[cat_cols] = encoder.transform(test[cat_cols])

predictions = xgb_model.predict(test.drop(columns=["id"]))


preds = le.inverse_transform(predictions)
submission = pd.DataFrame({
    "id": test["id"].values,
    "Personality": preds
})


submission.to_csv("submission.csv", index=False)




