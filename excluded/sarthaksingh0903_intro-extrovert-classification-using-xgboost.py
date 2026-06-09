import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
from datetime import datetime
import matplotlib.pyplot as plt


# Load the datasets
training_data = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
testing_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

print("Datasets loaded successfully.")



# Encode the target variable
target_encoder = LabelEncoder()
training_data["Personality_encoded"] = target_encoder.fit_transform(training_data["Personality"])

# Show class distribution
print("Target Class Distribution:")
print(training_data["Personality"].value_counts())




# Separate features and target
feature_set = training_data.drop(columns=["id", "Personality", "Personality_encoded"])
target_variable = training_data["Personality_encoded"]
final_test_features = testing_data.drop(columns=["id"])



# Combine train + test for consistent encoding
full_data = pd.concat([feature_set, final_test_features], axis=0)
categorical_features = full_data.select_dtypes(include="object").columns.tolist()

# Apply ordinal encoding to categorical columns
feature_encoder = OrdinalEncoder()
full_data[categorical_features] = feature_encoder.fit_transform(full_data[categorical_features])

# Split back
feature_set = full_data.iloc[:len(feature_set)].reset_index(drop=True)
final_test_features = full_data.iloc[len(feature_set):].reset_index(drop=True)

print("Feature encoding completed.")



# Detect classification type (binary vs multi-class)
num_classes = len(np.unique(target_variable))

# Set default binary classification params
model_hyperparams = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'lambda': 0.015,
    'alpha': 0.025,
    'max_depth': 5,
    'eta': 0.02,
    'gamma': 0.15,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 1,
    'seed': 42,
}

# Switch to multi-class if needed
if num_classes > 2:
    model_hyperparams["objective"] = "multi:softprob"
    model_hyperparams["eval_metric"] = "mlogloss"
    model_hyperparams["num_class"] = num_classes

print("Model hyperparameters configured.")



# Initialize CV and arrays
cross_validator = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
out_of_fold_predictions = np.zeros((len(feature_set), num_classes if num_classes > 2 else 1))
final_test_predictions = np.zeros((len(final_test_features), num_classes if num_classes > 2 else 1))



# Train model using Stratified K-Fold CV
for fold, (train_idx, val_idx) in enumerate(cross_validator.split(feature_set, target_variable)):
    X_train, X_val = feature_set.iloc[train_idx], feature_set.iloc[val_idx]
    y_train, y_val = target_variable.iloc[train_idx], target_variable.iloc[val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(final_test_features)

    model = xgb.train(
        params=model_hyperparams,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dval, "valid")],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    # Predict
    val_preds = model.predict(dval)
    test_preds = model.predict(dtest)

    # Store predictions
    if num_classes > 2:
        val_preds_class = np.argmax(val_preds, axis=1)
        out_of_fold_predictions[val_idx] = val_preds
        final_test_predictions += test_preds / cross_validator.n_splits
    else:
        val_preds_class = (val_preds > 0.5).astype(int)
        out_of_fold_predictions[val_idx] = val_preds.reshape(-1, 1)
        final_test_predictions += test_preds.reshape(-1, 1) / cross_validator.n_splits

    # Fold accuracy
    fold_acc = accuracy_score(y_val, val_preds_class)
    print(f"Fold {fold + 1} Accuracy: {fold_acc:.4f}")



# Overall CV score
if num_classes > 2:
    final_preds = np.argmax(out_of_fold_predictions, axis=1)
else:
    final_preds = (out_of_fold_predictions > 0.5).astype(int).flatten()

score = accuracy_score(target_variable, final_preds)
print(f"\nOverall CV Accuracy: {score:.4f}")

# Confusion matrix
cm = confusion_matrix(target_variable, final_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()



# Plot feature importance from final model
xgb.plot_importance(model, max_num_features=15, importance_type='gain')
plt.title("Top 15 Important Features")
plt.show()



# Final predictions
if num_classes > 2:
    final_submission_preds = np.argmax(final_test_predictions, axis=1)
else:
    final_submission_preds = (final_test_predictions > 0.5).astype(int).flatten()

# Decode predictions back to original labels
submission_df["Personality"] = target_encoder.inverse_transform(final_submission_preds)

# âœ… Save as submission.csv for Kaggle
submission_df.to_csv("submission.csv", index=False)

# Show accuracy and preview
print(f"\nâœ… Submission file created: submission.csv")
print(f"ğŸ“Š Final Cross-Validation Accuracy: {score:.4f}\n")
print("ğŸ”� Preview of submission:")
display(submission_df.head())

