!pip install tabpfn


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from tabpfn import TabPFNClassifier, TabPFNRegressor


# Set random seed for reproducibility
np.random.seed(42)

basepath = '/kaggle/input/playground-series-s5e3'
train_df = pd.read_csv(basepath + "/train.csv")
test_df = pd.read_csv(basepath + "/test.csv")


# Check if 'rainfall' is binary or continuous
rainfall_values = train_df['rainfall'].unique()
is_binary = len(rainfall_values) == 2 and set(rainfall_values) == {0, 1}

print(f"Rainfall is {'binary' if is_binary else 'continuous'} with values: {rainfall_values}")


# Define a function for preprocessing
def preprocess_data(df, is_train=True):
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Drop ID column as it's not useful for modeling
    if 'id' in df_copy.columns:
        df_copy.drop('id', axis=1, inplace=True)
    
    # Handle the target variable
    if is_train:
        y = df_copy['rainfall']
        X = df_copy.drop('rainfall', axis=1)
    else:
        y = None
        X = df_copy
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    return X_imputed, y

# Preprocess the training data
X_train, y_train = preprocess_data(train_df, is_train=True)


# Create a validation split for model evaluation
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_final), columns=X_train_final.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)


if is_binary:
    print("Building a TabPFN Classification model")
    model = TabPFNClassifier(device='cuda')
    model.fit(X_train_scaled, y_train_final)
    
    # Make predictions on validation set
    val_probs = model.predict_proba(X_val_scaled)
    val_preds = model.predict(X_val_scaled)
    
    # Evaluate the model
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    
    accuracy = accuracy_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds)
    precision = precision_score(y_val, val_preds)
    recall = recall_score(y_val, val_preds)
    roc_auc = roc_auc_score(y_val, val_probs[:, 1])
    
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation ROC AUC: {roc_auc:.4f}")
    
else:
    print("Building a TabPFN Regression model")
    try:
        model = TabPFNRegressor(device='cuda')
        model.fit(X_train_scaled, y_train_final)
        
        # Make predictions on validation set
        val_preds = model.predict(X_val_scaled)
        
        # Evaluate the model
        mse = mean_squared_error(y_val, val_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, val_preds)
        r2 = r2_score(y_val, val_preds)
        
        print(f"Validation MSE: {mse:.4f}")
        print(f"Validation RMSE: {rmse:.4f}")
        print(f"Validation MAE: {mae:.4f}")
        print(f"Validation R²: {r2:.4f}")
        
    except (AttributeError, NameError):
        print("TabPFNRegressor not available, using classification approach by binning target.")
        
        y_train_binned = (y_train_final > 0).astype(int)
        y_val_binned = (y_val > 0).astype(int)
        
        model = TabPFNClassifier(device='cuda')
        model.fit(X_train_scaled, y_train_binned)
        
        val_probs = model.predict_proba(X_val_scaled)
        val_preds = model.predict(X_val_scaled)
        

        val_preds_continuous = val_probs[:, 1]  # Using the probability of rain as a rough estimate
        
        accuracy = accuracy_score(y_val_binned, val_preds)
        f1 = f1_score(y_val_binned, val_preds)
        roc_auc = roc_auc_score(y_val_binned, val_probs[:, 1])
        
        print(f"Binary Classification Metrics:")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation F1 Score: {f1:.4f}")
        print(f"Validation ROC AUC: {roc_auc:.4f}")
        
        # Also evaluate how well our continuous conversion works
        mse = mean_squared_error(y_val, val_preds_continuous)
        rmse = np.sqrt(mse)
        
        print(f"\nContinuous Prediction Metrics (approximate):")
        print(f"Validation MSE: {mse:.4f}")
        print(f"Validation RMSE: {rmse:.4f}")

try:
    # Calculate feature importance using permutation importance
    perm_importance = permutation_importance(model, X_val_scaled, y_val, n_repeats=10, random_state=42)
    
    feature_importance = pd.DataFrame({
        'Feature': X_val_scaled.columns,
        'Importance': perm_importance.importances_mean
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance (Permutation Importance)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
except:
    print("\nFeature importance calculation not supported for this model.")

print("\nPerforming 5-Fold Cross-Validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []
fold = 1

for train_idx, val_idx in kf.split(X_train):
    # Split data
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Scale features
    X_train_fold_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_fold), 
        columns=X_train_fold.columns
    )
    X_val_fold_scaled = pd.DataFrame(
        scaler.transform(X_val_fold), 
        columns=X_val_fold.columns
    )
    
    # Train and evaluate model
    if is_binary:
        model_fold = TabPFNClassifier(device='cuda')
        model_fold.fit(X_train_fold_scaled, y_train_fold)
        val_preds = model_fold.predict(X_val_fold_scaled)
        score = accuracy_score(y_val_fold, val_preds)
        metric_name = "Accuracy"
    else:
        try:
            model_fold = TabPFNRegressor(device='cuda')
            model_fold.fit(X_train_fold_scaled, y_train_fold)
            val_preds = model_fold.predict(X_val_fold_scaled)
            score = r2_score(y_val_fold, val_preds)
            metric_name = "R²"
        except:
            # Fall back to classification approach
            y_train_fold_binned = (y_train_fold > 0).astype(int)
            y_val_fold_binned = (y_val_fold > 0).astype(int)
            
            model_fold = TabPFNClassifier(device='cuda')
            model_fold.fit(X_train_fold_scaled, y_train_fold_binned)
            val_preds = model_fold.predict(X_val_fold_scaled)
            score = accuracy_score(y_val_fold_binned, val_preds)
            metric_name = "Accuracy (binary)"
    
    cv_scores.append(score)
    print(f"Fold {fold}: {metric_name} = {score:.4f}")
    fold += 1

print(f"\nCross-Validation {metric_name}: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")


X_train_full_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

if is_binary:
    final_model = TabPFNClassifier(device='cuda')  
    final_model.fit(X_train_full_scaled, y_train)
else:
    try:
        final_model = TabPFNRegressor(device='cuda')
        final_model.fit(X_train_full_scaled, y_train)
    except:
        # Fall back to classification approach
        y_train_binned = (y_train > 0).astype(int)
        final_model = TabPFNClassifier(device='cuda')
        final_model.fit(X_train_full_scaled, y_train_binned)

# Process test data
X_test, _ = preprocess_data(test_df, is_train=False)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Make predictions on test data
if is_binary:
    test_preds = final_model.predict(X_test_scaled)
else:
    try:
        test_preds = final_model.predict(X_test_scaled)
    except:
        # Fall back to classification approach for prediction probability
        test_probs = final_model.predict_proba(X_test_scaled)
        test_preds = test_probs[:, 1]  # Use probability as continuous prediction

# 6. Prepare submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'rainfall': test_preds
})

submission.to_csv('tabpfn_submission.csv', index=False)
print("Submission file created: tabpfn_submission.csv")


plt.figure(figsize=(12, 6))

# Plot actual vs predicted on validation set
if is_binary:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_val, val_preds)
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.subplot(1, 2, 2)
    sns.histplot(val_probs[:, 1], bins=20)
    plt.title('Prediction Probability Distribution')
    plt.xlabel('Probability of Rainfall')
else:
    try:
        plt.subplot(1, 2, 1)
        plt.scatter(y_val, val_preds, alpha=0.5)
        plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--')
        plt.title('Actual vs Predicted')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        
        plt.subplot(1, 2, 2)
        plt.hist(val_preds, bins=20, alpha=0.5, label='Predicted')
        plt.hist(y_val, bins=20, alpha=0.5, label='Actual')
        plt.title('Distribution of Actual vs Predicted Values')
        plt.legend()
    except:
        # For binary fallback approach
        plt.subplot(1, 2, 1)
        sns.heatmap(confusion_matrix(y_val_binned, val_preds), annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Binary)')
        
        plt.subplot(1, 2, 2)
        sns.histplot(val_preds_continuous, bins=20)
        plt.title('Predicted Probability Distribution')

plt.tight_layout()
plt.savefig('prediction_analysis.png')
plt.show()

print("Model training and evaluation complete!")




