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

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
import xgboost as xgb
import lightgbm as lgb
#from imblearn.over_sampling import SMOTE
#from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns


df_train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


df_train.head(3)


df_train = df_train.drop(['id'],axis = 1)
df_test = df_test.drop(['id'],axis =1 )


df_train.head(3)


# Numeric columns
numeric_df = df_train.select_dtypes(include=['int64', 'float64'])

# Categorical columns
categorical_df = df_train.select_dtypes(include=['object'])



from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
def encode_categorical(df):
    """
    Encodes categorical columns :
    - One-hot encodes nominal features
    - Ordinal encodes ordered features
    - Maps grade_subgrade to numeric order
    Returns: dataframe with numeric columns only
    """
    
    # Make a copy to avoid changing original data
    df = df.copy()
    
    # Define column groups
    nominal_cols = ['gender', 'marital_status', 'employment_status', 'loan_purpose']
    ordinal_cols = ['education_level']
    
    # 1ï¸�âƒ£ Ordinal encoding for education_level
    education_order = [['High School', "Bachelor's", "Master's", 'PhD', 'Other']]
    ord_enc = OrdinalEncoder(categories=education_order)
    df['education_level'] = ord_enc.fit_transform(df[['education_level']])
    
    # 2ï¸�âƒ£ Custom mapping for grade_subgrade
    grade_map = {g: i for i, g in enumerate(
        ['A1','A2','A3','A4','A5',
         'B1','B2','B3','B4','B5',
         'C1','C2','C3','C4','C5',
         'D1','D2','D3','D4','D5',
         'E1','E2','E3','E4','E5',
         'F1','F2','F3','F4','F5'], start=1)}
    df['grade_subgrade'] = df['grade_subgrade'].map(grade_map)
    
    # 3ï¸�âƒ£ One-hot encode nominal columns
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True, dtype=int)
    
    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric)
    
    
    return df


df_train_encoded = encode_categorical(df_train)


df_train_encoded.head(3)


df_test_encoded = encode_categorical(df_test)


df_test_encoded.head(3)


df_test.columns


from sklearn.preprocessing import StandardScaler

features = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']

scaler = StandardScaler()

# Fit and transform the features
df_train_encoded[features] = scaler.fit_transform(df_train_encoded[features])
df_test_encoded[features] = scaler.transform(df_test_encoded[features])


df_train_encoded.head(3)


X = df_train_encoded.drop('loan_paid_back', axis=1)
y = df_train_encoded['loan_paid_back']



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


class_counts = np.bincount(y_train)
total = len(y_train)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Scale pos weight for XGBoost: {scale_pos_weight:.2f}\n")


models = {}
results = {}

# 1. XGBoost with scale_pos_weight
models['XGBoost'] = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='auc'
)

# 2. LightGBM with balanced class weights
models['LightGBM'] = lgb.LGBMClassifier(
    class_weight='balanced',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

# 3. Random Forest with balanced class weights
models['RandomForest'] = RandomForestClassifier(
    class_weight='balanced',
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)




print("All models initialized!")


from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for model_name, model in models.items():
    print(f"\n{'='*70}")
    print(f"TRAINING {model_name.upper()}")
    print('='*70)
    
    # Lists to store results across folds
    importances_list = []
    acc_scores = []
    auc_scores = []
    
    print(f"\nCross-Validation on Training Set:")
    print("-" * 50)
    
    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Create a fresh model instance for this fold
        from sklearn.base import clone
        fold_model = clone(model)
        
        # Train model
        fold_model.fit(X_tr, y_tr)
        
        # Permutation importance
        result = permutation_importance(
            fold_model, X_val, y_val, 
            scoring="roc_auc", 
            n_repeats=5, 
            random_state=42,
            n_jobs=-1
        )
        fold_imp = pd.Series(result.importances_mean, index=X_train.columns)
        importances_list.append(fold_imp)
        
        # Predictions
        y_pred = fold_model.predict(X_val)
        y_pred_proba = fold_model.predict_proba(X_val)[:, 1]
        
        # Evaluation
        acc = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        acc_scores.append(acc)
        auc_scores.append(auc)
        
        print(f"Fold {fold+1}: Accuracy={acc:.4f}, AUC={auc:.4f}")
    
    # Print CV summary
    print(f"\n{'-'*50}")
    print(f"CV Results Summary:")
    print(f"Mean Accuracy: {np.mean(acc_scores):.4f} (+/- {np.std(acc_scores)*2:.4f})")
    print(f"Mean AUC: {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores)*2:.4f})")
    
    # Calculate average feature importance
    importances_df = pd.concat(importances_list, axis=1)
    avg_importance = importances_df.mean(axis=1).sort_values(ascending=False)
    
    print(f"\nTop 10 Features (Permutation Importance):")
    for i, (feat, imp) in enumerate(avg_importance.head(10).items(), 1):
        print(f"  {i}. {feat}: {imp:.4f}")
    
    # Train final model on full training set
    print(f"\n{'-'*50}")
    print("Training final model on full training set...")
    final_model = clone(model)
    final_model.fit(X_train, y_train)
    print("Final model trained successfully!")
    
    # Evaluate on test set
    print(f"\n{'-'*50}")
    print("Evaluation on Held-Out Test Set:")
    print(f"{'-'*50}")
    
    y_test_pred = final_model.predict(X_test)
    y_test_pred_proba = final_model.predict_proba(X_test)[:, 1]
    
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Store results
    results[model_name] = {
        'model': final_model,
        'cv_auc_mean': np.mean(auc_scores),
        'cv_auc_std': np.std(auc_scores),
        'cv_acc_mean': np.mean(acc_scores),
        'cv_acc_std': np.std(acc_scores),
        'test_auc': test_auc,
        'test_acc': test_acc,
        'y_pred': y_test_pred,
        'y_pred_proba': y_test_pred_proba,
        'feature_importance': avg_importance
    }


print(f"\n{'='*70}")
print("MODEL COMPARISON SUMMARY")
print('='*70)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'CV AUC (meanÂ±std)': [f"{results[m]['cv_auc_mean']:.4f}Â±{results[m]['cv_auc_std']:.4f}" 
                           for m in results.keys()],
    'Test AUC': [f"{results[m]['test_auc']:.4f}" for m in results.keys()],
    'Test Accuracy': [f"{results[m]['test_acc']:.4f}" for m in results.keys()]
})

print(comparison_df.to_string(index=False))

# Find best model
best_model_name = max(results.keys(), key=lambda x: results[x]['test_auc'])
print(f"\nğŸ�† Best Model (by Test AUC): {best_model_name}")
print(f"   Test AUC: {results[best_model_name]['test_auc']:.4f}")


df_test_encoded.head(3)


submission_df = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


best_model_name = max(results.keys(), key=lambda x: results[x]['test_auc'])
print(f"Best Model: {best_model_name}")


best_model = results[best_model_name]['model']


y_pred_ensemble_test = best_model.predict(df_test_encoded)
y_pred_proba_ensemble_test = best_model.predict_proba(df_test_encoded)[:, 1]


prediction_results = pd.DataFrame({
    'id':  submission_df.id ,
    'loan_paid_back': y_pred_proba_ensemble_test
})
prediction_results.to_csv('submission.csv', index=False)

