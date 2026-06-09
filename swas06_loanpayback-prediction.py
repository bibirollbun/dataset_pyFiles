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
    
    # 1️⃣ Ordinal encoding for education_level
    education_order = [['High School', "Bachelor's", "Master's", 'PhD', 'Other']]
    ord_enc = OrdinalEncoder(categories=education_order)
    df['education_level'] = ord_enc.fit_transform(df[['education_level']])
    
    # 2️⃣ Custom mapping for grade_subgrade
    grade_map = {g: i for i, g in enumerate(
        ['A1','A2','A3','A4','A5',
         'B1','B2','B3','B4','B5',
         'C1','C2','C3','C4','C5',
         'D1','D2','D3','D4','D5',
         'E1','E2','E3','E4','E5',
         'F1','F2','F3','F4','F5'], start=1)}
    df['grade_subgrade'] = df['grade_subgrade'].map(grade_map)
    
    # 3️⃣ One-hot encode nominal columns
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
df_test_encoded[features] =scaler.fit_transform(df_test[features])


df_test_encoded.head(3)


df_test_encoded[features] =scaler.fit_transform(df_test_encoded[features])


X = df_train_encoded.drop('loan_paid_back', axis=1)
y = df_train_encoded['loan_paid_back']



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


class_counts = np.bincount(y_train)
total = len(y_train)
scale_pos_weight = class_counts[0] / class_counts[1]
print(f"Scale pos weight for XGBoost: {scale_pos_weight:.2f}")


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


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"TRAINING {name.upper()}")
    print('='*50)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Train on full training set
    model.fit(X_train, y_train)
    print(f"Model {name} trained successfully!")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    test_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Test AUC: {test_auc:.4f}")
    
    # Store results
    results[name] = {
        'model': model,
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'test_auc': test_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


print(f"\n{'='*50}")
print("CREATING ENSEMBLE MODEL")
print('='*50)

# Get top 3 models based on CV AUC
sorted_models = sorted(results.items(), key=lambda x: x[1]['cv_auc_mean'], reverse=True)[:3]
ensemble_models = [(name, results[name]['model']) for name, _ in sorted_models]

print(f"Ensemble will use: {[name for name, _ in ensemble_models]}")

# Create voting ensemble
ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')

# Train ensemble
cv_scores_ensemble = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"Ensemble CV AUC: {cv_scores_ensemble.mean():.4f} (+/- {cv_scores_ensemble.std()*2:.4f})")

ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_test)
y_pred_proba_ensemble = ensemble.predict_proba(X_test)[:, 1]
test_auc_ensemble = roc_auc_score(y_test, y_pred_proba_ensemble)

# Store ensemble results
results['Ensemble'] = {
    'model': ensemble,
    'cv_auc_mean': cv_scores_ensemble.mean(),
    'cv_auc_std': cv_scores_ensemble.std(),
    'test_auc': test_auc_ensemble,
    'y_pred': y_pred_ensemble,
    'y_pred_proba': y_pred_proba_ensemble
}

print(f"Ensemble Test AUC: {test_auc_ensemble:.4f}")
print("\nEnsemble Classification Report:")
print(classification_report(y_test, y_pred_ensemble))


print(f"\n{'='*60}")
print("FINAL MODEL COMPARISON")
print('='*60)

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'CV_AUC_Mean': [results[model]['cv_auc_mean'] for model in results.keys()],
    'CV_AUC_Std': [results[model]['cv_auc_std'] for model in results.keys()],
    'Test_AUC': [results[model]['test_auc'] for model in results.keys()]
}).sort_values('Test_AUC', ascending=False)

print(comparison_df.round(4))

best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
print(f"\nBest performing model: {best_model_name}")
print(f"Best Test AUC: {comparison_df.iloc[0]['Test_AUC']:.4f}")


fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. AUC Comparison
models_list = list(results.keys())
test_aucs = [results[model]['test_auc'] for model in models_list]
cv_aucs = [results[model]['cv_auc_mean'] for model in models_list]

x = np.arange(len(models_list))
axes[0,0].bar(x - 0.2, cv_aucs, 0.4, label='CV AUC', alpha=0.8)
axes[0,0].bar(x + 0.2, test_aucs, 0.4, label='Test AUC', alpha=0.8)
axes[0,0].set_xlabel('Models')
axes[0,0].set_ylabel('AUC Score')
axes[0,0].set_title('Model Performance Comparison')
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(models_list, rotation=45)
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. Confusion Matrix for best model
cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1])
axes[0,1].set_title(f'Confusion Matrix - {best_model_name}')
axes[0,1].set_xlabel('Predicted')
axes[0,1].set_ylabel('Actual')

# 3. Precision-Recall Curves
for model_name in results.keys():
    precision, recall, _ = precision_recall_curve(y_test, results[model_name]['y_pred_proba'])
    axes[1,0].plot(recall, precision, label=f'{model_name}')
axes[1,0].set_xlabel('Recall')
axes[1,0].set_ylabel('Precision')
axes[1,0].set_title('Precision-Recall Curves')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# 4. Feature importance for best tree-based model
tree_models = ['XGBoost', 'LightGBM', 'RandomForest']
best_tree_model = None
for model in tree_models:
    if model in results and hasattr(results[model]['model'], 'feature_importances_'):
        best_tree_model = model
        break

if best_tree_model:
    importances = results[best_tree_model]['model'].feature_importances_
    feature_names = [f'Feature_{i}' for i in range(len(importances))]
    
    # Get top 10 features
    top_indices = np.argsort(importances)[-10:]
    axes[1,1].barh(range(10), importances[top_indices])
    axes[1,1].set_yticks(range(10))
    axes[1,1].set_yticklabels([feature_names[i] for i in top_indices])
    axes[1,1].set_xlabel('Feature Importance')
    axes[1,1].set_title(f'Top 10 Features - {best_tree_model}')

plt.tight_layout()
plt.show()


df_test_encoded.head(3)


submission_df = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


y_pred_ensemble_test = ensemble.predict(df_test_encoded)
y_pred_proba_ensemble_test = ensemble.predict_proba(df_test_encoded)[:, 1]



# Save predictions to see results (Optional)
prediction_results = pd.DataFrame({
    'id':  submission_df.id ,
    'loan_paid_back': y_pred_proba_ensemble_test
})
prediction_results.to_csv('submission.csv', index=False)

