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


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
import xgboost as xgb
import lightgbm as lgb
#from imblearn.over_sampling import SMOTE
#from imblearn.pipeline import Pipeline as ImbPipeline



df_train=pd.read_csv('/kaggle/input/binary-prediction-of-smoker/train.csv')
df_test=pd.read_csv('/kaggle/input/binary-prediction-of-smoker/test.csv')


df_train.head(3)


df_train.smoking.value_counts()


df_train.columns


df_train['eyesight_avg'] = (df_train['eyesight(left)'] + df_train['eyesight(right)']) / 2

# Average hearing
df_train['hearing_avg'] = (df_train['hearing(left)'] + df_train['hearing(right)']) / 2

df_train['BMI'] = df_train['weight(kg)'] / (df_train['height(cm)'] / 100) ** 2

df_train['pulse_pressure'] = df_train['systolic'] - df_train['relaxation']

df_train['total_hdl_ratio'] = df_train['Cholesterol'] / df_train['HDL']
df_train['ldl_hdl_ratio'] = df_train['LDL'] / df_train['HDL']
df_train['triglyceride_hdl_ratio'] = df_train['triglyceride'] / df_train['HDL']
df_train['AST_ALT_ratio'] = df_train['AST'] / df_train['ALT']



df_train.head(3)


df_test['eyesight_avg'] = (df_test['eyesight(left)'] + df_test['eyesight(right)']) / 2

# Average hearing
df_test['hearing_avg'] = (df_test['hearing(left)'] + df_test['hearing(right)']) / 2

df_test['BMI'] = df_test['weight(kg)'] / (df_test['height(cm)'] / 100) ** 2

df_test['pulse_pressure'] = df_test['systolic'] - df_test['relaxation']

df_test['total_hdl_ratio'] = df_test['Cholesterol'] / df_test['HDL']
df_test['ldl_hdl_ratio'] = df_test['LDL'] / df_test['HDL']
df_test['triglyceride_hdl_ratio'] = df_test['triglyceride'] / df_test['HDL']
df_test['AST_ALT_ratio'] = df_test['AST'] / df_test['ALT']


df_test.head(3)


columns_to_drop = [
    'eyesight(left)',      
    'eyesight(right)',     
    'hearing(left)',       
    'hearing(right)', 
    'height(cm)',
    'weight(kg)',
    'Cholesterol',
    'HDL',
    'triglyceride',
    'LDL',                 
    'AST' ,
    'systolic',
    'relaxation',
    'ALT',
    'id'
]
df_train = df_train.drop(columns=columns_to_drop)
df_test = df_test.drop(columns=columns_to_drop)


df_train.head(3)


X = df_train.drop('smoking', axis=1)  # Features
y = df_train['smoking']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 80% train, 20% test
    random_state=42,    # For reproducibility
    stratify=y          # Maintains class distribution (use only for classification)
)


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 


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
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
    print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Train on full training set
    model.fit(X_train, y_train)
    print(f"Model {name} trained successfully!")
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
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
cv_scores_ensemble = cross_val_score(ensemble, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
print(f"Ensemble CV AUC: {cv_scores_ensemble.mean():.4f} (+/- {cv_scores_ensemble.std()*2:.4f})")

ensemble.fit(X_train_scaled, y_train)
y_pred_ensemble = ensemble.predict(X_test_scaled)
y_pred_proba_ensemble = ensemble.predict_proba(X_test_scaled)[:, 1]
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


df_test_scaled = scaler.transform(df_test)


sample_submission = pd.read_csv("/kaggle/input/binary-prediction-of-smoker/sample_submission.csv")


y_pred_ensemble_test = ensemble.predict(df_test_scaled)
y_pred_proba_ensemble_test = ensemble.predict_proba(df_test_scaled)[:, 1]

print("Predictions on new data:")
print(f"Predicted smokers: {sum(y_pred_ensemble_test)}")
print(f"Predicted smokers %: {sum(y_pred_ensemble_test)/len(y_pred_ensemble_test)*100:.2f}%")

# Save predictions to see results (Optional)
prediction_results = pd.DataFrame({
    'id': sample_submission.id,
    'smoking': y_pred_proba_ensemble_test
})
print(prediction_results.head())
prediction_results.to_csv('submission_ensemble.csv', index=False)
#print(sample_submission.head())

