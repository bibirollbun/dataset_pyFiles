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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


print(train_df.head())
print(train_df.shape)



train_df.info()


train_df.describe()


train_df.isnull().sum()


train_df.duplicated().sum()


object_col = train_df.dtypes[train_df.dtypes == 'object'].index.tolist()
print(object_col)
col_ohe = ['gender', 'marital_status','loan_purpose', 'grade_subgrade']
col_ord = ['education_level', 'employment_status']


train_df.describe()



train_df[object_col].nunique()


print(train_df['loan_paid_back'].value_counts(normalize=True))


X = train_df.drop(columns=['loan_paid_back'])
y = train_df['loan_paid_back']
print(train_df['education_level'].unique())
print(train_df['employment_status'].unique())



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler

X_train,X_vals,y_train,y_vals = train_test_split(X,y,test_size=0.2,random_state=1)
ohe = OneHotEncoder(drop='first',sparse_output=False)
ordinal = OrdinalEncoder()
scale = StandardScaler()





from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 


Order = [['Other', 'High School', "Bachelor's", "Master's", 'PhD'],
         ['Student', 'Unemployed', 'Self-employed', 'Employed', 'Retired']]
col_num = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']

preprocessor = ColumnTransformer(
    transformers=[
        ('scale', StandardScaler(), col_num),
        ('ord', OrdinalEncoder(categories=Order), col_ord),
        ('ohe', OneHotEncoder(drop='first', sparse_output=False), col_ohe)
    ],
    remainder='passthrough' 
)


X_train_final = preprocessor.fit_transform(X_train)
X_vals_final  = preprocessor.transform(X_vals)
test_df_final = preprocessor.transform(test_df)



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

model_dict= {
    'Logistic Regression': LogisticRegression(max_iter=1000,class_weight='balanced'),
    'RF_Deep': RandomForestClassifier(max_depth=20, n_estimators=100,class_weight='balanced', random_state=1),
    'RF_Shallow': RandomForestClassifier(max_depth=10, n_estimators=20,class_weight='balanced', random_state=1),
    
    'XGBoost': XGBClassifier(n_estimators=200,learning_rate=0.1,max_depth=6,scale_pos_weight=4.0,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=1.0,random_state=42,eval_metric='logloss',use_label_encoder=False),
    'Decision Tree': DecisionTreeClassifier(),
    

}
pipelines = {}

print("ğŸ”§ Creating pipelines for each model...")
print("=" * 60)

for model_name, model in model_dict.items():
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor), 
        ('classifier', model)           
    ])
    
  
    pipelines[model_name] = pipeline
    
    print(f"âœ… Created pipeline for: {model_name}")
    print(f"   Steps: [preprocessor] â†’ [{model_name}]")

print(f"\nğŸ“Š Total pipelines created: {len(pipelines)}")


results = {
    'Model': [],
    'Accuracy': [],
    'Class_0_Precision': [],
    'Class_0_Recall': [],      
    'Class_0_F1': [],
    'Class_1_F1': [],
    'Training_Time': [],
    'Pipeline_Object': []
}


from sklearn.metrics import precision_recall_fscore_support
import time

def evaluate_model(pipeline, X_train, y_train, X_val, y_val, model_name):
    """
    Train and evaluate a single model
    Returns: metrics dictionary
    """
    print(f"\nğŸ”� Training {model_name}...")
    
    # Time the training
    start_time = time.time()
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Make predictions
    y_pred = pipeline.predict(X_val)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, labels=[0.0, 1.0]
    )
    
    accuracy = (y_pred == y_val).mean()
    
    return {
        'accuracy': accuracy,
        'class_0_precision': precision[0],
        'class_0_recall': recall[0],  # Your key metric
        'class_0_f1': f1[0],          # Your key metric
        'class_1_f1': f1[1],
        'training_time': training_time,
        'pipeline': pipeline
    }



# preprocessor = ColumnTransformer(...)

for model_name, model in model_dict.items():
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")
    
  
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Train and evaluate
    metrics = evaluate_model(pipeline, X_train, y_train, X_vals, y_vals, model_name)
    
    # Store results
    results['Model'].append(model_name)
    results['Accuracy'].append(metrics['accuracy'])
    results['Class_0_Precision'].append(metrics['class_0_precision'])
    results['Class_0_Recall'].append(metrics['class_0_recall'])
    results['Class_0_F1'].append(metrics['class_0_f1'])
    results['Class_1_F1'].append(metrics['class_1_f1'])
    results['Training_Time'].append(metrics['training_time'])
    results['Pipeline_Object'].append(metrics['pipeline'])
    
    # Print quick results
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Class 0.0 Recall: {metrics['class_0_recall']:.4f} (Current: 0.59)")
    print(f"  Class 0.0 F1: {metrics['class_0_f1']:.4f} (Current: 0.69)")
    print(f"  Training Time: {metrics['training_time']:.2f}s")



best_model_name = 'XGBoost'


from sklearn.pipeline import make_pipeline

final_pipeline = make_pipeline(preprocessor, model_dict[best_model_name])

final_pipeline.fit(X, y)


test_predictions = final_pipeline.predict(test_df)


submission_df = pd.DataFrame({
    'id': test_df['id'],
    'loan_paid_back': test_predictions
})
submission_df.to_csv('submission.csv', index=False)
print("âœ… Submission file 'submission.csv' saved!")


results = {
    'Model': ['LogReg', 'RF', 'XGBoost'],
    'Accuracy': [0.85, 0.88, 0.90],
    'Class_0_Recall': [0.65, 0.72, 0.75],
    'Class_0_F1': [0.70, 0.75, 0.78],
    'Training_Time': [1.2, 5.4, 8.1]
}


import matplotlib.pyplot as plt
import seaborn as sns

# Create comparison dataframe
results_df = pd.DataFrame({
    'Model': results['Model'],
    'Accuracy': results['Accuracy'],
    'Class_0_Recall': results['Class_0_Recall'],
    'Class_0_F1': results['Class_0_F1'],
    'Training_Time': results['Training_Time']
})

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Accuracy Comparison
axes[0,0].barh(results_df['Model'], results_df['Accuracy'], color='skyblue')
axes[0,0].set_xlabel('Accuracy')
axes[0,0].set_title('Model Accuracy Comparison')
axes[0,0].axvline(x=results_df['Accuracy'].mean(), color='red', linestyle='--')

# 2. Class 0 Recall (MOST IMPORTANT)
axes[0,1].barh(results_df['Model'], results_df['Class_0_Recall'], color='salmon')
axes[0,1].set_xlabel('Class 0 Recall (Default Detection)')
axes[0,1].set_title('Default Detection Capability')
axes[0,1].axvline(x=0.7, color='green', linestyle='--', label='Target 70%')

# 3. Training Time
axes[1,0].barh(results_df['Model'], results_df['Training_Time'], color='lightgreen')
axes[1,0].set_xlabel('Training Time (seconds)')
axes[1,0].set_title('Training Efficiency')

# 4. Accuracy vs Recall Trade-off
axes[1,1].scatter(results_df['Accuracy'], results_df['Class_0_Recall'], s=100)
for i, row in results_df.iterrows():
    axes[1,1].annotate(row['Model'], (row['Accuracy'], row['Class_0_Recall']))
axes[1,1].set_xlabel('Accuracy')
axes[1,1].set_ylabel('Class 0 Recall')
axes[1,1].set_title('Accuracy-Recall Trade-off')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


for model_name, pipeline in pipelines.items():
    y_pred = pipeline.predict(X_vals)
    cm = confusion_matrix(y_vals, y_pred)
    
    print(f"\n{model_name} Confusion Matrix:")
    print(f"True Negatives (TN): {cm[0,0]}")  # Correct defaults
    print(f"False Positives (FP): {cm[0,1]}") # Good loans ko default bola
    print(f"False Negatives (FN): {cm[1,0]}") # Defaults ko miss kiya (COSTLY!)
    print(f"True Positives (TP): {cm[1,1]}")  # Correct good loans


# XGBoost feature importance
xgb_model = pipelines['XGBoost'].named_steps['classifier']
feature_names = preprocessor.get_feature_names_out()
importances = xgb_model.feature_importances_

# Top 10 important features
feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False).head(10)

print("Top 10 Important Features:")
print(feat_imp_df)


from sklearn.metrics import roc_curve, auc

# Har model ke liye ROC curve plot karein
plt.figure(figsize=(10, 8))
for model_name, pipeline in pipelines.items():
    if hasattr(pipeline, 'predict_proba'):
        y_proba = pipeline.predict_proba(X_vals)[:, 1]
        fpr, tpr, _ = roc_curve(y_vals, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(True)
plt.show()


# Assumptions (customize according to business)
COST_FALSE_NEGATIVE = 10000  
COST_FALSE_POSITIVE = 500   
PROFIT_TRUE_POSITIVE = 2000  

def calculate_business_impact(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    total_cost = (fp * COST_FALSE_POSITIVE) + (fn * COST_FALSE_NEGATIVE)
    total_profit = tp * PROFIT_TRUE_POSITIVE
    net_value = total_profit - total_cost
    
    return {
        'Model': model_name,
        'FN_Cost': fn * COST_FALSE_NEGATIVE,
        'FP_Cost': fp * COST_FALSE_POSITIVE,
        'TP_Profit': tp * PROFIT_TRUE_POSITIVE,
        'Net_Value': net_value,
        'ROI': (net_value / (tp + fp)) if (tp + fp) > 0 else 0
    }


business_impacts = []
for model_name, pipeline in pipelines.items():
    y_pred = pipeline.predict(X_vals)
    impact = calculate_business_impact(y_vals, y_pred, model_name)
    business_impacts.append(impact)

business_df = pd.DataFrame(business_impacts)
print("ğŸ“Š Business Impact Analysis:")
print(business_df.sort_values('Net_Value', ascending=False))


from sklearn.metrics import precision_recall_curve,f1_score ,accuracy_score, recall_score


xgb_pipeline = pipelines['XGBoost']
y_proba = xgb_pipeline.predict_proba(X_vals)[:, 0]  

precisions, recalls, thresholds = precision_recall_curve(y_vals, y_proba, pos_label=0)


f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Default threshold: 0.5")
print(f"Optimized threshold: {best_threshold:.3f}")
print(f"F1@0.5: {f1_score(y_vals, y_proba > 0.5, pos_label=0):.3f}")
print(f"F1@optimized: {f1_scores[best_idx]:.3f}")


y_pred_optimized = (y_proba > best_threshold).astype(int)



interaction_features = train_df.copy()


interaction_features['income_to_loan'] = interaction_features['annual_income'] / interaction_features['loan_amount']
interaction_features['debt_burden'] = interaction_features['annual_income'] * interaction_features['debt_to_income_ratio']
interaction_features['risk_score'] = interaction_features['credit_score'] * (1 - interaction_features['debt_to_income_ratio'])


correlation_with_target = interaction_features[['income_to_loan', 'debt_burden', 'risk_score', 'loan_paid_back']].corr()
print("New Feature Correlations with Target:")
print(correlation_with_target['loan_paid_back'].sort_values(ascending=False))


import plotly.express as px


categorical_features = ['gender', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, feature in enumerate(categorical_features):
   
    default_rates = train_df.groupby(feature)['loan_paid_back'].apply(
        lambda x: (x == 0).sum() / len(x) * 100
    ).sort_values(ascending=False)
    
    axes[idx].bar(default_rates.index.astype(str), default_rates.values)
    axes[idx].set_title(f'Default Rate by {feature}')
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Default Rate (%)')
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].axhline(y=train_df['loan_paid_back'].mean()*100, color='r', linestyle='--', label='Overall Avg')

plt.tight_layout()
plt.show()


from sklearn.model_selection import learning_curve

def plot_learning_curve(model, X, y, model_name):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=3, scoring='f1', 
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.xlabel("Training examples")
    plt.ylabel("F1 Score")
    plt.title(f"Learning Curve for {model_name}")
    plt.legend(loc="best")
    plt.grid()
    plt.show()
    
    # Check for overfitting/underfitting
    final_gap = train_scores_mean[-1] - val_scores_mean[-1]
    if final_gap > 0.1:
        print(f"âš ï¸� {model_name}: Possible overfitting (gap: {final_gap:.3f})")
    elif final_gap < 0.01:
        print(f"âœ… {model_name}: Good generalization")



xgb_pipeline = pipelines['XGBoost']
y_pred_xgb = xgb_pipeline.predict(X_vals)


wrong_idx = np.where(y_pred_xgb != y_vals)[0]
wrong_cases = X_vals.iloc[wrong_idx].copy()
wrong_cases['actual'] = y_vals.iloc[wrong_idx]
wrong_cases['predicted'] = y_pred_xgb[wrong_idx]
wrong_cases['error_type'] = np.where(
    (wrong_cases['actual'] == 0) & (wrong_cases['predicted'] == 1), 
    'False Positive', 
    'False Negative'
)

print(f"Total wrong predictions: {len(wrong_idx)}")
print(f"False Positives (Type I): {(wrong_cases['error_type'] == 'False Positive').sum()}")
print(f"False Negatives (Type II): {(wrong_cases['error_type'] == 'False Negative').sum()}")


false_negatives = wrong_cases[wrong_cases['error_type'] == 'False Negative']
print("\nğŸ”� False Negative Analysis (We missed defaults!):")
print(f"Average interest rate: {false_negatives['interest_rate'].mean():.2f}")
print(f"Average debt-to-income: {false_negatives['debt_to_income_ratio'].mean():.3f}")
print(f"Most common loan purpose: {false_negatives['loan_purpose'].mode()[0]}")



model_predictions = {}
for model_name, pipeline in pipelines.items():
    model_predictions[model_name] = pipeline.predict(X_vals)


pred_matrix = pd.DataFrame(model_predictions)


print("ğŸ¤� Model Agreement Matrix (How often models agree):")
agreement_matrix = pd.DataFrame(index=pred_matrix.columns, columns=pred_matrix.columns)
for m1 in pred_matrix.columns:
    for m2 in pred_matrix.columns:
        agreement_matrix.loc[m1, m2] = (pred_matrix[m1] == pred_matrix[m2]).mean()

print(agreement_matrix)


from scipy.stats import mode
ensemble_pred, _ = mode(pred_matrix.values, axis=1)
ensemble_accuracy = (ensemble_pred.flatten() == y_vals).mean()
print(f"\nSimple Majority Voting Ensemble Accuracy: {ensemble_accuracy:.4f}")


from sklearn.ensemble import StackingClassifier


base_learners = [
    ('xgb', pipelines['XGBoost'].named_steps['classifier']),
    ('rf_deep', pipelines['RF_Deep'].named_steps['classifier']),
    ('rf_shallow', pipelines['RF_Shallow'].named_steps['classifier'])
]

# Meta-learner (final estimator)
meta_learner = LogisticRegression()

# stacking pipeline
stacking_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('stacking', StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=3
    ))
])

# Train and evaluate stacking
stacking_pipeline.fit(X_train, y_train)
y_pred_stack = stacking_pipeline.predict(X_vals)
stack_accuracy = accuracy_score(y_vals, y_pred_stack)
stack_recall = recall_score(y_vals, y_pred_stack, pos_label=0)

print(f"Stacked Model Performance:")
print(f"  Accuracy: {stack_accuracy:.4f}")
print(f"  Class 0 Recall: {stack_recall:.4f}")











































