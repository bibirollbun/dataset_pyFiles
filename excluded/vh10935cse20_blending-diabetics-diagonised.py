import numpy as np
import pandas as pd


train=pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


def data(df):
    print(df.head(2))
    print('\n' + '-' * 50)
    print("\nDataType : ",df.dtypes)
    print("\nShape of the Dataset : ",df.shape)
    print('\n' + '*' * 50)
    print("\n",df.info())
    print('\n' + '-' * 50)
    print("\nTotal Null Values Present : \n",df.isna().sum())
    print("\nDescriptive : \n",df.describe())
    print('\n' + '*' * 50)


data(train)


data(test)


numerical_feat=['age', 'alcohol_consumption_per_week',
    'physical_activity_minutes_per_week', 'diet_score',
    'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
    'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
    'triglycerides']


import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='age', 
    y='bmi', 
    hue='diagnosed_diabetes', # Color points by target class
    data=train
)
plt.title('Age vs. BMI by Diabetes Status')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='systolic_bp', 
    y='diastolic_bp', 
    hue='diagnosed_diabetes', # Color points by target class
    data=train
)
plt.title('Systolic BP vs. Diastolic BP by Diabetes Status')
plt.show()


sns.violinplot(x='smoking_status', y='age', hue='diagnosed_diabetes', 
               data=train, split=True)
plt.title('Smoking Status vs. Age by Diabetes Status')
plt.show()


sns.boxplot(x='gender', y='bmi', hue='diagnosed_diabetes', 
            data=train)
plt.title('Gender vs. BMI by Diabetes Status')
plt.show()


pd.pivot_table(train, index='ethnicity', columns='employment_status', 
               values='diagnosed_diabetes').pipe(sns.heatmap, annot=True, fmt='.2f', cmap='viridis')
plt.show()


sns.catplot(
    x='income_level', 
    hue='hypertension_history', 
    col='diagnosed_diabetes', # This parameter works in sns.catplot
    data=train, 
    kind='count',             # Specify that you want a count plot
    height=5,                 # Height of each facet
    aspect=1.2                # Aspect ratio of each facet
)

# You can set the overall title on the resulting FacetGrid object (g)
plt.subplots_adjust(top=0.9) # Adjust space for title
plt.suptitle('Income Level & Hypertension History by Diabetes Status', fontsize=14)

plt.show()


corr_matrix = train[numerical_feat].corr()

# Visualize the correlation matrix
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm', 
            cbar=True,
            linewidths=0.5,
            linecolor='black')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()


from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def feat_eng(data):
    data['cholesterol_ratio']=data['cholesterol_total']/data['hdl_cholesterol']
    data['LDL-HDL_Gap']=data['ldl_cholesterol']- data['hdl_cholesterol']
    data['map']=data['diastolic_bp'] + 1/3*(data['systolic_bp'] - data['diastolic_bp'])
    return data


train=feat_eng(train)
test=feat_eng(test)


from sklearn.base import BaseEstimator, TransformerMixin

class BMIBinner(BaseEstimator, TransformerMixin):
    """Adds a categorical BMI risk feature based on standard clinical cutoffs."""
    def __init__(self):
        self.bins = [0, 18.5, 25.0, 30.0, float('inf')]
        self.labels = ['Underweight', 'Normal Weight', 'Overweight', 'Obese']
        # The column to operate on
        self.column_name = 'bmi'

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # Create the new feature using pandas.cut()
        X_out['bmi_category'] = pd.cut(
            X_out[self.column_name],
            bins=self.bins,
            labels=self.labels,
            right=False, # [a, b) interval notation
            include_lowest=True
        )
        return X_out


bmi_adder = BMIBinner()


cat_feat = [
    'gender', 'ethnicity', 'education_level',
    'income_level', 'smoking_status', 'employment_status','bmi_category'
]


num_feat = [
    'age', 'alcohol_consumption_per_week','cholesterol_ratio','map','LDL-HDL_Gap',
    'physical_activity_minutes_per_week', 'diet_score',
    'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
    'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
    'triglycerides'
]


bin_feat = [
    'family_history_diabetes', 'hypertension_history',
    'cardiovascular_history'
]


num_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])


cat_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipe, num_feat),
        ('cat',cat_pipe, cat_feat), 
        ('bin', 'passthrough', bin_feat)
    ],
    remainder='drop' 
)


X=train.drop(columns=['id','diagnosed_diabetes'],axis=1)
y=train['diagnosed_diabetes']


test_id=test['id']
test_fea=test.drop('id',axis=1)


X = bmi_adder.fit_transform(X)

test_fea = bmi_adder.transform(test_fea)


X_processed = preprocessor.fit_transform(X)
test_fea_proc=preprocessor.transform(test_fea)


from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier,early_stopping
from sklearn.model_selection import StratifiedKFold


cat_best_params={
    "verbose": 0,
    "random_seed": 42,
    "eval_metric": "AUC",
    "task_type": "GPU",
    "loss_function": "Logloss",
    'auto_class_weights': 'Balanced',
    'learning_rate': 0.08533248186362546, 
    'n_estimators': 3324, 
    'max_depth': 3, 
    'l2_leaf_reg': 0.0010415523414613289, 
    'border_count': 250, 
    'bagging_temperature': 0.37026279533351164
}


lgbm_best_params={
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "device": "gpu",
    "seed": 42,
    "verbose": -1, # Silences output
    "n_jobs": -1,
    "is_unbalance": True,
    'learning_rate': 0.019837293413664626, 
    'max_depth': 9, 
    'n_estimators': 1735, 
    'lambda_l1': 0.032964919534850534, 
    'lambda_l2': 0.41138289381409315, 
    'colsample_bytree': 0.315111657712345, 
    'subsample': 0.8415890647862263, 
    'min_child_samples': 5
}


N_SPLITS = 20


skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Arrays to store results
oof_preds = np.zeros(X_processed.shape[0])
test_preds = np.zeros(test_fea_proc.shape[0])
roc_auc_scores = []

print(f"Starting {N_SPLITS}-Fold Cross-Validation...")

for fold, (train_index, val_index) in enumerate(skf.split(X_processed, y)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
    
    X_train_fold, X_val_fold = X_processed[train_index], X_processed[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

    fold_model = CatBoostClassifier(**cat_best_params)

    fold_model.fit(
        X_train_fold, y_train_fold,
        eval_set=(X_val_fold, y_val_fold),
        early_stopping_rounds=100,
        verbose=False
    )
    
    # 1. OOF Predictions (for validation/score)
    oof_fold_preds = fold_model.predict_proba(X_val_fold)[:, 1]
    oof_preds[val_index] = oof_fold_preds
    
    # 2. Test Predictions (for final submission)
    test_preds_fold = fold_model.predict_proba(test_fea_proc)[:, 1]
    test_preds += test_preds_fold / N_SPLITS
    
    fold_auc = roc_auc_score(y_val_fold, oof_fold_preds)
    roc_auc_scores.append(fold_auc)
    print(f"Fold {fold+1} AUC: {fold_auc:.4f}")

# Final Scores
print("\n--- Cross-Validation Summary ---")
print(f"Mean CV ROC AUC: {np.mean(roc_auc_scores):.4f} +/- {np.std(roc_auc_scores):.4f}")
final_cv_auc_cat = roc_auc_score(y, oof_preds)
print(f"Overall OOF ROC AUC: {final_cv_auc_cat:.4f}")

final_cat = test_preds


skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Arrays to store results (X_processed and test_fea_proc assumed to be defined)
oof_preds = np.zeros(X_processed.shape[0])
test_preds = np.zeros(test_fea_proc.shape[0])
roc_auc_scores = []

print(f"Starting {N_SPLITS}-Fold Cross-Validation for LightGBM...")

for fold, (train_index, val_index) in enumerate(skf.split(X_processed, y)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
    
    X_train_fold, X_val_fold = X_processed[train_index], X_processed[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

    # 3. Change Model Initialization
    fold_model = LGBMClassifier(**lgbm_best_params) 

    fold_model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        # 2. CORRECT USAGE
        callbacks=[early_stopping(stopping_rounds=100, verbose=False)], 
    )
    
    # 1. OOF Predictions (for validation/score)
    oof_fold_preds = fold_model.predict_proba(X_val_fold)[:, 1]
    oof_preds[val_index] = oof_fold_preds
    
    # 2. Test Predictions (for final submission)
    test_preds_fold = fold_model.predict_proba(test_fea_proc)[:, 1]
    test_preds += test_preds_fold / N_SPLITS
    
    fold_auc = roc_auc_score(y_val_fold, oof_fold_preds)
    roc_auc_scores.append(fold_auc)
    print(f"Fold {fold+1} AUC: {fold_auc:.4f}")

# Final Scores
print("\n--- Cross-Validation Summary ---")
print(f"Mean CV ROC AUC: {np.mean(roc_auc_scores):.4f} +/- {np.std(roc_auc_scores):.4f}")
final_cv_auc_lgbm = roc_auc_score(y, oof_preds)
print(f"Overall OOF ROC AUC: {final_cv_auc_lgbm:.4f}")

final_lgbm = test_preds


tot_auc=final_cv_auc_cat+final_cv_auc_lgbm

weight_cat = final_cv_auc_cat/tot_auc
weight_lgbm = final_cv_auc_lgbm/tot_auc


final=(weight_cat*final_cat) + (weight_lgbm*final_lgbm)


submission = pd.DataFrame({
    'id': test_id, # Assumes test_id was correctly defined earlier
    'diagnosed_diabetes': final
})


submission.to_csv('submission.csv', index=False)


submission.head(3)

