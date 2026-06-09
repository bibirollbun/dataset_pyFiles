import pandas as pd
import numpy as np


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


train.columns


#test=test.drop(columns=['alcohol_consumption_per_week','screen_time_hours_per_day','employment_status','gender'])
#train=train.drop(columns=['alcohol_consumption_per_week','screen_time_hours_per_day','employment_status','gender'])


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


#plt.figure(figsize=(8, 5))
#sns.kdeplot(x='triglycerides', y='cholesterol_total', 
#            hue='diagnosed_diabetes', 
#            data=train, fill=True)
#plt.title('Triglycerides vs. Cholestrol Total by Diabetes Status')
#plt.show()


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


from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def feat_eng(data):
    data['cholesterol_ratio']=data['cholesterol_total']/data['hdl_cholesterol']
    data['LDL-HDL_Gap']=data['ldl_cholesterol']- data['hdl_cholesterol']
    data['Cholestrol_ratios']=data['ldl_cholesterol']/data['hdl_cholesterol']
    data['glucose_interaction']=data['bmi']*data['triglycerides']
    data['map']=data['diastolic_bp'] + 1/3*(data['systolic_bp'] - data['diastolic_bp'])
    data['pulse_pressure'] = data['systolic_bp'] - data['diastolic_bp']
    data['is_hypertensive'] = ((data['systolic_bp'] >= 140) | (data['diastolic_bp'] >= 90)).astype(int)
    
    # 2. Metabolic Syndrome Score
    # Diabetes is often part of Metabolic Syndrome (High BMI + High Trigs + High BP)
    data['metabolic_risk_factor'] = (
        (data['bmi'] > 30).astype(int) + 
        (data['triglycerides'] > 150).astype(int) + 
        (data['systolic_bp'] > 130).astype(int)
    )
    
    # 3. Healthy Living Ratio
    # Compare activity vs sedentary behavior
    # Note: Adding small epsilon to avoid division by zero
    data['activity_sleep_ratio'] = data['physical_activity_minutes_per_week'] / (data['sleep_hours_per_day'] * 60 + 1)
    
    # 4. Age-Based Risks
    # Risk factors usually matter more as we age
    data['age_bmi_interaction'] = data['age'] * data['bmi']
    data['age_glucose_risk'] = data['age'] * data['triglycerides'] 
    
    # 5. Log Transformations
    # Medical data like Triglycerides and Cholesterol are often "long-tailed"
    data['log_triglycerides'] = np.log1p(data['triglycerides'])
    data['log_cholesterol'] = np.log1p(data['cholesterol_total'])

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
    'income_level', 'smoking_status', 'employment_status'
]


num_feat = [
    'age','cholesterol_ratio','map','LDL-HDL_Gap',
    'physical_activity_minutes_per_week', 'diet_score','Cholestrol_ratios',
    'alcohol_consumption_per_week','screen_time_hours_per_day',
    'sleep_hours_per_day', 'bmi','glucose_interaction',
    'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
    'triglycerides','age_bmi_interaction','age_glucose_risk','activity_sleep_ratio',
    'metabolic_risk_factor','log_triglycerides','log_cholesterol']





bin_feat = [
    'family_history_diabetes', 'hypertension_history',
    'cardiovascular_history'
]


num_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', RobustScaler())
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


from lightgbm import LGBMClassifier,early_stopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


#scale_pos_weight=train['diagnosed_diabetes'].value_counts()[0] / train['diagnosed_diabetes'].value_counts()[1]


scale_pos_weight=train['diagnosed_diabetes'].value_counts()[0] / train['diagnosed_diabetes'].value_counts()[1]


"""""#Optuna 
import optuna
from optuna.samplers import TPESampler

def objective(trial, X_train, y_train, X_valid, y_valid):
    params={
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity":-1,
        "random_state":42,    
        "device_type":"gpu",

        "learning_rate":trial.suggest_float("learning_rate", 0.005, 0.02, log=True),
        "max_depth":trial.suggest_int("max_depth", 6, 12),
        "n_estimators":trial.suggest_int("n_estimators", 100, 2000),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 0.1),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 0.5),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    model=LGBMClassifier(**params)
    callbacks_list = [early_stopping(stopping_rounds=50, verbose=False)]
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=callbacks_list, # Add callbacks here
        #categorical_feature=cat_cols,
    )
    
    y_pred = model.predict_proba(X_valid)[:, 1]
    
    auc = roc_auc_score(y_valid, y_pred)
    return auc

X_train, X_valid, y_train, y_valid = train_test_split(X_processed, y, test_size=0.2, random_state=42)
# -----------------------------
# OPTUNA STUDY
# -----------------------------
sampler = TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(
     lambda trial: objective(trial, X_train, y_train, X_valid, y_valid),
     n_trials=50,
     show_progress_bar=True
 )
# -----------------------------
# BEST PARAMETERS
 # -----------------------------
print("="*60)
print("Best parameters:")
print(study.best_params)"""""


best_params={
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "device": "gpu",
    "seed": 42,
    "scale_pos_weight":scale_pos_weight,
    "verbose": -1, # Silences output
    "n_jobs": -1,
    #"is_unbalance": True,
    'learning_rate': 0.059216255749261655,
    'num_leaves': 26,
    'max_depth': 4,
    'lambda_l1': 1.3404844864067962,
    'lambda_l2': 3.1381681073903975e-07,
    'min_child_samples': 95,
    'subsample': 0.9745291249731525,
    'colsample_bytree': 0.5645863195919457,
    'random_state': 42,
    'n_estimators': 5000
}


from lightgbm.callback import early_stopping


#model=LGBMClassifier(**best_params)


from sklearn.model_selection import StratifiedKFold

N_SPLITS = 20
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
    fold_model = LGBMClassifier(**best_params) 

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
final_cv_auc = roc_auc_score(y, oof_preds)
print(f"Overall OOF ROC AUC: {final_cv_auc:.4f}")

final = test_preds


feat_importances = pd.Series(fold_model.feature_importances_, index=preprocessor.get_feature_names_out())
feat_importances.nlargest(20).plot(kind='barh')
plt.title("Top 20 Features")
plt.show()


submission= pd.DataFrame({
    'id': test_id,
    'diagnosed_diabetes': final
})

submission.to_csv('submission.csv', index=False)


submission.head(3)

