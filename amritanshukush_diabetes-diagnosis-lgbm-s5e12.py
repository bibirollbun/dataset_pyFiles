import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import optuna

import warnings
warnings.filterwarnings('ignore')


samp_sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
train_data = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

print("--------TRAIN DATA---------")
display(train_data.head(3))

print('\n')
print("\n--------TEST DATA----------")
display(test_data.head(3))

print('\n')
print("\n---------SAMPLE SUBMISSION---------")
display(samp_sub.head(3))


print('Train size : ' , train_data.shape)
print('Test size : ' , test_data.shape)


display(train_data.describe().T)


TARGET = 'diagnosed_diabetes'
NUM_COLS = train_data.select_dtypes(include=['int', 'float']).drop(['id', TARGET], axis=1).columns.tolist()
CAT_COLS = train_data.select_dtypes(include=['object']).columns.tolist()


print("Categorical Columns :", CAT_COLS)
print("\nNumerical Columns :", NUM_COLS)
print("\nTarget :", TARGET)


counts = train_data['diagnosed_diabetes'].value_counts()

plt.figure(figsize=(8,4))
sns.countplot(data=train_data, x='diagnosed_diabetes', palette=['red' , 'green'])
plt.title('Distribution of Diagnosed Diabetes')
plt.show()


plt.figure(figsize=(10,8))
corr_matrix = train_data[NUM_COLS].corr()

sns.heatmap(corr_matrix , annot = True, cmap='RdBu' , fmt = ".2f")
plt.title("Correlation between Numerical features")
plt.show()


train_x = train_data.drop(['id', TARGET], axis=1)
train_y = train_data[TARGET]

test_x = test_data.drop('id', axis=1)


class Preprocess:
    
    def __init__(self, train_df, test_df, CAT_COLS, NUM_COLS, TARGET):
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.CAT_COLS = CAT_COLS
        self.NUM_COLS = NUM_COLS
        self.TARGET = TARGET
        
        # Save encoders/scaler for reuse
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.final_columns = None  # Will store final feature columns

    def encoding(self):
        train = self.train_df.copy()
        test = self.test_df.copy()

        # 1. Label Encoding: gender
        train['gender'] = self.label_encoder.fit_transform(train['gender'])
        test['gender'] = self.label_encoder.transform(test['gender'])

        # 2. One-Hot: ethnicity
        train = pd.get_dummies(train, columns=['ethnicity'], prefix='ethnicity', dtype=int)
        test = pd.get_dummies(test, columns=['ethnicity'], prefix='ethnicity', dtype=int)
        train, test = train.align(test, join='outer', axis=1, fill_value=0)

        # 3. Ordinal: smoking_status
        smoking_map = {'Never': 0, 'Former': 1, 'Current': 2, 'Unknown': -1}
        train['smoking_status'] = train['smoking_status'].map(smoking_map).fillna(-1)
        test['smoking_status'] = test['smoking_status'].map(smoking_map).fillna(-1)

        # 4. Ordinal: education_level
        edu_map = {'No formal': 0, 'Highschool': 1, 'Graduate': 2, 'Postgraduate': 3}
        train['education_level'] = train['education_level'].map(edu_map)
        test['education_level'] = test['education_level'].map(edu_map)

        # 5. Ordinal: income_level
        inc_map = {'Low': 0, 'Lower-Middle': 1, 'Middle': 2, 'Upper-Middle': 3, 'High': 4}
        train['income_level'] = train['income_level'].map(inc_map)
        test['income_level'] = test['income_level'].map(inc_map)

        # 6. One-Hot: employment_status
        train = pd.get_dummies(train, columns=['employment_status'], prefix='employment', dtype=int)
        test = pd.get_dummies(test, columns=['employment_status'], prefix='employment', dtype=int)

        # aligning the columns in both datasets
        train, test = train.align(test, join='left', axis=1, fill_value=0)

        # Drop id and target from features
        if 'id' in train.columns:
            train = train.drop('id', axis=1)
        if 'id' in test.columns:
            test = test.drop('id', axis=1)
        if self.TARGET in train.columns:
            train = train.drop(self.TARGET, axis=1)

        # Save final column order 
        if self.final_columns is None:
            self.final_columns = train.columns.tolist()

        # Ensure test has exact same columns as train
        test = test.reindex(columns=self.final_columns, fill_value=0)

        return train, test

    def scaling(self):
        train_X, test_X = self.encoding()

        # Fit scaler only on train
        train_X[self.NUM_COLS] = self.scaler.fit_transform(train_X[self.NUM_COLS])
        test_X[self.NUM_COLS] = self.scaler.transform(test_X[self.NUM_COLS])

        return train_X, test_X

    def fit_transform(self):
        return self.scaling()


preprocess = Preprocess(train_x, test_x, CAT_COLS, NUM_COLS, TARGET)
train_x_preprocessed, test_x_preprocessed = preprocess.fit_transform()
train_x_preprocessed.head()


def lgb_objective(trial):
    lgb_params = {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
                  'learning_rate': trial.suggest_float('learning_rate' , 0.01 , 0.1 , log=True), 
                  'num_leaves': trial.suggest_int('num_leaves' , 5 , 50), 
                  'max_depth': trial.suggest_int('max_depth' , 3 , 15),
                  'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),         
                  'subsample': trial.suggest_float('subsample' , 0 , 1), 
                  'colsample_bytree': trial.suggest_float('colsample_bytree' , 0.2 , 1.0), 
                  'reg_alpha': trial.suggest_float('reg_alpha' , 0.01 , 0.1 , log=True), 
                  'reg_lambda': trial.suggest_float('reg_lambda' , 0.01 , 0.1 , log=True),               
                  'random_state': 42, 'verbosity': -1, 'device_type': 'gpu'}

    N_SPLITS = 5
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_aucs = []
    lgb_oof_preds = np.zeros(len(train_x_preprocessed))  # OOF probabilities
            
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_x_preprocessed, train_y)):
        
        X_train, X_val = train_x_preprocessed.iloc[train_idx], train_x_preprocessed.iloc[val_idx]
        y_train, y_val = train_y.iloc[train_idx], train_y.iloc[val_idx]
        
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_valid = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
            
        model = lgb.train(params=lgb_params, train_set=lgb_train, valid_sets=[lgb_valid],
            num_boost_round=10000,callbacks=[
                lgb.early_stopping(stopping_rounds=100, first_metric_only=True),
                lgb.log_evaluation(period=0)
            ])
        
        val_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
        lgb_oof_preds[val_idx] = val_pred_proba
        val_auc = roc_auc_score(y_val, val_pred_proba)
        fold_aucs.append(val_auc)

    return float(np.mean(fold_aucs))

# To run, you can un-comment below lines and run it
        
#study = optuna.create_study(direction='maximize')
#study.optimize(lgb_objective, n_trials=5, show_progress_bar=True)
#best_params = study.best_trial.params
#print("\nBest Hyperparameters from Optuna:")
#print(best_params)


# Optimal paramters according to Optuna
lgb_params = {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
              'learning_rate': 0.015263956955158248, 
              'num_leaves': 10, 
              'max_depth': 3,
              'min_child_samples': 21,         
              'subsample': 0.1737030598633309, 
              'colsample_bytree': 0.8307183018642812, 
              'reg_alpha': 0.07466896491431449, 
              'reg_lambda': 0.09755349265186107,               
              'random_state': 42, 'verbosity': -1,
              'device' : 'gpu'}


# Training model on optimized parameters
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
lgb_models, lgb_scores = [], []
lgb_oof_preds = np.zeros(len(train_x_preprocessed))  # OOF probabilities
print(".........Training LightGBM Model............")
        
for fold, (train_idx, val_idx) in enumerate(kf.split(train_x_preprocessed, train_y)):
    print(f"Fold {fold + 1}/{N_SPLITS}")
    X_train, X_val = train_x_preprocessed.iloc[train_idx], train_x_preprocessed.iloc[val_idx]
    y_train, y_val = train_y.iloc[train_idx], train_y.iloc[val_idx]
    # LightGBM Dataset for better performance & early stopping
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_valid = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        
    model = lgb.train(params=lgb_params, train_set=lgb_train, valid_sets=[lgb_valid],
        num_boost_round=10000,callbacks=[
            lgb.early_stopping(stopping_rounds=100, first_metric_only=True),
            lgb.log_evaluation(period=0)
        ])
    # Predict probabilities
    val_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
    lgb_oof_preds[val_idx] = val_pred_proba
    fold_auc = roc_auc_score(y_val, val_pred_proba)
    print(f"→ Fold {fold + 1} AUC: {fold_auc:.6f} (best iteration: {model.best_iteration})")
    lgb_models.append(model)
    lgb_scores.append(fold_auc)


print("\n" + "="*50)
print(f"Mean CV ROC AUC: {np.mean(lgb_scores):.5f} ± {np.std(lgb_scores):.5f}")
print(f"Best single fold: {max(lgb_scores):.5f}")
print("="*50)


# Deploying trained model on test set
pred_lgb = np.zeros(len(test_x_preprocessed))
for i in range(5):
    pred_lgb += lgb_models[i].predict(test_x_preprocessed) / 5


submission = pd.DataFrame({
    'id': test_data['id'],     # or .astype('float64') 
    'diagnosed_diabetes': pred_lgb
})
submission.to_csv('submission.csv', index=False)
display(submission.head())




