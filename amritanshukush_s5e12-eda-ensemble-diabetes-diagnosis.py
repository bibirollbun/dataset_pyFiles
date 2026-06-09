import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)


import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb
import catboost as catb


samp_sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
train_data = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

print("--------TRAIN DATA---------")
display(train_data.head(3))

print("--------TEST DATA----------")
display(test_data.head(3))

print("---------SAMPLE SUBMISSION---------")
display(samp_sub.head(3))


print('Train size : ' , train_data.shape)
print('Test size : ' , test_data.shape)

print("Columns :", train_data.columns.tolist())
display(train_data.describe().T)


TARGET = 'diagnosed_diabetes'

CAT_COLS = train_data.select_dtypes(include=['object']).columns.tolist()
NUM_COLS = train_data.select_dtypes(include=['int', 'float']).drop(['id', TARGET], axis=1).columns.tolist()

print("Categorical Columns :", CAT_COLS)
print("\nNumerical Columns :", NUM_COLS)


class EDA:

    def __init__(self, df, TARGET, CAT_COLS, NUM_COLS):
        self.df = df
        self.TARGET = TARGET
        self.CAT_COLS = CAT_COLS
        self.NUM_COLS = NUM_COLS


    #plot target distribution
    def target_plot(self):
        # count for each category of target
        counts = self.df[self.TARGET].value_counts()
        
        plt.figure(figsize=(8,4))
        sns.countplot(data=self.df, x=self.TARGET, palette=['red' , 'green'])
        plt.title('Distribution of Target')
        plt.xlabel(self.TARGET)
        plt.ylabel('Count')
        plt.show()
    
    # plot target vs categorical
    def target_vs_cat_plot(self):
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    
        # flatten the axes to get 2-d array for plotting
        axes = axes.flatten()

        # iterate over the ctegorical columns
        for i, col in enumerate(self.CAT_COLS):
            sns.countplot(data=self.df, x=self.TARGET, hue=col, ax=axes[i], palette='Set2')
            axes[i].set_title(f'diagnosed_diabetes vs {col}')
            axes[i].legend(title=col, fontsize=8, loc='upper right')

        # don't print unnecessary columns
        for j in range(len(self.CAT_COLS), len(axes)):
            axes[j].set_visible(False)
    
        plt.tight_layout()
        plt.show()

    # plot target vs numerical
    def target_vs_num_plot(self):
        n_cols_plot = 3     # no. of columns
        n_rows = math.ceil(len(self.NUM_COLS) / n_cols_plot)   # no. of rows
        
        fig, axes = plt.subplots(n_rows, n_cols_plot, figsize=(20, 5*n_rows))
        
        # Flatten axes for 2-d array 
        axes = axes.flatten()
        
        for i, col in enumerate(self.NUM_COLS):
            # Use histogram or boxplot for numerical data
            sns.histplot(data=self.df, x=col, hue=self.TARGET, ax=axes[i], 
                         palette='Set2', kde=True, alpha=0.6, multiple='stack')
            axes[i].set_title(f'{col} by diagnosed_diabetes')
        
        # Hide unused subplots
        for j in range(len(self.NUM_COLS), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.show()

    # correlation matrix of numerical columns
    def correlation_plot(self):
        plt.figure(figsize=(10,8))
        corr_matrix = self.df[self.NUM_COLS].corr()
        
        sns.heatmap(corr_matrix , annot = True, cmap='RdBu' , fmt = ".2f")
        plt.title("Correlation between Numerical features")
        plt.show()

    # ====PLOT ALL====
    def plot(self):
        print("1. Target Distribution")
        self.target_plot()
        
        print("\n2. Target vs Categorical Features")
        self.target_vs_cat_plot()
        
        print("\n3. Target vs Numerical Features")
        self.target_vs_num_plot()
        
        print("\n4. Correlation Matrix")
        self.correlation_plot()


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


class EnsembleModel:

    def __init__(self):
        #self.train_x = train_x
        #self.train_y = train_y
        #self.test_x = test_x
        #self.TARGET = TARGET
        self.N_SPLITS = 5

        self.lgb_models = []
        self.xgb_models = []
        self.catb_models = []
        self.lgb_scores = []
        self.xgb_scores = []
        self.catb_scores = []

        self.kf = KFold(n_splits=self.N_SPLITS, shuffle=True, random_state=42)

    def _lgboost(self, train_x, train_y):
        lgb_params = {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'learning_rate': 0.03, 'num_leaves': 128, 'max_depth': -1,'min_child_samples': 30,         
            'subsample': 0.85, 'colsample_bytree': 0.7, 'reg_alpha': 0.05, 'reg_lambda': 0.2,               
            'random_state': 42, 'verbosity': -1, 'device_type': 'gpu'}
        
        lgb_oof_preds = np.zeros(len(train_x))  # OOF probabilities
        print(".........Training LightGBM Model............")
        
        for fold, (train_idx, val_idx) in enumerate(self.kf.split(train_x, train_y)):
            print(f"Fold {fold + 1}/{self.N_SPLITS}")
            X_train, X_val = train_x.iloc[train_idx], train_x.iloc[val_idx]
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
            print(f"â†’ Fold {fold + 1} AUC: {fold_auc:.6f} (best iteration: {model.best_iteration})")
            self.lgb_models.append(model)
            self.lgb_scores.append(fold_auc)

    def _xgboost(self, train_x, train_y):
        xgb_params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'learning_rate': 0.03,'max_depth': 8,
            'min_child_weight': 5,'subsample': 0.85,'colsample_bytree': 0.7,'colsample_bylevel': 0.7,
            'reg_alpha': 0.05,'reg_lambda': 1.0,'random_state': 42,'verbosity': 0, 'tree_method': 'gpu_hist'}

        xgb_oof_preds = np.zeros(len(train_x))
        print("...........Training XGBoost Model...........")
        for fold, (train_idx, val_idx) in enumerate(self.kf.split(train_x, train_y)):
            print(f"Fold {fold + 1}/{self.N_SPLITS}")
        
            X_train, X_val = train_x.iloc[train_idx], train_x.iloc[val_idx]
            y_train, y_val = train_y.iloc[train_idx], train_y.iloc[val_idx]
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=train_x.columns.tolist())
            dvalid = xgb.DMatrix(X_val, label=y_val, feature_names=train_x.columns.tolist())

            model = xgb.train(params=xgb_params,dtrain=dtrain,num_boost_round=10000,
                evals=[(dvalid, 'valid')],early_stopping_rounds=100,verbose_eval=False,maximize=True)

            # Predict probabilities
            val_pred_proba = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1))
            xgb_oof_preds[val_idx] = val_pred_proba
            fold_auc = roc_auc_score(y_val, val_pred_proba)
            print(f"â†’ Fold {fold + 1} AUC: {fold_auc:.6f} (best iteration: {model.best_iteration})")
            self.xgb_models.append(model)
            self.xgb_scores.append(fold_auc)

    def _catboost(self, train_x, train_y):
        catb_params = {'loss_function': 'Logloss','eval_metric': 'AUC','learning_rate': 0.03,
            'depth': 8,'l2_leaf_reg': 3.0,'min_data_in_leaf': 5,'random_seed': 42,
            'verbose': False,'early_stopping_rounds': 100, 'task_type': 'GPU'}

        catb_oof_preds = np.zeros(len(train_x))        
        print("...........Training CatBoost Model.........")
        for fold, (train_idx, val_idx) in enumerate(self.kf.split(train_x, train_y)):
            print(f"Fold {fold + 1}/{self.N_SPLITS}")
            X_train, X_val = train_x.iloc[train_idx], train_x.iloc[val_idx]
            y_train, y_val = train_y.iloc[train_idx], train_y.iloc[val_idx]
            train_pool = catb.Pool(X_train, y_train)
            valid_pool = catb.Pool(X_val, y_val)
            model = catb.CatBoost(params=catb_params)

            model.fit(train_pool, eval_set=valid_pool, use_best_model=True,
                verbose=False,plot=False)
            # Predict probabilities
            val_pred_proba = model.predict(valid_pool, prediction_type='Probability')[:, 1]
            catb_oof_preds[val_idx] = val_pred_proba
            fold_auc = roc_auc_score(y_val, val_pred_proba)
            print(f"â†’ Fold {fold + 1} AUC: {fold_auc:.6f} (best iteration: {model.best_iteration_})")
            self.catb_models.append(model)
            self.catb_scores.append(fold_auc)
    
    def fit(self, train_x, train_y):
        self._lgboost(train_x, train_y)
        self._xgboost(train_x, train_y)
        self._catboost(train_x, train_y)
        print("========Training Done=========")
        return self

    def predict(self, test_x):
        print("Passing the Test Dataset to the Ensemble Model.....")
        pred_lgb = np.zeros(len(test_x))
        pred_xgb = np.zeros(len(test_x))
        pred_catb = np.zeros(len(test_x))
        
        for i in range(5):
            pred_lgb += self.lgb_models[i].predict(test_x) / 5
            pred_xgb += self.xgb_models[i].predict(xgb.DMatrix(test_x)) / 5
            pred_catb += self.catb_models[i].predict(catb.Pool(test_x), prediction_type='Probability')[:, 1] / 5
        
        # â€”â€”â€”â€” 2. Simple weighted average  â€”â€”â€”â€”
        # CatBoost usually strongest â†’ give it more weight
        final_pred = 0.35 * pred_lgb + 0.3 * pred_xgb + 0.35 * pred_catb
        print("Done! Final ensemble ready.")
        return final_pred


    def plot_feature_importance(self,data, top_n=20):
        features = list(data.columns)
        num_features = len(features)
        # ============== 1. Extract Importance ==============
        # LightGBM: gain (best for interpretability)
        lgb_imp = np.mean([m.feature_importance(importance_type='gain') for m in self.lgb_models], axis=0)
        
        # XGBoost: gain
        # Fix missing features in some folds
        xgb_full = np.zeros(num_features)
        
        for m in self.xgb_models:
            # FIX: Call get_score() DIRECTLY on the Booster object (m)
            scores = m.get_score(importance_type='gain') 
            
            for i, f_name in enumerate(features): 
                # This logic relies on feature_names being set in xgb.DMatrix
                xgb_full[i] += scores.get(f_name, 0.0) 
                
        xgb_imp = xgb_full / len(self.xgb_models)

        # CatBoost: PredictionValuesChange (most reliable)
        catb_imp = np.mean([m.get_feature_importance(type='PredictionValuesChange') for m in self.catb_models], axis=0)

        # Normalize to percentage
        lgb_imp = 100.0 * lgb_imp / lgb_imp.sum()
        xgb_imp = 100.0 * xgb_imp / xgb_imp.sum()
        catb_imp = 100.0 * catb_imp / catb_imp.sum()

        # Ensemble importance (weighted average)
        ensemble_imp = 0.4 * lgb_imp + 0.3 * xgb_imp + 0.3 * catb_imp
        ensemble_imp = 100.0 * ensemble_imp / ensemble_imp.sum()

        # ============== 2. Create DataFrame ==============
        imp_df = pd.DataFrame({
            'Feature': features,
            'LightGBM': lgb_imp,
            'XGBoost': xgb_imp,
            'CatBoost': catb_imp,
            'Ensemble': ensemble_imp
        }).sort_values(by='Ensemble', ascending=False).head(top_n)

        # ============== 3. Plot Horizontal Bar Chart ==============
        plt.figure(figsize=(12, 10))
        imp_melt = imp_df.melt(id_vars='Feature', value_vars=['LightGBM', 'XGBoost', 'CatBoost', 'Ensemble'],
                               var_name='Model', value_name='Importance (%)')

        sns.barplot(data=imp_melt, y='Feature', x='Importance (%)', hue='Model',
                    palette='viridis', edgecolor='black', linewidth=0.8)

        plt.title(f'Top {top_n} Feature Importance Comparison (Ensemble Weighted)', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Importance (%)', fontsize=12)
        plt.ylabel('')
        plt.legend(title='Model', fontsize=11, title_fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()


eda = EDA(train_data, TARGET, CAT_COLS, NUM_COLS)
eda.plot()


train_x = train_data.drop(['id', TARGET], axis=1)
train_y = train_data[TARGET]
test_x = test_data.drop('id', axis=1)

preprocess = Preprocess(train_x, test_x, CAT_COLS, NUM_COLS, TARGET)
train_preprocessed, test_preprocessed = preprocess.fit_transform()
train_preprocessed.head()


ensemble = EnsembleModel()
ensemble.fit(train_preprocessed, train_y)


ensemble.plot_feature_importance(train_preprocessed)


final_pred = ensemble.predict(test_preprocessed)


submission = pd.DataFrame({
    'id': test_data['id'],     # or .astype('float64') 
    'diagnosed_diabetes': final_pred
})
submission.to_csv('submission.csv', index=False)
display(submission.head())

