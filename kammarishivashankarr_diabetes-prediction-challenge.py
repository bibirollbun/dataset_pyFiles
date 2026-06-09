import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import chi2_contingency
from pandas import DataFrame

import warnings
warnings.filterwarnings('ignore')


class EDA:
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.target_col = target_col
        self.df_full = df
        self.df = df.drop(columns=target_col)
        self.plot_type = None
        self.numerical_columns = self.df.select_dtypes(include='number').columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=['object','category']).columns.tolist()
        self.num_col_length = len(self.numerical_columns)
        self.cat_col_length = len(self.categorical_columns)
        self.numerical_analysis = None
        self.categorical_analysis = None
        self.univarinat_column = None
        self.bivariate_analysis = False

    def plotData(self, plot_type: str = 'hist',column_type:str = 'numerical') -> None:
        if plot_type in ["corr", "target_corr"]:
            df_used = self.df_full
            if plot_type == "corr":
                corr_cols = self.numerical_columns + [self.target_col]
                corr = df_used[corr_cols].corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                plt.figure(figsize=(8,6))
                sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
                plt.title("Correlation Matrix")
                plt.show()
            elif plot_type == "target_corr":
                corr_cols = self.numerical_columns + [self.target_col]
                corr = df_used[corr_cols].corr()[self.target_col].drop(self.target_col)
                corr_sorted = corr.sort_values(ascending=False)
                plt.figure(figsize=(8,4))
                sns.barplot(x=corr_sorted.index, y=corr_sorted.values, palette="Set2")
                plt.title(f"Correlation of Numerical Features with {self.target_col}", fontsize=14)
                plt.ylabel("Correlation Coefficient")
                plt.xticks(rotation=45, ha='right')
                plt.show()
            return

        if column_type == 'numerical':
            columns = self.numerical_columns
        elif column_type == 'categorical':
            columns = self.categorical_columns
        else:
            raise ValueError("column_type must be 'numerical' or 'categorical'")
    
        if len(columns) == 0:
            print("No columns to plot.")
            return
        ncols = min(3, len(columns))
        nrows = math.ceil(len(columns) / ncols)
        fig, ax = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        ax = np.array(ax).reshape(-1)
        df_used = self.df_full if self.bivariate_analysis else self.df
        for idx, col in enumerate(columns):
            if plot_type == "hist" and col in self.numerical_columns:
                if self.bivariate_analysis:
                    sns.histplot(data=df_used, x=col, hue=self.target_col,
                                 ax=ax[idx], kde=True, palette="Set2")
                else:
                    sns.histplot(df_used[col], bins=20, ax=ax[idx],
                                 color='skyblue', edgecolor='black')
                ax[idx].set_title(col)
            elif plot_type == "box" and col in self.numerical_columns:
                if self.bivariate_analysis:
                    sns.boxplot(data=df_used, x=self.target_col, y=col,
                                ax=ax[idx], palette="Set2")
                else:
                    sns.boxplot(y=df_used[col], ax=ax[idx], color='skyblue')
                ax[idx].set_title(col)
            elif plot_type == "count" and col in self.categorical_columns:
                if self.bivariate_analysis:
                    sns.countplot(data=df_used, x=col, hue=self.target_col,
                                  ax=ax[idx], palette="Set2")
                else:
                    sns.countplot(x=df_used[col], ax=ax[idx], palette="pastel")
                ax[idx].set_title(f"Frequency of {col}")
                ax[idx].tick_params(axis='x', rotation=45)
                
            elif plot_type == "target_proportion" and col in self.categorical_columns:
                ct = pd.crosstab(self.df_full[col], self.df_full[self.target_col], normalize='index')
                ct = ct.reset_index().melt(id_vars=col, var_name=self.target_col, value_name='proportion')
                sns.barplot(x=col, y='proportion', hue=self.target_col,
                            data=ct, ax=ax[idx], palette="Set2")
                ax[idx].set_title(f"Target distribution by {col}")
                ax[idx].tick_params(axis='x', rotation=45)
        for j in range(len(columns), len(ax)):
            ax[j].set_visible(False)
        plt.tight_layout()
        plt.show()

    def chi_square_tests(self):
        results = []
        for col in self.categorical_columns:
            ct = pd.crosstab(self.df_full[col], self.df_full[self.target_col])
            chi2, p, dof, expected = chi2_contingency(ct)
            results.append((col, chi2, round(p,3)))
        return pd.DataFrame(results, columns=['Column', 'Chi2 Statistic', 'p-value'])

    def explore_numerical_columns(self):
        self.numerical_analysis = True
        print(self.df[self.numerical_columns].describe())
        self.plotData('hist','numerical')
        self.plotData('box','numerical')
        self.plotData('corr','numerical')
        self.plotData('target_corr','numerical')

    def explore_categorical_columns(self):
        self.categorical_analysis = True
        print(self.df[self.categorical_columns].describe())
        self.plotData('count','categorical')
        self.plotData('target_proportion','categorical')

    def univariate_analysis(self, column: str, plot_type: str = 'hist'):
        plt.figure(figsize=(6,4))
        if plot_type == 'hist':
            sns.histplot(self.df_full[column], kde=True, color='skyblue')
            plt.title(f"Distribution of {column}")
            plt.show()
            print(self.df_full[column].describe())
        elif plot_type == 'count':
            sns.countplot(x=self.df_full[column], palette="pastel")
            plt.title(f"Frequency of {column}")
            plt.show()
            print(f"{self.target_col} distribution in %")
            print(self.df_full[column].value_counts(normalize=True) * 100)
        elif plot_type == 'box':
            sns.boxplot(y=self.df_full[column], color='skyblue')
            plt.title(f"Boxplot of {column}")
            plt.show()
            print(self.df_full[column].describe())

def optimize_dataframe(df):
    """
    Reduce memory footprint by downcasting numeric columns
    and converting object columns with few distinct values to category.
    """
    for col in df.columns:
        col_type = df[col].dtype
        
        # Numeric columns
        if np.issubdtype(col_type, np.number):
            col_min, col_max = df[col].min(), df[col].max()
            
            if np.issubdtype(col_type, np.integer):
                # Downcast integers
                if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                # Downcast floats
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Object/string columns
        elif col_type == object:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            
            # If few distinct values, convert to category
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')
    
    return df



df_train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

df_train.head()


print(f"training data has {df_train.shape[0]} rows and {df_train.shape[1]} columns.\n")
print(f"test data has {df_test.shape[0]} rows and {df_test.shape[1]} columns.\n")
df_train.info()


for c in ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']:
    print(f"{c} : {df_train[c].unique()}")


df_train['family_history_diabetes'] = df_train['family_history_diabetes'].astype('object')
df_train['hypertension_history'] = df_train['hypertension_history'].astype('object')
df_train['cardiovascular_history'] = df_train['cardiovascular_history'].astype('object')

df_test['family_history_diabetes'] = df_test['family_history_diabetes'].astype('object')
df_test['hypertension_history'] = df_test['hypertension_history'].astype('object')
df_test['cardiovascular_history'] = df_test['cardiovascular_history'].astype('object')


df_train.info()


df_train = optimize_dataframe(df_train)
df_test = optimize_dataframe(df_test)


df_train.info()


df_train.duplicated().sum()


# droppping the id column
df_train.drop(columns=['id'],inplace=True)
df_test.drop(columns=['id'],inplace=True)


df_train.columns = df_train.columns.str.strip()
df_test.columns = df_test.columns.str.strip()


target_col = 'diagnosed_diabetes'


eda_obj = EDA(df_train,target_col)


print(eda_obj.numerical_columns,'\n',f"Number of numerical columns {eda_obj.num_col_length}")
print(eda_obj.categorical_columns,'\n',f"Number of numerical columns {eda_obj.cat_col_length}")


eda_obj.target_col


eda_obj.univariate_analysis(plot_type = 'count',column = target_col)


eda_obj.bivariate_analysis=True
eda_obj.explore_numerical_columns()


eda_obj.plotData('corr')


eda_obj.explore_categorical_columns()


chi=eda_obj.chi_square_tests()
chi


for col in eda_obj.numerical_columns:
    train_mean = df_train[col].mean()
    train_std = df_train[col].std()
    # lower, upper = train_mean - 3*train_std , train_mean + 3*train_std
    lower = df_train[col].quantile(0.05) 
    upper = df_train[col].quantile(0.95)
        
    df_train[col] = np.where(df_train[col] < lower, lower,
                           np.where(df_train[col] > upper, upper, df_train[col]))
                             
    df_test[col] = np.where(df_test[col] < lower, lower,
                           np.where(df_test[col] > upper, upper, df_test[col]))
    


eda_obj1 = EDA(df_train,target_col)
eda_obj1.plotData('box','numerical')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer


x = df_train.drop(columns=target_col)
y = df_train[target_col]

x_train,x_test, y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.25,stratify=y)


x_train.columns


ohe_cols = ['gender','ethnicity','smoking_status','employment_status']
ordinal_cols = ['education_level','income_level']
education_col_order = ['No formal','Highschool','Graduate','Postgraduate']
income_level_order = ['Low','Lower-Middle','Middle','Upper-Middle','High']

trf = ColumnTransformer(transformers = [
    ('ohe',OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),ohe_cols),
    ('oe',OrdinalEncoder(categories=[education_col_order,income_level_order]),ordinal_cols),
        ],verbose_feature_names_out=False,
    remainder='passthrough'
        )

trf.set_output(transform='pandas')


x_train_trfed = trf.fit_transform(x_train)
x_test_trfed = trf.transform(x_test)
x_test_trfed2 = trf.transform(df_test)


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score


xgb = XGBClassifier(enable_categorical=True,random_state=0)
rf = RandomForestClassifier(random_state=0)
lgbm = LGBMClassifier(objective='binary', boosting_type='gbdt', random_state=0)



xgb.fit(x_train_trfed,y_train)
y_pred_xgb = xgb.predict(x_test_trfed)

lgbm.fit(x_train_trfed,y_train)
y_pred_lgbm = lgbm.predict(x_test_trfed)


accuracy_score(y_pred_xgb,y_test), accuracy_score(y_pred_lgbm,y_test)


# y_pred = xgb.predict_proba(x_test_trfed2)
# submission['diagnosed_diabetes'] = y_pred
# submission.head()


# submission.to_csv(
#     'submission.csv',
#     index=False
# )


from sklearn.model_selection import GridSearchCV


param_grid = {
    'max_depth': [5], #Initial : [5,7]
    'learning_rate': [0.2,0.3,0.35], #Initial : [0.01,0.1,0.2]
    'n_estimators': [300,350,400], #Inital [100, 300, 500],
    'subsample': [1.0],
    'colsample_bytree': [1.0]
}

grid = GridSearchCV(XGBClassifier(eval_metric='auc',enable_categorical=True), param_grid, cv=3, scoring='accuracy')
grid.fit(x_train_trfed,y_train)


print(grid.best_params_)


y_pred_xgb = grid.predict(x_test_trfed)
accuracy_score(y_pred_xgb,y_test)


pip install --upgrade lightgbm



import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
from lightgbm import early_stopping, log_evaluation


# def objective(trial):
#     params = {
#         'objective': 'binary',
#         'boosting_type': 'gbdt',
#         'metric': 'auc',
#         'verbosity': -1,
#         'random_state': 42,
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
#         'num_leaves': trial.suggest_int('num_leaves', 31, 256),
#         'max_depth': trial.suggest_int('max_depth', -1, 20),
#         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
#         'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
#         'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
#         'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
#         'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0)
#     }

#     skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
#     auc_scores = []

#     for train_idx, valid_idx in skf.split(x_train_trfed, y_train):
#         X_tr, X_val = x_train_trfed.iloc[train_idx], x_train_trfed.iloc[valid_idx]
#         y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

#         model = lgb.LGBMClassifier(**params, n_estimators=1000)
#         model.fit(
#             X_tr, y_tr,
#             eval_set=[(X_val, y_val)],
#             eval_metric='auc',
#             callbacks=[early_stopping(100), log_evaluation(0)]
#         )

#         y_pred = model.predict_proba(X_val)[:, 1]
#         auc_scores.append(roc_auc_score(y_val, y_pred))

#     return np.mean(auc_scores)

# # Run Optuna study
# study = optuna.create_study(direction='maximize')  # maximize AUC
# study.optimize(objective, n_trials=4)


best_params_ = {'learning_rate': 0.03264502198024162,
 'num_leaves': 227,
 'max_depth': 6,
 'min_data_in_leaf': 136,
 'feature_fraction': 0.6031375242005526,
 'bagging_fraction': 0.7291665416262125,
 'bagging_freq': 2,
 'lambda_l2': 0.27297535115026084}


best_params = best_params_
final_model = lgb.LGBMClassifier(**best_params, n_estimators=1000,random_state=42)
final_model.fit(x_train_trfed, y_train)


pred_lgbm = final_model.predict(x_test_trfed)
accuracy_score(pred_lgbm,y_test)


y_pred = final_model.predict_proba(x_test_trfed2)[:,1]
submission['diagnosed_diabetes'] = y_pred
submission.head()

submission.to_csv(
    'submission.csv',
    index=False
)





