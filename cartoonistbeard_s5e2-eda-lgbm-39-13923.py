import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import math
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from lightgbm import early_stopping
from xgboost import DMatrix
from catboost import Pool
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import LabelEncoder, FunctionTransformer, OneHotEncoder
from xgboost import XGBRegressor, XGBRFRegressor, DMatrix
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import KFold
import numpy as np
import time
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
from xgboost import DMatrix
from catboost import Pool
from lightgbm import early_stopping


train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


train_df.drop(columns=['id'],inplace=True)
test_df.drop(columns=['id'],inplace=True)


display(train_df.head(2))
display(test_df.head(2))


dfs = [train_df, test_df]
titles = ["Train NULL Values", "Test NULL Values"]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))  # 1x2 grid for train & test

for i, df in enumerate(dfs):
    null_perc = (df.isnull().sum() / df.shape[0]) * 100  # Convert to percentage
    null_perc = null_perc[null_perc > 0]  # Only plot non-zero nulls

    ax = axes[i]
    sns.barplot(x=null_perc.index, y=null_perc.values, ax=ax, palette="viridis")

    # Annotate bars with exact percentage values
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2, p.get_height() + 0.5, f'{p.get_height():.2f}%', 
                ha='center', va='bottom', fontsize=10, color='black')

    ax.set_title(titles[i])
    ax.set_xlabel("Columns")
    ax.set_ylabel("Percentage NULL")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")  # Rotate labels

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


train_df.info()


cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style']
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))  # 3x2 grid
axes = axes.flatten()  # Flatten axes to iterate easily

for i, col in enumerate(cols):
    ax = axes[i]
    value_counts = train_df[col].value_counts()

    # Create bar plot
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax, palette="viridis")

    # Annotate bars with actual counts
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2, p.get_height() + 1, f'{int(p.get_height())}', 
                ha='center', va='bottom', fontsize=10, color='black')

    ax.set_title(f"{col} Distribution")
    ax.set_xlabel("Categories")
    ax.set_ylabel("Frequency")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")  # Rotate for better readability

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


num_cols = train_df.select_dtypes(include=['float64', 'int64']).columns.to_list()
_, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    if i >= len(axes):  
        break
    val_cnt = train_df[col].value_counts().sort_index()  
    ax = axes[i]
    sns.lineplot(x=val_cnt.index, y=val_cnt.values, ax=ax, marker="o", palette="viridis")
    ax.set_title(f"{col} Distribution")
    ax.set_xlabel("Values")
    ax.set_ylabel("Frequency")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")  

plt.tight_layout()
plt.show()


class load_data:
    def __init__(self, file_path=None,file_df=None):
        if file_path is not None:
            self.df = pd.read_csv(file_path)
        elif file_df is not None:
            self.df = file_df
        #self.id = self.df['id']
        #self.df.drop(columns=['id'], inplace=True)
        print("Shape = ", self.df.shape)

    def summarize(self, include='all'):
        print("=" * 50, 'SUMMARY', '=' * 50)
        if include == 'numerical':
            summarize_df = self.df.describe(include=['number']).T
        elif include == 'categorical':
            summarize_df = self.df.describe(include=['object', 'category']).T
        else:
            summarize_df = self.df.describe(include='all').T

        summarize_df['dtype'] = self.df.dtypes
        summarize_df['missing'] = self.df.isnull().sum()
        summarize_df['unique'] = self.df.nunique()
        summarize_df['duplicates'] = self.df.duplicated().sum()
        summarize_df['most_frequent'] = self.df.select_dtypes(include=['object', 'category']).apply(
            lambda col: col.value_counts().idxmax() if col.nunique() > 0 else None
        )

        def highlight(val):
            if isinstance(val, (int, float)):
                if val > 100000:
                    return 'background-color: red'
                elif val > 50000:
                    return 'background-color: orange'
                elif val > 10000:
                    return 'background-color: blue'
                elif val < 1000:
                    return 'background-color: green'
            return ''

        summarize_df.drop(columns=['25%', '50%', '75%', 'count', 'most_frequent'], inplace=True)
        styled_df = summarize_df.style.applymap(highlight, subset=['missing', 'unique'])
        return styled_df

    def visualize(self, include='all', sample=10000, exclude=[]):
        sample_df = self.df.sample(sample)

        if include == 'numerical':
            columns_to_plot = self.df.select_dtypes(include=['number']).columns
        elif include == 'categorical':
            columns_to_plot = self.df.select_dtypes(include=['object', 'category']).columns
        else:
            columns_to_plot = self.df.columns

        columns_to_plot = [col for col in columns_to_plot if col not in exclude]

        if 'numerical' in include or 'all' in include:
            print("=" * 50, 'Visualizing Numerical Features', '=' * 50)
            numerical_cols = self.df.select_dtypes(include=['number']).columns
            for col in numerical_cols:
                if col not in exclude:
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=(f'{col} - Histogram', f'{col} - Boxplot'),
                        column_widths=[0.5, 0.5]
                    )

                    hist = px.histogram(sample_df, x=col, title=f'{col} - Histogram')
                    fig.add_trace(hist.data[0], row=1, col=1)

                    box = px.box(sample_df, y=col, title=f'{col} - Boxplot')
                    fig.add_trace(box.data[0], row=1, col=2)

                    fig.update_layout(
                        title=f'{col} - Distribution and Boxplot',
                        showlegend=False
                    )
                    fig.update_xaxes(title_text=col, row=1, col=1)
                    fig.update_yaxes(title_text='Frequency', row=1, col=1)
                    fig.update_yaxes(title_text=col, row=1, col=2)

                    fig.show()

        if 'categorical' in include or 'all' in include:
            print("=" * 50, 'Visualizing Categorical Features', '=' * 50)
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if col not in exclude:
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=(f'{col} - Count Plot', f'{col} - Pie Chart'),
                        column_widths=[0.5, 0.5],
                        specs=[[{"type": "bar"}, {"type": "pie"}]]
                    )

                    count_data = sample_df[col].value_counts().reset_index()
                    count_data.columns = [col, 'count']
                    count = px.bar(count_data, x=col, y='count', title=f'{col} - Count Plot')
                    fig.add_trace(count.data[0], row=1, col=1)

                    pie = px.pie(sample_df, names=col, title=f'{col} - Pie Chart')
                    fig.add_trace(pie.data[0], row=1, col=2)

                    fig.update_layout(
                        title=f'{col} - Distribution and Pie Chart',
                        showlegend=True
                    )
                    fig.update_xaxes(title_text=col, row=1, col=1)
                    fig.update_yaxes(title_text='Count', row=1, col=1)

                    fig.show()

    def impute_columns(self, strategies=None, constant_values=None):
        if strategies is None:
            print("No strategies provided. Automatically imputing missing values.")
            strategies = {}

            for col in self.df.columns:
                if self.df[col].dtype in ['int64', 'float64']:
                    strategies[col] = 'mean'
                else:
                    strategies[col] = 'mode'

        for col, strategy in strategies.items():
            if col not in self.df.columns:
                print(f"Warning: Column '{col}' not found in the dataframe. Skipping.")
                continue

            if strategy == 'mean':
                if self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                    print(f"Imputed '{col}' with mean.")
                else:
                    print(f"Skipping '{col}' as it's not numeric for mean imputation.")

            elif strategy == 'median':
                if self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                    print(f"Imputed '{col}' with median.")
                else:
                    print(f"Skipping '{col}' as it's not numeric for median imputation.")

            elif strategy == 'mode':
                if not self.df[col].isnull().all():
                    self.df[col].fillna(self.df[col].mode().iloc[0], inplace=True)
                    print(f"Imputed '{col}' with mode.")
                else:
                    print(f"Cannot compute mode for '{col}' as all values are NaN.")

            elif strategy == 'constant':
                if constant_values and col in constant_values:
                    self.df[col].fillna(constant_values[col], inplace=True)
                    print(f"Imputed '{col}' with constant value '{constant_values[col]}'.")
                else:
                    print(f"Error: Provide a constant value for column '{col}' in 'constant_values' dictionary.")

            else:
                print(f"Error: Unsupported strategy '{strategy}' for column '{col}'. Use 'mean', 'median', 'mode', or 'constant'.")

    def feature_target_dependence(self, target_col, exclude=[]):
        """
        Analyze the dependence of the target column on other features.

        Parameters:
        - target_col (str): The target column name.
        - exclude (list): List of feature column names to exclude from analysis.

        Returns:
        - pd.DataFrame: Summary of the dependence analysis.
        """
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in the dataset.")
        
        dependence_summary = []

        for col in self.df.columns:
            if col == target_col or col in exclude:
                continue
            
            # Numerical target
            if pd.api.types.is_numeric_dtype(self.df[target_col]):
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    # Drop rows with NaN in either column
                    valid_data = self.df[[col, target_col]].dropna()
                    if len(valid_data) > 1:
                        stat, p_value = stats.pearsonr(valid_data[col], valid_data[target_col])
                        dependence_summary.append([col, 'numerical', 'Pearson Correlation', stat, p_value])
                    else:
                        dependence_summary.append([col, 'numerical', 'Pearson Correlation', 'Insufficient Data', 'N/A'])
                elif pd.api.types.is_categorical_dtype(self.df[col]) or pd.api.types.is_object_dtype(self.df[col]):
                    # Perform ANOVA
                    valid_data = self.df[[col, target_col]].dropna()
                    if len(valid_data[col].unique()) > 1 and len(valid_data) > 1:
                        groups = [valid_data[valid_data[col] == level][target_col] for level in valid_data[col].unique()]
                        stat, p_value = stats.f_oneway(*groups)
                        dependence_summary.append([col, 'categorical', 'ANOVA', stat, p_value])
                    else:
                        dependence_summary.append([col, 'categorical', 'ANOVA', 'Insufficient Data', 'N/A'])

            # Categorical target
            elif pd.api.types.is_categorical_dtype(self.df[target_col]) or pd.api.types.is_object_dtype(self.df[target_col]):
                if pd.api.types.is_categorical_dtype(self.df[col]) or pd.api.types.is_object_dtype(self.df[col]):
                    # Perform Chi-Square Test
                    contingency_table = pd.crosstab(self.df[col], self.df[target_col])
                    if contingency_table.size > 1:
                        stat, p_value, _, _ = stats.chi2_contingency(contingency_table)
                        dependence_summary.append([col, 'categorical', 'Chi-Square Test', stat, p_value])
                    else:
                        dependence_summary.append([col, 'categorical', 'Chi-Square Test', 'Insufficient Data', 'N/A'])
                elif pd.api.types.is_numeric_dtype(self.df[col]):
                    # Perform ANOVA
                    valid_data = self.df[[col, target_col]].dropna()
                    if len(valid_data[target_col].unique()) > 1 and len(valid_data) > 1:
                        groups = [valid_data[valid_data[target_col] == level][col] for level in valid_data[target_col].unique()]
                        stat, p_value = stats.f_oneway(*groups)
                        dependence_summary.append([col, 'numerical', 'ANOVA', stat, p_value])
                    else:
                        dependence_summary.append([col, 'numerical', 'ANOVA', 'Insufficient Data', 'N/A'])

        df =  pd.DataFrame(dependence_summary, columns=['Feature', 'Feature Type', 'Test Used', 'Statistic', 'p-value'])
        df['p-value'] = df['p-value'].apply(lambda x:round(x,5))
        def highlight(val):
            if isinstance(val, (int, float)):
                if val <0.05:
                    return 'background-color:green'     
            return ''

        dep_df = df.style.applymap(highlight, subset=['p-value'])
        return dep_df
        
    def get_df(self):
        return self.df


train_load = load_data(file_df=train_df)
test_load = load_data(file_df=test_df)


train_load.summarize()


test_load.summarize()


train_load.feature_target_dependence(target_col='Price')


train_load.impute_columns()


test_load.impute_columns()


train_df = train_load.get_df()
test_df = test_load.get_df()


display(train_df.head(2))
display(test_df.head(2))


X = train_df.copy()
y = X.pop("Price")
test = test_df.copy()


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]), num_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), cat_features)
])

X_train = preprocessor.fit_transform(X)
X_test = preprocessor.transform(test)


print(X_train.shape,X_test.shape)


import optuna
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as rmse
from sklearn.datasets import fetch_openml


x_t,x_v,y_t,y_v = train_test_split(X_train,y,test_size=0.2,random_state=42)

def objective(trial):
    """Optimize LightGBM hyperparameters for regression"""
    params = {
        "objective": "regression",  
        "metric": "rmse", 
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
        "min_gain_to_split": trial.suggest_loguniform("min_gain_to_split", 1e-8, 1.0),
    }


    model = lgb.LGBMRegressor(**params, random_state=42)
    model.fit(x_t, y_t)

    y_pred = model.predict(x_v)
    score = mape(y_v, y_pred) 

    return score  

study = optuna.create_study(direction="minimize") 
study.optimize(objective, n_trials=50)

print("Best Parameters:", study.best_params)
print("Best RMSE:", study.best_value)


lgb_model = study.best_params


best_params = {'learning_rate': 0.23932876262018543,
 'num_leaves': 277,
 'max_depth': 3,
 'min_data_in_leaf': 58,
 'feature_fraction': 0.9861212484501595,
 'bagging_fraction': 0.5003675499851594,
 'bagging_freq': 1,
 'lambda_l1': 5.8530014595688824e-08,
 'lambda_l2': 3.8526054550389e-06,
 'min_gain_to_split': 2.4428729828044025e-05}


def model_trainer(model, X, y, test, n_splits=5, random_state=42, verbose=-1, model_name=None):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    print("="*80)
    model_name_ = model[-1].__class__.__name__ if isinstance(model, Pipeline) else model.__class__.__name__
    print(f"Model: {model_name_}")
    print("="*80 + '\n')

    oof_rmse = []
    oof_test_preds = np.zeros(len(test))
    oof_train_preds = np.zeros(len(y))
    
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]
        if model_name == 'xgb':
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=verbose)
            booster = model.get_booster()
            y_pred = booster.predict(DMatrix(X_valid), iteration_range=(0, model.best_iteration+1))
            test_pred = booster.predict(DMatrix(test), iteration_range=(0, model.best_iteration+1))
            oof_train_preds[train_idx] = booster.predict(DMatrix(X_train), iteration_range=(0, model.best_iteration+1))

        elif model_name == 'cat':
            trainPool = Pool(X_train ,y_train)
            testPool = Pool(test)
            validPool = Pool(X_valid, y_valid)

            model.fit(X=trainPool, eval_set=validPool, verbose=verbose, early_stopping_rounds=200)
            y_pred = model.predict(validPool)
            test_pred = model.predict(testPool)
            oof_train_preds[train_idx] = model.predict(Pool(X_train))
        elif model_name == 'lgb':
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='rmse', callbacks=[early_stopping(200, verbose=0)])
            y_pred = model.predict(X_valid, num_iteration=model.best_iteration_)
            test_pred = model.predict(test, num_iteration=model.best_iteration_)
            oof_train_preds[train_idx] = model.predict(X_train, num_iteration=model.best_iteration_)

        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            test_pred = model.predict(test)
            oof_train_preds[train_idx] = model.predict(X_train)

        oof_test_preds += test_pred
        fold_res = rmse(y_valid, y_pred)
        print(f"Fold {fold+1} --> RMSE: {fold_res:.4f}")
        oof_rmse.append(fold_res)

    print(f"Average Fold RMSE: {np.mean(oof_rmse):.4f} \xb1 {np.std(oof_rmse):.4f}")
    return oof_test_preds/n_splits, oof_train_preds


lgb_reg = LGBMRegressor(**best_params,verbosity=-1)

preds,train_oof = model_trainer(
    lgb_reg,
    X_train, y, X_test, random_state=42, model_name='lgb'
)


sub_df = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
sub_df['Price'] = preds
sub_df.head()


sub_df.to_csv("PS5E2LGBMV1.csv",index=False)

