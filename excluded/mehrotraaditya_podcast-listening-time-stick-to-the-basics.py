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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold , train_test_split,GridSearchCV
from tqdm.auto import tqdm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
import scipy.stats as stats
from skopt import BayesSearchCV
import itertools
import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


df_train.head()


df_test.head()


print(f"Number of Unique Podcasts: {df_train['Podcast_Name'].nunique()}")


top_5_podcasts = df_train.groupby('Podcast_Name').agg({'Listening_Time_minutes':'sum'}).sort_values(by='Listening_Time_minutes', ascending=False)[:5]
plt.figure(figsize=(10,6))
sns.barplot(x=top_5_podcasts.index , y= top_5_podcasts['Listening_Time_minutes'],palette='viridis')
plt.xlabel("Podcasts")
plt.ylabel("Total Listening Time in min")
plt.title("Top 5 Podcasts v/s Total Listening Time")
plt.show()


print( f"Average length of an episode is : {df_train['Episode_Length_minutes'].mean()} mins")


plt.figure(figsize=(10,6))
sns.histplot(df_train['Episode_Length_minutes'], kde=True, bins=40)
plt.xlabel("Listening Time in mins")
plt.ylabel("Count of Episodes")
plt.title("Distribution of Episode Durations")
plt.show()


# Step 1: Get value counts and convert to DataFrame
genre_counts = df_train['Genre'].value_counts().reset_index()
genre_counts.columns = ['Genre', 'Count']  # Rename columns

# Step 2: Plot with Seaborn
plt.figure(figsize=(12, 6))  # Set figure size
sns.set_style("whitegrid")   # Clean background with grid

# Create bar plot
ax = sns.barplot(
    data=genre_counts,
    x='Count',
    y='Genre',
    palette="viridis",  # Color palette (try "rocket", "mako", or "flare")
    edgecolor="black"   # Add borders to bars
)



df_train.columns


plt.figure(figsize=(15,6))
temp_df = df_train.groupby('Genre').agg({'Listening_Time_minutes':'sum'}).reset_index()
sns.barplot(temp_df , x="Genre" , y= "Listening_Time_minutes", palette="flare", edgecolor="black" )
plt.show()


df_train.head()


def create_combination_columns(df, cat_cols, comb_sizes=[2, 3]):
    df = df.copy()
    new_comb_cols = []

    for comb_size in comb_sizes:
        combs = list(itertools.combinations(cat_cols, comb_size))
        
        for comb in combs:
            new_col_name = '_'.join(comb)
            df[new_col_name] = df[list(comb)].astype(str).agg('_'.join, axis=1)
            new_comb_cols.append(new_col_name)

    return df, new_comb_cols


df_train_comb, comb_cols = create_combination_columns(
    df_train,
    cat_cols=['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'],
    comb_sizes=[2,3]  # or [2, 3] if you want pairs and triplets both
)


df_test_comb, test_comb_cols = create_combination_columns(
    df_test,
    cat_cols=['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'],
    comb_sizes=[2,3]  # or [2, 3] if you want pairs and triplets both
)


def kfold_target_encoding_train(df, cat_cols, target_col, n_splits=5):
    df = df.copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    encoding_maps = {}  # Dictionary to store encoding mappings


    for col in cat_cols:
        df[f'tar_enc_{col}'] = np.nan
        encoding_maps[col] = {}

        for train_idx, valid_idx in kf.split(df):
            xtrain, xvalid = df.iloc[train_idx], df.iloc[valid_idx]
            mean_target = xtrain.groupby(col)[target_col].mean()

            # Assign to validation set
            df.loc[valid_idx, f'tar_enc_{col}'] = xvalid[col].map(mean_target)

        # Store final encoding for test data
        encoding_maps[col] = df.groupby(col)[target_col].mean().to_dict()

    return df, encoding_maps


cat_cols = df_train_comb.select_dtypes(include=['O']).columns


cat_cols


# Train target encoding
df_train_encoded, train_encoding_maps = kfold_target_encoding_train(df_train_comb,cat_cols, target_col='Listening_Time_minutes')


df_train_encoded


for col in cat_cols:
        df_test_comb[f'tar_enc_{col}'] = df_test_comb[col].map(train_encoding_maps[col])


df_test_encoded = df_test_comb.copy()


df_train_encoded.isna().sum()


for col in df_train_encoded.columns:
    if(df_train_encoded[col].isna().sum()>0):
        df_train_encoded[col] = df_train_encoded[col].fillna(df_train_encoded[col].median())        


df_train_encoded.isna().sum()


for col in df_test_encoded.columns:
    if(df_test_encoded[col].isna().sum()>0):
        df_test_encoded[col] = df_test_encoded[col].fillna(df_test_encoded[col].median())        


def remove_outliers(data,col,threshold=2):
    # Calculate Q1, Q3, and IQR
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    data = data[(data[col]>=lower_bound) & (data[col]<=upper_bound)]
    return data
    


for col in df_train_encoded.columns:
    if((col!='Listening_Time_minutes') & (df_train_encoded[col].dtype!='O')):
        df_train_encoded = remove_outliers(df_train_encoded,col,)


df_train_encoded.shape , df_test_encoded.shape


df_train_encoded.drop('id',axis=1,inplace=True)


plt.figure(figsize=(12,6))
sns.heatmap(df_train_encoded.select_dtypes(include=['float64']).corr(),annot=True)
plt.show()


df_train_encoded.columns


def plot_normality_checks(data,col, figsize=(10, 4)):
        
        # Create figure with two subplots
        plt.figure(figsize=figsize)
        
        # Plot 1: Histogram with normal curve
        plt.subplot(1, 2, 1)
        sns.histplot(data[col],bins=40,kde=True)
    
        plt.title(f'Normal Distribution Check: {col}')
        
        # Plot 2: Q-Q plot
        plt.subplot(1, 2, 2)
        stats.probplot(data[col], dist="norm", plot=plt)
        plt.title(f'Q-Q Plot: {col}')
        plt.tight_layout()
        
        plt.show()


# for col in df_train_encoded:
#     if(df_train_encoded[col].dtype!='O'):
#         plot_normality_checks(df_train_encoded,col,)


def find_best_normalization(df, alpha=0.05, test='shapiro', plot_results=False, verbose=False):
    """
    Returns a DataFrame with:
    - Best normalization method for each column
    - Normality test scores for all methods
    - % improvement over original data
    """
    
    # Define transformations (with improved stability)
    transformations = {
        'original': lambda x: x,
        'log': lambda x: np.log1p(x - x.min() + 1e-6),
        'sqrt': np.sqrt,
        'cube_root': lambda x: np.cbrt(x - x.min() + 1e-6),
        'boxcox': lambda x: stats.boxcox(x - x.min() + 1)[0] if (x > x.min()).all() else np.nan,
        'yeojohnson': lambda x: PowerTransformer(method='yeo-johnson').fit_transform(x.values.reshape(-1, 1)).flatten(),
        'quantile_normal': lambda x: QuantileTransformer(output_distribution='normal').fit_transform(x.values.reshape(-1, 1)).flatten()
    }

    # Normality test functions
    test_funcs = {
        'shapiro': lambda x: stats.shapiro(x)[1],  # p-value (higher = more normal)
        'normaltest': lambda x: stats.normaltest(x)[1],
        'anderson': lambda x: -stats.anderson(x).statistic  # Negative stat (higher = more normal)
    }
    test_func = test_funcs[test]
    
    results = []
    
    for col in df.select_dtypes(include=np.number).columns:
        if col!='Listening_Time_minutes':   # skipping Target Variable
            original_data = df[col].dropna()
            if len(original_data) < 3:
                if verbose:
                    print(f"Skipping {col}: insufficient data")
                continue
                
            col_results = {'column': col}
            original_score = test_func(original_data)
            best_score = original_score
            best_transform = 'original'
            
            for name, transform in transformations.items():
                try:
                    transformed = transform(original_data.copy())
                    
                    # Skip invalid transformations
                    if np.any(np.isnan(transformed)) or np.any(np.isinf(transformed)):
                        col_results[f'{name}_score'] = np.nan
                        col_results[f'{name}_normal'] = False
                        continue
                
                    # Calculate normality score
                    score = test_func(transformed)
                    is_normal = (score > alpha) if test != 'anderson' else (score > stats.anderson(original_data).critical_values[2])
                    
                    # Store results
                    col_results[f'{name}_score'] = score
                    col_results[f'{name}_normal'] = is_normal
                    
                    # Track best improvement
                    if score > best_score:
                        best_score = score
                        best_transform = name
                        
                    if plot_results:
                        plot_transformation_comparison(col, original_data, transformed, name)
                        
                except Exception as e:
                    if verbose:
                        print(f"{col} - {name} failed: {str(e)}")
                    col_results[f'{name}_score'] = np.nan
                    col_results[f'{name}_normal'] = False
        
            # Calculate % improvement
            if test == 'anderson':
                improvement_pct = ((original_score - best_score) / np.abs(original_score + 1e-6)) * 100
            else:
                improvement_pct = ((best_score - original_score) / (1 - original_score + 1e-6)) * 100
            
            col_results.update({
                'best_transform': best_transform,
                'best_score': best_score,
                'original_score': original_score,
                'improvement_pct': improvement_pct
            })
            results.append(col_results)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Column order: Key metrics first, then all scores
    metric_cols = ['column', 'best_transform', 'improvement_pct', 'original_score', 'best_score']
    score_cols = [f'{t}_score' for t in transformations.keys()] + [f'{t}_normal' for t in transformations.keys()]
    
    return results_df[metric_cols + score_cols]

def plot_transformation_comparison(col_name, original, transformed, transform_name):
    """Plot before/after distributions with Q-Q plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original distribution
    ax1.hist(original, bins=30, alpha=0.7, color='blue')
    ax1.set_title(f'Original: {col_name}')
    
    # Transformed distribution
    ax2.hist(transformed, bins=30, alpha=0.7, color='green')
    ax2.set_title(f'After {transform_name}')
    
    plt.tight_layout()
    plt.show()


results = find_best_normalization(df_train_encoded, test='shapiro', verbose=True)


results[['column', 'best_transform', 'improvement_pct']]


def apply_best_normalizations(df, normalization_results):
    """
    Applies the best normalization method to each column in the DataFrame
    based on the results from find_best_normalization().
    
    Returns a new DataFrame with transformed columns.
    """
    # Make a copy of the original data
    normalized_df = df.copy()
    
    # Define all available transformations
    transformations = {
        'original': lambda x: x,
        'log': lambda x: np.log1p(x - x.min() + 1e-6),
        'sqrt': np.sqrt,
        'cube_root': lambda x: np.cbrt(x - x.min() + 1e-6),
        'boxcox': lambda x: stats.boxcox(x - x.min() + 1)[0] if (x > x.min()).all() else x,
        'yeojohnson': lambda x: PowerTransformer(method='yeo-johnson').fit_transform(x.values.reshape(-1, 1)).flatten(),
        'quantile_normal': lambda x: QuantileTransformer(output_distribution='normal').fit_transform(x.values.reshape(-1, 1)).flatten()
    }
    
    for _, row in normalization_results.iterrows():
        col = row['column']
        best_transform = row['best_transform']
        
        # Skip non-numeric columns or columns that should remain original
        if not np.issubdtype(df[col].dtype, np.number) or best_transform == 'original' or col=='Listening_Time_minutes':
            continue
            
        # Apply the transformation
        try:
            normalized_df[col] = transformations[best_transform](df[col])
        except Exception as e:
            print(f"Failed to transform {col} with {best_transform}: {str(e)}")
            # Keep original if transformation fails
            normalized_df[col] = df[col]
    
    return normalized_df


df_train_normalized = apply_best_normalizations(df_train_encoded, results)


df_train_normalized.head()


df_test_normalized = apply_best_normalizations(df_test_encoded, results)


df_test_normalized.head()


df_train_normalized.head()


df_train_normalized.info()


X=df_train_normalized.select_dtypes('float64').drop(['Listening_Time_minutes'],axis=1)
y= df_train_normalized['Listening_Time_minutes']


X.shape , y.shape


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25, random_state=42)


X_train.shape , X_test.shape


# X_train.drop(['Guest_Popularity_percentage', 'Number_of_Ads'],axis=1,inplace=True)
# X_test.drop(['Guest_Popularity_percentage', 'Number_of_Ads'],axis=1,inplace=True)


def train_boosting_models(X_train, y_train, X_test, y_test, n_splits=5, n_iter=50, random_state=42):
    models = {
        "XGBoost": XGBRegressor(random_state=random_state, verbosity=1),
        "LightGBM": LGBMRegressor(random_state=random_state, verbose=-1),
        "CatBoost": CatBoostRegressor(random_state=random_state, verbose=0)
    }

    param_spaces = {
        "XGBoost": {
            "n_estimators": (100, 500),
            "max_depth": (3, 10),
            "learning_rate": (0.01, 0.2, 'log-uniform'),
            "subsample": (0.3, 1.0, 'uniform'),
            "colsample_bytree": (0.3, 1.0, 'uniform')
        },
        "LightGBM": {
            "n_estimators": (100, 500),
            "max_depth": (-1, 10),
            "learning_rate": (0.01, 0.2, 'log-uniform'),
            "num_leaves": (31, 150),
            "subsample": (0.7, 1.0, 'uniform')
        },
        "CatBoost": {
            "iterations": (100, 500),
            "depth": (4, 10),
            "learning_rate": (0.01, 0.2, 'log-uniform'),
            "l2_leaf_reg": (1, 10),
            "bootstrap_type": ['Bayesian', 'Bernoulli', 'MVS']
        }
    }

    best_models = {}

    for name, model in models.items():
        print(f"\nğŸ”� Bayesian Tuning {name}...")
        param_space = param_spaces[name]

        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        search = BayesSearchCV(
            model, search_spaces=param_space, n_iter=n_iter,
            cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1,
            random_state=random_state, verbose=2
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        print(f"âœ… Best Params for {name}: {search.best_params_}")

        preds = best_model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)
        print(f"ğŸ“Š {name} RMSE: {rmse:.4f}, RÂ²: {r2:.4f}")

        best_models[name] = {
            "model": best_model,
            "rmse": rmse,
            "r2": r2,
            "params": search.best_params_
        }

    return best_models


# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Linear Regression RMSE: {rmse_lr:.4f}")
print(f"Linear Regression R2 Score: {r2_lr:.4f}")



# best_models = train_boosting_models(X_train, y_train,X_test,y_test)


ids = df_test_normalized['id']
df_test_final = df_test_normalized[X.columns]


df_test_final.shape


xgb_params = {'colsample_bytree':1.0, 'learning_rate':0.03197053486756569, 'max_depth': 10, 'n_estimators':426, 'subsample': 0.8822652394618917}
xgb_reg = XGBRegressor(**xgb_params)
xgb_reg.fit(X,y)
y_pred = xgb_reg.predict(df_test_final)


sample_submission.head()


sub_dict = {'id':ids, 'Listening_Time_minutes':y_pred}
pd.DataFrame(sub_dict,columns= sub_dict.keys()).to_csv('submission.csv',index=False)




