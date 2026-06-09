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

data = pd.read_csv('/kaggle/input/big-data-analytics-certification-kr-2024-3/train.csv')
data.columns




data.Street.head(5)


# Data Exploration
import seaborn as sns 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

class data_exploration():
    def __init__(self, train_df, test_df):
        self.original_data = train_df
        self.df = train_df
        self.test_df = test_df
        self.categorical_feats = self.df.select_dtypes(include=['object']).columns
        self.numerical_feats = self.df.select_dtypes(include=['int64', 'float64']).drop(columns=['Id', 'SalePrice']).columns
        self.interest_feats = ['MSZoning', 'HouseStyle', 'BldgType', 'Neighborhood', 'LotConfig','Street']
        self.high_corr_feats = None 
        print('Ready for data exploration..!')

    def info(self):
        print(f'Shape of dataframe(rows, columns): {self.original_data.shape}')
        print(f'\nData types & non-null counts:') #--.isnull().values.any()
        print(self.original_data.info())
        print(f'\nStatistical summary:')
        print(self.original_data.describe())
        print(f'\nPeek at first few rows:')
        print(self.original_data.head())

    def missing(self):
        missing = self.df.isnull().sum()
        print(f'Missing values: {missing}') 
        print(f' -> see detail:{missing[missing > 0].sort_values(ascending=False)}')

    def cardinality_check(self, cat_feats=None, cardinality_thresh=10):
        cat_feats = cat_feats or self.categorical_feats
        low_card, high_card = [], []
        for col in cat_feats:
            n_unique = self.df[col].nunique()
            (low_card if n_unique <= cardinality_thresh else high_card).append(col)
        print("ğŸ§® Low-cardinality (One-hot):", low_card)
        print("ğŸš¨ High-cardinality (LabelEncode):", high_card)
        return low_card, high_card

    def multicollinearity_check(self, vif_thresh=5):
        vif_data = pd.DataFrame()
        vif_data["feature"] = self.numerical_feats
        vif_data["VIF"] = [variance_inflation_factor(self.df[self.numerical_feats].values, i) for i in range(len(self.numerical_feats))]
        print("Multicollinearity check - Variance Inflation Factor:")
        print(vif_data.sort_values("VIF", ascending=False))
        high_vif_feats = vif_data[vif_data["VIF"] > vif_thresh]["feature"].tolist()
        return high_vif_feats
        

    def features(self):
        print('Categorical features:')
        print(f'{self.categorical_feats}') 
        self.cardinality_check()
        print('\n\n')
        
        print('Numerical features:')
        print(f'{self.numerical_feats}') 
        numeric_df = self.df.select_dtypes(include=[np.number])
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()    
        print('\n\n')
        
        print('Target value plotting:')
        plt.figure(figsize=(8,5)) 
        sns.histplot(self.df['SalePrice'], kde=True)  
        plt.xlabel('SalePrice')
        plt.title('Price Distribution')
        plt.show()
        print(f'Target value skewness: {self.df["SalePrice"].skew()}')
        print('\n\n')
        
        print('Feature-Target correlation:')
        # numeric_df = self.df.select_dtypes(include=[np.number])
        correlation = numeric_df.corr()['SalePrice'].sort_values(ascending=False)
        print("Top (absolute) correlations, greater than 0.3:")
        print(correlation[abs(correlation) >= 0.3])
        self.high_corr_feats = correlation.index
        print('\n\n')

        print('Feature-Target plots: do visual analysis')
        ff = self.interest_feats #list(self.categorical_feats)
        plot_boxplots(self.df, features=ff)
        #print('\n\n') -- not print after plotting. It will hide the plot.

    def log1p_transform(self,column:str):
        self.df[f'{column}_log'] = np.log1p(self.df[column])
        print('Applying log1p_transform... Done!')

    def feature_engineering(self, vif_thresh=5,feats_to_log=False):
        # 1.a. Drop the columns with too many missing values
        missing = self.df.isnull().mean()
        low_info_cols = missing[missing > 0.8].index.tolist()
        self.df.drop(columns=low_info_cols, inplace=True) 
        self.test_df.drop(columns=low_info_cols, inplace=True)

        # 1.b. Impute missing numerical values 
        # Impute missing numerical values before VIF
        imputer = SimpleImputer(strategy='median')
        self.df[self.numerical_feats] = imputer.fit_transform(self.df[self.numerical_feats])
        
        # Save changes
        self.categorical_feats = self.df.select_dtypes(include=['object']).columns
        self.numerical_feats = self.df.select_dtypes(include=['int64', 'float64']).drop(columns=['Id', 'SalePrice'], errors='ignore').columns
        
        # 2.Cardinality check 
        low_card, high_card = self.cardinality_check()
        
        # One-hot encoding for low-cardinality features
        self.df = pd.get_dummies(self.df, columns=low_card, drop_first=True)
        self.test_df = pd.get_dummies(self.test_df, columns=low_card, drop_first=True) 

        # Label encoding for high-cardinality features
        le = LabelEncoder()
        for col in high_card:
            try:
                self.df[col] = le.fit_transform(self.df[col])
                self.test_df[col] = le.fit_transform(self.test_df[col])
            except:
                self.df[col] = self.df[col].astype(str).fillna('None')
                self.test_df[col] = self.test_df[col].astype(str).fillna('None')
                self.df[col] = le.fit_transform(self.df[col]) 
                self.test_df[col] = le.fit_transform(self.test_df[col]) 

        # 3.Normalize numeric features
        scaler = StandardScaler()
        self.df[self.numerical_feats] = scaler.fit_transform(self.df[self.numerical_feats])
        self.test_df[self.numerical_feats] = scaler.transform(self.test_df[self.numerical_feats])

        # 4.Multicollinearity
        print("Checking VIF for multicollinearity...")
        high_vif_feats = self.multicollinearity_check() 
        print("Dropping high VIF features:", high_vif_feats)
        self.df.drop(columns=high_vif_feats, inplace=True)
        self.test_df.drop(columns=high_vif_feats, inplace=True)

        # 5.Log transform 
        if feats_to_log:
            for f in feats_to_log:
                self.log1ptransform(f)
    
    def prepare_test_data(self):
        X_train_features = self.df.drop(columns='SalePrice')
        X_test_aligned = de.test_df.reindex(columns=X_train_features.columns, fill_value=0)
        self.test_df = X_test_aligned 
        

# Class definition ends
#===============================================================================


def collapse_rare_categories(series, min_freq=50):
    value_counts = series.value_counts()
    return series.where(series.isin(value_counts[value_counts >= min_freq].index), other='Other')

def plot_boxplots(df, features, target='SalePrice',
                  collapse_rare=False, min_freq=50,
                  n_cols=2, figsize=(14, 6), rotate_xticks=True):
    """
    Plots tidy boxplots for categorical features vs. target variable.
    
    Parameters:
    - df: DataFrame
    - features: List of categorical column names
    - target: Target column to plot against (default: 'SalePrice')
    - collapse_rare: Whether to collapse rare categories into 'Other'
    - min_freq: Minimum frequency to keep a category (if collapse_rare=True)
    - n_cols: Number of columns in subplot grid
    - figsize: Tuple of figure size
    - rotate_xticks: Whether to rotate x-tick labels
    """
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1]*n_rows))

    axes = axes.flatten()
    
    for ax, col in zip(axes, features):
        if collapse_rare:
            temp_col = f'{col}_cleaned'
            df[temp_col] = collapse_rare_categories(df[col], min_freq=min_freq)
            plot_col = temp_col
        else:
            plot_col = col

        # Order categories by median target value
        order = df.groupby(plot_col)[target].median().sort_values().index

        # Draw boxplot
        sns.boxplot(x=plot_col, y=target, data=df, order=order, ax=ax)
        ax.set_title(f'{col} vs {target}')
        if rotate_xticks:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Hide unused axes
    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


train_df= pd.read_csv('/kaggle/input/big-data-analytics-certification-kr-2024-3/train.csv')
test_df = pd.read_csv('/kaggle/input/big-data-analytics-certification-kr-2024-3/test.csv')
de = data_exploration(train_df, test_df) 
de.info()


de.features()


from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd 

def train_xgboost_model(df, target_col='SalePrice', model_type='xgboost'):
    # 1ï¸�âƒ£ Separate target and features
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2ï¸�âƒ£ Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3ï¸�âƒ£ Initialize model
    if model_type == 'xgboost':
        model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    elif model_type == 'catboost':
        model = CatBoostRegressor(verbose=100, random_seed=42)
    else:
        raise ValueError("model_type must be 'xgboost' or 'catboost'")

    # 4ï¸�âƒ£ Fit the model
    if model_type == 'catboost':
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    else:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)

    # 5ï¸�âƒ£ Make predictions
    preds = model.predict(X_val)

    # 6ï¸�âƒ£ Evaluate performance
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f'âœ… RMSE on validation set: {rmse:.2f}')

    # 7ï¸�âƒ£ Plot feature importance (optional)
    try:
        importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
        importance_df = importance_df.sort_values('Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title(f'Top 20 Feature Importances - {model_type.upper()}')
        plt.tight_layout()
        plt.show()
    except:
        print("âš ï¸� Could not extract feature importances.")

    return model



train_df = pd.read_csv('/kaggle/input/big-data-analytics-certification-kr-2024-3/train.csv')
test_df = pd.read_csv('/kaggle/input/big-data-analytics-certification-kr-2024-3/test.csv')
de = data_exploration(train_df, test_df)
de.feature_engineering()
model = train_xgboost_model(de.df, model_type='xgboost')

de.prepare_test_data()
pred = model.predict(de.test_df)  
print(pred)
result = pd.DataFrame({'id':test_df['Id'],'SalePrice':pred})
result.to_csv('/kaggle/working/result.csv', index=False) 

