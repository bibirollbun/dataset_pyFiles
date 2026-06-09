from tqdm.notebook import tqdm
from pathlib import Path
import joblib 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
from sklearn import preprocessing
from sklearn.base import clone
from datetime import datetime, date


import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb


import warnings
warnings.filterwarnings("ignore")


%load_ext autoreload
%autoreload 2
%matplotlib inline

sns.set()
SNS_CMAP = 'Pastel1'
sns.set_palette(SNS_CMAP)

colors = sns.palettes.color_palette(SNS_CMAP)
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)




train = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv")
test = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/test.csv")

submission = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/sample_submission.csv")


target = 'sale_price'
print('The dimension of the train dataset is:', train.shape)
print('The dimension of the test dataset is:', test.shape)


train.sample(5)


agg_df = train.agg(["nunique", "unique", lambda x:x.isna().sum(), "dtypes"]).T
agg_df['unique'] = agg_df['unique'].apply(lambda x: x if len(x)<10 else x[:10])
agg_df.style.apply(lambda s: [f'background-color: rgba({colors[0][0]*255}, {colors[0][1]*255}, {colors[0][2]*255}, 0.5)' if i % 2 == 0 else f'background-color: rgba({colors[3][0]*255}, {colors[3][1]*255}, {colors[3][2]*255}, 0.5)' for i in range(len(s))])


def winkler_score(y_true, lower, upper, alpha=0.1, return_coverage=False):
    """Compute the Winkler Interval Score for prediction intervals.

    Args:
        y_true (array-like): True observed values.
        lower (array-like): Lower bounds of prediction intervals.
        upper (array-like): Upper bounds of prediction intervals.
        alpha (float): Significance level (e.g., 0.1 for 90% intervals).
        return_coverage (bool): If True, also return empirical coverage.

    Returns:
        score (float): Mean Winkler Score.
        coverage (float, optional): Proportion of true values within intervals.
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    width = upper - lower
    penalty_lower = 2 / alpha * (lower - y_true)
    penalty_upper = 2 / alpha * (y_true - upper)

    score = width.copy()
    score += np.where(y_true < lower, penalty_lower, 0)
    score += np.where(y_true > upper, penalty_upper, 0)

    if return_coverage:
        inside = (y_true >= lower) & (y_true <= upper)
        coverage = np.mean(inside)
        return np.mean(score), coverage

    return np.mean(score)


# --- 2. EDA & Preprocessing ---
print("\nPreprocessing...")

# Target variable
TARGET = 'sale_price'

# Log transform the target (common for prices)
#train_df[TARGET] = np.log1p(train_df[TARGET])

def preprocess(df):
    df_copy = df.copy() # Work on a copy to avoid modifying the original DataFrame

   

    # --- Date Features ---
    # Convert 'sale_date' to datetime objects to extract useful components
    df_copy['sale_date'] = pd.to_datetime(df_copy['sale_date'])
    df_copy['sale_year'] = df_copy['sale_date'].dt.year
    df_copy['sale_month'] = df_copy['sale_date'].dt.month
    df_copy['sale_dayofyear'] = df_copy['sale_date'].dt.dayofyear # Day number within the year (1-366)
    df_copy['sale_dayofweek'] = df_copy['sale_date'].dt.dayofweek # Day of the week (Monday=0, Sunday=6)
    df_copy['sale_weekofyear'] = df_copy['sale_date'].apply(lambda x: x.isocalendar()[1])
    df_copy['sale_hour'] = df_copy['sale_date'].dt.hour


    # --- Simple Feature Engineering ---
    today = datetime.today()
    # Age of the property at the time of sale
    df_copy["days_since_sale"] = (today - df_copy["sale_date"]).dt.days
    df_copy['age_at_sale'] = df_copy['sale_year'] - df_copy['year_built']
    df_copy['age_at_sale'] = df_copy['age_at_sale'].clip(lower=0) # Ensure age isn't negative (e.g., if sale_year < year_built due to data error)
    df_copy = df_copy.drop('sale_date', axis=1) # Original date string no longer needed

    # Age since renovation at the time of sale
    df_copy['reno_age_at_sale'] = np.where( df_copy['year_reno'] > 0, df_copy['sale_year'] - df_copy['year_reno'], df_copy['age_at_sale'] )
    df_copy['reno_age_at_sale'] = df_copy['reno_age_at_sale'].clip(lower=0)
    df_copy['was_renovated'] = (df_copy['year_reno'] > 0).astype(int) # Binary flag: 1 if renovated, 0 otherwise

    # Combined square footage and per-story square footage
    view_cols = [c for c in df_copy.columns if 'view_' in c]
    if view_cols:
        df_copy['total_view_score'] = df_copy[view_cols].sum(axis=1)
        
    sqft_cols = [c for c in df_copy.columns if 'sqft_' in c]
    if sqft_cols:
        df_copy['total_sqft'] = df_copy[sqft_cols].sum(axis=1)

    df_copy['total_baths'] = (df_copy['bath_full'] + df_copy['bath_3qtr'] * 0.75 + df_copy['bath_half'] * 0.5)   
    df_copy['sqft_per_story'] = df_copy['sqft'] / df_copy['stories'].replace(0,1) # Avoid division by zero if 'stories' is 0
    df_copy['total_val'] = df_copy['land_val'] + df_copy['imp_val']
    df_copy['year_gap'] = df_copy['join_year'] - df_copy['sale_year']

    df_copy["submarket"] = df_copy["submarket"].fillna("Unknown")
    df_copy["sale_nbr"] = df_copy["sale_nbr"].fillna(df_copy['sale_nbr'].median())
    df_copy["subdivision"] = df_copy["subdivision"].fillna("Unknown")
    
    #rounding off
    df_copy['sale_nbr'] = df_copy['sale_nbr'].astype(int)
    df_copy['stories'] = df_copy['stories'].astype(int)

    # --- Handling Categorical Features ---

    for col in df_copy.select_dtypes(exclude=['number']).columns:
        # Convert to pandas 'category' dtype. LightGBM can handle this efficiently.
        df_copy[col] = df_copy[col].astype('category')
            
    return df_copy


train = preprocess(train)
test = preprocess(test)


train['source'] = "1"
test['source'] = "0"

all_data = pd.concat([train, test], axis =0)
print(all_data.shape)

for feature in ["sale_hour", "sale_dayofweek", "sale_weekofyear", "sale_month", "sale_dayofyear"]:
        min_f = all_data[feature].min() 
        max_f = all_data[feature].max()
        
        rel_diff = (all_data[feature] - min_f) / (max_f - min_f)
        all_data[f'sin_{feature}'] = np.sin(2 * np.pi * rel_diff)
        all_data[f'cos_{feature}'] = np.cos(2 * np.pi * rel_diff)
    


#for accounting Inflation in sale_price

cpi_data = {
    1999: 166.6, 2000: 172.2, 2001: 177.1, 2002: 179.9, 2003: 184.0,
    2004: 188.9, 2005: 195.3, 2006: 201.6, 2007: 207.3, 2008: 215.3,
    2009: 214.5, 2010: 218.1, 2011: 224.9, 2012: 229.6, 2013: 232.9,
    2014: 236.7, 2015: 237.0, 2016: 240.0, 2017: 245.1, 2018: 251.1,
    2019: 255.7, 2020: 258.8, 2021: 270.9, 2022: 292.7, 2023: 304.0,
    2024: 312.0, 2025: 315.0
}

all_data['cpi'] = all_data['sale_year'].map(cpi_data)



def map_zoning(zoning_code):
    code = str(zoning_code).upper()

    if any(x in code for x in ['RS', 'SF', 'R1', 'R 1', 'R-1']):
        return 'SF'
    elif any(x in code for x in ['RM', 'MF', 'MR', 'RA', 'RMA']):
        return 'MF'
    elif any(x in code for x in ['MU', 'MUR', 'MIO', 'GC-MU', 'CC-MU']):
        return 'MIXED'
    elif any(x in code for x in ['C', 'CB', 'NC', 'CC', 'RC', 'CA']):
        return 'COM'
    elif any(x in code for x in ['I', 'IC', 'IM', 'IG']):
        return 'IND'
    elif any(x in code for x in ['A', 'AG', 'A10', 'RA10', 'RA5']):
        return 'AGR'
    elif any(x in code for x in ['PUD', 'PLA', 'PR', 'MPD', 'DCE']):
        return 'PLN'
    elif any(x in code for x in ['OS', 'UF', 'PBZ', 'UC']):
        return 'PUB'
    elif any(x in code for x in ['TC', 'UV', 'DT', 'UR', 'MIO']):
        return 'TRANS'
    else:
        return 'SPECIAL'

all_data['zoning_category'] = all_data['zoning'].apply(map_zoning)



train = all_data[all_data['source'] == '1']
test = all_data[all_data['source'] == '0']



plt.figure(figsize =(15,5))
yearly = train.groupby('sale_year')['sale_price'].agg(['mean','median']).reset_index()
sns.lineplot(data=yearly, x='sale_year', y=yearly['mean']/1_000_000, color = "orange", label = 'mean')
sns.lineplot(data=yearly, x='sale_year', y=yearly['median']/1_000_000, color = 'blue', label = 'median')
plt.ylabel("Avg Sale Price in Million")
plt.xlabel("Sale Year")
plt.title("Sale Price based on Year", loc = 'center')
plt.show()



# CPI for base year (2024)
#base_cpi = cpi_data[2024]

# Adjust sale price to 2024 dollars
#train['sale_price_adj'] = train['sale_price'] * (base_cpi / train['cpi'])


df = train.groupby(['age_at_sale'], as_index = False)['sale_price'].agg(['count', 'mean', 'median'])

cols = ['count', 'mean', 'median']
fig, ax = plt.subplots(3,1, figsize =(20,18))
for i,col in enumerate(cols):
    
    sns.lineplot(data=df, x='age_at_sale', y=col, ax =ax[i])
    ax[i].set_title(f'{col.title()} of Sale Price by Age at Sale')

plt.show()


df = train[train['sale_price'] < 2000000]
sns.distplot(df['sale_price'], kde = True)


df = train[(train['total_sqft'] < 500000)]
sns.jointplot(x= df['total_sqft']/1000, y = df['sale_price']/1000000 , sizes = 0.2)
plt.ylabel("Sale Price in Million")
plt.xlabel("Area SqFt in K")
plt.title("Sale Price based on Total Area", loc = 'center' ,y = 1.3)
plt.show()


df = train.groupby(['city'], as_index = False)['sale_price'].agg(['count', 'mean', 'median'])

cols = ['count', 'mean', 'median']
fig, ax = plt.subplots(3,1, figsize =(25,36))
for i,col in enumerate(cols):
    
    sns.barplot(x= df['city'], y = df[col], ax = ax[i])
    ax[i].set_title(f'{col.title()} of Sale Price by City')
    ax[i].set_xticklabels(df['city'], rotation=60)

plt.show()


cities = ['BEAUX ARTS', 'CLYDE HILL', 'MEDINA', 'YARROW POINT', 'HUNTS POINT', 'MERCER ISLAND']

fig,ax = plt.subplots(1, len(cities), figsize = (len(cities)*5, 8) )
fig.suptitle('Sale Price in Millions')

for i,city in enumerate(cities):
    df = train[train['city'] == city]
    sns.histplot(df['sale_price']/1000000, ax= ax[i])
    ax[i].set_xlabel(f'{city}')
    
    


plt.figure(figsize = (20,5))
sns.boxplot(x = train['zoning_category'], y = train['sale_price'] )
plt.ylabel("Sale Price in Million")
plt.xlabel("Type of Building")
plt.title("Sale Price based on Type of Property", loc = 'center')
plt.show()


plt.figure(figsize = (16,6))
sns.boxplot(x = train['join_status'], y = train['sale_price'] )
plt.ylabel("Sale Price in Million")
plt.xlabel("Join Status")
plt.title("Sale Price based on Join Status", loc = 'center')
plt.show()


"""
def simplify_join_status(val):
    if val == 'rebuilt - before':
        return 'HighValue'
    elif val == 'reno - before':
        return 'AboveAvg'
    elif val in ['new', 'rebuilt - after']:
        return 'Moderate'
    else:
        return 'Low'

df['join_status_grouped'] = df['join_status'].apply(simplify_join_status)

"""


cols = ['sale_nbr', 'grade', 'fbsmt_grade', 'condition']

fig, ax = plt.subplots(4, 1, figsize = (15,24))
for i,col in enumerate(cols):
    sns.boxplot(x = train[col], y = train['sale_price'] ,ax = ax[i])
    ax[i].set_ylabel("Sale Price in Million")
    ax[i].set_xlabel(f"{col}")
    ax[i].set_title(f"Sale Price based on {col}", loc = 'center')
    
plt.show()


cols = ['beds', 'stories', 'total_baths']

fig, ax = plt.subplots(3, 1, figsize = (18,18))
for i,col in enumerate(cols):
    if col == 'total_baths':
        sns.stripplot(data=train, x=col, y='sale_price', jitter=True, alpha=0.5, palette='Set1', ax = ax[i])
    else:
        sns.boxplot(x = train[col], y = train['sale_price'] , palette='Set1', ax = ax[i])
    ax[i].set_ylabel("Sale Price in Million")
    ax[i].set_xlabel(f"{col}")
    ax[i].set_title(f"Sale Price based on {col}", loc = 'center')
    
plt.show()




view_cols = ['wfnt', 'golf', 'greenbelt', 'noise_traffic',
             'view_rainier', 'view_olympics', 'view_cascades',
             'view_territorial', 'view_skyline', 'view_sound',
             'view_lakewash', 'view_lakesamm', 'view_otherwater',
             'view_other']

n_cols = 4
n_rows = -(-len(view_cols) // n_cols)  # Ceiling division
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
axes = axes.flatten()

for i, col in enumerate(view_cols):
    ax = axes[i]
    sns.boxplot(data=df, x=col, y='sale_price', ax=ax, palette='Set2')
    ax.set_title(f'Sale Price vs {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Sale Price')

for j in range(len(view_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



categorical_cols = ['sale_nbr', 'sale_warning', 'join_status', 'area', 'city', 'subdivision', 'present_use', 'grade', 'fbsmt_grade', 'was_renovated',
                    'condition', 'stories', 'beds', 'golf', 'greenbelt', 'noise_traffic', 'view_rainier', 'view_olympics', 'view_cascades', 'zoning_category',
                    'view_territorial', 'view_skyline', 'view_sound', 'view_lakewash', 'view_lakesamm', 'view_otherwater', 'view_other', 'submarket',
                    'sale_year', 'sale_month', 'sale_dayofyear', 'sale_dayofweek', 'sale_weekofyear', 'sale_hour', 'age_at_sale', 'reno_age_at_sale']

numerical_cols = ['land_val', 'imp_val', 'sqft_lot', 'sqft', 'sqft_1', 'sqft_fbsmt', 'garb_sqft', 'gara_sqft', 'wfnt', 'total_view_score', 'total_sqft', 
                  'sqft_per_story', 'total_val', 'sin_sale_hour', 'cos_sale_hour', 'sin_sale_dayofweek', 'cos_sale_dayofweek', 'sin_sale_weekofyear', 
                  'cos_sale_weekofyear', 'sin_sale_month', 'cos_sale_month', 'sin_sale_dayofyear', 'cos_sale_dayofyear', 'cpi', 'total_baths']

drop_cols = [ 'join_year', 'year_built', 'year_reno', 'bath_full', 'bath_3qtr', 'bath_half', 'source', 'zoning', 'latitude', 'longitude', 'days_since_sale', 'year_gap']

for col in categorical_cols:
    train[col] = train[col].astype('object')
    test[col] = test[col].astype('object')



train.drop(drop_cols, axis= 1, inplace =True)
test.drop(drop_cols, axis= 1, inplace =True)


cols = ['land_val', 'imp_val', 'sqft_lot', 'sqft', 'sqft_1', 'sqft_fbsmt', 'garb_sqft', 'gara_sqft', 'wfnt', 'total_view_score', 'total_sqft', 'sqft_per_story', 'total_val', 'cpi']

sns.heatmap(train[cols].corr())
plt.show()


from sklearn.model_selection import train_test_split

x = train.drop(["sale_price", "id"], axis = 1)
y = train["sale_price"]
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2, random_state = 42)
print("Xtrain:",x_train.shape, "Xval:", x_val.shape)
print("ytrain:",y_train.shape, "yval:", y_val.shape)


from catboost import CatBoostRegressor

cats = x_train.select_dtypes(include=['object', 'category']).columns.tolist()

model_high_cat = CatBoostRegressor(
    loss_function='Quantile:alpha=0.95',
    iterations=10000,
    learning_rate=0.05,
    verbose=500,
    grow_policy = "Depthwise",
    min_data_in_leaf = 1000,
    l2_leaf_reg = 100,
    od_type="IncToDec",
    od_pval=0.1,
    task_type="GPU"
)
model_low_cat = CatBoostRegressor(
    loss_function='Quantile:alpha=0.05',
    iterations=10000,   
    learning_rate=0.05, 
    verbose=500,
    grow_policy = "Depthwise",
    min_data_in_leaf = 1000,
    l2_leaf_reg = 100, 
    od_type="IncToDec",
    od_pval=0.1,
    task_type="GPU"
)

high = model_high_cat.fit(x_train, y_train,cat_features=cats).predict(x_val)
low = model_low_cat.fit(x_train, y_train,cat_features=cats).predict(x_val)

winkler_score(y_val, low, high, return_coverage = True)


""""from lightgbm import LGBMRegressor

model_low_lgbm=LGBMRegressor(n_estimators=500, learning_rate = 0.1, min_data_in_leaf=100,objective='quantile',alpha = 0.05,  task_type="GPU")
model_high_lgbm=LGBMRegressor(n_estimators=500, learning_rate = 0.1,min_data_in_leaf=100,objective='quantile',alpha = 0.95,  task_type="GPU")

high = model_high_lgbm.fit(x_train, y_train, categorical_feature=cats).predict(x_val)
low = model_low_lgbm.fit(x_train, y_train, categorical_feature=cats).predict(x_val)

winkler_score(y_val, low, high, return_coverage = True)"""


import plotly.graph_objects as go

# Sort by actual values
sorted_idx = np.argsort(y_val)
y_sorted = y_val.iloc[sorted_idx].values
low_sorted = low[sorted_idx]
high_sorted = high[sorted_idx]
x = np.arange(len(y_val))

# Create figure
fig = go.Figure()

# Add shaded area for prediction interval
fig.add_trace(go.Scatter(
    x=x, y=high_sorted,
    line=dict(color='rgba(255,0,0,0)'),  # Invisible line
    showlegend=False,
    hoverinfo='skip'
))

fig.add_trace(go.Scatter(
    x=x, y=low_sorted,
    fill='tonexty',  # Fill between this trace and the previous
    fillcolor='rgba(128,128,128,0.2)',
    line=dict(color='rgba(0,0,255,0)'),
    name='Prediction Interval (90%)',
    hoverinfo='skip'
))

# Add lower bound
fig.add_trace(go.Scatter(
    x=x, y=low_sorted,
    line=dict(color='blue', dash='dash', width=0.5),
    name='Lower Bound (5%)'
))

# Add upper bound
fig.add_trace(go.Scatter(
    x=x, y=high_sorted,
    line=dict(color='red', dash='dash', width=0.5),
    name='Upper Bound (95%)'
))

# Add actual values
fig.add_trace(go.Scatter(
    x=x, y=y_sorted,
    line=dict(color='black', width=3),
    name='Actual'
))

# Customize layout
fig.update_layout(
    title='Prediction Intervals with Actual Values',
    xaxis_title='Sample Index (sorted by actual values)',
    yaxis_title='Target Value',
    legend=dict(x=0.01, y=0.99),
    template='simple_white',
    height=500,
    width=900
)

fig.show(renderer='iframe')


x_test = test.drop(["id", "sale_price"],axis =1)

low = np.maximum(0,model_low_cat.predict(x_test))
high = np.maximum(0,model_high_cat.predict(x_test))

out = pd.DataFrame({
    "id": test["id"],
    "pi_lower": low,
    "pi_upper": high
})

out.head()


out.describe().T


out.to_csv("submission.csv", index= False)

