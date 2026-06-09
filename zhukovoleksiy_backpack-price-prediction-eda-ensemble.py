# Misc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import random
import os
import gc
import warnings
import time
from typing import List
from math import sqrt
import polars as pl

# Sklearn classes for model selection, cross validation, and performance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


# Hypertuning
import optuna

# Gradient boosting
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from catboost import CatBoost, CatBoostRegressor
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from copy import deepcopy
from sklearn.metrics import mean_squared_error

# Seaborn
rc = {
    #FAEEE9
    "axes.facecolor": "#243139",
    "figure.facecolor": "#243139",
    "axes.edgecolor": "#000000",
    "grid.color": "#000000",
    "font.family": "arial",
    "axes.labelcolor": "#FFFFFF",
    "xtick.color": "#FFFFFF",
    "ytick.color": "#FFFFFF",
    "grid.alpha": 0.4,
}
sns.set(rc=rc)
#sns.set_palette("YlOrRd")

# Useful line of code to set the display option so we could see all the columns in pd dataframe
pd.set_option('display.max_columns', None)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Functions
def print_sl():
    print("=" * 50)
    print()


# Load Data
train_PATH    = '/kaggle/input/playground-series-s5e2/train.csv'
train_ex_PATH = '/kaggle/input/playground-series-s5e2/training_extra.csv'
test_PATH     = '/kaggle/input/playground-series-s5e2/test.csv'
sub_PATH      = '/kaggle/input/playground-series-s5e2/sample_submission.csv'

train_df      = pd.read_csv(train_PATH)
train_ex_df   = pd.read_csv(train_ex_PATH)
test_df       = pd.read_csv(test_PATH)
sub_df        = pd.read_csv(sub_PATH)

train_df.drop('id',axis=1,inplace=True)
train_ex_df.drop('id',axis=1,inplace=True)
test_df.drop('id',axis=1,inplace=True)

print('Data Loaded Succesfully!')
print_sl()

# Fast Data Check
print(f'Train Data Shape: {train_df.shape}')
print(f'Are there any null values in train? - {train_df.isnull().any().any()}\n')

print(f'Train Data Shape: {train_ex_df.shape}')
print(f'Are there any null values in train? - {train_ex_df.isnull().any().any()}\n')

print(f'Test Data Shape:  {test_df.shape}')
print(f'Are there any null values in test? - {test_df.isnull().any().any()}\n')
print_sl()

# Traget
target = 'Price'

train_df.head()


# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(train_df[target], bins=7, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Prices in train_df', color='white')
axes[0].set_xlabel('Price')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=train_df[target], ax=axes[1])
axes[1].set_title('Box plot of Prices in train_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(train_ex_df[target], bins=7, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Prices in train_ex_df', color='white')
axes[0].set_xlabel('Price')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=train_ex_df[target], ax=axes[1])
axes[1].set_title('Box plot of Prices in train_ex_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


train_ex_df['Compartments'].unique()


# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(train_df['Compartments'], bins=10, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Compartments in train_df', color='white')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=train_df['Compartments'], ax=axes[1])
axes[1].set_title('Box plot of Compartments in train_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(train_ex_df['Compartments'], bins=10, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Compartments in train_ex_df', color='white')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=train_ex_df['Compartments'], ax=axes[1])
axes[1].set_title('Box plot of Compartments in train_ex_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(test_df['Compartments'], bins=10, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Compartments in test_df', color='white')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=test_df['Compartments'], ax=axes[1])
axes[1].set_title('Box plot of Compartments in test_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(train_df['Weight Capacity (kg)'], bins=30, kde=True, ax=axes[0])
axes[0].set_title('Weight Capacity (kg) Distribution in train_df', color='white')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=train_df['Weight Capacity (kg)'], ax=axes[1])
axes[1].set_title('Box plot of Weight Capacity (kg) in train_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(train_ex_df['Weight Capacity (kg)'], bins=30, kde=True, ax=axes[0])
axes[0].set_title('Weight Capacity (kg) Distribution in train_ex_df', color='white')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=train_ex_df['Weight Capacity (kg)'], ax=axes[1])
axes[1].set_title('Box plot of Weight Capacity (kg) in train_ex_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(test_df['Weight Capacity (kg)'], bins=30, kde=True, ax=axes[0])
axes[0].set_title('Weight Capacity (kg) Distribution in test_df', color='white')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=test_df['Weight Capacity (kg)'], ax=axes[1])
axes[1].set_title('Box plot of Weight Capacity (kg) in test_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


# Create a copy of the dataframe
df_encoded = pd.concat([train_ex_df, train_df], axis=0).reset_index(drop=True).copy()

# Assuming these are your categorical variables, including 'outcome'
categorical_vars = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment','Waterproof', 'Style', 'Color']

# Label encode categorical columns
label_encoders = {}
for column in categorical_vars:
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(df_encoded[column])
    label_encoders[column] = le

def plot_correlation_heatmap(df: pd.core.frame.DataFrame, title_name: str = 'Train correlation') -> None:
    excluded_columns = ['id']
    columns_without_excluded = [col for col in df.columns if col not in excluded_columns]
    corr = df[columns_without_excluded].corr()
    
    fig, axes = plt.subplots(figsize=(14, 10))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, linewidths=.5, cmap='mako', annot=True, annot_kws={"size": 6})
    plt.title(title_name, color='white')
    plt.show()

# Plot correlation heatmap for encoded dataframe
plot_correlation_heatmap(df_encoded, 'Encoded Dataset Correlation')


def plot_count(df: pd.core.frame.DataFrame, col: str, title_name: str='Train') -> None:
    # Set background color
    f, ax = plt.subplots(1, 2, figsize=(16, 7))
    plt.subplots_adjust(wspace=0.2)

    s1 = df[col].value_counts()
    N = len(s1)

    outer_sizes = s1
    inner_sizes = s1/N

    colors = sns.color_palette("mako")
    # hex_colors = [matplotlib.colors.to_hex(color) for color in colors]
    # print(hex_colors)
    
    outer_colors = ['#2e1e3b', '#413d7b', '#37659e', '#348fa7', '#40b7ad', '#8bdab2']
    inner_colors = ['#2e1e3b', '#413d7b', '#37659e', '#348fa7', '#40b7ad', '#8bdab2']
    #inner_colors = ['#59b3a3',] #'#433C64']

    ax[0].pie(
        outer_sizes,colors=outer_colors, 
        labels=s1.index.tolist(), 
        startangle=90, frame=True, radius=1.3, 
        explode=([0.05]*(N-1) + [.3]),
        wedgeprops={'linewidth' : 1, 'edgecolor' : 'black'}, 
        textprops={'fontsize': 12, 'weight': 'bold', 'color': 'white'}
    )

    textprops = {
        'size': 13, 
        'weight': 'bold', 
        'color': 'white'
    }

    ax[0].pie(
        inner_sizes, colors=inner_colors,
        radius=1, startangle=90,
        autopct='%1.f%%', explode=([.1]*(N-1) + [.3]),
        pctdistance=0.8, textprops=textprops
    )

    center_circle = plt.Circle((0,0), .68, color='black', fc='#243139', linewidth=0)
    ax[0].add_artist(center_circle)

    x = s1
    y = s1.index.tolist()
    sns.barplot(
        x=x, y=y, ax=ax[1],
        palette=colors, orient='horizontal'
    )

    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].tick_params(
        axis='x',         
        which='both',      
        bottom=False,       
        labelbottom=False
    )

    for i, v in enumerate(s1):
        ax[1].text(v, i+0.1, str(v), color='white', fontweight='bold', fontsize=12)

    plt.setp(ax[1].get_yticklabels(), fontweight="bold")
    plt.setp(ax[1].get_xticklabels(), fontweight="bold")
    ax[1].set_xlabel(col, fontweight="bold", color='white')
    ax[1].set_ylabel('count', fontweight="bold", color='white')

    f.suptitle(f'{title_name}', fontsize=14, fontweight='bold', color='white')
    plt.tight_layout() 
    plt.show()


#train_tg = pd.concat([train_ex_df, train_df], axis=0).reset_index(drop=True).copy()
train_tg = train_df.reset_index(drop=True).copy()
train_tg.head()


plot_count(train_tg, 'Brand', 'Brand Distribution of Train Data')


plot_count(train_tg, 'Material', 'Material Distribution of Train Data')


plot_count(train_tg, 'Size', 'Size Distribution of Train Data')


plot_count(train_tg, 'Laptop Compartment', 'Laptop Compartment Distribution of Train Data')


plot_count(train_tg, 'Waterproof', 'Waterproof Distribution of Train Data')


plot_count(train_tg, 'Style', 'Style Distribution of Train Data')


plot_count(train_tg, 'Color', 'Color Distribution of Train Data')


train_df = train_tg


orig = pd.read_csv("/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv")
orig = orig.loc[(orig["Weight Capacity (kg)"]>5)&(orig["Weight Capacity (kg)"]<30)]
orig.columns = [f"orig_{c}" for c in orig.columns]
train = train_df.merge(orig.iloc[:,:], left_on="Weight Capacity (kg)", right_on="orig_Weight Capacity (kg)", how="left")
#train = train_df.drop("id",axis=1)
test = test_df.merge(orig.iloc[:,:], left_on="Weight Capacity (kg)", right_on="orig_Weight Capacity (kg)", how="left")
train.head()


# Testing
dict_fen = {'Material':'NaN','Style':'NaN','Brand':'NaN','Size':'NaN','Waterproof':'NaN','Color':'NaN','Laptop Compartment':'NaN'}

def feh(df):
    df = df.fillna(dict_fen)

    map_size       = {'Small':1.1,'Medium':1.2,'Large':1.3,'NaN':0}
    map_brand      = {'Jansport':1.1,'Adidas':1.2,'Nike':1.3,'Puma':1.4,'Under Armour':1.5,'NaN':0}
    map_color      = {'Black':1.1,'Green':1.2,'Red':1.3,'Blue':1.4,'Gray':1.05,'Pink':1.5,'NaN':0}
    map_style      = {'Messenger':1.1,'Backpack':1.2,'Tote':1.3,'NaN':0}
    map_material   = {'Polyester':1.1,'Leather':1.2,'Nylon':1.3,'Canvas':1.4,'NaN':0}
    map_waterproof = {'Yes':1.1,'No':1.0,'NaN':0}
    map_laptop     = {'Yes':1.1,'No':1.0,'NaN':0}
    
    df['Size_map']        = df['Size'].map(map_size)
    df['Brand_map']       = df['Brand'].map(map_brand)
    df['Color_map']       = df['Color'].map(map_color)
    df['Style_map']       = df['Style'].map(map_style)
    df['Material_map']    = df['Material'].map(map_material)
    df['Waterproof_map']  = df['Waterproof'].map(map_waterproof)
    df['Laptop_map']      = df['Laptop Compartment'].map(map_laptop)
    df['Compartments_map']= df['Compartments'].apply(lambda x: x/1.1)

    df = df.rename(columns={'Size_map':'x1', 'Brand_map':'x2', 'Color_map':'x3', 
                            'Style_map':'x4', 'Material_map':'x5', 'Waterproof_map':'x6', 
                            'Laptop_map':'x7', 'Compartments_map':'x8'}) 

    polar_df = pl.from_pandas(df)
    polar_df = polar_df.with_columns(
        _2_1=((pl.col('x1')-pl.col('x3'))**2 + (pl.col('x2')-pl.col('x4'))**2).sqrt(),
        _2_2=((pl.col('x1')-pl.col('x5'))**2 + (pl.col('x2')-pl.col('x6'))**2).sqrt(),
        _2_3=((pl.col('x1')-pl.col('x7'))**2 + (pl.col('x2')-pl.col('x8'))**2).sqrt(),
        _3_1=((pl.col('x1')-pl.col('x4'))**2 + (pl.col('x2')-pl.col('x5'))**2 + (pl.col('x3')-pl.col('x6'))**2).sqrt(),
        _3_2=((pl.col('x1')-pl.col('x7'))**2 + (pl.col('x2')-pl.col('x8'))**2).sqrt(),
        _3_3=((pl.col('x4')-pl.col('x7'))**2 + (pl.col('x5')-pl.col('x8'))**2).sqrt(),
        _4_1=((pl.col('x1')-pl.col('x5'))**2 + (pl.col('x2')-pl.col('x6'))**2 + 
              (pl.col('x3')-pl.col('x7'))**2 + (pl.col('x4')-pl.col('x8'))**2).sqrt()
    )

    # Standardize the weight capacity. Alternatively, you could create bins.
    scaler = StandardScaler()
    df["Weight_Capacity_scaled"] = scaler.fit_transform(df[["Weight Capacity (kg)"]])
    df["Weight_Capacity_scaled"] = scaler.transform(df[["Weight Capacity (kg)"]])
    
    df = polar_df.to_pandas()
    return df
    
#train = feh(train_tg)
#test  = feh(test_df)

train = feh(train)
test  = feh(test)


# # Define imputation strategies
# categorical_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
# numerical_features = ["Weight Capacity (kg)"]

# # Fill categorical missing values with mode (most frequent value)
# for col in categorical_features:
#     train_tg[col].fillna(train_tg[col].mode()[0], inplace=True)
#     test_df[col].fillna(test_df[col].mode()[0], inplace=True)

# # Fill numerical missing values with median
# for col in numerical_features:
#     train_tg[col].fillna(train_tg[col].median(), inplace=True)
#     test_df[col].fillna(test_df[col].median(), inplace=True)


# Calculate aggregated statistics for each brand in training data
# brand_stats = train_tg.groupby("Brand")["Price"].agg(["mean", "median", "std"]).reset_index()
# brand_stats.columns = ["Brand", "Brand_mean_price", "Brand_median_price", "Brand_std_price"]

brand_stats = train.groupby("Brand")["Price"].agg(["mean", "median", "std"]).reset_index()
brand_stats.columns = ["Brand", "Brand_mean_price", "Brand_median_price", "Brand_std_price"]

# Merge aggregated stats to both train and test datasets
#train_data = train_tg.merge(brand_stats, on="Brand", how="left")
#test_data = test_df.merge(brand_stats, on="Brand", how="left")

train_data = train.merge(brand_stats, on="Brand", how="left")
test_data = test.merge(brand_stats, on="Brand", how="left")


# In test_data, if any brand is missing (new brand), you might fill with global stats:
global_mean = train_data["Price"].mean()
global_median = train_data["Price"].median()
global_std = train_data["Price"].std()
test_data["Brand_mean_price"].fillna(global_mean, inplace=True)
test_data["Brand_median_price"].fillna(global_median, inplace=True)
test_data["Brand_std_price"].fillna(global_std, inplace=True)

# Calculate average price per material and create bins
material_avg = train_data.groupby("Material")["Price"].mean().reset_index().rename(columns={"Price": "Material_avg_price"})
train_data = train_data.merge(material_avg, on="Material", how="left")
test_data = test_data.merge(material_avg, on="Material", how="left")

# Bin the materials based on quantiles of average price
material_bins = pd.qcut(train_data["Material_avg_price"], q=3, labels=["Low", "Medium", "High"])
train_data["Material_price_bin"] = material_bins

# For test set, use the same bin edges as training (here, using pd.cut with bins from quantiles)
bin_edges = pd.qcut(train_data["Material_avg_price"], q=3, retbins=True)[1]
test_data["Material_price_bin"] = pd.cut(test_data["Material_avg_price"], bins=bin_edges, labels=["Low", "Medium", "High"], include_lowest=True)

# Convert Labtop Compartment and Waterproof
binary_map = {"Yes": 1, "No": 0}
train_data["Laptop Compartment"] = train_data["Laptop Compartment"].map(binary_map)
test_data["Laptop Compartment"] = test_data["Laptop Compartment"].map(binary_map)
train_data["Waterproof"] = train_data["Waterproof"].map(binary_map)
test_data["Waterproof"] = test_data["Waterproof"].map(binary_map)

# Standardize the weight capacity. Alternatively, you could create bins.
scaler = StandardScaler()
train_data["Weight_Capacity_scaled"] = scaler.fit_transform(train_data[["Weight Capacity (kg)"]])
test_data["Weight_Capacity_scaled"] = scaler.transform(test_data[["Weight Capacity (kg)"]])

train = train_data.copy()
test = test_data.copy()


# # One-hot Encoding:
# cat_features_to_encode = ["Brand", "Material", "Size", "Style", "Color", "Material_price_bin"]

# # Use pandas get_dummies for simplicity
# train_data_encoded = pd.get_dummies(train_data, columns=cat_features_to_encode, drop_first=True)
# test_data_encoded = pd.get_dummies(test_data, columns=cat_features_to_encode, drop_first=True)

# # Ensure both train and test have the same dummy columns
# train_cols = set(train_data_encoded.columns)
# test_cols = set(test_data_encoded.columns)
# for col in train_cols - test_cols:
#     test_data_encoded[col] = 0
# for col in test_cols - train_cols:
#     train_data_encoded[col] = 0

# # Sort columns to maintain consistent order
# train_data_encoded = train_data_encoded.sort_index(axis=1)
# test_data_encoded = test_data_encoded.sort_index(axis=1)

# # Now, your data is ready for model training.
# print("Feature engineering and encoding complete!")
# print_sl()


# train = train_data_encoded.copy()

# # For testing, ensure the target column is dropped if it exists
# if "Price" in test_data_encoded.columns:
#     test = test_data_encoded.drop("Price", axis=1)
# else:
#     test = test_data_encoded.copy()

# y_train = train['Price']
# X_train = train.drop(['Price'], axis=1)

# print("Train and test sets are ready for model training!")
# print_sl()

# X_train.head()


# One-hot Encoding:
cat_features_to_encode = ["Material_price_bin", 'Material', 'Style', 'Brand', 'Size', 'Waterproof', 'Color', 'Laptop Compartment', 
                         'orig_Material', 'orig_Style', 'orig_Brand', 'orig_Size', 'orig_Waterproof', 'orig_Color', 'orig_Laptop Compartment']

# Use pandas get_dummies for simplicity
train = pd.get_dummies(train, columns=cat_features_to_encode, drop_first=True)
test = pd.get_dummies(test, columns=cat_features_to_encode, drop_first=True)

# Ensure both train and test have the same dummy columns
train_cols = set(train.columns)
test_cols = set(train.columns)
for col in train_cols - test_cols:
    train[col] = 0
for col in test_cols - train_cols:
    train[col] = 0

# Sort columns to maintain consistent order
train = train.sort_index(axis=1)
test = test.sort_index(axis=1)

# Now, your data is ready for model training.
print("Feature engineering and encoding complete!")
print_sl()


# For testing, ensure the target column is dropped if it exists
if "Price" in test.columns:
    test = test.drop("Price", axis=1)
else:
    test = test.copy()

y_train = train['Price']
X_train = train.drop(['Price'], axis=1)

print("Train and test sets are ready for model training!")
print_sl()


for col in X_train.columns:
    if X_train[col].dtype in ['float64', 'int64']:
        X_train[col].fillna(X_train[col].median(), inplace=True)
    else:
        X_train[col].fillna(X_train[col].mode()[0], inplace=True)

for col in test.columns:
    if test[col].dtype in ['float64', 'int64']:
        test[col].fillna(test[col].median(), inplace=True)
    else:
        test[col].fillna(test[col].mode()[0], inplace=True)

X_train.head()


#plot_correlation_heatmap(train, 'Preprocessed Dataset Correlation')


# NOT RELEVANT FOR NOW

class Regressor:
    def __init__(self, n_estimators=100, device="cpu", random_state=0):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.models = self._define_model()
        self.models_name = list(self._define_model().keys())
        self.len_models = len(self.models)
        
    def _define_model(self):
        
        xgb_params = {
            'n_estimators': self.n_estimators,
            'max_depth': 7,
            'learning_rate': 0.0116,
            'colsample_bytree': 1,
            'subsample': 0.6085,
            'min_child_weight': 9,
            'reg_lambda': 4.879e-07,
            'max_bin': 431,
            'n_jobs': -1,
            'eval_metric': 'mae',
            'objective': "reg:squarederror",
            'verbosity': 0,
            'random_state': self.random_state,
        }
        if self.device == 'gpu':
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['predictor'] = 'gpu_predictor'
        xgb_exact_params = xgb_params.copy()
        xgb_exact_params['tree_method'] = 'exact'
        xgb_approx_params = xgb_params.copy()
        xgb_approx_params['tree_method'] = 'approx'
        
        lgb_params = {
            'n_estimators': self.n_estimators,
            'max_depth': 7,
            "num_leaves": 16,
            'learning_rate': 0.05,
            'subsample': 0.60,
            'colsample_bytree': 1,
            'reg_alpha': 0.25,
            'reg_lambda': 5e-07,
            'objective': 'regression_l1',
            'metric': 'mean_squared_error',
            'boosting_type': 'gbdt',
            'device': self.device,
            'verbosity': -1,
            'random_state': self.random_state
        }
        lgb2_params = {
            'n_estimators': self.n_estimators,
            'num_leaves': 93, 
            'min_child_samples': 20, 
            'learning_rate': 0.05533790147941807, 
            'colsample_bytree': 0.8809128870084636, 
            'reg_alpha': 0.0009765625, 
            'reg_lambda': 0.015589408048174165,
            'objective': 'regression_l1',
            'metric': 'mean_absolute_error',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state
        }
        lgb3_params = {
            'n_estimators': self.n_estimators,
            'num_leaves': 45,
            'max_depth': 13,
            'learning_rate': 0.0684383311038932,
            'subsample': 0.5758412171285148,
            'colsample_bytree': 0.8599714680300794,
            'reg_lambda': 1.597717830931487e-08,
            'objective': 'regression_l1',
            'metric': 'mean_absolute_error',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state,
            'force_col_wise': True
        }
        lgb_goss_params = lgb_params.copy()
        lgb_goss_params['boosting_type'] = 'goss'
        lgb_dart_params = lgb_params.copy()
        lgb_dart_params['boosting_type'] = 'dart'
        lgb_dart_params['n_estimators'] = 500
                
        cb_params = {
            'iterations': self.n_estimators,
            'depth': 8,
            'learning_rate': 0.01,
            'l2_leaf_reg': 0.7,
            'random_strength': 0.2,
            'max_bin': 200,
            'od_wait': 65,
            'one_hot_max_size': 70,
            'grow_policy': 'Depthwise',
            'bootstrap_type': 'Bayesian',
            'od_type': 'Iter',
            'eval_metric': 'RMSE',
            'loss_function': 'RMSE',
            'task_type': self.device.upper(),
            'random_state': self.random_state
        }
        cb2_params = {
            'iterations': self.n_estimators,
            'depth': 9, 
            'learning_rate': 0.456,
            'l2_leaf_reg': 8.41,
            'random_strength': 0.18,
            'max_bin': 225, 
            'od_wait': 58, 
            'grow_policy': 'Lossguide',
            'bootstrap_type': 'Bayesian',
            'od_type': 'Iter',
            'eval_metric': 'MAE',
            'loss_function': 'MAE',
            'task_type': self.device.upper(),
            'random_state': self.random_state
        }
        cb3_params = {
            'n_estimators': self.n_estimators,
            'depth': 11,
            'learning_rate': 0.08827842054729117,
            'l2_leaf_reg': 4.8351074756668864e-05,
            'random_strength': 0.21306687539993183,
            'max_bin': 483,
            'od_wait': 97,
            'grow_policy': 'Lossguide',
            'bootstrap_type': 'Bayesian',
            'od_type': 'Iter',
            'eval_metric': 'MAE',
            'loss_function': 'MAE',
            'task_type': self.device.upper(),
            'random_state': self.random_state,
            'silent': True
        }
        cb_sym_params = cb_params.copy()
        cb_sym_params['grow_policy'] = 'SymmetricTree'
        cb_loss_params = cb_params.copy()
        cb_loss_params['grow_policy'] = 'Lossguide'
        
        models = {
            #"xgb": xgb.XGBRegressor(**xgb_params),
            #"xgb_exact": xgb.XGBRegressor(**xgb_exact_params),
            #"xgb_approx": xgb.XGBRegressor(**xgb_approx_params),
            #"lgb": lgb.LGBMRegressor(**lgb_params),
            "lgb_": lgb.LGBMRegressor(verbosity=-1),
            #"lgb2": lgb.LGBMRegressor(**lgb2_params),
            #"lgb3": lgb.LGBMRegressor(**lgb3_params),
            "cat": CatBoostRegressor(**cb_params),
            #"cat2": CatBoostRegressor(**cb2_params),
            #"cat3": CatBoostRegressor(**cb3_params),
            #"cat_sym": CatBoostRegressor(**cb_sym_params),
            #"cat_loss": CatBoostRegressor(**cb_loss_params),
            #"Ridge": RidgeCV(),
            #"Lasso": LassoCV(),
            #"RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=self.random_state, n_jobs=-1),
            #"PLSRegression": PLSRegression(n_components=10, max_iter=2000),
            #"PassiveAggressiveRegressor": PassiveAggressiveRegressor(max_iter=3000, tol=1e-3, n_iter_no_change=30, random_state=self.random_state),
            #"GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=2000, learning_rate=0.05, loss="absolute_error", random_state=self.random_state),
            #"HistGradientBoostingRegressor": HistGradientBoostingRegressor(max_iter=self.n_estimators, learning_rate=0.01, loss="absolute_error", n_iter_no_change=300,random_state=self.random_state),
            #"ARDRegression": ARDRegression(n_iter=1000),
            #"HuberRegressor": HuberRegressor(max_iter=3000),
            #"KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
        }
        
        return models


# # drop everything insignificant
# cols_to_keep = ["Brand_mean_price", "Brand_median_price", "Brand_std_price", 
#                 "Laptop Compartment_1.0", "Material_avg_price", 
#                 "Weight Capacity (kg)", "Weight_Capacity_scaled", 
#                 "orig_Weight Capacity (kg)", "orig_Price"]

# # Keep only the desired columns
# X_train = X_train[cols_to_keep]
# test = test[cols_to_keep]


# cats = ['Brand_Jansport','Brand_NaN', 'Brand_Nike',	'Brand_Puma', 'Brand_Under Armour', 'Color_Blue', 'Color_Gray',
#         'Color_Green', 'Color_NaN', 'Color_Pink', 'Color_Red', 'Laptop Compartment_No',	'Laptop Compartment_Yes',
#         'Material_Leather',	'Material_NaN', 'Material_Nylon', 'Material_Polyester',	'Size_Medium', 'Size_NaN', 
#         'Size_Small', 'Style_Messenger', 'Style_NaN', 'Style_Tote', 'Waterproof_No', 'Waterproof_Yes']
# # Configurations
kfold = True
n_splits = 5 if kfold else 1
random_state = 42
n_estimators = 700
early_stopping_rounds = 666
verbose = False

# Assuming X_train, y_train, and test are already defined
kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# Arrays to store predictions
test_pred_total = np.zeros(test.shape[0])
oof_pred_total = np.zeros(X_train.shape[0])
ensemble_scores = []

fold_idx = 0
start_time = time.time()

print("Starting ensemble training...")

for train_index, val_index in kf.split(X_train):
    fold_start_time = time.time()
    fold_idx += 1
    print(f"\n=== Fold {fold_idx} ===")
    
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

    # Define models
    models = {
        "LGBM": lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            random_state=random_state,
            n_estimators=1500,
            max_depth=7,
            num_leaves=16,
            learning_rate=0.05,
            subsample=0.60,
            colsample_bytree=1,
            reg_alpha=0.25,
            reg_lambda=5e-07,
            boosting_type='gbdt',
            verbosity = -1,
        ),
        
        "CatBoost_optuna": cb.CatBoostRegressor(
            random_state = random_state,
            learning_rate = 0.2757572018124821,
            depth = 5, 
            l2_leaf_reg = 7.247059958722575, 
            n_estimators = 666,
            #cat_features = cats
        ),
        
        "LGBM_optuna": lgb.LGBMRegressor(
            learning_rate = 0.03125634355299445, 
            max_depth = 4, 
            num_leaves = 69, 
            min_data_in_leaf = 84, 
            reg_alpha = 0.0003940341792750523, 
            reg_lambda = 2.2071774712491142e-08, 
            n_estimators = 1000,
            verbosity = -1,
        ),

        "CatBoost_optuna1": cb.CatBoostRegressor(
            iterations = 1038,
            depth = 7, 
            learning_rate = 0.05402571213309853,
            l2_leaf_reg = 0.25707861461291215,
            random_strength = 0.5892798244734105,
            bagging_temperature = 0.3700245272388676, 
            border_count = 223
        ),
    }

    val_preds_list = []
    test_preds_list = []

    for name, model in models.items():
        print(f"Training {name}...")
        if isinstance(model, lgb.LGBMRegressor):
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose)])
        else:
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=verbose)

        # Predictions
        val_preds = model.predict(X_val)
        test_preds = model.predict(test)

        val_preds_list.append(val_preds)
        test_preds_list.append(test_preds)

    # -----------------
    # Simple Ensemble: Average Predictions
    # -----------------
    val_preds_ensemble = np.mean(val_preds_list, axis=0)
    test_preds_ensemble = np.mean(test_preds_list, axis=0)

    # Store predictions
    oof_pred_total[val_index] = val_preds_ensemble
    test_pred_total += test_preds_ensemble / n_splits

    # Calculate RMSE for current fold
    fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds_ensemble))
    ensemble_scores.append(fold_rmse)
    print(f"Fold {fold_idx} RMSE: {fold_rmse:.5f} (Fold time: {time.time()-fold_start_time:.1f} sec)")

    gc.collect()

total_time = time.time() - start_time
print(f"\nTraining complete. Total time: {total_time/60:.2f} minutes.")
print(f"Average RMSE: {np.mean(ensemble_scores):.5f}")

# test_pred_total holds the ensemble test predictions



# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(test_pred_total, bins=10, kde=True, ax=axes[0])
axes[0].set_title('Distribution of preds')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=test_pred_total, ax=axes[1])
axes[1].set_title('Box plot of preds')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


sub = pd.read_csv(sub_PATH)
#sub[f'{target}'] = test_pred_total

# public solutions
SAE_TE_XGB = pd.read_csv("/kaggle/input/s5e2-sae-te-xgb/submission.csv")
ps_A_XGB = pd.read_csv("/kaggle/input/ps-s5e2-hyperspace-as-feats-a-xgboost-lightgbm/submission.csv")

ps_A_XGB.columns = ps_A_XGB.columns.str.strip()
SAE_TE_XGB.columns = SAE_TE_XGB.columns.str.strip()


sub['Price'] = ps_A_XGB['Price'] * 0.25 + SAE_TE_XGB['Price'] * 0.5 + test_pred_total * 0.25

sub.to_csv('submission.csv', index=False)
sub

