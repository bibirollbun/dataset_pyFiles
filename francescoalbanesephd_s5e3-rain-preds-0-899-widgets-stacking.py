# %%capture
!pip install itables

import pandas as pd
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from itables import init_notebook_mode, show
init_notebook_mode(all_interactive=False,connected=True)

# Sets the seed for reproducibility in numpy, random, torch CPU, and CUDA.
SEED = 12
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED) # For multi-GPU setups.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False #May slightly slow down training, but ensures reproducibility


# Import datasets
TRAIN_DF = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv',index_col='id')
TEST_DF = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv',index_col='id')
ORIGINAL_DATA = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")


for dataset in [TRAIN_DF, TEST_DF, ORIGINAL_DATA]:
    print(dataset.shape)


print("="*30,"Show Training Dataset for initial data assessment","="*30)
show(TRAIN_DF)


import missingno as msno
plt.figure(figsize=(5,8))

# Visualize missing values as a matrix 
for df, df_name in [
    (TRAIN_DF, "Train"), 
    (TEST_DF, "Test"), 
    (ORIGINAL_DATA, "Original")]:
    
    msno.matrix(df,figsize=(20,8))
    plt.title(f"Missing Data in {df_name} Dataset: {df.isnull().sum().sum()}", fontsize = 30)
    plt.show()


# Visualize missing values as a matrix 
for df, df_name in [
    (TRAIN_DF, "Train"), 
    (TEST_DF, "Test"), 
    (ORIGINAL_DATA, "Original")]:
    
    print("\n")
    print("="*50,f"Description of {df_name} Dataset","="*50)
    unique_df = pd.DataFrame(df.nunique(),columns=['n_unique'])

    # Round the describe output before styling
    descriptive_stats = df.describe().T
    
    display(pd.concat([descriptive_stats,unique_df],axis=1).T
            .T.style.background_gradient(cmap='Blues'))


# Median impute TEST_DF missing value since there is only one
median_test_winddirection = TEST_DF['winddirection'].median()
TEST_DF.fillna(median_test_winddirection,inplace=True)


# Identify Target
target = 'rainfall'

# Countplot
plt.figure(figsize=(8,5))
sns.set_theme(style='whitegrid')

sns.countplot(x = TRAIN_DF[target],palette='coolwarm_r')

plt.xlabel('Rainfall')
plt.ylabel('Count')
plt.title('Distribution of Target Variable')
plt.show()


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Analysis of all NUMERICAL features
# Define a custom color palette
custom_palette = ['#219ebc', '#c1121f']

# Function to create and display plots for a single numerical variable
def create_variable_plots(train, test, variable):

    # Merge data for visualization (without modifying original DataFrames)
    train_temp = train.copy()
    test_temp = test.copy()
    train_temp["Dataset"] = "Train"
    test_temp["Dataset"] = "Test"
    combined_data = pd.concat([train_temp, test_temp])

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    annot_kws = {'xy': (0.03, 0.75), 'xycoords': 'axes fraction', 'fontsize': 10}

    # Box plot
    sns.boxplot(data=combined_data, x=variable, y="Dataset", palette=custom_palette, ax=axes[0])
    axes[0].set_xlabel(variable)
    axes[0].set_title(f"Box Plot of {variable}")

    # Histogram
    sns.histplot(data=train, x=variable, color=custom_palette[0], kde=True, bins=30, label="Train", ax=axes[1])
    sns.histplot(data=test, x=variable, color=custom_palette[1], kde=True, bins=30, label="Test", ax=axes[1])
    axes[1].set_xlabel(variable)
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Histogram of {variable} [Train, Test]")
    axes[1].legend()
    axes[1].annotate(f"Skewness (TRAIN): {train[variable].skew():.2f}\nKurtosis (TRAIN): {train[variable].kurt():.2f}",
                     xy=annot_kws['xy'], xycoords=annot_kws['xycoords'], fontsize=annot_kws['fontsize'])


    # Adjust spacing and show
    plt.tight_layout()
    plt.show()

# Perform univariate analysis for each numerical variable
for variable in TRAIN_DF.columns.difference(["rainfall"]):
    create_variable_plots(TRAIN_DF, TEST_DF, variable)


# KDE plot for Feature-Target Relationship
plt.figure(figsize=(14, 10))
for i, col in enumerate(TRAIN_DF.columns.difference(['day','rainfall']), 1):
    plt.subplot(3, 4, i)
    sns.kdeplot(data=TRAIN_DF, x=col, hue='rainfall', fill=True, common_norm=False, alpha=0.2) # common_norm=False: Normalize each KDE curve independently for each 'rainfall' category
    plt.title(f'Distribution of {col} by Rainfall')
plt.tight_layout()
plt.show()


# Pairplot to visualize relationships between variables in the TRAIN_DF DataFrame
sns.pairplot(TRAIN_DF[TRAIN_DF.columns.difference(['day'])], hue='rainfall', palette='coolwarm', kind='scatter', diag_kind='kde')
plt.show()


# Create a heatmap to visualize the correlation matrix of the TRAIN_DF DataFrame
plt.figure(figsize=(12,8))
sns.heatmap(data=TRAIN_DF.corr().round(4), annot=True, cmap="coolwarm", linewidth = 0.3,)


from sklearn.model_selection import train_test_split

# Define X and y
X = TRAIN_DF.copy()
y = X.pop(target)

# Define the training and validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=SEED, shuffle=False)

# Print the shapes of the datasets
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}")

# Rename test_df (optional)
X_test = TEST_DF.copy()


def iqr_outlier_capping(train, valid=None, test=None, columns=None):
    """
    Applies IQR-based outlier capping to specified columns of one, two, or three DataFrames.

    Parameters:
        train (pd.DataFrame): The training DataFrame used to calculate IQR thresholds.
        valid (pd.DataFrame, optional): The validation DataFrame to cap using train thresholds.
        test (pd.DataFrame, optional): The test DataFrame to cap using train thresholds.
        columns (list, optional): List of column names to apply capping to. If None, applies to all numerical columns.

    Returns:
        tuple: A tuple containing:
            - train_capped (pd.DataFrame): Capped training DataFrame.
            - valid_capped (pd.DataFrame or None): Capped validation DataFrame (if provided).
            - test_capped (pd.DataFrame or None): Capped test DataFrame (if provided).

    Note: Make sure there are no nans
    """
    train_capped = train.copy() # Avoid modifying the original DataFrame
    valid_capped = valid.copy() if valid is not None else None
    test_capped = test.copy() if test is not None else None
    
    if columns is None:
        columns = train.select_dtypes(include='number').columns.tolist()  # All numerical columns
    
    # Calculate IQR-based thresholds from the training set
    for col in columns:
        Q1 = np.percentile(train[col], 25)
        Q3 = np.percentile(train[col], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Show Values
        print(f'Columns {col}: \tLower Bound is: {lower_bound:.2f} \tUpper Bound is: {upper_bound:.2f}')
    
        # Cap outliers in the training set
        train_capped[col] = np.clip(train_capped[col], lower_bound, upper_bound)
    
        # If validation set is provided, cap using training set thresholds
        if valid is not None:
            valid_capped[col] = np.clip(valid[col], lower_bound, upper_bound)
    
        # If test set is provided, cap using training set thresholds
        if test is not None:
            test_capped[col] = np.clip(test[col], lower_bound, upper_bound)

    return train_capped, valid_capped, test_capped

# Cap target outliers in train and validation sets
train_capped, valid_capped, test_capped = iqr_outlier_capping(X_train, X_valid, X_test)


from scipy import stats

def recommend_transformations(data, variables, transformations):
    """
    Checks transformations for each variable and recommends the one that minimizes absolute skewness.

    Args:
        data (pd.DataFrame): The DataFrame containing the variables.
        variables (list): List of variable names to check.
        transformations (list): List of transformation names to apply.

    Returns:
        pd.DataFrame: DataFrame with variable names and recommended transformations.
    """
    df = data.copy()
    
    results = []
    for variable in variables:
        skewness = {'original': abs(df[variable].skew())}
        
        for transformation in transformations:
            transformed_variable = f"{variable}_{transformation}"
            if transformation == 'log1p':
                df[transformed_variable] = np.log1p(df[variable])
            elif transformation == 'sqrt':
                df[transformed_variable] = np.sqrt(df[variable])
            elif transformation == 'square':
                df[transformed_variable] = np.square(df[variable])
            elif transformation == 'yeojohnson':
                df[transformed_variable], _ = stats.yeojohnson(df[variable])
            elif transformation == 'boxcox':
                # Box-Cox requires positive data
                if(df[variable] > 0).all():
                    df[transformed_variable], _ = stats.boxcox(data[variable])
                else:
                    skewness[transformation] = float('inf') # set to inf if boxcox cannot be used
                    continue
            elif transformation == 'reflect_log':
                min_val = df[variable].min()
                df[transformed_variable] = np.log1p(df[variable] - min_val)
            elif transformation == 'reflect_sqrt':
                min_val = df[variable].min()
                df[transformed_variable] = np.sqrt(df[variable] - min_val)
            else: 
                continue
            skewness[transformation] = abs(df[transformed_variable].skew())

        best_transformation = min(skewness, key=skewness.get)
        results.append({'variable': variable, 'abs_original_skewness':skewness['original'], 'recommended_transformation': best_transformation,'abs_transformation_skewness':skewness[best_transformation]})

    return pd.DataFrame(results)

# Run function
transform_df = recommend_transformations(data = train_capped, 
                                         variables = train_capped.columns.difference(['rainfall']), 
                                         transformations = ['log1p', 'sqrt', 'square', 'yeojohnson', 'boxcox', 'reflect_log', 'reflect_sqrt']).set_index(['variable'])

transform_df.style.background_gradient(cmap='coolwarm',vmin=0, vmax=1)


# Create a dictionary to store recommended transformations for skewed features.
transform_dict = {}

# Iterate through columns in the training data (excluding 'rainfall' and 'winddirection').
for var in train_capped.columns.difference(['rainfall', 'winddirection']):
    if transform_df['abs_original_skewness'].loc[var] > 0.5:
        transform_dict[var] = transform_df['recommended_transformation'].loc[var]

transform_dict


from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def transform_var(data, transformation, variable, custom_palette):
    """
    Plots the original and transformed variable histograms with skewness and kurtosis annotations.
    Supports various transformations: log1p, sqrt, square, yeojohnson, boxcox, and reflect.
    """
    df = data.copy()
    transformed_variable = f"{variable}_{transformation}"
 
    if transformation == 'log1p':
        df[transformed_variable] = np.log1p(df[variable])
    elif transformation == 'sqrt':
        df[transformed_variable] = np.sqrt(df[variable])
    elif transformation == 'square':
        df[transformed_variable] = np.square(df[variable])
    elif transformation == 'yeojohnson':
        df[transformed_variable], _ = stats.yeojohnson(df[variable])
    elif transformation == 'boxcox':
        df[transformed_variable], _ = stats.boxcox(df[variable])
    elif transformation == 'reflect_log':
        min_val = df[variable].min()
        df[transformed_variable] = np.log1p(df[variable] - min_val)
    elif transformation == 'reflect_sqrt':
        min_val = df[variable].min()
        df[transformed_variable] = np.sqrt(df[variable] - min_val)
    else:
        raise ValueError(f"Unsupported transformation: {transformation}")

    # Original var stats
    original_skew = df[variable].skew()
    original_kurt = df[variable].kurt()

    # Transformed var stats
    transformed_skew = df[transformed_variable].skew()
    transformed_kurt = df[transformed_variable].kurt()

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    annot_kws = {'xy': (0.06, 0.8), 'xycoords': 'axes fraction', 'fontsize': 10}

    # Plot the first histogram (original data)
    sns.histplot(x=df[variable], color=custom_palette[0], kde=True, bins=30, label="Original Var", ax=axes[0])
    axes[0].set_xlabel(variable)
    axes[0].set_ylabel("Frequency (Original)")
    axes[0].legend()
    axes[0].annotate(f"Skewness: {original_skew:.2f}\nKurtosis: {original_kurt:.2f}",
                     xy=annot_kws['xy'], xycoords=annot_kws['xycoords'], fontsize=annot_kws['fontsize'])

    # Plot the second histogram (transformed data)
    sns.histplot(x=df[transformed_variable], color=custom_palette[1], kde=True, bins=30, label="Transformed Var", ax=axes[1])
    axes[1].set_xlabel(transformed_variable)
    axes[1].set_ylabel("Frequency (Transformed)")
    axes[1].legend()
    axes[1].annotate(f"Skewness: {transformed_skew:.2f}\nKurtosis: {transformed_kurt:.2f}",
                     xy=annot_kws['xy'], xycoords=annot_kws['xycoords'], fontsize=annot_kws['fontsize'])

    plt.tight_layout()
    plt.show()

# Use widget to circle through different var*transformation combinations
interact(transform_var,
         data=fixed(train_capped),
         transformation = ['log1p', 'sqrt', 'square', 'yeojohnson', 'boxcox', 'reflect_log', 'reflect_sqrt'],
         variable = train_capped.columns.difference(['rainfall']),
         custom_palette=fixed(custom_palette))


import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler

def dayofyear_to_month(df):
    """
    Converts day of year (1-365) to month (1-12).
    """
    month_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    
    def get_month(day):
        for month in range(1, 13):
            if day <= month_days[month]:
                return month
    df['month'] = df['day'].apply(get_month)
    
    return df

def lag_and_roll(df,variables,lags=[1], rolls=[3,7]):
    """
    Generates lag and roll features
    """    
    for var in variables:
        # ----------------------
        # Lagged Features (Previous day's values for key predictors)
        # ----------------------
        for lag in lags:
            new_var = var+f"_lag{str(lag)}"
            df[new_var] = df[var].shift(lag).fillna(0)

        # ----------------------
        # Rolling Statistics (n-day averages for trend analysis)
        # ----------------------
        for roll in rolls:
            new_var = var+f"_roll{str(roll)}"
            df[new_var] = df[var].rolling(window=roll, min_periods=1).mean().fillna(method='bfill') # Backfill if needed so the first rows won't be NaN. 

    return df

def transform_winddirection(df):
    """
    Transforms winddirection (degrees) into sine and cosine components.
    """
    radians = np.radians(df['winddirection'])
    df['winddirection'+'_sin'] = np.sin(radians)
    df['winddirection'+'_cos'] = np.cos(radians)
    df.drop('winddirection', axis=1, inplace=True)
    

def circular_features_time(df):
    """
    Applies sin & cos transformations to time variables 
    """
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)
    df['month_sin'] = np.sin(2* np.pi * df['month']/ 12)
    df['month_cos'] = np.cos(2* np.pi * df['month']/ 12)
    df.drop('day', axis=1, inplace=True) # Optional 
    
    return df

def transform_var(df, transform_dict):
    """
    Transforms specified variables in a DataFrame based on provided transformations.

    """

    for variable, transformation in transform_dict.items():
        transformed_variable = f"{variable}_{transformation}"
     
        if transformation == 'log1p':
            df[transformed_variable] = np.log1p(df[variable])
        elif transformation == 'sqrt':
            df[transformed_variable] = np.sqrt(df[variable])
        elif transformation == 'square':
            df[transformed_variable] = np.square(df[variable])
        elif transformation == 'yeojohnson':
            df[transformed_variable], _ = stats.yeojohnson(df[variable])
        elif transformation == 'boxcox':
            df[transformed_variable], _ = stats.boxcox(df[variable])
        elif transformation == 'reflect_log':
            min_val = df[variable].min()
            df[transformed_variable] = np.log1p(df[variable] - min_val)
        elif transformation == 'reflect_sqrt':
            min_val = df[variable].min()
            df[transformed_variable] = np.sqrt(df[variable] - min_val)
        else:
            raise ValueError(f"Unsupported transformation: {transformation}")
            
        df.drop(variable, axis=1, inplace=True)
   
    return df

# Feature Engineering Function
def perform_feature_engineering(df):
    """
    Applies feature engineering to the dataframe, creating new features for weather prediction.
    """
    
    # ----------------------
    # 1. Get month
    # ----------------------  
    dayofyear_to_month(df)

    # ----------------------
    # 2. Lag & Roll Features
    # ----------------------    
    lag_and_roll(df,['cloud','sunshine','humidity','pressure','dewpoint',],lags=[1], rolls=[3])
    
    # ----------------------
    # 3. Cyclical Seasonal Features (day & month)
    # ----------------------
    circular_features_time(df)

    # ----------------------
    # 4. Transform Skewed Variables
    # ----------------------
    transform_var(df, transform_dict)
    
    # ----------------------
    # 5. Interaction, Rations, and New Features
    # ----------------------
    df['cloud*humidity'] = (df['cloud_yeojohnson'] * df['humidity']).fillna(0)
    df['humidity*sunshine'] = df["humidity"] * df['sunshine_yeojohnson']
    df['sunshine/cloud'] = (df['sunshine_yeojohnson'] / (df['cloud_yeojohnson'] + 1e-5)).fillna(0)  # Avoid division by zero
    df['temp_range'] = (df['maxtemp_yeojohnson'] - df['mintemp_yeojohnson']).fillna(df['maxtemp_yeojohnson'].median())
    df['pressure_diff'] = df['pressure'].diff().fillna(0) # 1 day
    df['cloud_diff'] = df['cloud_yeojohnson'].diff().fillna(0)
    df['sunshine_diff'] = df['sunshine_yeojohnson'].diff().fillna(0)
    df['humidity_diff'] = df['humidity'].diff().fillna(0)

    # ----------------------
    # 6. Interaction, Rations, and New Features
    # ----------------------

    return df

# Create a heatmap to visualize the correlation matrix of the engineered train_capped DataFrame
plt.figure(figsize=(12,10))
sns.heatmap(data=pd.concat([perform_feature_engineering(train_capped.copy()),y_train],axis=1).corr().T.round(4), 
            annot=False, 
            cmap="coolwarm", 
            linewidth = 0.3,
            annot_kws={"rotation": 90}
           )

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.title('Correlation Heatmap (Full)')
plt.show()

# Create a heatmap to visualize the correlation matrix of the engineered train_capped DataFrame
plt.figure(figsize=(22,2))
sns.heatmap(data=pd.concat([perform_feature_engineering(train_capped.copy()),y_train],axis=1).corr()[['rainfall']].T.round(4), 
            annot=True, 
            cmap="coolwarm", 
            linewidth = 0.3,
            annot_kws={"rotation": 90}
           )

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.title('Correlation Heatmap (Rainfall only)')
plt.show()



# Engineer features
for df in [train_capped, valid_capped, test_capped]:
    perform_feature_engineering(df)


# import numpy as np
# import imblearn
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_classif, chi2 
# from sklearn.linear_model import LogisticRegression
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier
# from sklearn.metrics import accuracy_score, roc_auc_score

# models = {
#     "Logistic Regression": LogisticRegression(random_state=SEED, max_iter=1000),
#     "Random Forest": RandomForestClassifier(random_state=SEED, n_estimators=100),
#     "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
#     "Support Vector Machine": SVC(probability=True, random_state=SEED),
#     "K-Nearest Neighbors": KNeighborsClassifier(),
#     "Neural Network": MLPClassifier(random_state=SEED, max_iter=400),
#     "XGBoost": XGBClassifier(random_state=SEED, n_estimators=100, learning_rate=0.05, max_depth=6,),
#     "CatBoost": CatBoostClassifier(random_state=SEED, iterations=100, learning_rate=0.14, depth=6, verbose=0)
# }

# results = {}

# for name, model in models.items():
#     # Define Parameter Distributions for RandomizedSearchCV
#     param_dist = {
#         'sample': [SMOTE(random_state=SEED, k_neighbors=5), RandomUnderSampler(random_state=SEED)],
#         'selector__percentile': np.arange(70, 96) # Percentiles from 70 to 95
#     }
#     pipeline = imblearn.pipeline.Pipeline([
#         ('scaler', MinMaxScaler()),
#         ('sample', RandomUnderSampler(random_state=SEED)),
#         # ('sample', SMOTE(random_state=SEED,k_neighbors=5)),
#         ('selector', SelectPercentile(mutual_info_classif, percentile=95)),
#         ('model', model)
#     ])

#     pipeline.fit(train_capped, y_train)
#     y_pred = pipeline.predict(valid_capped)
#     accuracy = accuracy_score(y_valid, y_pred)
#     auc = roc_auc_score(y_valid, pipeline.predict_proba(valid_capped)[:, 1])
#     results[name] = {"accuracy": accuracy, "auc": auc}
#     print(f"{name}: Accuracy = {accuracy:.4f}, AUC = {auc:.4f}")

# # Stacking
# estimators = [
#     (name, 
#      imblearn.pipeline.Pipeline([('scaler', MinMaxScaler()), 
#                ('sample', RandomUnderSampler(random_state=SEED)), 
#                # ('sample', SMOTE(random_state=SEED,k_neighbors=5)), 
#                ('selector', SelectPercentile(mutual_info_classif, percentile=95)), 
#                ('model', model)])
#     ) for name, model in models.items()]
# stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(random_state=SEED, max_iter=1000))

# stacking_model.fit(train_capped, y_train)
# stacking_pred = stacking_model.predict(valid_capped)
# stacking_auc = roc_auc_score(y_valid, stacking_model.predict_proba(valid_capped)[:,1])
# stacking_accuracy = accuracy_score(y_valid, stacking_pred)
# print("="*50)
# print(f"Stacking Model: Accuracy = {stacking_accuracy:.4f}, AUC = {stacking_auc:.4f}")


import numpy as np
import imblearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
import random

models = {
    "Logistic Regression": LogisticRegression(random_state=SEED, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    "Support Vector Machine": SVC(probability=True, random_state=SEED),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Neural Network": MLPClassifier(random_state=SEED, max_iter=400),
    "XGBoost": XGBClassifier(random_state=SEED),
    "CatBoost": CatBoostClassifier(random_state=SEED, verbose=0)
}

results = {}

for name, model in models.items():
    # Define Parameter Distributions for RandomizedSearchCV
    param_dist = {
        'sample': [SMOTE(random_state=SEED, k_neighbors=5), RandomUnderSampler(random_state=SEED)],
        'selector__percentile': np.arange(70, 98+1, 2),  # Percentiles from 70 to 96, steps of 2
        # 'model__n_estimators': np.arange(50, 201) if name in ["Random Forest", "XGBoost", "CatBoost"] else [None],
        # 'model__learning_rate': np.logspace(-3, 0, 10) if name in ["XGBoost", "CatBoost", "Gradient Boosting"] else [None],
        # 'model__max_depth': np.arange(3, 11) if name in ["XGBoost", "CatBoost", "Gradient Boosting"] else [None],
        # 'model__C': np.logspace(-3, 3, 7) if name == "Logistic Regression" else [None],
        # 'model__kernel': ['linear', 'rbf', 'poly'] if name == "Support Vector Machine" else [None],
        # 'model__gamma': ['scale', 'auto'] if name == "Support Vector Machine" else [None],
        # 'model__n_neighbors': np.arange(3, 11) if name == "K-Nearest Neighbors" else [None],
        # 'model__alpha': np.logspace(-5, -1, 5) if name == "Neural Network" else [None],
    }

    pipeline = imblearn.pipeline.Pipeline([
        ('scaler', MinMaxScaler()),
        ('sample', SMOTE(random_state=SEED, k_neighbors=5)),  # Default sampler, will be changed by random search
        ('selector', SelectPercentile(mutual_info_classif)), # Default selector, percentile will be changed
        ('model', model)
    ])

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=30,  
        cv=3, 
        random_state=SEED,
        n_jobs=-1,
        scoring='roc_auc',
        verbose=1
    )

    random_search.fit(train_capped, y_train)
    y_pred = random_search.predict(valid_capped)
    accuracy = accuracy_score(y_valid, y_pred)
    auc = roc_auc_score(y_valid, random_search.predict_proba(valid_capped)[:, 1])
    results[name] = {"accuracy": accuracy, "auc": auc, "best_params": random_search.best_params_}
    print(f"{name}: Accuracy = {accuracy:.4f}, AUC = {auc:.4f}, Best Params: {random_search.best_params_}")

# Stacking using the best estimators found through RandomizedSearchCV
estimators = []
for name, model in models.items():
    best_params = results[name]["best_params"]
    pipeline = imblearn.pipeline.Pipeline([
        ('scaler', MinMaxScaler()),
        ('sample', best_params['sample']),
        ('selector', SelectPercentile(mutual_info_classif, percentile=best_params['selector__percentile'])),
        ('model', models[name].set_params(**{k.split('model__', 1)[1]: v for k, v in best_params.items() if k.startswith('model__')}))
    ])
    estimators.append((name, pipeline))

stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(random_state=SEED, max_iter=1000))

stacking_model.fit(train_capped, y_train)
stacking_pred = stacking_model.predict(valid_capped)
stacking_auc = roc_auc_score(y_valid, stacking_model.predict_proba(valid_capped)[:, 1])
stacking_accuracy = accuracy_score(y_valid, stacking_pred)
print("=" * 50)
print(f"Stacking Model: Accuracy = {stacking_accuracy:.4f}, AUC = {stacking_auc:.4f}")


stacking_model


# Create submission file
submission_df = pd.DataFrame({
    'id': list(test_capped.index),
    'rainfall': stacking_model.predict_proba(test_capped)[:, 1],  # Predicted probabilities for rainfall
})

# # Save to CSV
submission_df.to_csv("submission.csv", index=False)

# Display first 5 rows
submission_df.head(5)

