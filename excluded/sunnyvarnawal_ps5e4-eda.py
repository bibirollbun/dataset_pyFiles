# Importing Libraries
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import optuna
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, median_absolute_error
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import catboost as cb
from scipy.optimize import minimize


train_data = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


train_data.head()


train_data=train_data.drop(columns=['Episode_Title', 'id'])
test_data = test_data.drop(columns=['Episode_Title', 'id'])


# Checking the number of rows and columns

num_train_rows, num_train_columns = train_data.shape
num_test_rows, num_test_columns = test_data.shape

print("Training Data:")
print(f"Number of Rows: {num_train_rows}")
print(f"Number of Columns: {num_train_columns}\n")

print("Test Data:")
print(f"Number of Rows: {num_test_rows}")
print(f"Number of Columns: {num_test_columns}\n")


# Creating a table for missing values, unique values and data types of the features

missing_values_train = pd.DataFrame({'Feature': train_data.columns,
                              '[TRAIN] No. of Missing Values': train_data.isnull().sum().values,
                              '[TRAIN] % of Missing Values': ((train_data.isnull().sum().values)/len(train_data)*100)})

missing_values_test = pd.DataFrame({'Feature': test_data.columns,
                             '[TEST] No.of Missing Values': test_data.isnull().sum().values,
                             '[TEST] % of Missing Values': ((test_data.isnull().sum().values)/len(test_data)*100)})

unique_values = pd.DataFrame({'Feature': train_data.columns,
                              'No. of Unique Values[FROM TRAIN]': train_data.nunique().values})

feature_types = pd.DataFrame({'Feature': train_data.columns,
                              'DataType': train_data.dtypes})

merged_df = pd.merge(missing_values_train, missing_values_test, on='Feature', how='left')
merged_df = pd.merge(merged_df, unique_values, on='Feature', how='left')
merged_df = pd.merge(merged_df, feature_types, on='Feature', how='left')

merged_df


# Count duplicate rows in train_data and test_data
train_duplicates = train_data.duplicated().sum()
test_duplicates = test_data.duplicated().sum()

print(f"Number of duplicate rows in train_data: {train_duplicates}")
print(f"Number of duplicate rows in test_data: {test_duplicates}")


train_data.describe().T\
.style.bar(subset=['mean'], color=px.colors.qualitative.G10[2])\
.background_gradient(subset=['std'], cmap='Blues')\
.background_gradient(subset=['50%'], cmap='BuGn')


cat_cols = train_data.select_dtypes(include='object').columns.tolist()
num_cols = [col for col in test_data.columns if col not in cat_cols and col!='id']
target_col='Listening_Time_minutes'


custom_palette = ['#3498db', '#e74c3c','#2ecc71']
def num_univar_plots(df, var, bins=10, figsize=(12,4)):
    sns.set_style('whitegrid')
    data = df[var].dropna().copy()  # Keep it as a Series

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax = ax.ravel()

    # histogram
    sns.histplot(data=data, bins=bins, ax=ax[0], kde=True, color=custom_palette[0])
    sns.rugplot(data=data, ax=ax[0], color='black')
    ax[0].set(title='Histogram')

    # box-plot
    sns.boxplot(x=data, color=custom_palette[1], ax=ax[1])
    ax[1].set(title='Boxplot')

    fig.suptitle(f'Univariate Analysis of {var}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


num_univar_plots(train_data, 'Episode_Length_minutes')


num_univar_plots(train_data, 'Host_Popularity_percentage')


num_univar_plots(train_data, 'Guest_Popularity_percentage')


num_univar_plots(train_data, 'Number_of_Ads')


num_univar_plots(train_data, 'Listening_Time_minutes')


train_data['Number_of_Ads'].value_counts()


numeric_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage']
target_col = 'Listening_Time_minutes'  # Make sure this is a single column name (string, not a list)
data = train_data.dropna()

# Select only the required columns
selected_cols = numeric_cols + [target_col]
data = data[selected_cols]

# Create pairplot
sns.pairplot(data, diag_kind='kde', corner=True, plot_kws={'alpha': 0.5})

plt.suptitle("Pairwise Relationship Between Numeric Features & Target", fontsize=14, fontweight='bold')
plt.show()


pie_chart_palette = ['#33638d', '#28ae80', '#d3eb0c', '#ff9a0b', '#7e03a8', '#35b779', '#fde725', '#440154', '#90d743', '#482173', '#22a884', '#f8961e']

countplot_color = '#5C67A3'

# Function to create and display a row of plots for a single categorical variable
def create_categorical_plots(variable):
    sns.set_style('whitegrid')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Pie Chart
    plt.subplot(1, 2, 1)
    train_data[variable].value_counts().plot.pie(
        autopct='%1.1f%%', colors=pie_chart_palette, wedgeprops=dict(width=0.3), startangle=140
    )
    plt.title(f"Pie Chart for {variable}")

    # Bar Graph
    plt.subplot(1, 2, 2)
    sns.countplot(
        data=pd.concat([train_data.dropna(), test_data.dropna()]), 
        x=variable, 
        color=countplot_color,  # Using a single color for the countplot
        alpha=0.8  # Setting 80% opacity
    )
    plt.xlabel(variable)
    plt.ylabel("Count")
    plt.title(f"Bar Graph for {variable} [TRAIN, TEST]")

    plt.tight_layout()
    plt.show()

for col in cat_cols:
    if col not in ['Podcast_Name', 'Episode_Title']:
        create_categorical_plots(col)


train_data['Podcast_Name'].value_counts()


# train_data['Episode_Title'].value_counts()



NUM_cols = num_cols + ['Listening_Time_minutes']
corr = train_data[NUM_cols].dropna().corr()
mask = np.triu(np.ones_like(corr, dtype='bool'))
plt.figure(figsize=(7, 7))
heatmap=sns.heatmap(corr, mask=mask, annot=True, square=True, cmap='viridis')


train_data['Episode_Length_minutes'].fillna(train_data['Episode_Length_minutes'].mean(), inplace=True)
train_data['Guest_Popularity_percentage'].fillna(train_data['Guest_Popularity_percentage'].mean(), inplace=True)
train_data['Number_of_Ads'].fillna(train_data['Number_of_Ads'].mean(), inplace=True)

test_data['Episode_Length_minutes'].fillna(test_data['Episode_Length_minutes'].mean(), inplace=True)
test_data['Guest_Popularity_percentage'].fillna(test_data['Guest_Popularity_percentage'].mean(), inplace=True)


def perform_feature_engineering(df):
    data = df.copy()
    data['Is_weekend'] = data['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    
    data['Popularity_Diff'] = data['Host_Popularity_percentage'] - data['Guest_Popularity_percentage']
    data['Total_Popularity'] = (data['Guest_Popularity_percentage'] + data['Host_Popularity_percentage'])/2

    data['Host_Popularity_Tier'] = pd.cut(data['Host_Popularity_percentage'], bins=[0, 50, 70, 85, 120], labels=['Low', 'Medium', 'High', 'Top'])

    data['Ads_per_Minute'] = data['Number_of_Ads'] / (data['Episode_Length_minutes']+1)
    data['Ads_per_Minute'].replace([np.inf, -np.inf], np.nan, inplace=True)  
    
    data['Genre_Sentiment'] = data['Genre'] + "_" + data['Episode_Sentiment']

    data['Episode_Length_Category'] = pd.cut(data['Episode_Length_minutes'], bins=[0, 60, 90, np.inf], labels=['Short', 'Medium', 'Long'])
    return data
    

train = perform_feature_engineering(train_data)
test = perform_feature_engineering(test_data)


train['Episode_Length_Category'].fillna(train['Episode_Length_Category'].mode()[0], inplace=True)
# train = train_data.copy()
# test = test_data.copy()


# Function to remove outliers using IQR and visualize
def remove_outliers_iqr_with_plot(data, column):
    Q1 = data[column].quantile(0.15)
    Q3 = data[column].quantile(0.85)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the data
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    # Calculate the number of rows deleted
    rows_deleted = len(data) - len(filtered_data)
    
    # Plot the distribution with outliers
    plt.figure(figsize=(7, 3))
    sns.boxplot(x=data[column], color='lightblue', flierprops={'marker': 'o', 'markersize': 5, 'markerfacecolor': 'red'})
    
    # Highlight Q1 and Q3
    plt.axvline(Q1, color='green', linestyle='--', label='Q1 (10th Percentile)')
    plt.axvline(Q3, color='blue', linestyle='--', label='Q3 (90th Percentile)')
    
    # Highlight lower and upper bounds
    plt.axvline(lower_bound, color='red', linestyle='-', label='Lower Bound')
    plt.axvline(upper_bound, color='red', linestyle='-', label='Upper Bound')

    plt.title(f'Outlier Detection for {column}')
    plt.legend()
    plt.xlabel(column)
    plt.show()
    
    return filtered_data, rows_deleted

# Apply function to each numerical column and visualize
rows_deleted_total = 0

for column in num_cols:
    train, rows_deleted = remove_outliers_iqr_with_plot(train, column)
    rows_deleted_total += rows_deleted
    print(f"Rows deleted for {column}: {rows_deleted}")

print(f"Total rows deleted: {rows_deleted_total}")


skewed_features = train[num_cols].skew()[train[num_cols].skew() > 0.75].index.values

# Print the list of variables to be transformed
print("Features to be transformed (skewness > 0.75):")
display(skewed_features)

# Plot skewed features before transformation
for feature in skewed_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(train[feature], bins=50, kde=True, color='blue')
    plt.title(f'Distribution of {feature} before log transformation')
    plt.show()

# Apply log1p transformation to skewed features
train[skewed_features] = np.log1p(train[skewed_features])

# Plot skewed features after transformation
for feature in skewed_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(train[feature], bins=50, kde=True, color='green')
    plt.title(f'Distribution of {feature} after log transformation')
    plt.show()


skewed_features = test[num_cols].skew()[test[num_cols].skew() > 0.75].index.values

# Print the list of variables to be transformed
print("Features to be transformed (skewness > 0.75):")
display(skewed_features)

# Plot skewed features before transformation
for feature in skewed_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(test[feature], bins=50, kde=True, color='blue')
    plt.title(f'Distribution of {feature} before log transformation')
    plt.show()

# Apply log1p transformation to skewed features
test[skewed_features] = np.log1p(test[skewed_features])

# Plot skewed features after transformation
for feature in skewed_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(test[feature], bins=50, kde=True, color='green')
    plt.title(f'Distribution of {feature} after log transformation')
    plt.show()


categorical_features = train.select_dtypes(include=['category', 'object']).columns.tolist()
numerical_features = [col for col in test.columns if col not in categorical_features]
y = train['Listening_Time_minutes']
X = train.drop(columns=["Listening_Time_minutes"], axis=1)


for c in categorical_features:
    X[col] = X[col].astype("category")
    test[col] = test[col].astype("category")


X.info()


for c in numerical_features:
    m = train[c].mean()
    s = train[c].std()
    test[c] = (train[c]-m)/s
    test[c] = train[c].fillna(0)
    if test[c].dtype=='float64':
        test[c].astype('float32')
    if test[c].dtype=='int64':
        test[c].astype('int32')


for c in numerical_features:
    m = train[c].mean()
    s = train[c].std()
    X[c] = (train[c] - m) / s
    X[c] = train[c].fillna(0)  # This overwrites the normalization, which might not be intended
    
    # Ensure data types are optimized
    if X[c].dtype == 'float64':
        X[c] = X[c].astype('float32')
    if X[c].dtype == 'int64':
        X[c] = X[c].astype('int32')


from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import RepeatedKFold
import numpy as np

# Define repeated K-Fold
skf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42)

# Lists to store scores and predictions
scores = []
test_preds = np.zeros((len(test), skf.get_n_splits()))

# Loop through each fold
for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    
    # Train-Validation Split
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Create CatBoost Pool objects
    d_train = Pool(X_train, y_train, cat_features=categorical_features)
    d_val = Pool(X_val, y_val, cat_features=categorical_features)

    # Initialize and Train Model
    model = CatBoostRegressor(
        iterations=1000, 
        learning_rate=0.05,
        depth=10,
        eval_metric='RMSE',
        random_seed=42,
        verbose=0, 
        task_type="GPU"
    )
    model.fit(d_train, eval_set=d_val, early_stopping_rounds=100)

    # Predictions
    y_pred = model.predict(X_val)
    test_preds[:, i] = model.predict(test)  # Store predictions per fold
    
    # Compute RMSE for this fold
    score = np.sqrt(np.mean((y_val - y_pred) ** 2))
    scores.append(score)
    print(f"Fold {i+1} RMSE: {score:.4f}")

# Compute mean and std deviation of RMSE scores
print(f"Mean RMSE: {np.mean(scores):.4f} Â± {np.std(scores):.4f}")

# Get final test predictions (mean over folds)
final_test_pred = test_preds.mean(axis=1)



%%time
submission = pd.read_csv("../input/playground-series-s5e4/sample_submission.csv")
submission["Listening_Time_minutes"] = test_preds
submission.to_csv('submission.csv', index=False)
submission.head()




