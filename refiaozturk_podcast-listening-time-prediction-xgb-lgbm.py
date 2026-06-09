import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix

import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")


df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df = df_train.copy()
df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


df.head()


df_test.head()


def data_overview(df):
    from IPython.display import display, Markdown

    def print_md_title(title, emoji):
        display(Markdown(f"**{emoji} {title}**"))
        print("=" * 35)
        
    print()
    print_md_title("Duplicate Rows Check", "ğŸ—ƒï¸�")
    dup_count = df.duplicated().sum()
    if dup_count == 0:
        print("âœ… No duplicate rows found.")
    else:
        print(f"âš ï¸� Found {dup_count} duplicate rows. Dropping them...")
        df.drop_duplicates(keep="first", inplace=True)
        print("âœ… Duplicate rows dropped.")
    
    print()
    print_md_title("Shape & Columns", "ğŸ§¾")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print()
    print_md_title("Dataset Info", "ğŸ“‹")
    print()
    df.info()

    print()
    print_md_title("Numerical Features Summary", "ğŸ”¢")
    display(df.describe().T)

    print()
    print_md_title("Categorical Features Summary", "ğŸ”¤")
    display(df.describe(include="object").T)

    print()
    print_md_title("Missing Values", "â�“")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    total_cells = df.size
    missing_percentage = (total_missing / total_cells) * 100

    if total_missing == 0:
        print("âœ… No missing values found.")
    else:
        print(f"âš ï¸� Missing values detected:")
        print(missing[missing > 0])
        print(f"\nTotal Missing: {total_missing} values ({missing_percentage:.2f}% of the dataset)")


data_overview(df)


data_overview(df_test)


# Numerical features
numeric_columns = df.select_dtypes(include=['number']).columns

# Categorical features
categoric_features = df.select_dtypes(include=['object', 'category']).columns.tolist()


# Checking out unique values in categorical features
for col in categoric_features:
    print(f"{col}")
    print("-" * 20)
    print("\nTotal unique values:", df[col].nunique())
    print("\n".join(df[col].unique().astype(str)))
    print("\n")


# Count the number of occurrences of each podcast in the Podcast_Name column
top_10_podcasts = df['Podcast_Name'].value_counts().head(10)

# Set plot style
sns.set(style="whitegrid")

# Create a barplot for the top 10 most published podcasts
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=top_10_podcasts.values, y=top_10_podcasts.index, palette="Set2")

# Set the title and labels
plt.title('Top 10 Most Published Podcasts')
plt.xlabel('Number of Episodes Published')
plt.ylabel('Podcast Name')

# Add values on top of the bars
for p in ax.patches:
    ax.annotate(f'{p.get_width():.0f}', 
                (p.get_width(), p.get_y() + p.get_height() / 2.), 
                ha='left', va='center', 
                fontsize=10, color='black', 
                xytext=(5, 0), textcoords='offset points')

# Display the plot
plt.show()


# List of categorical columns
categorical_columns = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

# Set plot style
sns.set(style="whitegrid")

# Create subplots with 2 rows and 2 columns (for 4 categorical columns)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Iterate through each categorical column and plot the value counts
for i, column in enumerate(categorical_columns):
    ax = axes[i//2, i%2]  # Select the subplot position
    value_counts = df[column].value_counts()
    
    # Create a barplot for each categorical column
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax, palette="Set2")
    
    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=30)

    # Add values on top of the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.0f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    fontsize=10, color='black', 
                    xytext=(0, 5), textcoords='offset points')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# Calculate the number of rows based on the specified number of columns
num_columns = 3
num_rows = (len(numeric_columns) // num_columns) + (1 if len(numeric_columns) % num_columns != 0 else 0)

# Create a figure with specified size
fig, axes = plt.subplots(num_rows, num_columns, figsize=(16, 4 * num_rows))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot each numeric column
for x, col in enumerate(numeric_columns):
    sns.boxplot(data=df[col], color='mediumseagreen', ax=axes[x])
    axes[x].set_title(col)

# Hide any unused axes (if there are any)
for i in range(x + 1, len(axes)):
    axes[i].axis('off')

plt.tight_layout() 
plt.show()


# Calculate the number of rows based on the specified number of columns
num_rows = (len(numeric_columns) // num_columns) + (1 if len(numeric_columns) % num_columns != 0 else 0)

# Create a figure with specified size
fig, axes = plt.subplots(num_rows, num_columns, figsize=(16, 4 * num_rows))
axes = axes.flatten()

for i, column in enumerate(numeric_columns):
    sns.histplot(df[column], kde=True, ax=axes[i], color='mediumseagreen')
    axes[i].set_title(column)
    
for i in range(len(numeric_columns), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


df.Episode_Length_minutes.isnull().sum()


# Group by categorical columns and calculate the median Episode_Length_minutes for each group
grouped_median = df.groupby(['Genre', 'Publication_Day', 'Publication_Time'])['Episode_Length_minutes'].median()

# Check and fill missing Episode_Length_minutes values based on the group median and Listening_Time_minutes
df['Episode_Length_minutes'] = df.apply(
    lambda row: min(row['Listening_Time_minutes'], grouped_median[row['Genre'], row['Publication_Day'], row['Publication_Time']]) 
    if pd.isna(row['Episode_Length_minutes']) 
    else row['Episode_Length_minutes'], axis=1
)

# Check the result
print("Number of missing values in df: ", df['Episode_Length_minutes'].isna().sum())


# Group by categorical columns and calculate the median Episode_Length_minutes for each group
grouped_median_dftest = df_test.groupby(['Genre', 'Publication_Day', 'Publication_Time'])['Episode_Length_minutes'].median()

# Check and fill missing Episode_Length_minutes values based on the group median
df_test['Episode_Length_minutes'] = df_test.apply(
    lambda row: grouped_median_dftest[row['Genre'], row['Publication_Day'], row['Publication_Time']] 
    if pd.isna(row['Episode_Length_minutes']) 
    else row['Episode_Length_minutes'], axis=1
)

# Check the result
print("Number of missing values in df_test: ", df_test['Episode_Length_minutes'].isna().sum())


df.Guest_Popularity_percentage.isnull().sum()


# Group by 'Genre' and calculate the median Guest_Popularity_percentage for each group
grouped_median_guest_popularity = df.groupby('Genre')['Guest_Popularity_percentage'].median()

# Check and fill missing Guest_Popularity_percentage values based on the group median
df['Guest_Popularity_percentage'] = df.apply(
    lambda row: grouped_median_guest_popularity[row['Genre']] 
    if pd.isna(row['Guest_Popularity_percentage']) 
    else row['Guest_Popularity_percentage'], axis=1
)

# If there's a test dataset (df_test), apply the same process here
df_test['Guest_Popularity_percentage'] = df_test.apply(
    lambda row: grouped_median_guest_popularity[row['Genre']] 
    if pd.isna(row['Guest_Popularity_percentage']) 
    else row['Guest_Popularity_percentage'], axis=1
)

# Check the result
print("Number of missing values in df (Guest_Popularity_percentage): ", df['Guest_Popularity_percentage'].isna().sum())
print("Number of missing values in df_test (Guest_Popularity_percentage): ", df_test['Guest_Popularity_percentage'].isna().sum())


df.Number_of_Ads.isnull().sum()


# Median value of Number_of_Ads column
median_num_ads = df.Number_of_Ads.median()

# Filling NaN values with median
df.Number_of_Ads.fillna(median_num_ads, inplace=True)


# Check the result
df.Number_of_Ads.isnull().sum()


# saving clean data
df.to_csv("train_clean.csv", index=False)


# saving clean data
df_test.to_csv("test_clean.csv", index=False)


df = pd.read_csv("/kaggle/working/train_clean.csv")
df_test = pd.read_csv("/kaggle/working/test_clean.csv")
df.head()


# dropping unnecessary features from df and df_test
df.drop(columns="id", inplace=True)
df_test.drop(columns="id", inplace=True)


X = df.drop(columns="Listening_Time_minutes")
y = df.Listening_Time_minutes


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


categoric_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
categoric_features


from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder1 = OrdinalEncoder(categories=[['Negative', 'Neutral', 'Positive']])

# Fitting train set only
X_train['Episode_Sentiment'] = ordinal_encoder1.fit_transform(X_train[['Episode_Sentiment']])

# Transforming test data
X_test['Episode_Sentiment'] = ordinal_encoder1.transform(X_test[['Episode_Sentiment']])

df_test['Episode_Sentiment'] = ordinal_encoder1.fit_transform(df_test[['Episode_Sentiment']])

# -------------------------------------------------

# Publication_Day
ordinal_encoder2 = OrdinalEncoder(categories=[['Tuesday', 'Monday', 'Wednesday', 'Saturday',
                                               'Friday', 'Thursday', 'Sunday']])
X_train['Publication_Day'] = ordinal_encoder2.fit_transform(X_train[['Publication_Day']])
X_test['Publication_Day'] = ordinal_encoder2.transform(X_test[['Publication_Day']])

df_test['Publication_Day'] = ordinal_encoder2.fit_transform(df_test[['Publication_Day']])

# -------------------------------------------------

# Publication_Time
ordinal_encoder3 = OrdinalEncoder(categories=[['Night', 'Afternoon', 'Morning', 'Evening']])
X_train['Publication_Time'] = ordinal_encoder3.fit_transform(X_train[['Publication_Time']])
X_test['Publication_Time'] = ordinal_encoder3.transform(X_test[['Publication_Time']])

df_test['Publication_Time'] = ordinal_encoder3.fit_transform(df_test[['Publication_Time']])


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

cat_onehot = ['Podcast_Name', 'Episode_Title', 'Genre']
column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), cat_onehot),
    remainder='passthrough'
)


X_train = pd.DataFrame(column_trans.fit_transform(X_train), columns=column_trans.get_feature_names_out())
X_test = pd.DataFrame(column_trans.transform(X_test), columns=column_trans.get_feature_names_out())

df_test = pd.DataFrame(column_trans.fit_transform(df_test), columns=column_trans.get_feature_names_out())


X_train.head()


X_test.head()


df_test.head()


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Convert to pandas format
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)
test = pd.DataFrame(df_test)

# KFold parameters
n_splits = 8
SEED = 101
kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)


xgb_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 5000,
    'learning_rate': 0.08,
    'max_depth': 15,
    'subsample': 1.0,
    'colsample_bytree': 0.7,
    'reg_alpha': 1,
    'reg_lambda': 8,
    'random_state': SEED,
    'tree_method': 'auto'
}

xgb_scores = []
xgb_test_preds = []

for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        early_stopping_rounds=100,
        verbose=1000
    )
    
    val_pred = model.predict(X_val, iteration_range=(0, model.best_iteration + 1))
    rmse = mean_squared_error(y_val, val_pred, squared=False)
    xgb_scores.append(rmse)
    
    test_pred = np.maximum(model.predict(test, iteration_range=(0, model.best_iteration + 1)), 0)
    xgb_test_preds.append(test_pred)
    
    print(f"Fold {i+1} RMSE: {rmse:.4f}")

print(f"\nXGBoost Mean RMSE: {np.mean(xgb_scores):.4f}")


# Since the predictions made on the test set are taken as many predictions with KFold,
# the final predictions should be taken as an average.
final_xgb_test_pred = np.mean(xgb_test_preds, axis=0)


# create submission_xgb file
submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
submission["Listening_Time_minutes"] = final_xgb_test_pred


submission.head(5)


# save submission_xgb file
submission.to_csv("submission_xgb.csv", index=False)


import lightgbm as lgb

lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 5000,
    'learning_rate': 0.08,
    'max_depth': 15,
    'num_leaves': 64,
    'reg_alpha': 1,
    'reg_lambda': 8,
    'colsample_bytree': 0.7,
    'subsample': 1.0,
    'subsample_freq': 6,
    'seed': SEED,
    'verbose': -1,
    'device': 'cpu'
}

lgbm_scores = []
lgbm_test_preds = []
categoric_features = list(set(X_train.columns) & set(categoric_features))

for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        categorical_feature=categoric_features,
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(period=1000)
        ]
    )
    
    val_pred = model.predict(X_val, num_iteration=model.best_iteration_)
    rmse = mean_squared_error(y_val, val_pred, squared=False)
    lgbm_scores.append(rmse)
    
    test_pred = np.maximum(model.predict(test, num_iteration=model.best_iteration_), 0)
    lgbm_test_preds.append(test_pred)
    
    print(f"Fold {i+1} RMSE: {rmse:.4f}")

print(f"\nLightGBM Mean RMSE: {np.mean(lgbm_scores):.4f}")


final_lgbm_test_pred = np.mean(lgbm_test_preds, axis=0)


# create submission_lgbm file
submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
submission["Listening_Time_minutes"] = final_lgbm_test_pred


submission.head(5)


# save submission_lgbm file
submission.to_csv("submission_lgbm.csv", index=False)

