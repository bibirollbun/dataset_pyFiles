import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_percentage_error
import logging



#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import gc


#load data
train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
gc.collect()


#Set id as index

train.set_index('id', inplace=True)
test.set_index('id', inplace=True)


train.head()


test.head()


train.info()


test.info()


train.shape


test.shape


train.isnull().sum()


test.isnull().sum()


#Plot the target variable distribution
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.histplot(train['num_sold'], bins=30, kde=True, color='blue')
plt.title('Distribution of Target Variable')
plt.xlabel('num_sold')
plt.ylabel('Frequency')
plt.show()

#Plot the relationship between the target variable and date
plt.figure(figsize=(14, 7))
train['date'] = pd.to_datetime(train['date'])
train.groupby('date')['num_sold'].sum().plot()
plt.title('num_sold Over Time')
plt.xlabel('Date')
plt.ylabel('Total num_sold')
plt.show()



def plot_boxplot(df, target_col):
   
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, y=target_col, palette="Set2")
    plt.title(f"Boxplot of {target_col}", fontsize=14)
    plt.ylabel(target_col, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

plot_boxplot(train, 'num_sold')



# Statistical distribution of class features
categorical_cols = ['country', 'store', 'product']

for col in categorical_cols:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=train, x=col, order=train[col].value_counts().index, palette="viridis")
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()



# The mean of class features and targets
for col in categorical_cols:
    plt.figure(figsize=(8, 6))
    sns.barplot(data=train, x=col, y='num_sold', estimator=np.mean, ci=None, palette="magma")
    plt.title(f'{col} vs Average Target')
    plt.xlabel(col)
    plt.ylabel('Average num_sold')
    plt.xticks(rotation=45)
    plt.show()




for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    
    train_dist = train[col].value_counts(normalize=True)
    test_dist = test[col].value_counts(normalize=True)
    
    dist_df = pd.DataFrame({'Train': train_dist, 'Test': test_dist}).fillna(0)
    
    # Draw a distribution comparison chart
    dist_df.plot(kind='bar', figsize=(10, 6), alpha=0.7, color=['blue', 'orange'])
    plt.title(f'Category Distribution of {col} in Train vs Test')
    plt.xlabel(col)
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.legend(['Train', 'Test'])
    plt.show()




train['date'] = pd.to_datetime(train['date'], errors='coerce')
test['date'] = pd.to_datetime(test['date'], errors='coerce')

print(train['date'].describe())
print(test['date'].describe())

# Draw a date distribution comparison
plt.figure(figsize=(12, 6))

sns.histplot(train['date'], bins=30, label='Train', color='blue', kde=False, alpha=0.6)
sns.histplot(test['date'], bins=30, label='Test', color='orange', kde=False, alpha=0.6)

plt.title('Date Distribution: Train vs Test')
plt.xlabel('Date')
plt.ylabel('Frequency')
plt.legend()
plt.show()




def plot_monthly_distribution(train, test, date_col='date'):
    
    train[date_col] = pd.to_datetime(train[date_col], errors='coerce')
    test[date_col] = pd.to_datetime(test[date_col], errors='coerce')

    # Monthly aggregation
    train_monthly = train[date_col].dt.to_period('M').value_counts().sort_index()
    test_monthly = test[date_col].dt.to_period('M').value_counts().sort_index()

    # Distribution mapping
    plt.figure(figsize=(12, 6))
    train_monthly.plot(label='Train', color='blue', marker='o')
    test_monthly.plot(label='Test', color='orange', marker='o')
    plt.title('Monthly Distribution: Train vs Test')
    plt.xlabel('Year-Month')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.show()
    
plot_monthly_distribution(train, test)



# Delete rows containing NaN in the num_sold column
train = train.dropna(subset=['num_sold'])



def process_date_features(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col])
    
    df['Year'] = df[date_col].dt.year
    df['Month'] = df[date_col].dt.month
    df['Day'] = df[date_col].dt.day
    
    df['YearSin'] = np.sin(2 * np.pi * df['Year'] / 4)
    df['YearCos'] = np.cos(2 * np.pi * df['Year'] / 4)
    df['MonthSin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['MonthCos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DaySin'] = np.sin(2 * np.pi * df['Day'] / 30)
    df['DayCos'] = np.cos(2 * np.pi * df['Day'] / 30)
    
    df['Season'] = df['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else
                                                'Spring' if x in [3, 4, 5] else
                                                'Summer' if x in [6, 7, 8] else
                                                'Autumn')
    df = pd.get_dummies(df, columns=['Season'], prefix=['Season'])
    df = df.drop(date_col, axis=1)
    
    return df

train = process_date_features(train, 'date')
test = process_date_features(test, 'date')


train.head()


cat_cols = ['country', 'store', 'product', 'Season_Winter', 'Season_Spring', 'Season_Summer', 'Season_Autumn']


def one_hot_encode(df, columns):
    encoded_df = pd.get_dummies(df, columns=columns, drop_first=True)
    return encoded_df

train = one_hot_encode(train, cat_cols)
test = one_hot_encode(test, cat_cols)

test = test.reindex(columns=train.columns, fill_value=0)



train.head()


test.head()


train.columns



logging.getLogger('lightgbm').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

X = train.drop(columns=['num_sold'])
y = np.log1p(train['num_sold'])
test = test[X.columns]

def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

def cross_val_lgbm_mape(X, y, test, groups, n_splits=5, **params):
    group_kf = GroupKFold(n_splits=n_splits)
    mape_scores = []
    preds = []

    for train_index, valid_index in group_kf.split(X, y, groups):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_valid)
        score = mape(y_valid, y_pred)
        mape_scores.append(score)

        preds.append(model.predict(test))

    test_preds_mean = np.mean(preds, axis=0)

    return np.mean(mape_scores), test_preds_mean

model_params = {
    "objective": "regression",
    "metric": "rmse",
    "seed": 42,
    "verbose": -1,
}

# Try different grouping strategies
# 1. Grouping by Year
# CV score 0.0157
groups_year = train['Year']

# 2. Group by country and year
# Converts the year column to a string type
#groups_country_year = train['country'].astype(str) + "_" + train['Year'].astype(str)

# 3. Group by store
# CV score 0.0671
# groups_store = train['store']

# Select group strategy
# CV score  0.0145
groups = groups_year  # The best group can be selected according to the experimental results

average_mape, lgb_preds = cross_val_lgbm_mape(X, y, test, groups, n_splits=3, **model_params)
print(f"Average MAPE across folds: {average_mape:.4f}")



test_submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
lgb_preds_original = np.expm1(lgb_preds)
submission = pd.DataFrame({
    'id': test_submission['id'],
    'num_sold': lgb_preds_original
})
submission.to_csv('submission_lgb.csv', index=False)
print("Submission file created:")
print(submission.head())

