import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
print('Imports Check!')


train = pd.read_csv(r'/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s5e4/test.csv')
train


nan_counts = train.isna().sum()
print("NaN counts per column:")
print(nan_counts)


for df in [test,train]:
    df['Episode_Length_minutes'] = df.groupby('Podcast_Name')['Episode_Length_minutes'].transform(lambda x: x.fillna(x.median()))
    df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median(), inplace = True)
    df['Number_of_Ads'] = df.groupby('Podcast_Name')['Number_of_Ads'].transform(lambda x: x.fillna(x.median()))


cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    category_target_rate = train.groupby(col)['Listening_Time_minutes'].mean()
    print(category_target_rate)


train.set_index('id',inplace = True)
test.set_index('id',inplace = True)


plt.figure(figsize=(5,3))
sns.jointplot(x='Host_Popularity_percentage',y='Listening_Time_minutes',data=train,kind='hex')
sns.set_style('dark')


plt.figure(figsize=(5,3))
sns.jointplot(x='Guest_Popularity_percentage',y='Listening_Time_minutes',data=train,kind='hex')
sns.set_style('dark')


plt.figure(figsize=(5,3))
sns.boxplot(x='Genre',y='Listening_Time_minutes',data=train)
sns.set_style('dark')


plt.figure(figsize=(5,3))
sns.jointplot(x='Episode_Length_minutes',y='Listening_Time_minutes',data=train,kind='hex')
sns.set_style('dark')


numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = train[numeric_cols].corr()

plt.figure(figsize=(3, 3))
sns.heatmap(corr_matrix[['Guest_Popularity_percentage']].sort_values(by='Guest_Popularity_percentage', ascending=False),
            annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation with Guest_Popularity_percentage")


bins = [0, 10, 50, 90, 100]
labels = ['Bottom 10%', 'Mid 40%', 'Top 40%', 'Top 10%']
for df in [train,test]:
    df['Host_Popularity_bin'] = pd.cut(df['Host_Popularity_percentage'], bins=bins, labels=labels)


bins = [0, 10, 50, 90, 100]
labels = ['Bottom 10%', 'Mid 40%', 'Top 40%', 'Top 10%']
import re
choices = ['Solo','Power_Pair','Strong_Pair']


for df in [train,test]:
    df['Host_Popularity_bin'] = pd.cut(df['Host_Popularity_percentage'], bins=bins, labels=labels)
    
    df['Publish_Day_Time'] = df['Publication_Day'] + '_' + df['Publication_Time']
    
    df['Episode_Num'] = df['Episode_Title'].str.extract(r'(\d+)').astype(float)

    df['No_Guest'] = df['Guest_Popularity_percentage'].isna().astype(int)
    df['Guest_Popularity_bin'] = pd.cut(df['Guest_Popularity_percentage'], bins=bins, labels=labels).cat.add_categories('No_Guest').fillna('No_Guest')

    df['Guest_Host_interaction'] = (df['Guest_Popularity_percentage'].astype(float))*(df['Host_Popularity_percentage'].astype(float))
    
    conditions = [
        (df['No_Guest'] == 1),
        (df['Host_Popularity_bin'] == 'Elite') & (df['Guest_Popularity_bin'].isin(['High', 'Elite'])),
        (df['Host_Popularity_bin'] == 'High') & (df['Guest_Popularity_bin'] == 'High')
    ]
    df['Host_Guest_Tier'] = np.select(conditions, choices, default='Standard')
    
    df['Ads_per_minute'] = df['Number_of_Ads']/df['Episode_Length_minutes']
    
    df['No_Ads'] = (df['Ads_per_minute'] == 0).astype(int)
    
    df['Ad_Density_Bin'] = pd.qcut(df['Ads_per_minute'], 4, labels=['Low', 'Mid', 'High'], duplicates='drop')

    df['Popularity_Match'] = (df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']).abs()
    
    df['Episode_Length_bin'] = pd.qcut(df['Episode_Length_minutes'],q=4,labels=["snack", "small", "medium", "large"],duplicates='drop')
    


sns.boxplot(x='Host_Popularity_bin', y='Listening_Time_minutes', data=train)
plt.title("Listener Duration by Host Popularity Bin")


sns.pointplot(x='Host_Popularity_bin', y='Listening_Time_minutes', data=train, capsize=0.1)
plt.title("Mean Listener Duration by Host Popularity Bin")


sns.pointplot(x='Publish_Day_Time', y='Listening_Time_minutes', data=train, capsize=0.1)
plt.title("Mean Listener Duration by Host Popularity Bin")
plt.xticks(rotation=90)


sns.pointplot(x='Ad_Density_Bin', y='Listening_Time_minutes', data=train, capsize=0.1)
plt.title("Mean Listener Duration by Host Popularity Bin")
plt.xticks(rotation=90)


sns.pointplot(x='Episode_Length_bin', y='Listening_Time_minutes', data=train, capsize=0.1)
plt.title("Mean Listener Duration by Host Popularity Bin")
plt.xticks(rotation=90)



pivot = train.pivot_table(
    index='Episode_Sentiment', 
    columns='Publication_Time', 
    values='Listening_Time_minutes', 
    aggfunc='median'
)
sns.heatmap(pivot, cmap='YlGnBu', annot=True, fmt=".1f")



pivot = train.pivot_table(
    index='Episode_Sentiment', 
    columns='Genre', 
    values='Listening_Time_minutes', 
    aggfunc='median'
)
sns.heatmap(pivot, cmap='YlGnBu', annot=True, fmt=".1f")


pivot = train.pivot_table(
    index='Episode_Sentiment', 
    columns='Host_Popularity_bin', 
    values='Listening_Time_minutes', 
    aggfunc='median'
)
sns.heatmap(pivot, cmap='YlGnBu', annot=True, fmt=".1f")


pivot = train.pivot_table(
    index='Episode_Sentiment', 
    columns='Guest_Popularity_bin', 
    values='Listening_Time_minutes', 
    aggfunc='median'
)
sns.heatmap(pivot, cmap='YlGnBu', annot=True, fmt=".1f")


pivot = train.pivot_table(
    index='Genre', 
    columns='Host_Popularity_bin', 
    values='Listening_Time_minutes', 
    aggfunc='median'
)
sns.heatmap(pivot, cmap='YlGnBu', annot=True, fmt=".1f")


pivot = train.pivot_table(
    index='Genre', 
    columns='Host_Popularity_bin', 
    values='Listening_Time_minutes', 
    aggfunc='median'
)
sns.heatmap(pivot, cmap='YlGnBu', annot=True, fmt=".1f")


X = train.drop(columns = 'Listening_Time_minutes')
y = train['Listening_Time_minutes']


X


test


for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category')
    test[col] = test[col].astype('category')


sns.histplot(train['Listening_Time_minutes'], kde=True)


sns.boxplot(train['Listening_Time_minutes'])       


from scipy.stats import skew
skewness = skew(train['Listening_Time_minutes'])
print(f"Skewness: {skewness:.2f}")


from sklearn.preprocessing import OrdinalEncoder

# Convert object columns to category if not already
cat_cols = X.select_dtypes(include=['object', 'category']).columns

oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[cat_cols] = oe.fit_transform(X[cat_cols])
test[cat_cols] = oe.transform(test[cat_cols])


params = {
    'device':'cuda',
    'objective': 'reg:squarederror',
    'colsample_bytree': 0.6,
    'gamma': 5.0,
    'learning_rate': 0.01,
    'max_depth': 3,
    'min_child_weight': 1,
    'reg_alpha': 0.001,
    'reg_lambda': 10.0,
    'subsample': 0.611,
    'n_estimators': 500,  # Locked-in value
}


import shap
from xgboost import XGBRegressor

# Fit the model again on full dataset
model = XGBRegressor(**params)
model.fit(X, y)

# Use TreeExplainer
explainer = shap.Explainer(model)

# Compute SHAP values (this can take time on large data)
shap_values = explainer(X)

# Summary plot: which features matter most
shap.plots.beeswarm(shap_values)


y.describe()


# Define the interaction pairs explicitly
inter_pairs = [
    ('Episode_Length_minutes', 'Episode_Length_bin'),
    ('Episode_Length_minutes', 'Ads_per_minute'),
    ('Episode_Length_minutes', 'Host_Popularity_percentage'),
    ('Episode_Length_minutes', 'Number_of_Ads'),
    ('Episode_Length_minutes', 'Ad_Density_Bin'),
    ('Episode_Length_minutes', 'Episode_Sentiment'),
    ('Episode_Length_minutes', 'No_Ads'),
    
    ('Episode_Length_bin', 'Ads_per_minute'),
    ('Episode_Length_bin', 'Host_Popularity_percentage'),
    ('Episode_Length_bin', 'Number_of_Ads'),
    ('Episode_Length_bin', 'Ad_Density_Bin'),
    ('Episode_Length_bin', 'Episode_Sentiment'),
    ('Episode_Length_bin', 'No_Ads'),

    ('Ads_per_minute', 'Host_Popularity_percentage'),
    ('Ads_per_minute', 'Number_of_Ads'),
    ('Ads_per_minute', 'Ad_Density_Bin'),
    ('Ads_per_minute', 'Episode_Sentiment'),
    ('Ads_per_minute', 'No_Ads'),

    ('Host_Popularity_percentage', 'Number_of_Ads'),
    ('Host_Popularity_percentage', 'Ad_Density_Bin'),
    ('Host_Popularity_percentage', 'Episode_Sentiment'),
    ('Host_Popularity_percentage', 'No_Ads'),

    ('Number_of_Ads', 'Ad_Density_Bin'),
    ('Number_of_Ads', 'Episode_Sentiment'),
    ('Number_of_Ads', 'No_Ads'),

    ('Ad_Density_Bin', 'Episode_Sentiment'),
    ('Ad_Density_Bin', 'No_Ads'),

    ('Episode_Sentiment', 'No_Ads'),
]

# Create interaction features
for col1, col2 in inter_pairs:
    new_col = f"{col1}_{col2}"
    X[new_col] = X[col1].astype(str) + "_" + X[col2].astype(str)
    test[new_col] = test[col1].astype(str) + "_" + test[col2].astype(str)

    X[new_col] = X[new_col].astype('category')
    test[new_col] = test[new_col].astype('category')


from sklearn.preprocessing import OrdinalEncoder

# Convert object columns to category if not already
cat_cols = X.select_dtypes(include=['object', 'category']).columns

oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[cat_cols] = oe.fit_transform(X[cat_cols])
test[cat_cols] = oe.transform(test[cat_cols])


from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import numpy as np

# Define the model
xgb = XGBRegressor(
    objective='reg:squarederror',
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    device='cuda',
    verbosity=1,
    sampling_method='gradient_based',
    max_bin=512, 
    single_precision_histogram=True,
    grow_policy='lossguide',
    max_depth=0,
    eval_metric='rmse',
    enable_categorical=False
)

search_space = {
    'learning_rate': Real(0.01, 0.2, prior='log-uniform'), 
    'max_leaves': Integer(32, 128),
    'subsample': Real(0.6, 1.0),
    'gamma': Real(0.1, 5.0),
    'colsample_bytree': Real(0.3, 0.9),
    'colsample_bylevel': Real(0.5, 1.0),
    'reg_alpha': Real(1e-3, 10.0, prior='log-uniform'),
    'reg_lambda': Real(1e-3, 10.0, prior='log-uniform'),
    'min_child_weight': Integer(1, 10),
    'n_estimators': Integer(200, 800)
}

# Set up BayesSearchCV
opt = BayesSearchCV(
    estimator=xgb,
    search_spaces=search_space,
    n_iter=60,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=3,
    random_state=42
)

# Run the optimization
opt.fit(X, y)

# Results
print("Best Params:", opt.best_params_)
print("Best RMSE:", -opt.best_score_)

