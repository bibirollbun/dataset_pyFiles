import numpy as np          
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from xgboost import XGBRegressor  
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder 

pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', 100) 


# Import packages for warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

from IPython.core.display import display, HTML    
def color_text(text, color):
    display(HTML(f'<p style="color:{color}; font-weight:bold; margin: 0;">{text}</p>'))                


train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')    


print('Train:', train_df.shape)
print('Test:', test_df.shape)   


print('Train:', train_df.duplicated().sum())
print('Test:', test_df.duplicated().sum())


# Change all columns to lower case
train_df.columns = train_df.columns.str.lower()
test_df.columns = test_df.columns.str.lower()

print(f'Columns in Train Dataset:\n {train_df.columns}\n')
print('Columns in Test Dataset', test_df.columns)


train_df.head()


test_df.head()  


# Check for infinite values in the `premium_amount` column
train_df['listening_time_minutes'].isin([np.inf, -np.inf]).any()     


# Check whether both datasets train_df and test_df have same categorical columns
a = train_df.select_dtypes('object').columns.tolist() == test_df.select_dtypes('object').columns.tolist()

if a:
    print('âœ… Both Train & Test dataset have same categorical columns.')
else:
    print('Mismatch detected! The categorical columns differ between Train & Test datasets.')  


# Check whether both datasets have same unique categories for categorical variables
# Extract all catgorical variables
cat_col = train_df.select_dtypes('object').columns.tolist() 

# Creat dictionary to hold unqiue categories for each categorical variable for train and test datasets
categorical_train = {var: set(train_df[var].dropna().unique()) for var in cat_col}
categorical_test = {var: set(test_df[var].dropna().unique()) for var in cat_col}

if categorical_train == categorical_test:
    print('âœ… Both Train & Test datasets have the same unique categories for each categorical variable.')
else:
    print('Mismatch detected! Some categorical variables have different unique categories.')


# Extract all numerical variables
num_col = [col for col in train_df.columns if col not in cat_col]
num_col = [col for col in num_col if col not in ['listening_time_minutes', 'id']]  


# Descriptive Statistics - (train set)
train_df[num_col].describe()


# Count the number of rows where 'episode_length_minutes' is equal to 0
train_df[train_df['episode_length_minutes'] == 0]       


# Replace `0` in episode_length_minutes with NaN 
train_df.loc[train_df['episode_length_minutes'] == 0, 'episode_length_minutes'] = np.nan     


# Check episode_length_minutes with `0` value
train_df[train_df['episode_length_minutes']==0].shape[0]       


# Check if `episode_length_minutes` is less than `listening_time_minutes` 
train_df[train_df['episode_length_minutes'] < train_df['listening_time_minutes']].shape[0]    


# Descriptive Statistics - (Test set)
test_df[num_col].describe()     


# Check rows with `episode_length_minutes` above 350 
test_df[test_df['episode_length_minutes']>350]  


# Calculate median from train set for `episode_length_minutes`          
episode_length_train_median = train_df['episode_length_minutes'].median()
episode_length_train_max = train_df['episode_length_minutes'].max()

# Impute abnormal test values in `episode_length_minutes` with train median in test set 
test_df.loc[test_df['episode_length_minutes'] > episode_length_train_max, 'episode_length_minutes'] = episode_length_train_median


# Count occurrences in `number_of_ads` to check for anomalies for train_df   
train_df['number_of_ads'].value_counts()     


# Count occurrences in `number_of_ads` to check for anomalies for test_df   
test_df['number_of_ads'].value_counts()     


# Replace `number_of_ads` values greater than 3 with 0
for df in [train_df, test_df]:
    df.loc[df['number_of_ads'] > 3, 'number_of_ads'] = 0


agg_data = train_df.groupby(['podcast_name', 'episode_title', 'genre'])[[
                             'episode_length_minutes', 'number_of_ads']].agg([
                             'min', 'max', 'nunique'])

agg_data.head(4)   


train_df[(train_df['podcast_name'] == "Athlete's Arena") & (train_df['episode_title'] == "Episode 1")].head(2)


# Check how many episodes are shared between train and test
train_episodes = set(train_df['podcast_name'].str.strip().str.lower() + '_' + train_df[
                     'episode_title'].str.strip().str.lower())
test_episodes = set(test_df['podcast_name'].str.strip().str.lower() + '_' + test_df[
                    'episode_title'].str.strip().str.lower())

overlapping_episodes = train_episodes.intersection(test_episodes)
color_text(f'âž© Number of overlapping podcast episodes between Train & Test Dataset: {len(overlapping_episodes)}', 'black')    


def podcast_episode(df, name):
    episode_cnt_per_podcast = df.groupby(['podcast_name'])['episode_title'].nunique()
    all_have_100 = (episode_cnt_per_podcast == 100).all()
    podcast_count = df['podcast_name'].nunique()
    if all_have_100:
        return(f'âœ… There are {podcast_count} podcasts that have exactly 100 episodes each in the {name}.')
    else:
        return(f'Not all podcasts have exactly 100 episodes in the {name}.')

print(f'{podcast_episode(train_df, "training dataset")}\n') 
print(f'{podcast_episode(test_df, "test dataset")}') 


# Check without duplicates without `id`
print('Duplicates without id in Train:', test_df.drop(columns=['id']).duplicated().sum())
print('Duplicates without id in Test:', train_df.drop(columns=['id']).duplicated().sum())


# Check missing values in train_df
train_miss = train_df.isna().sum()
train_perct = train_df.isna().sum() / len(train_df) * 100

test_miss = test_df.isna().sum()
test_perct = test_df.isna().sum() / len(train_df) * 100

# Create a DataFrame for missing value summary for train & test datasets
missing_summary = pd.DataFrame({
    'train_count': train_miss,
    'train_%': train_perct,
    'test_count': test_miss,
    'test_%': test_perct
})

missing_summary = missing_summary[missing_summary['train_count'] > 0]
missing_summary = missing_summary[missing_summary['test_count'] > 0]

print('\n' + '='*75) 
print(f"{'SUMMARY OF MISSING VALUES: TRAIN & TEST DATASETS':^75}") 
print('='*75 + '\n')

missing_summary


# Create Histogram for Numerical Variables  
# Setup subplots
fig, axes = plt.subplots(len(num_col), 2, figsize = (13, 5 * len(num_col)))

# Color for datasets
palette = {'Train':'green', 'Test':'orange'}

# Plot Histogram
for i, var in enumerate(num_col):
    axes[i, 0].hist(train_df[var], alpha=0.5, color=palette['Train'], label='Train')
    axes[i, 0].hist(test_df[var], alpha=0.5, color=palette['Test'], label='Test')
    axes[i, 0].set_title(f'Histogram for {var}', weight='bold')
    axes[i, 0].legend()

    # Prepare data for boxplot 
    combined = pd.concat([train_df[var].to_frame().assign(dataset='Train'),
                          test_df[var].to_frame().assign(dataset='Test')
                         ])
    # Plot Boxplot
    sns.boxplot(data=combined, x='dataset', y=var, ax=axes[i, 1], palette=palette)
    axes[i, 1].set_title(f'Boxplot for {var}', weight='bold')

plt.tight_layout()
plt.show() 


# Set subplots 
fig, axes = plt.subplots(1, 2, figsize=(10, 4)) 

# Plot Histogram 
sns.histplot(train_df['listening_time_minutes'], ax=axes[0], kde=True) 
axes[0].set_title('Histogram of listening_time_minutes',fontsize=10, weight='bold') 

# Boxplot for the original dataset 
sns.boxplot(x=train_df['listening_time_minutes'], ax=axes[1]) 
axes[1].set_title('Boxplot of listening_time_minutes', fontsize=10, weight='bold') 

# Adjust layout 
plt.tight_layout() 
plt.show()


train_df['listening_time_minutes'].describe()


# Count of rows where listening_time_minutes is 0
train_df[train_df['listening_time_minutes']==0].shape[0]


def heatmap(df, df_name):
    plt.figure(figsize=(8, 5))
    df = df.drop('id', axis=1)
    sns.heatmap(df.corr(method='pearson', numeric_only=True), annot=True, cmap='coolwarm')
    plt.title(f'Correlation Heatmap for {df_name} Data', fontsize=14, weight='bold')
    plt.show();

heatmap(train_df, 'Train')
heatmap(test_df, 'Test')


import math   

# Total number of numerical variables
n_vars = len(num_col)

# Compute number of rows needed
n_cols = 2
n_rows = math.ceil(n_vars / n_cols)

# Create subplots
fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))

# Flatten the axes array to iterate over each subplot
axs = axs.flatten()

# Plot Scatterplot for each variables based on listening_time_minutes
for i, var in enumerate(num_col):
    ax = axs[i]
    sns.scatterplot(data=train_df, x=var, y='listening_time_minutes', ax=ax)
    ax.set_title(f'Scatterplot of {var} vs listening_time_minutes', size=8, weight='bold')

# Hide extra subplots
for j in range(len(num_col), len(axs)):
    axs[j].axis('off')     


# Check rows with `episode_length_minutes` above `121` 
train_df[train_df['episode_length_minutes'] > 121]      


# Filter data 
filtered_df = train_df[train_df['episode_length_minutes'] < 121]
# Create Regplot
plt.figure(figsize=(8, 3))  
sns.regplot(data=filtered_df, x='episode_length_minutes', y='listening_time_minutes',
            scatter_kws = dict(alpha=0.3, s=20, edgecolors='white'), line_kws={'color': 'red'})
plt.title('Regplot: episode_length_minutes vs. listening_time_minutes', fontsize=10, weight='bold');   


def boxplot(var):
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=train_df, x=var, y='listening_time_minutes', palette='Set2')
    plt.title(f'Boxplot of {var} vs listening_time_minutes', size=12, weight='bold')   


boxplot('number_of_ads')


# Calculate medians for `listening_time_minutes` grouped by number_of_ads 
median_values = train_df.groupby('number_of_ads')['listening_time_minutes'].median().reset_index()

# Update data dictionary with these median values 
data = {'number_of_ads': median_values['number_of_ads'].tolist(), 
        'median_listening_time_minutes': median_values['listening_time_minutes'].tolist()}

df = pd.DataFrame(data)

# Regplot
plt.figure(figsize=(8, 3)) 
sns.regplot(data=df, x='number_of_ads', y='median_listening_time_minutes', ci=None) 
plt.title('Regplot: median_listening_time_minutes vs. number_of_ads', 
          fontsize=12, weight='bold') 
plt.grid(True)
plt.show()      


sns.lmplot(data=filtered_df, x='episode_length_minutes',y='listening_time_minutes',hue='number_of_ads', 
           scatter_kws=dict(alpha=0.3, s=20, edgecolors='white'),
           line_kws={'linewidth': 2}, height=4, aspect=2)
plt.title('Effect of Episode Duration and Ad Count on Listening Time', fontsize=10, weight='bold')
plt.grid(True)
plt.show();       


boxplot('genre')   

train_df.groupby('genre')['listening_time_minutes'].describe()


boxplot('publication_day') 
train_df.groupby('publication_day')['listening_time_minutes'].describe()


boxplot('episode_sentiment')  
train_df.groupby('episode_sentiment')['listening_time_minutes'].describe()


boxplot('publication_time') 
train_df.groupby('publication_time')['listening_time_minutes'].describe()


def ordinal(df):
    sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    df['episode_sentiment_enco'] = df['episode_sentiment'].map(sentiment_mapping)  
    return df

train_df = ordinal(train_df)
test_df = ordinal(test_df)  

def create_features(df):
    df['ad_density'] = df['number_of_ads'] / df['episode_length_minutes']
    # Do ads have less negative impact if the guest or host are popular
    df['ad_density_host'] = df['ad_density'] * df['host_popularity_percentage']
    df['ad_density_guest'] = df['ad_density'] * df['guest_popularity_percentage']
    # Do ads feel more irritating or acceptable depending on the episode's overall sentiment
    df['ad_density_sentiment'] = df['ad_density'] * df['episode_sentiment_enco']
    # Popularity interaction 
    df['total_popularity'] = df['guest_popularity_percentage'] * df['host_popularity_percentage']
    # Does large diff in popularity affects engagement
    df['popularity_diff'] = df['guest_popularity_percentage'] - df['host_popularity_percentage']
    # Ad exposure in an episode 
    df['epi_length_ads'] = df['episode_length_minutes'] * df['number_of_ads']
    # Popularity ratio
    df['popularity_ratio'] = df['guest_popularity_percentage'] / df['host_popularity_percentage']
    # Interaction: Combine length and host popularity to potentially capture engagement trends 
    df['episode_length_host'] = df['episode_length_minutes'] * df['host_popularity_percentage']
    # Polynomial Feature: Squared popularity to model potentially accelerating returns
    df['host_pop_squared'] = df['host_popularity_percentage'] ** 2
    df['guest_pop_squared'] = df['guest_popularity_percentage'] ** 2
    return df


# Extract categorical columns excluding `episode_title`
group_col = [col for col in cat_col if col != "episode_title"]

# Create Group-wise features
def create_group_features(df):
    for i, col1 in enumerate(group_col):
        for col2 in group_col[i+1:]:
            df[f"{col1}_{col2}"] = df[col1] + "_" + df[col2]

# Apply
create_group_features(train_df)
create_group_features(test_df)               

# Extract episode number from `episode_title`
train_df['episode_title_num'] = train_df['episode_title'].str.extract(r'(\d+)').astype(float)  
test_df['episode_title_num'] = test_df['episode_title'].str.extract(r'(\d+)').astype(float)  

# Create group feature: `podcast_name` + `episode_title`
train_df['podcast_name_episode_title'] = train_df['podcast_name'] + "_" + train_df['episode_title']
test_df['podcast_name_episode_title'] = test_df['podcast_name'] + "_" + test_df['episode_title']

# Extract all categorical variables 
cat_col = train_df.select_dtypes(include='object').columns.tolist()   

# Excluding 'genre' and 'podcast_name_episode_title' from categorical features 
# because we want to apply additional target aggregation methods (mean, median, standard deviation) to these features.
# The remaining categorical features will be used for Target Encoding, which only applies mean aggregation.
cat_col = [col for col in cat_col if col not in ['genre', 'podcast_name_episode_title']]   


train_df.head(1)          


# Covert data type to category from object
def change_type(df):
    for col in df.select_dtypes('object').columns:
        df[col] = df[col].astype('category')

change_type(train_df)
change_type(test_df)        


def add_group_target_encoding(df_train, df_valid, df_test, group_cols, target_col='listening_time_minutes'):
    # Calculate group stats only in training set
    for col in group_cols:   
        agg_funcs = ['mean', 'median', 'std']
        agg_stats = df_train.groupby(col)[target_col].agg(agg_funcs).reset_index()
        agg_stats.columns = [col] + [f"{col}_{func}" for func in agg_funcs]

        # Merge into train, valid, test
        df_train = df_train.merge(agg_stats, on=col, how='left')
        df_valid = df_valid.merge(agg_stats, on=col, how='left')
        df_test = df_test.merge(agg_stats, on=col, how='left')

    return df_train, df_valid, df_test


def add_listen_ratio_encoding(df_train, df_valid, df_test, group_cols):
    # Calculate listen_ratio only in training set
    df_train['listen_ratio'] = df_train['listening_time_minutes'] / df_train['episode_length_minutes']
    
    for col in group_cols:
        agg_funcs = ['mean', 'median', 'std']
        agg_stats = df_train.groupby(col)['listen_ratio'].agg(agg_funcs).reset_index()
        agg_stats.columns = group_cols + [f"{col}_listen_{func}" for func in agg_funcs]

        # Merge into train, valid, & test
        df_train = df_train.merge(agg_stats, on=col, how='left')
        df_valid = df_valid.merge(agg_stats, on=col, how='left')
        df_test = df_test.merge(agg_stats, on=col, how='left')

    # Drop listen_ratio after aggregation
    df_train.drop(columns=['listen_ratio'], inplace=True)

    return df_train, df_valid, df_test


def add_episode_length_group_stats(df_train, df_valid, df_test, target_col='listening_time_minutes'):
    # Create quantile bin (100 groups)
    df_train['episode_len_bin'] = pd.qcut(df_train['episode_length_minutes'], q=100, duplicates='drop')

    # Map bins to valid/test
    bin_map = df_train[['episode_length_minutes', 'episode_len_bin']].drop_duplicates()
    df_valid = df_valid.merge(bin_map, on='episode_length_minutes', how='left')
    df_test = df_test.merge(bin_map, on='episode_length_minutes', how='left')

    # Aggregation
    agg_funcs = ['mean', 'median', 'std'] 
    agg_df = df_train.groupby('episode_len_bin')[target_col].agg(agg_funcs).reset_index()
    agg_df.columns = ['episode_len_bin'] + [f"episode_len_bin_{func}" for func in agg_funcs]

    # Merge into train/valid/test
    df_train = df_train.merge(agg_df, on='episode_len_bin', how='left')
    df_valid = df_valid.merge(agg_df, on='episode_len_bin', how='left')
    df_test = df_test.merge(agg_df, on='episode_len_bin', how='left')

    # Drop bin column after feature creation
    df_train.drop(columns=['episode_len_bin'], inplace=True)
    df_valid.drop(columns=['episode_len_bin'], inplace=True)
    df_test.drop(columns=['episode_len_bin'], inplace=True)

    return df_train, df_valid, df_test  


# Features & Target
# Keeping the target variable for computing target statistics (mean, median, std).
# The target variable will be removed during KFold cross-validation after feature creation.
X = train_df.drop(columns=['id'])   
y = train_df['listening_time_minutes']

X_test = test_df.drop(columns=['id'])

# Creating list of columns with missing values
num_missing = ['episode_length_minutes', 'guest_popularity_percentage', 'number_of_ads']

# Data Dimension 
X.shape, X_test.shape, y.shape                        


# CatBoost hyper-parameter             
catboost_params = {
    'iterations': 12000,
    'learning_rate': 0.01,
    'depth': 16,
    'l2_leaf_reg': 7.0,
    'random_seed': 42,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'task_type': 'GPU'  
}        


# Function to compute RMSE     
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Initialize OOF arrays      
oof_preds = np.zeros(len(X))
# Test prediction
cat_test_preds = np.zeros(len(X_test))
# Store RMSE
rmse_scores = []   

kf = KFold(n_splits=5, shuffle=True, random_state=42)  

# KFold cross-validation
color_text("### Training CatBoost Model ###", "black")
for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
    color_text(f"\n### Training Fold {fold+1} ###", "black")

    # Train-valid Split 
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]  

    # Reset index after split
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_valid = X_valid.reset_index(drop=True)
    y_valid = y_valid.reset_index(drop=True)   

    # Missing value Imputation 
    imputer = SimpleImputer(strategy='mean')
    
    # Fit imputer to the train data
    imputer.fit(X_train[num_missing])
    
    # Transform to the train, validation, & test data
    X_train[num_missing] = imputer.transform(X_train[num_missing])
    X_valid[num_missing] = imputer.transform(X_valid[num_missing])

    X_test_fold = X_test.copy()     
    X_test_fold[num_missing] = imputer.transform(X_test_fold[num_missing])

    # Add aggregated stats by groups
    for col in ['genre', 'number_of_ads', 'podcast_name_episode_title']:
        X_train, X_valid, X_test_fold = add_group_target_encoding(
            df_train=X_train,  
            df_valid=X_valid,
            df_test=X_test_fold,
            group_cols=[col],
            target_col='listening_time_minutes')
    
    # Add listen ratio aggregation stats
    for col in ['genre', 'number_of_ads', 'podcast_name_episode_title']:
        X_train, X_valid, X_test_fold = add_listen_ratio_encoding(
            df_train=X_train,
            df_valid=X_valid,
            df_test=X_test_fold,
            group_cols=[col])  

    # Add episode_length group aggregation
    X_train, X_valid, X_test_fold = add_episode_length_group_stats(
        df_train=X_train,
        df_valid=X_valid,
        df_test=X_test_fold,
        target_col='listening_time_minutes')   

    # Create new features
    X_train = create_features(X_train)
    X_valid = create_features(X_valid)
    X_test_fold = create_features(X_test_fold)

    # Drop features 
    X_train.drop(columns=['listening_time_minutes', 'genre', 'podcast_name_episode_title'], inplace=True)
    X_valid.drop(columns=['listening_time_minutes', 'genre', 'podcast_name_episode_title'], inplace=True)
    X_test_fold.drop(columns=['genre', 'podcast_name_episode_title'], inplace=True)

    # Categorical Target Encoding 
    enco_cat = cat_col  
    te = TargetEncoder(cols=enco_cat, smoothing=10)
    X_train = te.fit_transform(X_train, y_train)
    X_valid = te.transform(X_valid)
    X_test_fold = te.transform(X_test_fold)

    # Impute missing values 
    final_imputer = SimpleImputer(strategy='mean')
    X_train = pd.DataFrame(final_imputer.fit_transform(X_train), columns=X_train.columns)
    X_valid = pd.DataFrame(final_imputer.transform(X_valid), columns=X_valid.columns)
    X_test_fold = pd.DataFrame(final_imputer.transform(X_test_fold), columns=X_test_fold.columns)
    
    # Training CatBoost Model
    cat_model = CatBoostRegressor(**catboost_params, verbose=0)

    cat_model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=True,             
        early_stopping_rounds=100, verbose=0)
        
    # OOF Predictions 
    y_valid_pred = cat_model.predict(X_valid) 
    oof_preds[valid_idx] = y_valid_pred
    
    # Test Set Predictions (Averaged over folds)
    cat_test_preds += cat_model.predict(X_test_fold) / kf.n_splits  
    
    # Calculate RMSE for each fold
    fold_rmse = rmse(y_valid, y_valid_pred)
    rmse_scores.append(fold_rmse)
    color_text(f"Fold {fold+1} RMSE: {fold_rmse:.6f}", "green") 

# Mean RMSE across folds
mean_rmse = np.mean(rmse_scores)
color_text(f"Mean RMSE across folds: {mean_rmse:.6f}", "blue")
# Compute final CV RMSE
cv_score = rmse(y, oof_preds)
color_text(f"Overall CV (Out-of-Fold) RMSE: {cv_score:.6f}", "blue")   

color_text(f"\nâž© Total Number of Features Used: {X_train.shape[1]}", "black")
color_text(f"âž© Features Used:\n{X_train.columns}", "black")  

# Create the submission DataFrame      
submission = pd.DataFrame({
    'id': test_df['id'],
    'Listening_Time_minutes': cat_test_preds
})

# Save the submission file
submission.to_csv('submission.csv', index=False)
color_text("Final submission file created", "green")  

submission.head()         


# Get feature importances
importances = cat_model.get_feature_importance()
feature_names = X_train.columns

# Combine into a DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

feature_importance_df.reset_index(drop=True, inplace=True)  

plt.figure(figsize=(8, 10))
sns.barplot(data=feature_importance_df, x='Importance', y='Feature')
plt.title('CatBoost Feature Importance', weight='bold');   


# Test prediction           
plt.figure(figsize=(8, 4))
plt.hist(data=submission, x='Listening_Time_minutes', bins=100)
plt.title('Data Distribution of Test Prediction', size=12, weight='bold');      

