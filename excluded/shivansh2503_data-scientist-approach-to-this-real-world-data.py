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


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


os.listdir("/kaggle/input/podcast-listening-time-prediction-dataset")


df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df1=df.copy()
df2 = pd.read_csv("/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv")
df3= pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


def get_data(df):

    #Podcast name not important
    df.drop('Podcast_Name',axis=1,inplace=True)

    #Only getting the episode number from episode title
    df["Episode_Title"]=int(df["Episode_Title"].str.split()[0][1])

    #Filling the values on nan with median
    df["Episode_Length_minutes"].fillna(df["Episode_Length_minutes"].median(), inplace=True)
    df["Guest_Popularity_percentage"].fillna(df["Guest_Popularity_percentage"].median(), inplace=True)

    #One hot encoding the genre column
    genre_dummies = pd.get_dummies(df["Genre"], prefix="Genre", dtype=int)
    df = pd.concat([df, genre_dummies], axis=1)
    df.drop("Genre", axis=1, inplace=True)

    #Label encoding the publication day
    label_encoder = LabelEncoder()
    df["Publication_Day"] = label_encoder.fit_transform(df["Publication_Day"])
    
    # Apply label encoding to Publication_Time
    df["Publication_Time"] = label_encoder.fit_transform(df["Publication_Time"])
    df["Episode_Sentiment"] = label_encoder.fit_transform(df['Episode_Sentiment'])

    #There is one missing value in number of ads so removing that row won't make any major change
    df.dropna(subset=['Number_of_Ads'], inplace=True)
    columns_to_convert = df.columns[10:20]
    
    # Convert the selected 'genre' columns to integers (0 and 1) as type of them might be bool sometimes as we one hot encoded using pandas(which encode in true/false)
    df[columns_to_convert] = df[columns_to_convert].astype(int)
    

    return df


df = get_data(df)


df


def get_model(df, model):
    # Define the features and target variable
    X = df.drop(columns=['Listening_Time_minutes', 'id'])  # Remove non-relevant columns
    y = df['Listening_Time_minutes']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("RMSE: ",rmse)
    print("R2: ",r2)

    return model, X_train, X_test, y_train, y_test


model = xgb.XGBRegressor()
model,_,_,_,_ = get_model(df, model)


df3 = get_data(df3)  #df3 is test.csv


df3


X = df3.drop(columns=[ 'id'])  # Remove non-relevant columns
y = df3['id']

#using the same model which we trained earlier
op=model.predict(X)

submission = pd.DataFrame({
    'id': df3['id'],
    'Listening_Time_minutes': op  
})

# Save it to a CSV file
submission.to_csv('submission.csv', index=False)


#df2 is the original dataset from the kaggle itself(The dataset provided in this competition is the synthetic one as stated in the description of this competition)
df2_copy = df2.copy()

#  As the original dataset has missing values in the target itself therefore dropping them
df_to_predict_later =df2_copy[df2_copy['Listening_Time_minutes'].isna()].copy()
df2_copy = df2_copy[~df2_copy['Listening_Time_minutes'].isna()].copy()


# Add a new `id` column to df2, continuing from the last id in df
df2_copy['id'] = range(df1['id'].max() + 1, df1['id'].max() + 1 + len(df2_copy))

# Reorder columns in df2 to match df
df2_copy = df2_copy[df1.columns]
combined_df = pd.concat([df1, df2_copy], ignore_index=True)


com_df = get_data(combined_df)


com_df.info()


model = lgb.LGBMRegressor()
model_stack,_,_,_,_ = get_model(com_df, model)


df3= pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
df3 = get_data(df3)

X = df3.drop(columns=[ 'id'])  # Remove non-relevant columns
y = df3['id']

op=model_stack.predict(X)

submission = pd.DataFrame({
    'id': df3['id'],
    'Listening_Time_minutes': op  # predicted values from Linear Regression
})

# Save it to a CSV file
submission.to_csv('submission.csv', index=False)


df_synthetic = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df1=df_synthetic.copy()
df_original = pd.read_csv("/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv")
df_test= pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


df2_copy = df_original.copy()

#  As the original dataset has missing values in the target itself therefore dropping them
df_to_predict_later =df2_copy[df2_copy['Listening_Time_minutes'].isna()].copy()
df2_copy = df2_copy[~df2_copy['Listening_Time_minutes'].isna()].copy()


# Add a new `id` column to df2, continuing from the last id in df
df2_copy['id'] = range(df1['id'].max() + 1, df1['id'].max() + 1 + len(df2_copy))

# Reorder columns in df2 to match df
df2_copy = df2_copy[df1.columns]
combined_df = pd.concat([df1, df2_copy], ignore_index=True)


combined_df.shape


combined_df["Listening_Time_minutes"].skew()


sns.histplot(combined_df["Listening_Time_minutes"], kde=True)
plt.show()


combined_df.info()





#Checking for duplicates
combined_df.duplicated().sum()


# Seperating numerical and categorical columns
target_col = "Listening_Time_minutes"
cat_columns = [col for col in combined_df.columns if combined_df[col].dtype == 'object']
num_columns = [col for col in combined_df.columns if combined_df[col].dtype != 'object' and col != target_col]


cat_columns


num_columns


combined_df[num_columns].describe()


for col in cat_columns:
    print(f"\nValue Counts for Column: {col}")
    print(combined_df[col].value_counts())
    print(len(combined_df[col].value_counts()))


# Analazying the target variable

import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(combined_df[target_col],kde=True)
plt.show()


sns.kdeplot(combined_df[target_col])
plt.show()


listning_time_0=combined_df[combined_df[target_col]==0].shape[0]
listning_time_0_Percent=(listning_time_0/combined_df.shape[0])*100


print(listning_time_0, listning_time_0_Percent)


combined_df[target_col].describe()


# Getting outliers using IQR strategy
q3 = combined_df[target_col].describe()[6]
q1 = combined_df[target_col].describe()[4]

iqr = q3-q1

lower_fense = q1-1.5*(iqr)
upper_fense =  q3+1.5*(iqr)
outliers = combined_df[(combined_df[target_col] > upper_fense) | (combined_df[target_col]<lower_fense)]


outliers


sns.boxplot(combined_df[target_col])


combined_df[combined_df['Host_Popularity_percentage'] > 100].describe()


from statsmodels.nonparametric.smoothers_lowess import lowess

for feature in num_columns[1:]:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=combined_df[feature], y=combined_df[target_col], alpha=0.2)
    
    # LOWESS smoothing line to capture non-linear trend
    lowess_smoothed = lowess(combined_df[target_col], combined_df[feature], frac=0.3)
    plt.plot(lowess_smoothed[:, 0], lowess_smoothed[:, 1], color='red', linewidth=2, label='LOWESS Curve')

    plt.title(f'{feature} vs. {target_col}')
    plt.xlabel(feature)
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#Making vcalues >100 to 100 in host_popularity as percentage must not be greater than 100
combined_df['Host_Popularity_percentage'] = combined_df['Host_Popularity_percentage'].clip(upper=100)


combined_df[num_columns].describe()


# MAking values >100 to 100 in guest_popularity as well and since there are missing values 
combined_df["Guest_Popularity_percentage"] = combined_df["Guest_Popularity_percentage"].clip(upper=100)

#Creating new feature making model to learn that whether the guest was there or not
combined_df['Has_Guest'] = combined_df['Guest_Popularity_percentage'].notna().astype(int)

#Making nan values to 0 assumning nans as that there is no gues present in the podcast
combined_df["Guest_Popularity_percentage"].fillna(0, inplace=True)





combined_df


# Rounding off the ads into integer as it can't be in decimals
# Also there is one nan in ads so removing it as it won't make any change

combined_df = combined_df[combined_df['Number_of_Ads'].notna()]

combined_df['Number_of_Ads'] = combined_df['Number_of_Ads'].round().astype(int)

sns.boxplot(combined_df["Number_of_Ads"])


#Clipping the values like 103 to 10 as 103 ads in a single 30min podcast is not possible
combined_df['Number_of_Ads'] = combined_df['Number_of_Ads'].clip(upper=10)


sns.boxplot(combined_df["Number_of_Ads"])


combined_df["Number_of_Ads"].describe()


combined_df[cat_columns[3]].unique()


combined_df[cat_columns[4]].unique()


# Mapping days to numbers (0 = Monday, ..., 6 = Sunday)
day_map = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
    'Friday': 4, 'Saturday': 5, 'Sunday': 6
}
combined_df['Day_Num'] = combined_df['Publication_Day'].map(day_map)

# Apply sin and cos transformation for cyclic encoding
combined_df['Day_Sin'] = np.sin(2 * np.pi * combined_df['Day_Num'] / 7)
combined_df['Day_Cos'] = np.cos(2 * np.pi * combined_df['Day_Num'] / 7)


# Mapping time of day to numbers
time_map = {
    'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3
}
combined_df['Time_Num'] = combined_df['Publication_Time'].map(time_map)

# Apply cyclic encoding
combined_df['Time_Sin'] = np.sin(2 * np.pi * combined_df['Time_Num'] / 4)
combined_df['Time_Cos'] = np.cos(2 * np.pi * combined_df['Time_Num'] / 4)

#Now since we have captured the cyclic nature now we shouyld drop original columns as keeping them can create a risk of multicollinearity!
combined_df.drop(["Day_Num", "Time_Num", "Publication_Time", "Publication_Day"],axis=1, inplace=True)


combined_df["Episode_Sentiment"].isna().sum()


combined_df = pd.get_dummies(combined_df, columns = ["Episode_Sentiment"], dtype=int)
# = pd.get_dummies(combined_df["Episode_Sentiment"])


combined_df["Episode_Length_minutes"].describe()


combined_df["Episode_Length_minutes"].skew()


sns.histplot(combined_df["Episode_Length_minutes"], kde=True)


combined_df[combined_df["Episode_Length_minutes"]==0]


combined_df[combined_df['Episode_Length_minutes'] == 0].shape


combined_df = combined_df[combined_df["Episode_Length_minutes"] !=0 ]


q1 = combined_df["Episode_Length_minutes"].quantile(0.25)
q3 = combined_df["Episode_Length_minutes"].quantile(0.75)

iqr = q3-q1
upper_fense = q3 + 1.5*(iqr)

combined_df["Episode_Length_minutes"] = combined_df["Episode_Length_minutes"].clip(upper=upper_fense)


sns.boxplot(combined_df["Episode_Length_minutes"])


combined_df['Episode_Length_missing'] = combined_df['Episode_Length_minutes'].isna().astype(int)


combined_df["Episode_Length_minutes"].fillna(combined_df["Episode_Length_minutes"].median(),inplace=True)


combined_df["Episode_Length_minutes"]


combined_df[combined_df['Episode_Length_missing'] ==1].shape


combined_df.head()


combined_df["Episode_Title"]


# combined_df['Episode_Title'].str.extract(r'(\d+)').astype(float)
combined_df['Episode_Number'] = combined_df['Episode_Title'].str.extract(r'(\d+)').astype(int)
combined_df.drop('Episode_Title', axis=1, inplace=True)


combined_df.head()


combined_df.info()


combined_df['Genre'].value_counts(dropna=False)


# pd.get_dummies(combined_df, columns=['Genre'], prefix='Genre',dtype=int)
combined_df = pd.get_dummies(combined_df, columns=['Genre'], prefix='Genre', dtype=int, drop_first=True)



combined_df.info()


combined_df['Guest_Host_Popularity_Ratio'] = combined_df['Guest_Popularity_percentage'] / (combined_df['Host_Popularity_percentage'] + 1e-6)

# 2. Ads Per Minute
combined_df['Ads_Per_Minute'] = combined_df['Number_of_Ads'] / (combined_df['Episode_Length_minutes'] + 1e-6)

# 3. Popularity Difference
combined_df['Popularity_Difference'] = combined_df['Guest_Popularity_percentage'] - combined_df['Host_Popularity_percentage']


combined_df_2 = combined_df.copy()

combined_df_2 = pd.get_dummies(combined_df_2, columns = ["Podcast_Name"], prefix = "Podcast", dtype=int)


combined_df_2.head()


def get_clean_data(df):
    
    combined_df = df
    
    target_col = "Listening_Time_minutes"
    podcast_name_column = 'Podcast_Name'

    
    cat_columns = [col for col in combined_df.columns if combined_df[col].dtype == 'object']
    num_columns = [col for col in combined_df.columns if combined_df[col].dtype != 'object' and col != target_col]

    #Making vcalues >100 to 100 in host_popularity as percentage must not be greater than 100
    combined_df['Host_Popularity_percentage'] = combined_df['Host_Popularity_percentage'].clip(upper=100)

    
    # MAking values >100 to 100 in guest_popularity as well and since there are missing values 
    combined_df["Guest_Popularity_percentage"] = combined_df["Guest_Popularity_percentage"].clip(upper=100)
    
    #Creating new feature making model to learn that whether the guest was there or not
    combined_df['Has_Guest'] = combined_df['Guest_Popularity_percentage'].notna().astype(int)
    
    #Making nan values to 0 assumning nans as that there is no gues present in the podcast
    combined_df["Guest_Popularity_percentage"].fillna(0, inplace=True)
    
    # Rounding off the ads into integer as it can't be in decimals
    # Also there is one nan in ads so removing it as it won't make any change
    
    combined_df = combined_df[combined_df['Number_of_Ads'].notna()]
    
    combined_df['Number_of_Ads'] = combined_df['Number_of_Ads'].round().astype(int)
    
    #Clipping the values like 103 to 10 as 103 ads in a single 30min podcast is not possible
    combined_df['Number_of_Ads'] = combined_df['Number_of_Ads'].clip(upper=10)
    
    
    # Mapping days to numbers (0 = Monday, ..., 6 = Sunday)
    day_map = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    combined_df['Day_Num'] = combined_df['Publication_Day'].map(day_map)

    # Apply sin and cos transformation for cyclic encoding
    combined_df['Day_Sin'] = np.sin(2 * np.pi * combined_df['Day_Num'] / 7)
    combined_df['Day_Cos'] = np.cos(2 * np.pi * combined_df['Day_Num'] / 7)
    
    
    # Mapping time of day to numbers
    time_map = {
        'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3
    }
    combined_df['Time_Num'] = combined_df['Publication_Time'].map(time_map)
    
    # Apply cyclic encoding
    combined_df['Time_Sin'] = np.sin(2 * np.pi * combined_df['Time_Num'] / 4)
    combined_df['Time_Cos'] = np.cos(2 * np.pi * combined_df['Time_Num'] / 4)
    
    #Now since we have captured the cyclic nature now we shouyld drop original columns as keeping them can create a risk of multicollinearity!
    combined_df.drop(["Day_Num", "Time_Num", "Publication_Time", "Publication_Day"],axis=1, inplace=True)
    
    combined_df = pd.get_dummies(combined_df, columns = ["Episode_Sentiment"], dtype=int)
    
    combined_df = combined_df[combined_df["Episode_Length_minutes"] !=0 ]
    
    q1 = combined_df["Episode_Length_minutes"].quantile(0.25)
    q3 = combined_df["Episode_Length_minutes"].quantile(0.75)
    
    iqr = q3-q1
    upper_fense = q3 + 1.5*(iqr)
    
    combined_df["Episode_Length_minutes"] = combined_df["Episode_Length_minutes"].clip(upper=upper_fense)
    
    combined_df['Episode_Length_missing'] = combined_df['Episode_Length_minutes'].isna().astype(int)
    
    combined_df["Episode_Length_minutes"].fillna(combined_df["Episode_Length_minutes"].median(),inplace=True)
    
    # combined_df['Episode_Title'].str.extract(r'(\d+)').astype(float)
    combined_df['Episode_Number'] = combined_df['Episode_Title'].str.extract(r'(\d+)').astype(int)
    combined_df.drop('Episode_Title', axis=1, inplace=True)
    
    # pd.get_dummies(combined_df, columns=['Genre'], prefix='Genre',dtype=int)
    combined_df = pd.get_dummies(combined_df, columns=['Genre'], prefix='Genre', dtype=int, drop_first=True)
    
    combined_df['Guest_Host_Popularity_Ratio'] = combined_df['Guest_Popularity_percentage'] / (combined_df['Host_Popularity_percentage'] + 1e-6)
    
    # 2. Ads Per Minute
    combined_df['Ads_Per_Minute'] = combined_df['Number_of_Ads'] / (combined_df['Episode_Length_minutes'] + 1e-6)
    
    # 3. Popularity Difference
    combined_df['Popularity_Difference'] = combined_df['Guest_Popularity_percentage'] - combined_df['Host_Popularity_percentage']
    
    combined_df = pd.get_dummies(combined_df, columns = ["Podcast_Name"], prefix = "Podcast", dtype=int)
    
    return combined_df


### Stacking again 

df2_copy = df_original.copy()

#  As the original dataset has missing values in the target itself therefore dropping them
df_to_predict_later =df2_copy[df2_copy['Listening_Time_minutes'].isna()].copy()
df2_copy = df2_copy[~df2_copy['Listening_Time_minutes'].isna()].copy()


# Add a new `id` column to df2, continuing from the last id in df
df2_copy['id'] = range(df1['id'].max() + 1, df1['id'].max() + 1 + len(df2_copy))

# Reorder columns in df2 to match df
df2_copy = df2_copy[df1.columns]
combined_df = pd.concat([df1, df2_copy], ignore_index=True)


df = get_clean_data(combined_df)


df.head()


df.info()


sns.histplot(df["Listening_Time_minutes"],kde=True)
plt.show()


from xgboost import XGBRegressor, plot_importance


model,X_train, X_test,y_train, y_test = get_model(df, XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor',random_state=42))


plot_importance(model, max_num_features=20, height=0.5)
plt.show()


from sklearn.model_selection import RandomizedSearchCV


param_dist = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}



xgb = XGBRegressor(objective='reg:squarederror', random_state=42)


search = RandomizedSearchCV(
    xgb,
    param_distributions=param_dist,
    n_iter=30,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)


# search.fit(X_train, y_train)

# print("Best Parameters:\n", search.best_params_)


# best_params = search.best_params_
best_params = {'subsample': 1.0, 'reg_lambda': 2, 'reg_alpha': 1, 'n_estimators': 500, 'max_depth': 7, 'learning_rate': 0.1, 'gamma': 0, 'colsample_bytree': 1.0}

print(best_params)


# X_train_small, X_val, y_train_small, y_val = train_test_split(
#     X_train, y_train, test_size=0.2, random_state=42)


# model = XGBRegressor(**best_params,tree_method='gpu_hist', predictor='gpu_predictor', random_state=42)

# eval_set = [(X_train_small, y_train_small), (X_val, y_val)]
# model.fit(X_train_small, y_train_small,
#           eval_metric="rmse",
#           eval_set=eval_set,
#           verbose=False)

# # Plot learning curve
# results = model.evals_result()

# epochs = len(results['validation_0']['rmse'])
# x_axis = range(0, epochs)

# plt.figure(figsize=(10, 6))
# plt.plot(x_axis, results['validation_0']['rmse'], label='Train')
# plt.plot(x_axis, results['validation_1']['rmse'], label='Validation')
# plt.xlabel("Epoch")
# plt.ylabel("RMSE")
# plt.title("XGBoost Learning Curve")
# plt.legend()
# plt.grid(True)
# plt.show()


# # Therefore dropping very rarely used features by xgboost

# importances = model.feature_importances_
# feature_names = X_train.columns
# importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})


# # 3. Drop least important features (e.g., bottom 10%)
# threshold = np.percentile(importances, 10)
# features_to_keep = importance_df[importance_df['importance'] > threshold]['feature'].tolist()


# len(features_to_keep)


# X_train_reduced = X_train_small[features_to_keep]
# X_val_reduced = X_val[features_to_keep]
# X_test_reduced = X_test[features_to_keep]


# # Fit XGB with early stopping



# xgb_final = XGBRegressor(**best_params, tree_method='gpu_hist', predictor='gpu_predictor',random_state=42)

# xgb_final.fit(
#     X_train_reduced, y_train_small,
#     eval_set=[(X_val_reduced, y_val)],
#     eval_metric='rmse',
#     early_stopping_rounds=30,
#     verbose=True
# )

# # Save the best number of estimators
# best_n_estimators = xgb_final.best_iteration
# print(f"Best number of trees: {best_n_estimators}")


# from sklearn.ensemble import StackingRegressor
# from sklearn.linear_model import Ridge
# from sklearn.ensemble import ExtraTreesRegressor
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor



# # Your best XGBRegressor
# xgb_model = XGBRegressor(**best_params, tree_method='gpu_hist', predictor='gpu_predictor',random_state=42)

# # LightGBM and CatBoost (silent training)
# lgb_model = LGBMRegressor(device='gpu', random_state=42)
# cat_model = CatBoostRegressor(verbose=0,task_type='GPU', random_state=42)


# # Final meta-model (Ridge is great for blending)
# meta_model = Ridge(alpha=1.0)

# # Define the stacking regressor
# stack = StackingRegressor(
#     estimators=[
#         ('xgb', xgb_model),
#         ('lgb', lgb_model),
#         ('cat', cat_model),
#     ],
#     final_estimator=meta_model,
#     passthrough=True,
#     n_jobs=1
# )

# # Fit on full training data
# stack.fit(X_train_reduced, y_train_small)

# # Predict on test set
# stack_preds = stack.predict(X_test_reduced)
# rmse_stack = np.sqrt(mean_squared_error(y_test, stack_preds))
# print(f"✅ Stacking RMSE: {rmse_stack:.4f}")








