import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import FileLink
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler 

warnings.simplefilter("ignore")


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


traindataset = '/kaggle/input/playground-series-s5e4/train.csv'

train_df = pd.read_csv(traindataset)


testdataset = '/kaggle/input/playground-series-s5e4/test.csv'

test_df = pd.read_csv(testdataset)


train_df.info()


train_df.describe()


pd.isnull(train_df).sum()


train_df.dtypes


train_df.head(10)


train_df['Podcast_Name'].value_counts().count()


plt.figure(figsize=(20,16))

train_df['Podcast_Name'].value_counts().plot(kind='pie', startangle=140, autopct='%1.3f%%')

plt.show()


unique_podcast = train_df['Podcast_Name'].value_counts()

unique_podcast


train_df['Podcast_Name'].count()


plt.figure(figsize=(20,8))
sns.boxplot(x='Podcast_Name', y= 'Listening_Time_minutes', data=train_df)
plt.show()


plt.figure(figsize=(60, 48))

train_df['Episode_Title'].value_counts().plot(kind='pie', startangle=140, autopct='%1.3f%%')

plt.show()


plt.figure(figsize=(20,8))
sns.boxplot(x='Episode_Title', y='Listening_Time_minutes', data=train_df)
plt.show()


# os.getcwd()

# files = os.listdir('/kaggle/working')

# files


unique_episodetitle = train_df['Episode_Title'].unique()

unique_episodetitle.sort()

unique_episodetitle


unique_episodetitle_count = train_df['Episode_Title'].value_counts()

unique_episodetitle_count.to_dict()


episodetitle_count = train_df['Episode_Title'].count()

episodetitle_count


# File = FileLink('sns_hist_episode_title.pdf')

# File


genre_count = train_df['Genre'].value_counts()

print(genre_count)


genre_obs = train_df['Genre'].count()

genre_obs


plt.figure(figsize=(10,8))
sns.boxplot(x='Genre', y='Listening_Time_minutes', data=train_df)
plt.title("Genre Vs Listening_Time_minutes")
plt.show()


genre_count = train_df['Publication_Day'].value_counts()

genre_count


genre_obs = train_df['Publication_Day'].count()

genre_obs 


plt.figure(figsize=(10,8))
sns.boxplot(x='Publication_Day', y='Listening_Time_minutes', data=train_df)
plt.title("Genre Vs Listening_Time_minutes")
plt.show()


train_df['Publication_Time'].value_counts()


pub_obs = train_df['Publication_Time'].count()

print(pub_obs)


plt.figure(figsize=(10,8))
sns.boxplot(x='Publication_Time', y='Listening_Time_minutes', data=train_df)
plt.show()


episode_sentiment = train_df['Episode_Sentiment'].value_counts()

episode_sentiment


episode_sentiment_obs = train_df['Episode_Sentiment'].count()

episode_sentiment_obs


plt.figure(figsize=(10,8))
sns.boxplot(x='Episode_Sentiment', y='Listening_Time_minutes', data=train_df)
plt.show()


train_df.info()


# creating a dataframe with categorical variables
cat_train_df = train_df.drop(columns = ['id','Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes'])

print(cat_train_df)


# Plotting the categorical variables 

col_cat_train_df  = cat_train_df.columns.to_list()

n_columns = 3 

n_row = -(-len(col_cat_train_df) // n_columns)

plt.figure(figsize=(7 * n_columns, 6 * n_row))

for i, column in enumerate(col_cat_train_df, 1):
    plt.subplot(n_row, n_columns, i)
    sns.countplot(x=cat_train_df[column])
    plt.title(column)

plt.show()


num_train_df = train_df.drop(columns = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'])

print(num_train_df)


num_train_df.describe()


scaler = StandardScaler()

cols_to_impute = ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']

scaled_data = scaler.fit_transform(num_train_df[cols_to_impute])



simple_imputer = SimpleImputer(strategy='mean')

imputed_scaled_data = simple_imputer.fit_transform(scaled_data)

imputed_data = scaler.inverse_transform(imputed_scaled_data)

num_train_df[cols_to_impute] = imputed_data


corr = num_train_df.corr()

sns.heatmap(corr, annot=True, cmap="crest", fmt=".2f")


plt.figure(figsize=(10,8))
sns.scatterplot(data=num_train_df, x='Episode_Length_minutes', y='Listening_Time_minutes', alpha=1)
plt.title('Episode_Length_minutes Vs Listening_Time_minutes')
plt.show()


plt.figure(figsize=(10,8))
sns.lineplot(data=num_train_df, x='Episode_Length_minutes', y='Listening_Time_minutes', alpha=1)
plt.show()


num_train_df.describe()


num_train_df.median()


# sns.histplot(x=num_train_df['Guest_Popularity_percentage'])

# plt.figure(figsize=(20,16))

num_train_df.hist(figsize=(20,16), bins=500)

# plt.show()


pd.plotting.scatter_matrix(num_train_df, figsize=(40, 36), diagonal='kde')


train_df = train_df.drop(columns=['Episode_Title'])


# Define mappings for non-time strings
custom_time_mapping = {
    'Morning': 9,
    'Afternoon': 15,
    'Evening': 18,
    'Night': 21,
    'Noon': 12,
    'Midnight': 0
}

def extract_hour(val):
    try:
        return pd.to_datetime(val, format='%H:%M').hour
    except:
        return custom_time_mapping.get(val, np.nan)

train_df['Publication_Hour'] = train_df['Publication_Time'].apply(extract_hour)

train_df = train_df.drop(columns=['Publication_Time'])


from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder

# Target encode Podcast_Name
target_encoder = TargetEncoder()
train_df['Podcast_Name_encoded'] = target_encoder.fit_transform(train_df['Podcast_Name'], train_df['Listening_Time_minutes'])

# One-hot encode Genre, Publication_Day, Episode_Sentiment
ohe_cols = ['Genre', 'Publication_Day', 'Episode_Sentiment']
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
ohe_encoded = ohe.fit_transform(train_df[ohe_cols])
ohe_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(ohe_cols))

# Combine with original DataFrame
train_df = pd.concat([train_df.drop(columns=ohe_cols + ['Podcast_Name']), ohe_df], axis=1)



train_df.info()


# Drop overlapping columns from num_train_df
numerical_df = num_train_df.drop(columns=['id', 'Listening_Time_minutes'])

# Keep target separately
target = num_train_df['Listening_Time_minutes']

# Final training features
X = pd.concat([numerical_df.reset_index(drop=True), train_df.drop(columns=['id', 'Listening_Time_minutes']).reset_index(drop=True)], axis=1)
y = target



from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')



duplicates = X.columns[X.columns.duplicated()]
print("Duplicate columns:", list(duplicates))

# Drop duplicate columns
X = X.loc[:, ~X.columns.duplicated()]


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Fit and transform on the training data
X_train_imputed = imputer.fit_transform(X_train)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_imputed, y_train)


# Fit and transform on the training data
X_val_imputed = imputer.fit_transform(X_val)

# Predict and evaluate
y_pred = model.predict(X_val_imputed)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f'Validation RMSE: {rmse:.4f}')


from sklearn.metrics import r2_score

# R² score tells you how well the model explains the variance in the data
r2 = r2_score(y_val, y_pred)
print(f'R² Score: {r2:.4f}')



from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Cross-Validation with Random Forest
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
cv_scores = cross_val_score(model_rf, X_train_imputed, y_train, cv=2, scoring='neg_root_mean_squared_error')

print(f'Random Forest CV RMSE Scores: {-cv_scores}')
print(f'Average CV RMSE: {-np.mean(cv_scores):.4f}')



# Train on full training data and evaluate on validation set 
model_rf.fit(X_train_imputed, y_train)

# Impute validation set
X_val_imputed = imputer.transform(X_val)

y_pred_rf = model_rf.predict(X_val_imputed)

rmse_rf = mean_squared_error(y_val, y_pred_rf, squared=False)
r2_rf = r2_score(y_val, y_pred_rf)

print(f'\nRandom Forest Validation RMSE: {rmse_rf:.4f}')
print(f'Random Forest R² Score: {r2_rf:.4f}')



# Feature Importance
importances = model_rf.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.tight_layout()
plt.show()


# Train and Evaluate XGBoost 
from xgboost import XGBRegressor

model_xgb = XGBRegressor(n_estimators=100, random_state=42)
model_xgb.fit(X_train_imputed, y_train)

y_pred_xgb = model_xgb.predict(X_val_imputed)
rmse_xgb = mean_squared_error(y_val, y_pred_xgb, squared=False)
r2_xgb = r2_score(y_val, y_pred_xgb)

print(f'\nXGBoost Validation RMSE: {rmse_xgb:.4f}')
print(f'XGBoost R² Score: {r2_xgb:.4f}')


new_model_xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

new_model_xgb.fit(
    X_train_imputed, y_train,
    early_stopping_rounds=10,
    eval_set=[(X_val_imputed, y_val)],
    verbose=False
)


new_y_pred_xgb = new_model_xgb.predict(X_val_imputed)
new_rmse_xgb = mean_squared_error(y_val, new_y_pred_xgb, squared=False)
new_r2_xgb = r2_score(y_val, new_y_pred_xgb)

print(f'\nXGBoost Validation RMSE: {new_rmse_xgb:.4f}')
print(f'XGBoost R² Score: {new_r2_xgb:.4f}')


from sklearn.ensemble import VotingRegressor

ensemble = VotingRegressor([('rf', model_rf), ('xgb', new_model_xgb)])
ensemble.fit(X_train_imputed, y_train)

y_pred_ens = ensemble.predict(X_val_imputed)
rmse_ens = mean_squared_error(y_val, y_pred_ens, squared=False)
r2_ens = r2_score(y_val, y_pred_ens)

print(f"\nEnsemble RMSE: {rmse_ens:.4f}")
print(f"Ensemble R² Score: {r2_ens:.4f}")



import numpy as np
import pandas as pd


test_df = test_df.drop_duplicates()


test_df.info()



test_df = test_df.drop(columns=['Episode_Title'])


test_df['Publication_Hour'] = test_df['Publication_Time'].apply(extract_hour)

test_df = test_df.drop(columns=['Publication_Time'])


test_df['Podcast_Name_encoded'] = target_encoder.transform(test_df['Podcast_Name'])

ohe_encoded_test = ohe.transform(test_df[ohe_cols])
ohe_df_test = pd.DataFrame(ohe_encoded_test, columns=ohe.get_feature_names_out(ohe_cols))


test_df_final = pd.concat([test_df.drop(columns=ohe_cols + ['Podcast_Name']), ohe_df_test], axis=1)

test_df_final = test_df_final.reindex(columns=X.columns, fill_value=0)


test_predictions = ensemble.predict(test_df_final)



submission_df = pd.DataFrame({
    'id': test_df['id'], 
    'Listening_Time_minutes': test_predictions
})

# Save to CSV
submission_df.to_csv('submission.csv', index=False)


import os

print(os.getcwd())


import os

files = os.listdir('/kaggle/working')
print(files)


from IPython.display import FileLink

# Provide a clickable link to download the file
FileLink('submission.csv')




