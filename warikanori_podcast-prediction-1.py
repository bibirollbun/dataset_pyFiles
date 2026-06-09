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

train_link = "/kaggle/input/playground-series-s5e4/train.csv"
test_link = "/kaggle/input/playground-series-s5e4/test.csv"



# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# import sklearn
from sklearn.preprocessing import LabelEncoder

# warnings 
import warnings
warnings.filterwarnings('ignore')


train_podcast = pd.read_csv(train_link)
test_podcast = pd.read_csv(test_link)

# shape of train and test 
train_podcast.shape, test_podcast.shape


train_podcast.info()


# join train and test dataframes
podcast = pd.concat([train_podcast, test_podcast], sort=False).reset_index(drop=True)
print(podcast.shape)
podcast.head(3)


# check the unique from dataframes
for i in podcast.columns:
    print(f"{i} ({podcast[i].nunique()}): {podcast[i].unique()}")
    print()


# drop the id
podcast = podcast.drop("id", axis=1)
podcast.sample(3)


podcast.isnull().sum()


podcast.describe()


podcast.info()


# check the unique from dataframes
for i in podcast.columns:
    print(f"{i} : {podcast[i].nunique()}")


# # split the data into train and test for train to model 
# new_train = podcast.iloc[:len(train_podcast)]
# new_test = podcast.iloc[len(train_podcast):].drop('Listening_Time_minutes', axis=1)

# # check the shape of new_train and new_test
# new_train.shape, new_test.shape


podcast


# Extract numbers from Episode_Title
podcast['Episode_Number'] = (
    podcast['Episode_Title']
    .astype(str)
    .str.extract(r'(\d+)')[0]
    .astype(int)
)

podcast[['Episode_Title', 'Episode_Number']]


# Convert Podcast_Name and Genre to numeric codes
podcast['Podcast_Name'] = pd.factorize(podcast['Podcast_Name'])[0]
podcast['Genre'] = pd.factorize(podcast['Genre'])[0]

podcast.sort_values(['Podcast_Name','Episode_Number'])


#fill the value with me
podcast['Episode_Length_minutes'] = podcast.groupby(['Podcast_Name', 'Episode_Number'])['Episode_Length_minutes'].transform(lambda x: x.fillna(x.mean()))
podcast['Guest_Popularity_percentage'] = podcast.groupby(['Podcast_Name', 'Episode_Number'])['Guest_Popularity_percentage'].transform(lambda x: x.fillna(x.mean()))
podcast['Number_of_Ads'] = podcast.groupby(['Podcast_Name', 'Episode_Number'])['Number_of_Ads'].transform(lambda x: x.fillna(x.median()))
#podcast['Guest_Popularity_percentage'].fillna(podcast['Guest_Popularity_percentage'].median(), inplace=True)
#podcast['Listening_Time_minutes'].fillna(podcast['Listening_Time_minutes'].median(), inplace=True)
#podcast['Number_of_Ads'].fillna(podcast['Number_of_Ads'].median(), inplace=True)

#check if there is any difference
podcast.isnull().sum()


podcast.loc[podcast['Episode_Length_minutes'] <= podcast['Listening_Time_minutes']].shape


# Original mappings (keep these)
podcast['Episode_Sentiment'] = podcast['Episode_Sentiment'].map({"Negative": -1, "Neutral":0, 'Positive':1})
podcast['Publication_Day'] = podcast['Publication_Day'].map({
    "Monday":1, "Tuesday":2, "Wednesday":3, "Thursday":4,
    "Friday":5, "Saturday":6, "Sunday":7
})

podcast['Publication_Time'] = podcast['Publication_Time'].map({
    "Morning":1, "Afternoon":2, "Evening":3, "Night":4
})

# New optimized combination
podcast['Publication_DayTime'] = (podcast['Publication_Day'] - 1) * 4 + podcast['Publication_Time']
podcast = podcast.drop('Episode_Title', axis=1)

podcast.head()


podcast.loc[podcast['Host_Popularity_percentage'] < 20,'Host_Popularity_percentage'] = 20
podcast.loc[podcast['Host_Popularity_percentage'] > 100,'Host_Popularity_percentage'] = 100
podcast.loc[podcast['Guest_Popularity_percentage'] < 0.01,'Guest_Popularity_percentage'] = 0.01
podcast.loc[podcast['Guest_Popularity_percentage'] > 100,'Guest_Popularity_percentage'] = 100
podcast['Guest_Popularity_percentage'] = podcast['Guest_Popularity_percentage'] / 100
podcast['Host_Popularity_percentage'] = podcast['Host_Popularity_percentage'] / 100


podcast['Popularity_Score_Mean'] = (podcast['Host_Popularity_percentage'] + podcast['Guest_Popularity_percentage'])/2
podcast['Is_Weekend'] = 0
podcast.loc[(podcast['Publication_Day']==6) | (podcast['Publication_Day']==7),'Is_Weekend'] = 1


podcast.loc[podcast['Number_of_Ads'] > 3,'Number_of_Ads'] = 3
podcast.loc[podcast['Episode_Length_minutes'] > 120,'Episode_Length_minutes'] = 120
podcast.loc[podcast['Episode_Length_minutes'] < 5,'Episode_Length_minutes'] = 5
podcast.loc[podcast['Episode_Length_minutes'] < podcast['Listening_Time_minutes'],'Listening_Time_minutes'] = podcast['Episode_Length_minutes']
podcast['Ad_Density'] = podcast['Number_of_Ads'] / podcast['Episode_Length_minutes']
podcast['Length_Gap'] = podcast['Episode_Length_minutes'] / (podcast['Number_of_Ads'] + 1)


podcast.loc[podcast['Episode_Length_minutes'] <= podcast['Listening_Time_minutes']]


podcast_copy = podcast.copy()
podcast_copy


podcast_copy.info()


podcast_copy.describe()


import pandas as pd
import numpy as np

# Assuming 'podcast' is your DataFrame
continuous_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Listening_Time_minutes', 'Ad_Density', 'Length_Gap', 'Popularity_Score_Mean']
categorical_cols = ['Podcast_Name','Genre']  # Add others if you want
ordinal_cols = ['Episode_Number','Publication_Day','Publication_Time','Episode_Sentiment', 'Publication_DayTime', 'Number_of_Ads', 'Is_Weekend']  # Manually define ordinal columns if they exist

sns.boxplot(x='Is_Weekend', y='Listening_Time_minutes', data=podcast_copy)
plt.title("Binary vs. Continuous")
plt.show()


# Pearson correlation (for continuous-continuous pairs)
pearson_corr = podcast_copy[continuous_cols].dropna().corr(method='pearson')

# Spearman correlation (for ordinal or non-normal continuous)
spearman_corr = podcast_copy[ordinal_cols].corr(method='spearman')

# For ordinal vs. continuous, use Spearman
mixed_corr = podcast_copy[continuous_cols + ordinal_cols].dropna().corr(method='spearman')


plt.figure(figsize=(10, 8))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title("Pearson Correlation (Continuous Variables)")
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title("Spearman Rank Correlation (Ordinal/Non-Normal Variables)")
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(mixed_corr, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title("Spearman Rank Correlation (All Variables)")
plt.show()


from scipy.stats import f_oneway

def anova_test(df, categorical_col, continuous_col):
    # Drop rows where either the categorical OR continuous column is NaN
    temp_df = df[[categorical_col, continuous_col]].dropna()
    
    # Check if any group has <2 samples (ANOVA requires at least 2 per group)
    group_counts = temp_df.groupby(categorical_col).size()
    if any(group_counts < 2):
        print(f"{categorical_col} vs {continuous_col}: SKIPPED (some groups have <2 samples)")
        return
    
    # Perform ANOVA
    groups = temp_df.groupby(categorical_col)[continuous_col].apply(list)
    f_stat, p_value = f_oneway(*groups)
    print(f"{categorical_col} vs {continuous_col}: p-value = {p_value:.4f}")

# Example usage (no need for .dropna() here now)
for cat_col in categorical_cols:
    for cont_col in continuous_cols:
        anova_test(podcast_copy, cat_col, cont_col)


for col in continuous_cols:
    sns.scatterplot(x=col, y='Listening_Time_minutes', data=podcast_copy)
    plt.title(f'Listening Time vs {col}')
    plt.xticks(rotation=45)  # rotate labels if needed
    plt.show()
    plt.show()



for col in continuous_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=podcast_copy[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


for cat_col in categorical_cols:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=cat_col, y='Listening_Time_minutes', data=podcast_copy)
    plt.title(f'Listening Time vs {cat_col}')
    plt.xticks(rotation=90)  # rotate labels if needed
    plt.show()

for cat_col in ordinal_cols:
    plt.figure(figsize=(15, 5))
    sns.boxplot(x=cat_col, y='Listening_Time_minutes', data=podcast_copy)
    plt.title(f'Listening Time vs {cat_col}')
    plt.xticks(rotation=90)  # rotate labels if needed
    plt.show()



from scipy.stats import normaltest  # Uses D'Agostino-Pearson

def dagostino_pearson_test(data, column, alpha=0.05):
    stat, p = normaltest(data[column].dropna())
    print(f"Column: {column}")
    print(f"D'Agostino-Pearson p-value = {p:.4f}")
    if p > alpha:
        print("✅ Normally distributed (fail to reject H₀)")
    else:
        print("❌ Not normally distributed (reject H₀)")
    print("---")

# Example usage
import pandas as pd
for col in continuous_cols:
    dagostino_pearson_test(podcast_copy, col)


col = ['Episode_Number', 'Ad_Density']
train_podcast = podcast[podcast['Listening_Time_minutes'].notnull()]
submission_podcast = podcast[podcast['Listening_Time_minutes'].isnull()]


from sklearn.model_selection import train_test_split
# First split: 60% train, 40% temp

# Separate features (X) and target (y)
y = train_podcast['Listening_Time_minutes']  # Target variable
X = train_podcast[col]  # Features (exclude target)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
y_submission = submission_podcast['Listening_Time_minutes']
X_submission = submission_podcast[col]

print(f"Train: {len(X_train)}, Test: {len(X_test)}, Submission:{len(X_submission)}")


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit AND transform train
X_test_scaled = scaler.transform(X_test)  # Only transform test (no fit!)
X_submission_scaled = scaler.transform(X_submission)


features = X_train_scaled.shape[1]
X_train_scaled.shape


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
from keras.losses import MeanSquaredError
import tensorflow.keras.backend as K

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

model = Sequential([
    tf.keras.layers.Input(shape=(features,)),       # Input layer with 15 features
    Dense(64, activation='relu'),
    Dropout(0.2),                              # Optional regularization
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)                                   # Output layer for regression
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error',      # or use 'mae' (Mean Absolute Error)
    metrics=[rmse, 'mae']
)

model.summary()


history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.3,           # 20% of training set used for validation
    epochs=20,
    batch_size=32,
    verbose=1
)


import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Progress')
plt.legend()
plt.show()

plt.plot(history.history['rmse'], label='Train RMSE')
plt.plot(history.history['val_rmse'], label='Val RMSE')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('Training RMSE Progress')
plt.legend()
plt.show()


test_loss, test_mae, test_rmse = model.evaluate(X_test_scaled, y_test)
print(f"Test MSE: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")


predictions = model.predict(X_submission_scaled)

# Convert to DataFrame for easier handling
pred_df = pd.DataFrame(predictions, columns=['Listening_Times_minutes'])
pred_df['id'] = pred_df.index
pred_df = pred_df[['id','istening_Times_minutes']]


# Save to CSV (optional)
pred_df.to_csv("submission.csv", index=False)

# Or just view a few
print(pred_df.head())

