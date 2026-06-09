# Basic Libraries
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Model Building
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# For Better UI
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings("ignore")


# Reading the train dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
train.head()


# Reading the test dataset
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
test.head()


print("Train Dataset Shape:", train.shape)
print("Test Dataset Shape:", test.shape)


train.info()


# Lets bifurcate the numeric and non-numeric columns
object_col = train.select_dtypes(include="object").columns
num_col = train.select_dtypes(exclude="object").columns

print("Non Numeric Columns Count - ", len(object_col))
print("Non Numeric Columns - ", list(object_col))

print()

print("Numeric Columns Count - ", len(num_col))
print("Numeric Columns - ", list(num_col))


# Non Numeric columns Unique Values and top 10 values
for i in object_col:
    plt.figure(figsize=(12, 5))
    
    print(i, " Unique values -" ,train[i].nunique())
    print("-"*50)
    if train[i].nunique() > 10:
        top_cat = train[i].value_counts().head(10)
        ax = sns.barplot(x = top_cat.index, y = top_cat.values)
    else:
        ax = sns.countplot(x = train[i], order=train[i].value_counts().index)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                (p.get_x() + p.get_width() / 2, p.get_height()),  
                ha='center', va='bottom', fontsize=9, color='black')

    plt.show()



pod_episode = train[['Podcast_Name', 'Episode_Title']]
print(pod_episode.shape)

pod_episode.drop_duplicates(inplace=True)

print(pod_episode.shape)


pod_episode['Episode_Number'] = pod_episode['Episode_Title'].str.split(" ").str[1].astype(int)
pod_episode.head()


pod_episode.groupby('Podcast_Name').agg({
    'Episode_Number' : ['min', 'max', pd.Series.nunique, 'count']
})


train[num_col].describe()


train_df = train.copy()


# Imputing Median in NULL Values
train_df.fillna(train_df.median(), inplace=True)
train_df[num_col].describe()


train_df[num_col].hist(figsize=(11, 8))
plt.show()


plt.figure(figsize=(12, 10))
sns.pairplot(train_df[num_col], y_vars=["Listening_Time_minutes"], 
             x_vars=train_df[num_col].columns.drop("Listening_Time_minutes"))
plt.show()


plt.figure(figsize=(8, 5))
sns.heatmap(train_df[num_col].corr().round(3), annot=True, cmap="coolwarm")
plt.show()


train_df.set_index('id', inplace=True)


train_df.head()


def feature_engg(df):
    df['Episode_Sentiment'] = df['Episode_Sentiment'].map({'Positive':1, 'Negative':-1, 'Neutral':0})
    df['Publication_Time'] = df['Publication_Time'].map({'Morning':1, 'Afternoon':2, 'Evening':3, 'Night':4})
    df['Publication_Day'] = df['Publication_Day'].map({'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 
                                                      'Friday':5, 'Saturday':6, 'Sunday':7})
    df['Episode_Title'] = df['Episode_Title'].apply(lambda x: x.split(' ')[1]).astype(int)
    df = pd.get_dummies(df, columns=['Genre'], drop_first=True)
    return df


train_df = feature_engg(train_df)
train_df.head()


train_df.info()


# Create TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
# Fit and Transform Podcast Name column
tfidf_matrix = tfidf.fit_transform(train_df['Podcast_Name'].fillna(''))

# Convert to DataFra*-me
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out(), index=train_df.index)

train_df_2 = train_df.drop(columns=['Podcast_Name'])
train_df_2 = pd.concat([train_df_2, tfidf_df], axis=1)
train_df_2.shape


X = train_df_2.drop('Listening_Time_minutes', axis=1)
y = train_df_2['Listening_Time_minutes']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# Create the model
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)

# Fit the model
xgb.fit(X_train, y_train)

# Predict the model
y_pred = xgb.predict(X_test)

# Evaluate the model
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("XGBoost Basic Model RMSE:", xgb_rmse)








test.info()


# Setting the ID column as index
test.set_index('id', inplace=True)


# Imputing Median in NULL Values
test.fillna(test.median(), inplace=True)
test.describe()


# Implementing same feature Engineering on Test
test = feature_engg(test)
test.head()


# Implementing same tfidf on Test
test_tfidf_matrix = tfidf.transform(test['Podcast_Name'].fillna(''))

# Convert to DataFrame
test_tfidf_df = pd.DataFrame(test_tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out(), index=test.index)

test_df = test.drop(columns=['Podcast_Name'])
test_df = pd.concat([test_df, test_tfidf_df], axis=1)
test_df.shape


# Final Predictions
final_test_pred = xgb.predict(test_df)

submission = pd.DataFrame({
    'id': test_df.index,  # or test_ids if you reset index
    'listening_time': final_test_pred
})

submission.to_csv('submission.csv', index=False)
















