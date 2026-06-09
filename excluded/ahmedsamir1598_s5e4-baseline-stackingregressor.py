!pip install --upgrade scikit-learn


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col = False)
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col = False)
sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv', index_col = False)
train.head()


train.info()


test.info()


train.describe().T


# ads has outliers
train['Number_of_Ads'].value_counts()


train[train['Number_of_Ads'].isnull()==True]


train['Publication_Day'].value_counts()


train['Publication_Time'].value_counts()


train['Podcast_Name'].value_counts()


train['Episode_Title'].value_counts()


train['Episode_Sentiment'].value_counts()


train['Genre'].value_counts()


test['Number_of_Ads'].value_counts()


sns.histplot(train['Episode_Length_minutes'], bins = 50, kde= True)


x = (train['Episode_Length_minutes'] - train['Episode_Length_minutes'].mean())/train['Episode_Length_minutes'].std()
sns.histplot(x, bins = 50, kde= True)


sns.histplot(train['Guest_Popularity_percentage'])


from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

def process(df, vectorizer=None, istrain = False):
    df = df.copy()  # Avoid modifying the original dataframe
    
    if istrain:
        # Remove outliers in Number_of_Ads
        df = df[df["Number_of_Ads"] <= 3]
    
    # Fill missing values with median
    df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median())
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].median())
    
    df.dropna(inplace=True)  # Drop any remaining missing values
    if not istrain:
        print(df.info())
    if istrain:
        # Apply TF-IDF to Podcast_Name (since it has two words)
        vectorizer = TfidfVectorizer()
        df_tfidf = vectorizer.fit_transform(df['Podcast_Name'])
        # Convert sparse matrix to DataFrame
        train_podcast_df = pd.DataFrame(df_tfidf.toarray(), columns=[f'Podcast_TFIDF_{i}' for i in range(df_tfidf.shape[1])])
        # Reset index to merge
        df = df.reset_index(drop=True)
        # Concatenate TF-IDF features
        df = pd.concat([df, train_podcast_df], axis=1)
    else:
        df_tfidf = vectorizer.transform(df['Podcast_Name'])
        test_podcast_df = pd.DataFrame(df_tfidf.toarray(), columns=[f'Podcast_TFIDF_{i}' for i in range(df_tfidf.shape[1])])
        df = df.reset_index(drop=True)
        df = pd.concat([df, test_podcast_df], axis=1)
    # Ordinal Encoding for ordered categorical variables
    time_order = ['Night', 'Morning', 'Afternoon', 'Evening']
    day_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    sentiment_order = ['Negative', 'Neutral', 'Positive']
    
    df['Publication_Time'] = df['Publication_Time'].map({v: i for i, v in enumerate(time_order)})
    df['Publication_Day'] = df['Publication_Day'].map({v: i for i, v in enumerate(day_order)})
    df['Episode_Sentiment'] = df['Episode_Sentiment'].map({v: i for i, v in enumerate(sentiment_order)})
    
    # Ordinal Encoding for unordered categorical variables
    ordinal_enc = OrdinalEncoder()
    df[['Genre', 'Episode_Title']] = ordinal_enc.fit_transform(df[['Genre', 'Episode_Title']])

    # FE
    df["SinWeekday"] = np.sin(2 * np.pi * df["Publication_Day"]/ 7)
    df["CosWeekday"] = np.cos(2 * np.pi * df["Publication_Day"]/ 7)
    df["SinTime"] = np.sin(2 * np.pi * df["Publication_Time"]/ 4)
    df["CosTime"] = np.cos(2 * np.pi * df["Publication_Time"]/ 4)

    df.drop([
        'id',
        'Podcast_Name',
    ], axis=1, inplace=True, errors='ignore')
    
    return df, vectorizer

train2, tfidf_vectorizer = process(train, istrain = True)
test2, _ = process(test, tfidf_vectorizer, istrain = False)
train2.info()


train2


test2.info()


X = train2.drop(columns=['Listening_Time_minutes'])
y = train2['Listening_Time_minutes']

kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_scorer = make_scorer(rmse)

lgbm = LGBMRegressor(
    boosting_type='gbdt',
    objective='regression',
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=31,
    max_depth=-1,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_data_in_leaf=20,
    random_state=42, 
    verbose = -1
)
cv_results = cross_val_score(lgbm, X, y, cv=kf, scoring=rmse_scorer)
print(f'Cross-Validation RMSE Scores: {cv_results}')
print(f'Mean RMSE: {np.mean(cv_results)}')


cbr = CatBoostRegressor(
        iterations=500,
        learning_rate=0.08777255350163136,
        depth=10,
        l2_leaf_reg=0.1259643500248322,
        bootstrap_type='Bayesian',
        random_strength=4.276181166674371e-08,
        bagging_temperature=0.35995482350907326,
        od_type='Iter',
        od_wait=39,
        verbose=0,
        allow_writing_files=False,
        random_seed=42,
    )

cv_results = cross_val_score(cbr, X, y, cv=kf, scoring=rmse_scorer)
print(f'Cross-Validation RMSE Scores: {cv_results}')
print(f'Mean RMSE: {np.mean(cv_results)}')


estimators = [
    ('catboost', cbr),
    ('lgbm', lgbm)
]
stacked_model = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(),
    cv=kf  
)

cv_results_stacked = cross_val_score(stacked_model, X, y, cv=kf, scoring=rmse_scorer)

print(f'Stacked Model Cross-Validation RMSE Scores: {-cv_results_stacked}')
print(f'Mean RMSE: {-np.mean(cv_results_stacked)}')


stacked_model.fit(X, y)
preds = stacked_model.predict(test2)
sub['Listening_Time_minutes'] = preds
sub.to_csv('submission.csv', index = False)

