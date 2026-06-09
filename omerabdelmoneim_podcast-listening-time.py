import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv", index_col='id')


train


train['Genre'].value_counts()


train['Episode_Title'].str.match(r'Episode \d*').mean()


test['Episode_Title'].str.match(r'Episode \d*').mean()


test


podcast_groups = train.groupby('Podcast_Name')
numeric_cols = train.select_dtypes(include='number').columns
podcast_agg = podcast_groups.agg(
    {'Genre': 'nunique', **{col: 'mean' for col in numeric_cols}, 'Episode_Title': 'count'}
)
podcast_agg['Main_Genre'] = podcast_groups['Genre'].agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else None)
podcast_agg['Publication_Day'] = podcast_groups['Publication_Day'].agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else None)
podcast_agg['Publication_Time'] = podcast_groups['Publication_Time'].agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else None)
podcast_agg.columns = podcast_agg.columns.map(lambda x: 'Podcast_'+x)
podcast_agg.reset_index(inplace=True)
podcast_agg


plt.figure(figsize=(15, 4))
sns.barplot(x=podcast_agg['Podcast_Name'],
            y=podcast_agg['Podcast_Listening_Time_minutes'])
plt.xticks(rotation=-60)
plt.show()


a = test.groupby('Podcast_Name')['Episode_Title'].count().reset_index()
b = podcast_agg[['Podcast_Name', 'Podcast_Episode_Title']].copy()
a['source'] = 'test'
b['source'] = 'train'
combined = pd.concat([a, b])
plt.figure(figsize=(12, 4))
sns.barplot(x=combined['Podcast_Name'],
            y=combined['Podcast_Episode_Title'], hue=combined['source'])
plt.xticks(rotation=-60)
plt.show()


plt.figure(figsize=(15, 4))
sns.barplot(x=podcast_agg['Podcast_Name'],
            y=podcast_agg['Podcast_Episode_Length_minutes'])
plt.xticks(rotation=-60)
plt.show()


from sklearn.model_selection import cross_val_score



def eval_model(model, X, y, cv=4):

    scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error',

                             cv=cv, n_jobs=1)

    print("Mean Score:", np.mean(scores))

    print("Median Score:", np.median(scores))

    print("Std Score:", np.std(scores))

    print("Cross-validation Scores:", list(map(float, scores)))


from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import r2_score


import numpy as np
import pandas as pd



class NormalizedDurationRegressor(BaseEstimator, RegressorMixin):


    def __init__(self, base_regressor):
        self.base_regressor = base_regressor


    def fit(self, X, y):
        X = X.copy()
        y = y.copy()
        eps_length = X['Episode_Length_minutes']
        mask = (eps_length == 0) | (eps_length.isna())
        safe_eps_length = eps_length.loc[~mask]
        # Calculate IQR
        Q1 = safe_eps_length.quantile(0.25)
        Q3 = safe_eps_length.quantile(0.75)
        IQR = Q3 - Q1

        # Calculate lower and upper bounds
        self.lower_bound_ = Q1 - 1.5 * IQR
        self.upper_bound_ = Q3 + 1.5 * IQR

        # Clip values to the bounds
        clipped_eps_length = eps_length.loc[~mask].clip(lower=self.lower_bound_, upper=self.upper_bound_)
        y_norm = y.loc[~mask] / clipped_eps_length
        self.base_regressor_ = clone(self.base_regressor)
        self.backup_regressor_ = clone(self.base_regressor)
        self.base_regressor_.fit(X.loc[~mask], y_norm)
        self.backup_regressor_.fit(X, y)
        return self


    def predict(self, X):
        X = X.copy()
        eps_length = X['Episode_Length_minutes']
        mask = (eps_length == 0) | (eps_length.isna())
        clipped_eps_length = eps_length.loc[~mask].clip(lower=self.lower_bound_, upper=self.upper_bound_)
        preds = pd.Series(index=X.index, dtype=float)
        # Predict using normalized model where valid
        if not mask.all():
            preds.loc[~mask] = self.base_regressor_.predict(
                X.loc[~mask]) * clipped_eps_length

        # Predict using fallback model where length is invalid
        if mask.any():
            preds.loc[mask] = self.backup_regressor_.predict(X.loc[mask])
            
        return preds

    def score(self, X, y):
        preds = self.predict(X)
        return r2_score(y, preds)


from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


def extract_episode_number(df: pd.DataFrame):
    episode_numbers = df['Episode_Title'].str.extract(
        r'Episode (\d*)').astype(np.int32)
    episode_numbers.columns = ['Episode_Number']
    return episode_numbers


podcast_agg.columns


train.columns


from xgboost import XGBRegressor
X = train.merge(podcast_agg[['Podcast_Name', 'Podcast_Number_of_Ads', 'Podcast_Host_Popularity_percentage']], on='Podcast_Name').drop(
    columns=['Podcast_Name', 'Listening_Time_minutes'])
y = train['Listening_Time_minutes']
reg = Pipeline(steps=[('columns', ColumnTransformer(remainder='passthrough', transformers=[
               ('cat_enc', OneHotEncoder(), ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']), ('episode', FunctionTransformer(extract_episode_number), ['Episode_Title'])])), ('reg', XGBRegressor(n_jobs=-1))])
normalized_reg = NormalizedDurationRegressor(reg)
eval_model(normalized_reg, X, y)


from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
forest = Pipeline(steps=[('columns', ColumnTransformer(remainder='passthrough', transformers=[
    ('cat_enc', OneHotEncoder(), ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']), ('episode', FunctionTransformer(extract_episode_number), ['Episode_Title'])])), ('impute', SimpleImputer()), ('reg', RandomForestRegressor(n_jobs=-1))])
normalized_forest = NormalizedDurationRegressor(forest)
eval_model(normalized_forest, X, y)


normalized_forest.fit(X, y)
pred = normalized_forest.predict(test.merge(podcast_agg[[
    'Podcast_Name', 'Podcast_Number_of_Ads', 'Podcast_Host_Popularity_percentage']], on='Podcast_Name')).drop(
    columns=['Podcast_Name'])
submission = pd.DataFrame(index=test.index, data={
                          'Listening_Time_minutes': pred.values})
submission.to_csv('submission.csv')
if (submission.isna().sum() != 0).any():
    print('Submission contains na!')
submission

