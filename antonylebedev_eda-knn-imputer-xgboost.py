# Uploading all the modules that we'll need
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import iplot
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


df.shape


test_df.shape


df.info()


df = df.drop(columns = ['id']) # Dropping the ID column, since it simply duplicates the indexes
test_df = test_df.drop(columns = ['id'])


# Creating lists for numeric features, categorical features and our target; this will come in handy later
numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
target = ['Personality']


df.isnull().sum()


test_df.isnull().sum()


df['Stage_fear'] = df['Stage_fear'].map({'Yes':1, 'No': 0})
df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes':1, 'No': 0})
df['Personality'] = df['Personality'].map({'Extrovert':1, 'Introvert': 0})

test_df['Stage_fear'] = test_df['Stage_fear'].map({'Yes':1, 'No': 0})
test_df['Drained_after_socializing'] = test_df['Drained_after_socializing'].map({'Yes':1, 'No': 0})


numeric_processor = Pipeline([('knn_imputer', KNNImputer(n_neighbors = 3)),
                              ('scaling', StandardScaler())])
categorical_processor = Pipeline([('knn_imputer', KNNImputer(n_neighbors = 3))])
target_processor = 'passthrough'


preprocessor_1 = ColumnTransformer([('categorical', categorical_processor, categorical_cols), 
                                 ('numeric', numeric_processor['knn_imputer'], numeric_cols),
                                 ('target', target_processor, target)])


eda_df = pd.DataFrame(data = preprocessor_1.fit_transform(df), columns = categorical_cols+numeric_cols+target)


# Making a couple of manipulations so that dataset is suitable for EDA
eda_df = eda_df.round()
eda_df['Stage_fear'] = eda_df['Stage_fear'].map({1:'Yes', 0:'No'})
eda_df['Drained_after_socializing'] = eda_df['Drained_after_socializing'].map({1:'Yes', 0:'No'})
eda_df['Personality'] = eda_df['Personality'].map({1:'Extrovert', 0:'Introvert'})


eda_df


eda_df.describe()


gr = eda_df.groupby('Personality').count()
gr


fig = px.histogram(eda_df, y = 'Personality')
fig.show(renderer = 'iframe')


fig = px.histogram(eda_df, y = 'Stage_fear')
fig.show(renderer = 'iframe')


fig = px.histogram(eda_df, y = 'Drained_after_socializing')
fig.show(renderer = 'iframe')


fig = make_subplots(rows = 3, cols = 2, start_cell = 'top-left', subplot_titles = numeric_cols)
fig.add_trace(go.Histogram(x = eda_df['Time_spent_Alone']), row = 1, col = 1)
fig.add_trace(go.Histogram(x = eda_df['Social_event_attendance']), row = 1, col = 2)
fig.add_trace(go.Histogram(x = eda_df['Going_outside']), row = 2, col = 1)
fig.add_trace(go.Histogram(x = eda_df['Friends_circle_size']), row = 2, col = 2)
fig.add_trace(go.Histogram(x = eda_df['Post_frequency']), row = 3, col = 1)
fig.update_layout(height=900, width=800, title_text="Numeric variables' distributions")
fig.show(renderer = 'iframe')


eda_df[numeric_cols].corr()


fig = px.box(eda_df, x = 'Personality', y = 'Time_spent_Alone')
fig.show(renderer = 'iframe')


fig = px.box(eda_df, x = 'Personality', y = 'Friends_circle_size')
fig.show(renderer = 'iframe')


gr = eda_df.groupby(['Personality', 'Stage_fear']).count().reset_index()
fig = px.bar(gr, x = 'Personality', y = 'Going_outside', color = 'Stage_fear', barmode = 'group')
fig.show(renderer = 'iframe')


gr = eda_df.groupby(['Personality', 'Drained_after_socializing']).count().reset_index()
fig = px.bar(gr, x = 'Personality', y = 'Going_outside', color = 'Drained_after_socializing', barmode = 'group')
fig.show(renderer = 'iframe')


preprocessor_2 = ColumnTransformer([('categorical', categorical_processor, categorical_cols), 
                                 ('numeric', numeric_processor, numeric_cols)])


log_pipe = make_pipeline(preprocessor_2, LogisticRegression())
cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 2, random_state = 42)
X = df.drop(columns = ['Personality'])
Y = df['Personality']


log_scores = cross_val_score(log_pipe, X, Y, cv = cv)
print(f"The scores are: {log_scores}\nThe mean score is {np.mean(log_scores)}")


rf_pipe = make_pipeline(preprocessor_2, RandomForestClassifier())
cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 2, random_state = 42)
rf_scores = cross_val_score(log_pipe, X, Y, cv = cv)
print(f"The scores are: {rf_scores}\nThe mean score is {np.mean(rf_scores)}")


rf_pipe = make_pipeline(preprocessor_2, RandomForestClassifier(n_estimators = 200, max_depth = 5))
cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 2, random_state = 42)
rf_scores = cross_val_score(log_pipe, X, Y, cv = cv)
print(f"The scores are: {rf_scores}\nThe mean score is {np.mean(rf_scores)}")


xgb_pipe = make_pipeline(preprocessor_2, XGBClassifier(n_estimators = 1000, learning_rate = .01, max_depth = 4, subsample = .9, n_jobs = -1))
cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 2, random_state = 42)
xgb_scores = cross_val_score(xgb_pipe, X, Y, cv = cv)
print(f"The scores are: {xgb_scores}\nThe mean score is {np.mean(xgb_scores)}")


xgb_pipe.fit(X, Y)
pred_Y = pd.Series(xgb_pipe.predict(test_df))


submission = pd.DataFrame({'id': pd.Series(range(18524, 24699)), 'Personality': pred_Y})
submission['Personality'] = submission['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
submission


submission.to_csv('submission-1.csv', index=False)

