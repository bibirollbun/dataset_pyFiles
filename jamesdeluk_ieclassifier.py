import pandas as pd
import math
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from xgboost import XGBClassifier


df_train_raw = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_train_raw


df_train_raw.describe(include='all')


df_train_raw.isnull().sum()


n_cols = 4
cols = df_train_raw.columns[1:]
n_plots = len(cols)
n_rows = math.ceil(n_plots / n_cols)

fig = make_subplots(
    rows=n_rows, cols=n_cols,
    subplot_titles=list(cols),
    horizontal_spacing=0.05,
    vertical_spacing=0.10
)

for i, col_name in enumerate(cols):
    row = i // n_cols + 1
    col = i % n_cols + 1
    fig.add_trace(
        go.Histogram(
            x=df_train_raw[col_name],
            # nbinsx=30,
            name=col_name,
            showlegend=False
        ),
        row=row,
        col=col
    )

fig.update_layout(
    height=300 * n_rows,      # e.g. 300px per row
    # width=300 * n_cols,       # 300px per column
    bargap=0.05
)

fig.show()


# df_train_raw.corr()


df_test_raw = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


df_test_raw.describe(include='all')


df_test_raw.isnull().sum()


n_cols = 4
cols = df_test_raw.columns[1:]
n_plots = len(cols)
n_rows = math.ceil(n_plots / n_cols)

fig = make_subplots(
    rows=n_rows, cols=n_cols,
    subplot_titles=list(cols),
    horizontal_spacing=0.05,
    vertical_spacing=0.10
)

for i, col_name in enumerate(cols):
    row = i // n_cols + 1
    col = i % n_cols + 1
    fig.add_trace(
        go.Histogram(
            x=df_test_raw[col_name],
            # nbinsx=30,
            name=col_name,
            showlegend=False
        ),
        row=row,
        col=col
    )

fig.update_layout(
    height=300 * n_rows,      # e.g. 300px per row
    width=300 * n_cols,       # 300px per column
    bargap=0.05
)

fig.show()


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
sample_submission


def process_data_1(df):
    df_ = df.copy()
    df_ = df_.dropna()
    df_ = df_.drop(columns='id')
    for col in df_.columns:
        if col in ['Stage_fear', 'Drained_after_socializing']:
            df_[col] = df[col].map({'Yes':1,'No':0})
        if col == 'Personality':
            df_[col] = df[col].map({'Extrovert':1,'Introvert':0})
    return df_


df_train_1 = process_data_1(df_train_raw)
df_train_1


df_test_1 = process_data_1(df_test_raw)
df_test_1


X_1 = df_train_1.drop(columns=['Personality'])
y_1 = df_train_1['Personality']


X_train_1, X_val_1, y_train_1, y_val_1 = train_test_split(X_1, y_1, test_size=0.2, random_state=42)


lr_clf_1 = LogisticRegression(random_state=42)
lr_clf_1.fit(X_train_1, y_train_1)


y_pred_lr_1 = lr_clf_1.predict(X_val_1)
print("Validation Accuracy:", accuracy_score(y_val_1, y_pred_lr_1))


df_test_1


df_test_lr_1 = df_test_1.copy()
df_test_lr_1['Predicted_Personality_LR'] = lr_clf_1.predict(df_test_1)
df_test_lr_1


# submission = df_test_raw[['id']].join(
#     df_test_lr_1['Predicted_Personality_LR']
#     .map({1:'Extrovert',0:'Introvert'})
#     .rename('Personality'))
# submission.to_csv('submission.csv', index=None)
# submission


xgb_clf_1 = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_clf_1.fit(X_train_1, y_train_1)


pd.DataFrame({
    'feature': X_1.columns,
    'importance': xgb_clf_1.feature_importances_
}).sort_values(by='importance', ascending=False)


y_pred_xgb_1 = xgb_clf_1.predict(X_val_1)
print("Validation Accuracy:", accuracy_score(y_val_1, y_pred_xgb_1))


df_test_xgb_1 = df_test_1.copy()
df_test_xgb_1['Predicted_Personality_XGB'] = xgb_clf_1.predict(df_test_1)
df_test_xgb_1


abs(df_test_lr_1['Predicted_Personality_LR'] - df_test_xgb_1['Predicted_Personality_XGB']).sum()


def process_data_2(df):
    df_ = df.copy()
    df_ = df_.drop(columns='id')
    for col in df_.columns:
        if col in ['Stage_fear', 'Drained_after_socializing']:
            df_[col] = df[col].map({'Yes':1,'No':0})
        if col == 'Personality':
            df_[col] = df[col].map({'Extrovert':1,'Introvert':0})
    df_ = df_.fillna(df_.mean())
    return df_


df_train_2 = process_data_2(df_train_raw)
X_2 = df_train_2.drop(columns=['Personality'])
y_2 = df_train_2['Personality']
X_train_2, X_val_2, y_train_2, y_val_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=42)

df_test_2 = process_data_2(df_test_raw)


lr_clf_2 = LogisticRegression(random_state=42)
lr_clf_2.fit(X_train_2, y_train_2)
y_pred_lr_2 = lr_clf_2.predict(X_val_2)
print()
print("Validation Accuracy:", accuracy_score(y_val_2, y_pred_lr_2))
print()
df_test_lr_2 = df_test_2.copy()
df_test_lr_2['Predicted_Personality_LR'] = lr_clf_2.predict(df_test_2)
df_test_lr_2


# submission = df_test_raw[['id']].join(
#     df_test_lr_2['Predicted_Personality_LR']
#     .map({1:'Extrovert',0:'Introvert'})
#     .rename('Personality'))
# submission.to_csv('submission.csv', index=None)
# submission


xgb_clf_2 = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_clf_2.fit(X_train_2, y_train_2)
print(pd.DataFrame({
    'feature': X_2.columns,
    'importance': xgb_clf_2.feature_importances_
}).sort_values(by='importance', ascending=False))
y_pred_xgb_2 = xgb_clf_2.predict(X_val_2)
print()
print("Validation Accuracy:", accuracy_score(y_val_2, y_pred_xgb_2))
print()
df_test_xgb_2 = df_test_2.copy()
df_test_xgb_2['Predicted_Personality_XGB'] = xgb_clf_2.predict(df_test_2)
df_test_xgb_2


df_train_3 = df_train_raw.copy()
X_3 = df_train_3.drop(columns=['id','Personality'])
y_3 = df_train_3['Personality']
X_train_3, X_val_3, y_train_3, y_val_3 = train_test_split(X_3, y_3, test_size=0.2, random_state=42)

df_test_3 = df_test_raw.copy()


num_cols = ['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']
cat_cols = ['Stage_fear','Drained_after_socializing']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
    ('scaler', StandardScaler()),
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols),
], remainder='drop')


lr_clf_3 = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(random_state=42)),
])

lr_clf_3.fit(X_train_3, y_train_3)
y_pred_lr_3 = lr_clf_3.predict(X_val_3)
print()
print("Validation Accuracy:", accuracy_score(y_val_3, y_pred_lr_3))
print()
df_test_lr_3 = df_test_3.copy()
df_test_lr_3['Predicted_Personality_LR'] = lr_clf_3.predict(df_test_3)
df_test_lr_3


# submission = df_test_raw[['id']].join(
#     df_test_lr_3['Predicted_Personality_LR']
#     .rename('Personality'))
# submission.to_csv('submission.csv', index=None)
# submission


xgb_clf_3 = Pipeline([
    ('pre', preprocessor),
    ('clf', XGBClassifier(eval_metric='logloss', random_state=42)),
])

xgb_clf_3.fit(X_train_3, y_train_3.map({'Introvert':0,'Extrovert':1}))
y_pred_xgb_3 = xgb_clf_3.predict(X_val_3)
print()
print("Validation Accuracy:", accuracy_score(y_val_3.map({'Introvert':0,'Extrovert':1}), y_pred_xgb_3))
print()
df_test_xgb_3 = df_test_3.copy()
df_test_xgb_3['Predicted_Personality_LR'] = xgb_clf_3.predict(df_test_3)
df_test_xgb_3


df_train_4 = df_train_raw.copy()

df_train_4['Stage_fear'] = df_train_4['Stage_fear'].map({'Yes':1,'No':0})
df_train_4['Drained_after_socializing'] = df_train_4['Drained_after_socializing'].map({'Yes':1,'No':0})
df_train_4['Personality'] = df_train_4['Personality'].map({'Extrovert':1,'Introvert':0})

df_train_4 = df_train_4.drop(columns=['id'])

X_4 = df_train_4.drop(columns=['Personality'])
y_4 = df_train_4['Personality']
X_train_4, X_val_4, y_train_4, y_val_4 = train_test_split(X_4, y_4, test_size=0.2, random_state=42)

df_test_4 = df_test_raw.copy()
df_test_4 = df_test_4.drop(columns=['id'])
df_test_4['Stage_fear'] = df_test_4['Stage_fear'].map({'Yes':1,'No':0})
df_test_4['Drained_after_socializing'] = df_test_4['Drained_after_socializing'].map({'Yes':1,'No':0})


(pd.DataFrame(
    make_pipeline(
        StandardScaler(),
        KNNImputer(n_neighbors=5, weights="distance")
        # IterativeImputer(estimator=None, max_iter=100, random_state=42)
    ).fit_transform(df_train_4),
    columns=df_train_4.columns,
    index=df_train_4.index
) - pd.DataFrame(
    make_pipeline(
        StandardScaler(),
        # KNNImputer(n_neighbors=5, weights="distance")
        IterativeImputer(estimator=None, max_iter=100, random_state=42)
    ).fit_transform(df_train_4),
    columns=df_train_4.columns,
    index=df_train_4.index
)).sum()


pipe_lr_4_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
    ("clf", LogisticRegression(random_state=42))
])

pipe_lr_4_knn.fit(X_train_4, y_train_4)
y_pred_lr_4_knn = pipe_lr_4_knn.predict(X_val_4)
print("Validation Accuracy:", accuracy_score(y_val_4, y_pred_lr_4_knn))


pipe_lr_4_ii = Pipeline([
    ("scaler", StandardScaler()),
    ("imputer", IterativeImputer(max_iter=1000, random_state=42)),
    ("clf", LogisticRegression(random_state=42))
])

pipe_lr_4_ii.fit(X_train_4, y_train_4)
y_pred_lr_4_ii = pipe_lr_4_ii.predict(X_val_4)
print("Validation Accuracy:", accuracy_score(y_val_4, y_pred_lr_4_ii))


df_test_4_lr_ii = df_test_4.copy()
df_test_4_lr_ii['Personality'] = pipe_lr_4_ii.predict(df_test_4)


# submission = (df_test_raw[['id']]
#               .join(df_test_4_lr_ii['Personality']
#               .map({1:'Extrovert',0:'Introvert'})))
# submission.to_csv('submission.csv', index=None)
# submission


pipe_xgb_4_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
    ("clf", XGBClassifier(eval_metric='logloss', random_state=42))
])

pipe_xgb_4_knn.fit(X_train_4, y_train_4)
y_pred_xgb_4_knn = pipe_lr_4_knn.predict(X_val_4)
print("Validation Accuracy:", accuracy_score(y_val_4, y_pred_xgb_4_knn))


pipe_xgb_4_ii = Pipeline([
    ("scaler", StandardScaler()),
    ("imputer", IterativeImputer(max_iter=100, random_state=42)),
    ("clf", XGBClassifier(eval_metric='logloss', random_state=42))
])

pipe_xgb_4_ii.fit(X_train_4, y_train_4)
y_pred_xgb_4_ii = pipe_xgb_4_ii.predict(X_val_4)
print("Validation Accuracy:", accuracy_score(y_val_4, y_pred_xgb_4_ii))


pd.DataFrame({
    'feature': X_4.columns,
    'importance': pipe_xgb_4_ii['clf'].feature_importances_
}).sort_values(by='importance', ascending=False)


from sklearn.base import BaseEstimator, TransformerMixin

class ThresholdToBinary(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert all values greater than threshold to 1, else 0
        return (X > self.threshold).astype(int)


num_cols = ['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']
cat_cols = ['Stage_fear','Drained_after_socializing']

numeric_transformer = Pipeline([
    ('imputer', IterativeImputer(max_iter=1000, random_state=42)),
    ('scaler', StandardScaler()),
])

categorical_transformer = Pipeline([
    ('imputer', IterativeImputer(max_iter=1000, random_state=42)),
    ('threshold', ThresholdToBinary(threshold=0.5)),
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols),
], remainder='drop')

pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(random_state=42)),
])

pipe.fit(X_train_4, y_train_4)
pred = pipe.predict(X_val_4)
print()
print("Validation Accuracy:", accuracy_score(y_val_4, pred))


# df_test_4_lr_ii = df_test_4.copy()
# df_test_4_lr_ii['Personality'] = pipe.predict(df_test_4)
# submission = (df_test_raw[['id']]
#               .join(df_test_4_lr_ii['Personality']
#               .map({1:'Extrovert',0:'Introvert'})))
# submission.to_csv('submission.csv', index=None)
# submission


num_cols = ['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']
cat_cols = ['Stage_fear','Drained_after_socializing']

numeric_transformer = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5, weights="distance")),
    ('scaler', StandardScaler()),
])

categorical_transformer = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5, weights="distance")),
    ('threshold', ThresholdToBinary(threshold=0.5)),
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols),
], remainder='drop')

kmeans_pipe = Pipeline([
    ('pre', preprocessor),
    ('clusterer', KMeans(n_clusters=2, random_state=42))
])

kmeans_pipe.fit(X_4)
X_4['Cluster'] = kmeans_pipe.named_steps['clusterer'].labels_
df_train_cluster = X_4.join(y_4)
df_train_cluster[['Cluster','Personality']]


df_train_cluster[['Cluster','Personality']].value_counts(normalize=True)


df_test_4_kmeans = df_test_4.copy()
df_test_4_kmeans['Personality'] = kmeans_pipe.predict(df_test_4_kmeans)
submission = (df_test_raw[['id']]
              .join(df_test_4_kmeans['Personality']
              .map({0:'Extrovert',1:'Introvert'})))
submission.to_csv('submission.csv', index=None)
submission


# import os
# import shutil

# working_dir = '/kaggle/working'

# for filename in os.listdir(working_dir):
#     file_path = os.path.join(working_dir, filename)
#     try:
#         if os.path.isfile(file_path) or os.path.islink(file_path):
#             os.unlink(file_path)  # delete file or link
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)  # delete directory
#     except Exception as e:
#         print(f'Failed to delete {file_path}. Reason: {e}')

