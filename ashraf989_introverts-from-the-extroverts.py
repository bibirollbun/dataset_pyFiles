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


pip install nbformat>=4.2.0



pip install optuna-integration[xgboost]


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots as sp
import plotly.figure_factory as ff
import plotly.offline as pyo
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


train= pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test= pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


train.head()


test.head()


columns_to_fill = ['Time_spent_Alone', 'Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']

for col in columns_to_fill:
    skewness= train[col].skew()
    if abs(skewness) > 1:
        print(f" -> {col} is skewed (|skewness| > 1")
    else:
        print(f" -> {col} is apporoximately normal")

fig, axes = plt.subplots(2,3, figsize=(15,10))
axes = axes.ravel()

for i, col in enumerate(columns_to_fill):
    axes[i].hist(train[col].dropna(), bins=20, alpha=0.7, edgecolor='black')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_ylabel('Frequency')

axes[5].remove()
plt.tight_layout()
plt.show()

train['Time_spent_Alone'].fillna(train['Time_spent_Alone'].median(), inplace=True)
train['Social_event_attendance'].fillna(train['Social_event_attendance'].mean(), inplace=True)
train['Going_outside'].fillna(train['Going_outside'].mean(),inplace=True)
train['Friends_circle_size'].fillna(train['Friends_circle_size'].median(),inplace=True)
train['Post_frequency'].fillna(train['Post_frequency'].median(),inplace=True)


def fill_categorical_missing(df, column, personality_weight=0.7):

    overall_mode = df[column].mode()[0]

    personality_modes = {}
    for personality in df['Personality'].unique():
        personality_data = df[(df['Personality'] == personality) & (df[column].notna())]
        if len(personality_data) > 0:
            personality_modes[personality] = personality_data[column].mode()[0]
        else:
            personality_modes[personality] = overall_mode

    df_filled = df.copy()
    for idx in df[df[column].isna()].index:
        personality = df.loc[idx, 'Personality']
        personality_mode = personality_modes[personality]

        if np.random.random() < personality_weight:
            df_filled.loc[idx, column] = personality_mode
        else:
            df_filled.loc[idx, column] = overall_mode
    
    return df_filled
    
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

def knn_impute_categorical(df, categorical_columns, numerical_columns):

    df_impute = df.copy()
    
    le_dict = {}
    for col in categorical_columns:
        if col in df_impute.columns:
            le = LabelEncoder()
            temp_fill = df_impute[col].fillna('MISSING')
            df_impute[col] = le.fit_transform(temp_fill)
            le_dict[col] = le
    
    feature_columns = numerical_columns + categorical_columns
    feature_data = df_impute[feature_columns].copy()
    
    imputer = KNNImputer(n_neighbors=5, weights='uniform')
    imputed_data = imputer.fit_transform(feature_data)

    df_imputed = df.copy()
    for i, col in enumerate(feature_columns):
        if col in categorical_columns:

            rounded_values = np.round(imputed_data[:, i]).astype(int)
            df_imputed[col] = le_dict[col].inverse_transform(rounded_values)
        else:
            df_imputed[col] = imputed_data[:, i]
    
    return df_imputed

train_filled_mode = fill_categorical_missing(train, 'Stage_fear')
train_filled_mode = fill_categorical_missing(train_filled_mode, 'Drained_after_socializing')

numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                  'Friends_circle_size', 'Post_frequency']
categorical_cols = ['Stage_fear', 'Drained_after_socializing']

for col in numerical_cols:
    if train[col].isna().sum() > 0:
        if abs(train[col].skew()) > 1:
            train[col].fillna(train[col].median(), inplace=True)
        else:
            train[col].fillna(train[col].mean(), inplace=True)

train_filled_knn = knn_impute_categorical(train, categorical_cols, numerical_cols)

train = fill_categorical_missing(train, 'Stage_fear')
train = fill_categorical_missing(train, 'Drained_after_socializing')


columns_to_fill = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

for col in columns_to_fill:
    if test[col].isna().sum() > 0:
        skewness = test[col].dropna().skew()
        
        if abs(skewness) > 1:
            print(f"  -> {col} is skewed (|skewness| > 1)")
        else:
            print(f"  -> {col} is approximately normal")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, col in enumerate(columns_to_fill):
    if test[col].isna().sum() > 0:
        axes[i].hist(test[col].dropna(), bins=20, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Distribution of {col} (Test)')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    else:
        axes[i].text(0.5, 0.5, f'{col}\nNo missing values', 
                    ha='center', va='center', transform=axes[i].transAxes)
        axes[i].set_title(f'{col} (Test)')

axes[5].remove()
plt.tight_layout()
plt.show()

test['Time_spent_Alone'].fillna(test['Time_spent_Alone'].median(), inplace=True)
test['Social_event_attendance'].fillna(test['Social_event_attendance'].mean(), inplace=True)
test['Going_outside'].fillna(test['Going_outside'].mean(), inplace=True)
test['Friends_circle_size'].fillna(test['Friends_circle_size'].median(), inplace=True)
test['Post_frequency'].fillna(test['Post_frequency'].median(), inplace=True)

categorical_cols_test = ['Stage_fear', 'Drained_after_socializing']

for col in categorical_cols_test:
    if test[col].isna().sum() > 0:
        mode_value = test[col].mode()[0]
        test[col].fillna(mode_value, inplace=True)

def knn_impute_test_data(df, categorical_columns, numerical_columns):

    df_impute = df.copy()
    
    le_dict = {}
    for col in categorical_columns:
        if col in df_impute.columns:
            le = LabelEncoder()
            temp_fill = df_impute[col].fillna('MISSING')
            df_impute[col] = le.fit_transform(temp_fill)
            le_dict[col] = le
    
    feature_columns = numerical_columns + categorical_columns
    feature_data = df_impute[feature_columns].copy()
    
    imputer = KNNImputer(n_neighbors=5, weights='uniform')
    imputed_data = imputer.fit_transform(feature_data)

    df_imputed = df.copy()
    for i, col in enumerate(feature_columns):
        if col in categorical_columns:
            rounded_values = np.round(imputed_data[:, i]).astype(int)
            df_imputed[col] = le_dict[col].inverse_transform(rounded_values)
        else:
            df_imputed[col] = imputed_data[:, i]
    
    return df_imputed

numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                  'Friends_circle_size', 'Post_frequency']
categorical_cols = ['Stage_fear', 'Drained_after_socializing']

test_filled_knn = knn_impute_test_data(test, categorical_cols, numerical_cols)


train.isna().sum()



test.isna().sum()



#train.head()
test.head()


train_encoded = train.copy()

le_personality = LabelEncoder()
le_stage_fear = LabelEncoder()
le_drained = LabelEncoder()

train_encoded['Personality_encoded'] = le_personality.fit_transform(train['Personality'])
train_encoded['Stage_fear_encoded'] = le_stage_fear.fit_transform(train['Stage_fear'])
train_encoded['Drained_after_socializing_encoded'] = le_drained.fit_transform(train['Drained_after_socializing'])

numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                  'Friends_circle_size', 'Post_frequency']

fig = sp.make_subplots(
    rows=2, cols=3,
    subplot_titles=['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                   'Friends_circle_size', 'Post_frequency'],
    specs=[[{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}],
           [{"type": "histogram"}, {"type": "histogram"}, {"type": "scatter"}]])

for i, col in enumerate(numerical_cols):
    row = (i // 3) + 1
    col_pos = (i % 3) + 1
    
    fig.add_trace(go.Histogram(x=train[col], name=col, nbinsx=30, showlegend=False),
        row=row, col=col_pos)

fig.update_layout(title="Distribution of Numerical Variables",
    height=600, showlegend=False)
fig.show()

fig = px.pie(train, names='Personality', 
    title='Distribution of Personality Types',
    color_discrete_sequence=px.colors.qualitative.Set3)

fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()

fig = sp.make_subplots(rows=1, cols=2,
    subplot_titles=['Stage Fear Distribution', 'Drained After Socializing Distribution'],
    specs=[[{"type": "bar"}, {"type": "bar"}]])

stage_fear_counts = train['Stage_fear'].value_counts()
fig.add_trace(go.Bar(x=stage_fear_counts.index, y=stage_fear_counts.values, name='Stage Fear'),
    row=1, col=1)

drained_counts = train['Drained_after_socializing'].value_counts()
fig.add_trace(go.Bar(x=drained_counts.index, y=drained_counts.values, name='Drained After Socializing'),
    row=1, col=2)

fig.update_layout(title="Distribution of Categorical Variables",
    height=400, showlegend=False)
fig.show()

fig = sp.make_subplots(rows=2, cols=3,
    subplot_titles=['Time_spent_Alone by Personality', 'Social_event_attendance by Personality', 
                   'Going_outside by Personality', 'Friends_circle_size by Personality', 
                   'Post_frequency by Personality'],
    specs=[[{"type": "box"}, {"type": "box"}, {"type": "box"}],
           [{"type": "box"}, {"type": "box"}, {"type": "scatter"}]])

for i, col in enumerate(numerical_cols):
    row = (i // 3) + 1
    col_pos = (i % 3) + 1
    
    for personality in train['Personality'].unique():
        data = train[train['Personality'] == personality][col]
        fig.add_trace(go.Box(y=data, name=f'{personality}', showlegend=False),
            row=row, col=col_pos)

fig.update_layout(title="Box Plots of Numerical Variables by Personality",height=600)
fig.show()

fig = sp.make_subplots(rows=2, cols=3,
    subplot_titles=['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                   'Friends_circle_size', 'Post_frequency'],
    specs=[[{"type": "violin"}, {"type": "violin"}, {"type": "violin"}],
           [{"type": "violin"}, {"type": "violin"}, {"type": "scatter"}]])

for i, col in enumerate(numerical_cols):
    row = (i // 3) + 1
    col_pos = (i % 3) + 1
    
    for personality in train['Personality'].unique():
        data = train[train['Personality'] == personality][col]
        fig.add_trace(go.Violin(y=data, name=f'{personality}', showlegend=False),
            row=row, col=col_pos)

fig.update_layout(title="Violin Plots of Numerical Variables by Personality",
    height=600)
fig.show()

fig = px.scatter_3d(train, x='Time_spent_Alone', y='Social_event_attendance',
    z='Going_outside', color='Personality',
    title="3D Scatter Plot: Time Alone vs Social Events vs Going Outside",
    opacity=0.7)

fig.update_layout(height=600)
fig.show()

all_numerical_cols = numerical_cols + ['Personality_encoded', 'Stage_fear_encoded', 'Drained_after_socializing_encoded']
target_correlations = train_encoded[all_numerical_cols].corr()['Personality_encoded'].abs().sort_values(ascending=False)

fig = px.bar(x=target_correlations.index,
    y=target_correlations.values,
    title="Feature Importance (Correlation with Personality)",
    labels={'x': 'Features', 'y': 'Absolute Correlation'})

fig.update_layout(height=500)
fig.show()

fig = sp.make_subplots(rows=2, cols=3,
    subplot_titles=['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                   'Friends_circle_size', 'Post_frequency'],
    specs=[[{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}],
           [{"type": "histogram"}, {"type": "histogram"}, {"type": "scatter"}]])

for i, col in enumerate(numerical_cols):
    row = (i // 3) + 1
    col_pos = (i % 3) + 1
    
    for personality in train['Personality'].unique():
        data = train[train['Personality'] == personality][col]
        fig.add_trace(go.Histogram(x=data, name=f'{personality}', opacity=0.7, showlegend=False),
            row=row, col=col_pos)

fig.update_layout(title="Distribution Comparison: Extrovert vs Introvert",
    height=600)
fig.show()

fig = px.scatter(train, x='Time_spent_Alone', y='Social_event_attendance',
    color='Personality', size='Friends_circle_size', hover_data=['Going_outside', 'Post_frequency', 'Stage_fear', 'Drained_after_socializing'],
    title="Interactive Scatter Plot with Multiple Features", opacity=0.7)
fig.update_layout(height=600)
fig.show()



numerical_columns = train.select_dtypes(include=[np.number]).columns.tolist()

correlation_matrix = train[numerical_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, 
            cmap='coolwarm', center=0,
            square=True, fmt='.2f',
            cbar_kws={'shrink': 0.8})

plt.title('Correlation Matrix - Train Dataset', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

correlation_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        col1 = correlation_matrix.columns[i]
        col2 = correlation_matrix.columns[j]
        corr_value = correlation_matrix.iloc[i, j]
        correlation_pairs.append((col1, col2, corr_value))

correlation_pairs.sort(key=lambda x: abs(x[2]), reverse=True)



test_encoded = test.copy()

le_stage_fear = LabelEncoder()
le_drained = LabelEncoder()

test_encoded['Stage_fear'] = test_encoded['Stage_fear'].fillna('Unknown')
test_encoded['Drained_after_socializing'] = test_encoded['Drained_after_socializing'].fillna('Unknown')

test_encoded['Stage_fear_encoded'] = le_stage_fear.fit_transform(test_encoded['Stage_fear'])
test_encoded['Drained_after_socializing_encoded'] = le_drained.fit_transform(test_encoded['Drained_after_socializing'])

numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                  'Friends_circle_size', 'Post_frequency']

fig = sp.make_subplots(rows=2, cols=3,
    subplot_titles=['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                   'Friends_circle_size', 'Post_frequency'],
    specs=[[{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}],
           [{"type": "histogram"}, {"type": "histogram"}, {"type": "scatter"}]])

for i, col in enumerate(numerical_cols):
    row = (i // 3) + 1
    col_pos = (i % 3) + 1
    
    fig.add_trace(go.Histogram(x=test[col], name=col, nbinsx=30, showlegend=False),
        row=row, col=col_pos)

fig.update_layout(title="Distribution of Numerical Variables (Test Dataset)",
    height=600, showlegend=False)
fig.show()

fig = sp.make_subplots(rows=1, cols=2,
    subplot_titles=['Stage Fear Distribution', 'Drained After Socializing Distribution'],
    specs=[[{"type": "bar"}, {"type": "bar"}]])

stage_fear_counts = test['Stage_fear'].value_counts()
fig.add_trace(go.Bar(x=stage_fear_counts.index, y=stage_fear_counts.values, name='Stage Fear'),
    row=1, col=1)

drained_counts = test['Drained_after_socializing'].value_counts()
fig.add_trace(go.Bar(x=drained_counts.index, y=drained_counts.values, name='Drained After Socializing'),
    row=1, col=2)

fig.update_layout(title="Distribution of Categorical Variables (Test Dataset)",
    height=400, showlegend=False)
fig.show()

fig = sp.make_subplots(rows=2, cols=3,
    subplot_titles=['Time_spent_Alone by Stage Fear', 'Social_event_attendance by Stage Fear', 
                   'Going_outside by Stage Fear', 'Friends_circle_size by Stage Fear', 
                   'Post_frequency by Stage Fear'],
    specs=[[{"type": "box"}, {"type": "box"}, {"type": "box"}],
           [{"type": "box"}, {"type": "box"}, {"type": "scatter"}]])

for i, col in enumerate(numerical_cols):
    row = (i // 3) + 1
    col_pos = (i % 3) + 1
    
    for stage_fear in test['Stage_fear'].unique():
        if pd.notna(stage_fear):
            data = test[test['Stage_fear'] == stage_fear][col]
            fig.add_trace(go.Box(y=data, name=f'{stage_fear}', showlegend=False),
                row=row, col=col_pos)

fig.update_layout(title="Box Plots of Numerical Variables by Stage Fear (Test Dataset)", height=600)
fig.show()

fig = sp.make_subplots(rows=2, cols=3,
    subplot_titles=['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                   'Friends_circle_size', 'Post_frequency'],
    specs=[[{"type": "violin"}, {"type": "violin"}, {"type": "violin"}],
           [{"type": "violin"}, {"type": "violin"}, {"type": "scatter"}]])

for i, col in enumerate(numerical_cols):
    row = (i // 3) + 1
    col_pos = (i % 3) + 1
    
    for stage_fear in test['Stage_fear'].unique():
        if pd.notna(stage_fear):
            data = test[test['Stage_fear'] == stage_fear][col]
            fig.add_trace(go.Violin(y=data, name=f'{stage_fear}', showlegend=False),
                row=row, col=col_pos)

fig.update_layout(title="Violin Plots of Numerical Variables by Stage Fear (Test Dataset)",
    height=600)
fig.show()

fig = px.scatter_3d(test, x='Time_spent_Alone', y='Social_event_attendance',
    z='Going_outside', color='Stage_fear',
    title="3D Scatter Plot: Time Alone vs Social Events vs Going Outside (Test Dataset)",
    opacity=0.7)

fig.update_layout(height=600)
fig.show()

fig = sp.make_subplots(rows=2, cols=3,
    subplot_titles=['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                   'Friends_circle_size', 'Post_frequency'],
    specs=[[{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}],
           [{"type": "histogram"}, {"type": "histogram"}, {"type": "scatter"}]])

for i, col in enumerate(numerical_cols):
    row = (i // 3) + 1
    col_pos = (i % 3) + 1
    
    for stage_fear in test['Stage_fear'].unique():
        if pd.notna(stage_fear):
            data = test[test['Stage_fear'] == stage_fear][col]
            fig.add_trace(go.Histogram(x=data, name=f'{stage_fear}', opacity=0.7, showlegend=False),
                row=row, col=col_pos)

fig.update_layout(title="Distribution Comparison: Stage Fear vs No Stage Fear (Test Dataset)",
    height=600)
fig.show()

fig = px.scatter(test, x='Time_spent_Alone', y='Social_event_attendance',
    color='Stage_fear', size='Friends_circle_size', 
    hover_data=['Going_outside', 'Post_frequency', 'Drained_after_socializing'],
    title="Interactive Scatter Plot with Multiple Features (Test Dataset)", 
    opacity=0.7)

fig.update_layout(height=600)
fig.show()



numerical_columns = test.select_dtypes(include=[np.number]).columns.tolist()

correlation_matrix = test[numerical_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, 
            cmap='coolwarm', center=0,
            square=True, fmt='.2f',
            cbar_kws={'shrink': 0.8})

plt.title('Correlation Matrix - Test Dataset', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

correlation_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        col1 = correlation_matrix.columns[i]
        col2 = correlation_matrix.columns[j]
        corr_value = correlation_matrix.iloc[i, j]
        correlation_pairs.append((col1, col2, corr_value))

correlation_pairs.sort(key=lambda x: abs(x[2]), reverse=True)


#train.head()
test.head()


cat_train = ["Stage_fear","Drained_after_socializing", "Personality"]
cat_test = ["Stage_fear","Drained_after_socializing"]

le_train = LabelEncoder()
for col in cat_train:
    train[col] = le_train.fit_transform(train[col].astype(str))

le_test = LabelEncoder()
for col in cat_test:
    test[col] = le_test.fit_transform(test[col].astype(str))


cat_train = ["Stage_fear","Drained_after_socializing", "Personality"]
cat_test = ["Stage_fear","Drained_after_socializing"]

le_train = LabelEncoder()
for col in cat_train:
    train[col] = le_train.fit_transform(train[col].astype(str))

le_test = LabelEncoder()
for col in cat_test:
    test[col] = le_test.fit_transform(test[col].astype(str))


from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold
import optuna
import lightgbm as lgb


from sklearn.model_selection import cross_val_score
import xgboost as xgb
from catboost import CatBoostClassifier


X = train.drop(columns=['Personality','id'])
y = train['Personality']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


kfold = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)


def objective_lgbm(trial):

    params = {'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbose': -1}
    
    model = lgb.LGBMClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    return scores.mean()

def objective_xgb(trial):

    params = {'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42}
    
    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    return scores.mean()

def objective_cb(trial):
    params = {'objective': 'Logloss',
        'eval_metric': 'Logloss',
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbose': False}
    
    model = CatBoostClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    return scores.mean()


print("\n1. Optimizing LightGBM...")
study_lgbm = optuna.create_study(direction='maximize')
study_lgbm.optimize(objective_lgbm, n_trials=50)

print(f"Best LightGBM CV Score: {study_lgbm.best_value:.4f}")
print(f"Best LightGBM Parameters: {study_lgbm.best_params}")

print("\n2. Optimizing XGBoost...")
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=50)

print(f"Best XGBoost CV Score: {study_xgb.best_value:.4f}")
print(f"Best XGBoost Parameters: {study_xgb.best_params}")

print("\n3. Optimizing CatBoost...")
study_cb = optuna.create_study(direction='maximize')
study_cb.optimize(objective_cb, n_trials=50)

print(f"Best CatBoost CV Score: {study_cb.best_value:.4f}")
print(f"Best CatBoost Parameters: {study_cb.best_params}")


study_lgbm.best_params



test



id = test["id"]
X_test = test.drop(columns=['id'])
X_test


best_lgbm_params = study_lgbm.best_params

best_lgbm_params.update({'objective': 'binary',
    'metric': 'binary_logloss', 'boosting_type': 'gbdt',
    'random_state': 42, 'verbose': -1})

final_lgbm_model = lgb.LGBMClassifier(**best_lgbm_params)

final_lgbm_model.fit(X_train, y_train)

Personality = final_lgbm_model.predict(X_test)


submission = pd.DataFrame({'id': id,
    'Personality': Personality})


submission.to_csv('submission.csv', index=False)





