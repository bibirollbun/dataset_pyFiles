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
import optuna

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

import xgboost as xgb

import json

import warnings
warnings.filterwarnings('ignore')


test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
test_df = test_data.copy()


train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
train_df = train_data.copy()
train_df.head()


train_df.info()


train_df.isna().sum()


train_df.describe().T


def feature_engineering(data):
    df = data.copy()
    
    map_humidity = {'dry': 0, 'optimal': 1, 'humid': 2}
    map_temperature = {'cold': 0, 'normal': 1, 'hot': 2, 'too hot': 3, 'insane hot': 4}
    
    df['Humidity Category'] = pd.cut(df['Humidity'], bins=[0, 40, 60, 100], labels=['dry', 'optimal', 'humid'])
    df['Temparature Category'] = pd.cut(df['Temparature'], bins=[-20, 5, 25, 35, 40, 50], labels=['cold', 'normal', 'hot','too hot', 'insane hot'])
    df['Humidity Level'] = df['Humidity Category'].map(map_humidity).astype(int)
    df['Temperature Level'] = df['Temparature Category'].map(map_temperature).astype(int)
    return df


cat_cols = train_df.select_dtypes(exclude=np.number).columns
num_cols = train_df.select_dtypes(include=np.number).columns
num_cols = num_cols[~np.isin(num_cols, 'id')]
print(f'Numeric variables: {list(num_cols)} \n, categoric variables: {list(cat_cols)}')


names = train_df['Fertilizer Name'].unique()
for name in names:
    fig, axes = plt.subplots(nrows=1, ncols=len(num_cols), figsize=(5*len(num_cols), 5))
    fig.suptitle(name, fontsize=30)
    for i, col in enumerate(num_cols):
        sns.boxplot(y=train_df[train_df['Fertilizer Name'] == name][col], ax=axes[i])
        axes[i].set_title(col)
    #plt.tight_layout()
    plt.show()




fig = plt.figure(figsize=(10, 7))
plt.pie(train_df['Fertilizer Name'].value_counts().values, labels=train_df['Fertilizer Name'].value_counts().index, autopct='%1.1f%%')

# show plot
plt.show()


mn = MinMaxScaler()
fertilizer_profile = train_df[np.append(num_cols, 'Fertilizer Name')].groupby('Fertilizer Name')[num_cols].mean().reset_index()
fertilizer_profile[num_cols] = mn.fit_transform(fertilizer_profile[num_cols])
fertilizer_profile


def radar_chart(data, metrics, title, size = (8, 8), color='b'):
    N = len(metrics)

    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    theta = np.append(theta, theta[0])

    values = data[metrics].values.flatten().tolist()
    values += values[:1]
    
    plt.figure(figsize=size)
    plt.subplot(polar=True)
    plt.plot(theta, values)
    plt.fill(theta, values, alpha=0.5, color=color)
    lines, labels = plt.thetagrids(np.degrees(theta[:N]), labels=metrics)
    plt.title(f"Fertilizer Profile: {title}")
    plt.show()


colors = ['b', 'r', 'g', 'c', 'm', 'y', 'b', 'r', 'g', 'c']
metrics = num_cols


for i in range(len(fertilizer_profile['Fertilizer Name'].values)):
    name = fertilizer_profile['Fertilizer Name'].values[i]
    radar_chart(fertilizer_profile[fertilizer_profile['Fertilizer Name'] == name], metrics, name, size=(5, 5), color=colors[i])


train_df.head()


   def calculate_map_at_k(model, x, y, k=3):
        y_pred_proba = model.predict_proba(x)
        scores = []
        
        for i in range(len(y)):
            top_k_idx = np.argsort(y_pred_proba[i])[-k:][::-1]
            ap = 0.0
            num_hits = 0
            
            for j, pred_idx in enumerate(top_k_idx):
                if pred_idx == y[i]:
                    num_hits += 1
                    ap += num_hits / (j + 1)
                    break
            
            scores.append(ap)
        
        return np.mean(scores)


class dropfeatureselector(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        X_dropped = X.drop(self.variables, axis = 1)
        return X_dropped


class featureselector(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        return X.loc[:,self.variables]

class simpleimputercustom(BaseEstimator, TransformerMixin):
    def __init__(self, variables, strategy):
        self.variables = variables
        self.strategy = strategy
        self.imp = SimpleImputer(missing_values=np.nan, strategy=self.strategy)
    def fit(self, X, y = None):
        X_ = X.loc[:,self.variables]
        self.imp.fit(X_)
        return self
    def transform(self, X):
        X_ = X.loc[:,self.variables]
        X_transformed = pd.DataFrame(self.imp.transform(X_), columns= self.variables)
        X.drop(self.variables, axis= 1, inplace=True)
        X[self.variables] = X_transformed[self.variables].values
        return X

class OneHotEncodercustom(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
        self.ohe = OneHotEncoder(drop='first', handle_unknown = 'ignore')
    def fit(self, X, y = None):
        X_ = X.loc[:,self.variables]
        self.ohe.fit(X_)
        return self
    def transform(self, X):
        X_ = X.loc[:,self.variables]
        X_transformed =  pd.DataFrame(self.ohe.transform(X_).toarray(), columns= self.ohe.get_feature_names_out())
        X.drop(self.variables, axis= 1, inplace=True)
        X[self.ohe.get_feature_names_out()] = X_transformed[self.ohe.get_feature_names_out()].values
        return X

class ModelSwitcher(BaseEstimator):
    
    def __init__(self, estimator = xgb.XGBClassifier(random_state=42)):
        self.estimator = estimator
    
    def fit(self, x, y=None, **kwargs):
        self.estimator.fit(x, y)
        return self
    
    def predict(self, x, y=None):
        return self.estimator.predict(x)
    
    def predict_proba(self, x):
        return self.estimator.predict_proba(x)
    
    def score(self, x):
        return self.estimator.score(x, y)



x = train_data.drop(columns=['Fertilizer Name'], axis=1)
y = train_data['Fertilizer Name']


cat_cols = x.select_dtypes(exclude=np.number).columns
num_cols = x.select_dtypes(include=np.number).columns
num_cols = num_cols[~np.isin(num_cols, 'id')]
print(f'Numeric variables: {list(num_cols)} \n, categoric variables: {list(cat_cols)}')


drop_features = dropfeatureselector(variables=['id'])
num_features = featureselector(variables=num_cols)
median_imputer = simpleimputercustom(variables=num_cols, strategy='median')
scaler = StandardScaler()

num_preprocess = Pipeline(
    steps=[('drop_features',drop_features),
           ('num_features_select', num_features),
           ('imputer', median_imputer)])


cat_features = featureselector(variables=cat_cols)
freq_imputer = simpleimputercustom(variables=cat_cols, strategy='most_frequent')
one_hot_encode = OneHotEncodercustom(variables=cat_cols)
cat_preprocess = Pipeline(
    steps=[('cat_feature_select', cat_features),
           ('imputer_cat', freq_imputer),
           ('one_hot', one_hot_encode)])


combined_preprocessing = FeatureUnion([
    ('numericals', num_preprocess),
    ('categoricals', cat_preprocess),
])


complete_pipeline = Pipeline([
        ('preprocessing', combined_preprocessing),
        ('StandardScaler', scaler),
        ('Model Training', ModelSwitcher())
    ])
display(complete_pipeline)




lb_target = LabelEncoder()
y = lb_target.fit_transform(y)



x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'tree_method': 'hist',
        'device': 'cuda'
    }

    pipeline = Pipeline([
        ('preprocessing', combined_preprocessing),
        ('StandardScaler', scaler),
        ('Model', xgb.XGBClassifier(**params, random_state=42))])
    pipeline.fit(x_train, y_train)
    #y_pred = model.predict_proba(x_valid)
    score = cross_val_score(pipeline, x, y, cv=5, scoring=calculate_map_at_k).mean()
    return score


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
print("Best trial:", study.best_trial.params)
print('Best value:', study.best_trial)


output_file = "best_params.json"
with open(output_file, "w") as f:
    json.dump(study.best_trial.params, f) # Using indent for readability

print(f"\nOptimized parameters successfully saved to {output_file}")



 with open(output_file, "r") as f:
        loaded_params = json.load(f)
print(loaded_params)
print(f"Verification successful: {loaded_params == study.best_trial.params}")


optimized_pipeline = Pipeline([
    ('preprocessing', combined_preprocessing),
    ('StandardScaler', scaler),
    ('Model', xgb.XGBClassifier(**study.best_trial.params, random_state=42))])
optimized_pipeline.fit(x, y)


calculate_map_at_k(optimized_pipeline, x, y)


predictions = optimized_pipeline.predict_proba(test_df)


np.argsort(predictions)


final = pd.DataFrame()
final['id'] = test_df['id']
final['Fertilizer Name'] = ''
final.head()


for i in range(1, 4):
    final['Fertilizer Name'] = final['Fertilizer Name'] + ' ' + lb_target.inverse_transform(np.argsort(predictions)[:, -i])


final.head()


final.to_csv('submission.csv', index=False)




