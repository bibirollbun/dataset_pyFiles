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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import  TransformerMixin
from sklearn.preprocessing import  MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import  LabelEncoder
from collections import defaultdict
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


import warnings
warnings.filterwarnings('ignore')


df  = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


df.shape


df.head()


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

numerical_cols = df.select_dtypes(include='number').columns.difference(['id'])

num_cols = 3  # Number of columns in the grid
num_rows = (len(numerical_cols) + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), constrained_layout=True)
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.kdeplot(df[col], ax=axes[i], color='blue', fill=True, label='Train')
    if col in test.columns:
        sns.kdeplot(test[col], ax=axes[i], color='red', fill=True, label='Test')
    axes[i].set_title(col)

    ax_box = axes[i].inset_axes([0.2, -0.4, 0.6, 0.2])  # [x, y, width, height]
    sns.boxplot(x=df[col], ax=ax_box, orient='h')
    ax_box.set(xlabel='')

# Remove unused subplots
for j in range(len(numerical_cols), len(axes)):
    fig.delaxes(axes[j])

# Add a legend in the top right corner of the figure
custom_lines = [
    Line2D([0], [0], color='blue', lw=4, label='Train'),
    Line2D([0], [0], color='red', lw=4, label='Test')
]
fig.legend(
    handles=custom_lines,
    loc='upper right',
    bbox_to_anchor=(1, 1),  # x=1 (right), y=1 (top)
    frameon=False
)

plt.show()



numerical_cols = df.select_dtypes(include='number').columns.difference(['id'])


corr_matrix = df[numerical_cols].corr()

# Affichage avec Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation matrix")
plt.show()


cat_columns = df.select_dtypes(exclude=np.number).columns
num_cols = 3  # Number of columns in the grid
num_rows = (len(cat_columns) + num_cols - 1) // num_cols

# Create the subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), constrained_layout=True)
axes = axes.flatten()
palette = sns.color_palette("Set2", len(df.iloc[:, 0].value_counts()))

for i, col in enumerate(cat_columns):
    df[col].value_counts().plot(kind='bar', ax=axes[i], color=palette)
    axes[i].set_title(col)
    axes[i].tick_params(axis='x', rotation=45, labelsize=8)
for j in range(len(cat_columns), len(axes)):
    fig.delaxes(axes[j])

plt.show()


class Custom_Scaler(TransformerMixin):
    def __init__(self, except_col=[], cols=[], strategy="MinMax"):
        super().__init__()
        self.except_col=except_col
        self.cols = cols if cols else []
        self.strategy = strategy

    def fit(self, df, y=None):
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        final_col =  numerical_cols.difference(self.except_col)
        self.col  =  final_col if not self.cols else self.cols
        if self.strategy=="RBT":
            self.scaler = RobustScaler().fit(df[self.col]) 
        elif self.strategy=="STD" :
            self.scaler = StandardScaler().fit(df[self.col])
        else :
            self.scaler =  MinMaxScaler().fit(df[self.col]) 
        return self
    
    def transform(self, data, y=None):
        df =data.copy()
        scaler_data =  self.scaler.transform(df[self.col])
        scaler_data_df = pd.DataFrame(scaler_data, columns=self.col, index=df.index)
        others_cols  =  df.columns.difference(self.col)
        return pd.concat([scaler_data_df, df[others_cols]], axis='columns')

class MultiColumnLabelEncoder(TransformerMixin):
    def __init__(self, except_col=[]):
        self.except_col = except_col
        self.label_encoders = defaultdict(LabelEncoder)

    def fit(self,X , y=None):
        df  = X.copy()
        cat_col =  df.select_dtypes(exclude=[np.number]).columns
        final_col =  cat_col.difference(self.except_col)
        self.columns = final_col
        for col in self.columns:
            self.label_encoders[col]
            self.label_encoders[col].fit(df[col])
        return self

    def transform(self, X):
        X_copy = X.copy()  # To avoid modifying the original dataframe
        for col in self.columns:
            X_copy[col] = X_copy[col].apply(lambda s: '<unknown>' if s not in self.label_encoders[col].classes_ else s)
            self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, '<unknown>')
            X_copy[col] = self.label_encoders[col].transform(X_copy[col])
        return X_copy

    def inverse_transform(self, X):
        X_copy = X.copy()  # To avoid modifying the original dataframe
        for col in self.columns:
            X_copy[col] = self.label_encoders[col].inverse_transform(X_copy[col])
        return X_copy


pipe  = Pipeline([('scaler', Custom_Scaler(except_col=['id', 'accident_risk'])), ('label_encoder', MultiColumnLabelEncoder())])


transform_data  = pipe.fit_transform(df)


test_transform =  pipe.transform(test.drop(['id'], axis='columns'))


X = transform_data.drop(['id', 'accident_risk'], axis='columns')
y =  transform_data['accident_risk']





import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from category_encoders import MEstimateEncoder 

feature_importances =[]
def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",   
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
        "verbose":-1
    }
    from sklearn.model_selection import KFold
    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = LGBMRegressor(**params)
        # Entraînement
        model.fit(X_train, y_train)
        
        # Prédiction
        y_pred = model.predict(X_test)
        
        # Calcul du score
        score = np.sqrt(mean_squared_error((y_test), (y_pred)))
        print(score)
        scores.append(score)
        feature_importances.append(model.feature_importances_)
        return score
    # Afficher les résultats
    print(f"Scores pour chaque fold : {scores}")
    print(f"Score moyen : {np.mean(scores):.4f}±{np.std(scores)}")
    return np.mean(scores) + np.std(scores)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)
print(study.best_value)
print(study.best_params)


LGBM_params ={'learning_rate': 0.03557928710293894, 'n_estimators': 264, 'max_depth': 14, 'num_leaves': 191, 'min_child_samples': 75, 'subsample': 0.8482020047135577, 'colsample_bytree': 0.7414288744230556, 'reg_alpha': 0.10403322095969704, 'reg_lambda': 0.2985515688566991, "verbose":-1}


skf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
prediction =  []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model = LGBMRegressor()
    # Entraînement
    model.fit(X_train, y_train)
    # Prédiction
    y_pred1 = model.predict(X_test)
    # Calcul du score
    score1 = np.sqrt(mean_squared_error((y_test), (y_pred1)))
    print(f"score model 1 : {score1}")
    scores.append(score1)
    prediction.append (model.predict(test_transform))
# Aficher les résultats
print(f"Scores pour chaque fold model 1 : {scores}")
print(f"Score moyen model 2: {np.mean(scores):.4f}±{np.std(scores)}")



submission  = pd.DataFrame([], columns=['id', 'accident_risk'])
submission.id  =  test.id
submission['accident_risk']  =np.mean((prediction), axis=0)


submission.head()


submission.to_csv('submission.csv', index=False)

