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


import warnings
import optuna
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import defaultdict
from matplotlib.lines import Line2D

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    LabelEncoder
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    cross_val_score
)
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    mean_squared_error
)

from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer



warnings.filterwarnings('ignore')


sns.palplot(sns.color_palette("bright", 10))
palette = sns.color_palette("bright", 10)


df  = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test  = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


df.head()


df.isna().sum()


test.isna().sum()


df.shape


numerical_cols = df.select_dtypes(include='number').columns.difference(['id'])

num_cols = 3  # Number of columns in the grid
num_rows = (len(numerical_cols) + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), constrained_layout=True)
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.kdeplot(df[col], ax=axes[i], color=palette[0], fill=True, label='Train')
    if col in test.columns:
        sns.kdeplot(test[col], ax=axes[i], color=palette[1], fill=True, label='Test')
    axes[i].set_title(col)

    ax_box = axes[i].inset_axes([0.2, -0.4, 0.6, 0.2])  # [x, y, width, height]
    sns.boxplot(x=df[col], ax=ax_box, orient='h')
    ax_box.set(xlabel='')

# Remove unused subplots
for j in range(len(numerical_cols), len(axes)):
    fig.delaxes(axes[j])

# Add a legend in the top right corner of the figure
custom_lines = [
    Line2D([0], [0], color=palette[0], lw=4, label='Train'),
    Line2D([0], [0], color=palette[1], lw=4, label='Test')
]
fig.legend(
    handles=custom_lines,
    loc='upper right',
    bbox_to_anchor=(1, 1),  # x=1 (right), y=1 (top)
    frameon=False
)

plt.show()


df.columns


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
#palette = sns.color_palette("Set2", len(df.iloc[:, 0].value_counts()))

for i, col in enumerate(cat_columns):
    df[col].value_counts().plot(kind='bar', ax=axes[i], color=palette)
    axes[i].set_title(col)
    axes[i].tick_params(axis='x', rotation=45, labelsize=8)
for j in range(len(cat_columns), len(axes)):
    fig.delaxes(axes[j])

plt.show()


df.groupby(by="loan_paid_back")[cat_columns].describe().T


df.groupby(by="employment_status")["education_level"].value_counts().sort_values(ascending=False).plot(kind="bar", color=palette)


df.groupby(by="education_level")["marital_status"].value_counts().sort_values(ascending=False).plot(kind="bar", color=palette)


df.groupby(by="loan_purpose")["gender"].value_counts().sort_values(ascending=False).plot(kind="bar", color=palette)


df.groupby(by="gender")["marital_status"].value_counts().sort_values(ascending=False).plot(kind="bar", color=palette)


def create_feature(df):
    df = df.copy()
    df['credit_score_per_interest_rate'] = df["credit_score"] /df["interest_rate"]
    df["debt_to_income_ratio_interest_rate"] = df["debt_to_income_ratio"]/df["interest_rate"]
    df["annual_income"] = np.log1p(df["annual_income"])
    joined_columns = [
        ("employment_status", "education_level"),
        ("education_level", "marital_status"),
        ("loan_purpose", "gender"),
        ("gender", "marital_status"),
        ("loan_purpose", "employment_status"),
    ]
    for cols in joined_columns:
        df[f'{cols[0]}_{cols[1]}'] =  df[cols[0]] +"-"+ df[cols[1]]
    return df


df = create_feature(df)


test = create_feature(test)


mapping_data= {
    "grade_subgrade":{key: index for index, key in enumerate(sorted(list(df["grade_subgrade"].value_counts().index)))},
    "education_level" : {
        "Other": 0,
        "High School": 1,
        "Bachelor's": 2,
        "Master's" : 3,
        "PhD": 4
    }
}


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


pipe  = Pipeline([('scaler', Custom_Scaler(except_col=['id', 'loan_paid_back'])), ('label_encoder', MultiColumnLabelEncoder())])


transform_data  = pipe.fit_transform(df)


test_transform =  pipe.transform(test.drop(['id'], axis='columns'))


X = transform_data.drop(['id', 'loan_paid_back'], axis='columns')
y =  transform_data['loan_paid_back']


# import warnings
# import optuna
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# from collections import defaultdict
# from matplotlib.lines import Line2D

# from sklearn.base import TransformerMixin
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import (
#     MinMaxScaler,
#     StandardScaler,
#     RobustScaler,
#     LabelEncoder
# )
# from sklearn.model_selection import (
#     train_test_split,
#     StratifiedKFold,
#     KFold,
#     cross_val_score
# )
# from sklearn.metrics import (
#     roc_auc_score,
#     accuracy_score,
#     mean_squared_error
# )

# from lightgbm import LGBMClassifier
# import lightgbm as lgb
# from sklearn.datasets import load_breast_cancer



Lgbm_params  = {'learning_rate': 0.029182725943328738, 'num_leaves': 206, 'max_depth': 6, 'min_data_in_leaf': 177, 'feature_fraction': 0.40297986920383033, 'bagging_fraction': 0.9847395789013972, 'bagging_freq': 7, 'lambda_l1': 2.4583908740117452e-08, 'lambda_l2': 6.059217643816623, 'device': 'gpu'}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
prediction =  []
importance = []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model = LGBMClassifier(**Lgbm_params, n_estimators=1000 ,verbose=-1)
    # Entraînement
    model.fit(X_train, y_train)
    # Prédiction
    y_pred1 = model.predict_proba(X_test)
    # Calcul du score
    score1 = roc_auc_score(y_test, y_pred1[:, 1])
    print(f"score model 1 : {score1}")
    scores.append(score1)
    prediction.append (model.predict_proba(test_transform))
    importance.append(model.feature_importances_)
# Aficher les résultats
print(f"Scores pour chaque fold model 1 : {scores}")
print(f"Score moyen model 2: {np.mean(scores):.4f}±{np.std(scores)}")


pd.DataFrame({
    "columns" : X.columns,
    "importance" : np.mean(importance, axis=0)
}).sort_values(by="importance")


submission  = pd.DataFrame([], columns=['id', 'loan_paid_back'])
submission.id  =  test.id
submission['loan_paid_back']  =np.mean(np.array(prediction)[:,:, 1], axis=0)


submission.head()


submission.to_csv("submission.csv", index=False)

