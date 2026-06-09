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
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
import math
import logging
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger().setLevel(logging.ERROR)
%matplotlib inline


df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv', index_col='id')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv', index_col='id')


# df.replace([np.inf, -np.inf], np.nan, inplace=True)
# df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
df.head()


X = df.drop(['y'], axis=1)
y = df['y']


# num_cols = X.select_dtypes(include=['number']).columns.to_list()
# cat_cols = X.select_dtypes(exclude=['number']).columns.to_list()
num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
# print(num_cols, cat_cols)


import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_graph(d_cols, type_plot):
    n = len(d_cols)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(d_cols):
        if type_plot == 'countplot':
            sns.countplot(data=df, x=col, ax=axes[i])
            axes[i].set_title(f'Count Plot for {col}')
            axes[i].tick_params(axis='x', rotation=45)

        elif type_plot == 'hue_countplot':
            sns.countplot(data=df, x=col, hue='y', ax=axes[i])
            axes[i].set_title(f'{col} vs y')
            axes[i].tick_params(axis='x', rotation=45)

        elif type_plot == 'bar_stacked':
            prop_df = df.groupby(col)['y'].value_counts(normalize=True).unstack()
            prop_df.plot(kind='bar', stacked=True, ax=axes[i], colormap='Set2', legend=False)
            axes[i].set_title(f'Proportion of y by {col}')
            axes[i].set_ylabel('Proportion')
            axes[i].tick_params(axis='x', rotation=45)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Shared legend only for bar_stacked
    if type_plot == 'bar_stacked':
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title='y', loc='upper center', ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.95] if type_plot == 'bar_stacked' else None)
    plt.show()



plot_graph(cat_cols, 'countplot')


plot_graph(cat_cols, 'hue_countplot')


plot_graph(cat_cols, 'bar_stacked')


num_pipeline = Pipeline(steps=[
    ('scaler', RobustScaler())
])

cat_pipeline = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Combine all preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num_pre', num_pipeline, num_cols),
    ('cat_pre', cat_pipeline, cat_cols)
])


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    # ('classifier', LogisticRegression(max_iter=1000, solver='lbfgs'))
])
# (max_iter=2000, solver='lbfgs')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Fit model
pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)

print("Accuracy:", roc_auc_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


scores = cross_val_score(pipeline, X, y, cv=3, scoring='roc_auc')
print("Cross-Validation Accuracy: %.4f ± %.4f" % (scores.mean(), scores.std()))


test_predictions = pipeline.predict(df_test)


submission = pd.DataFrame({'y': test_predictions}, index=df_test.index)
submission.to_csv('submission.csv')
submission.head()




