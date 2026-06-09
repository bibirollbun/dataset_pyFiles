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


import os
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')



# LOAD DATA

train_df = pd.read_csv(r'/kaggle/input/playground-series-s5e10/train.csv').drop(columns = 'id')
test_df  = pd.read_csv(r'/kaggle/input/playground-series-s5e10/test.csv').drop(columns = 'id')

print(f'Train : {train_df.shape[0]}')
print(f'Test  : {test_df.shape[0]}')



print(train_df.head()) # GET DESCRIPTIVE STATISTICS.


print(train_df.info())  # GET INFORMATION ABOUT THE DATA FRAME.


print(train_df.describe())  # GET DISCRIPTIVE STATISTICS.


display(train_df)
display(test_df)


# CHECK TARGET DISTRIBUTION

sns.histplot(train_df['accident_risk'], kde = True, color = 'orange', bins = 50)



# CHECK CORRELATION BETWEEN OOF

# GET ALL OOF AUTOGLUON COLUMNS
autogluon_features = train_df.iloc[:, 12:].columns

# HEATMAP CORRELATION
plt.figure(figsize = (5, 5))
corr_matrix = train_df[autogluon_features].corr(method = 'spearman')
sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f')



plt.figure(figsize = (5, 5))


corr_matrix = train_df.describe().corr(method = 'spearman')



sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f')


# FEATURE ENGINEERING 

def f(X):
    return \
    0.3 * X["curvature"] + \
    0.2 * (X["lighting"] == "night").astype(int) + \
    0.1 * (X["weather"] != "clear").astype(int) + \
    0.2 * (X["speed_limit"] >= 60).astype(int) + \
    0.1 * (X["num_reported_accidents"] > 2).astype(int)



def clip(f):
    def clip_f(X):
        sigma = 0.05
        mu = f(X)
        a, b = -mu/sigma, (1-mu)/sigma
        Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
        phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
        return mu*(Phi_b-Phi_a)+sigma*(phi_a-phi_b)+1-Phi_b
    return clip_f

train = clip(f)(train_df)
test = clip(f)(test_df)

train_df['score'] = train
test_df['score']  = test

train_df


%%capture
!pip install autogluon.tabular scikit-learn==1.5.2
#!pip install "autogluon.tabular[tabpfn]==1.4.0"


from autogluon.tabular import TabularPredictor

import warnings
warnings.filterwarnings('ignore')


# AUTOGLUON

# DEFINE AUTOGLUON
predictor = TabularPredictor(label = 'accident_risk',
                             problem_type = 'regression',
                             eval_metric = 'rmse')

# TRAIN AUTOGLUON
predictor.fit(train_df,
              presets = 'best_quality',
              time_limit = 3600 * 11,
              auto_stack = True,
              #num_bag_folds = 5,
              #num_bag_sets = 3,
              num_cpus = 4,
              verbosity = 1,
              ag_args_fit={'early_stopping_rounds': 300, 'num_cpus': 4, 'num_gpus': 0},
              #ag_args_fit={'time_limit': 900}
             )



%%time
# COMPARE MODELS
predictor.leaderboard(silent = True)


# CHECK FEATURE IMPORTANCES 

importance_df = predictor.feature_importance(merge_train[:500])

importance_df.style.background_gradient(subset=['importance', 'stddev'], cmap='Blues')



# PLOT FEATURE IMPORTANCE

imp = importance_df['importance'].sort_values(ascending=True)

plt.figure(figsize=(6, 8))
imp.plot(kind='barh', color='steelblue')
plt.title('Feature Importance (AutoGluon)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()




