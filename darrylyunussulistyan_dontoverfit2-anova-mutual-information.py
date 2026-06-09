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
%matplotlib inline

from sklearn.feature_selection import mutual_info_classif

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import CategoricalNB
from sklearn.multioutput import ClassifierChain
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import RobustScaler


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.feature_selection import mutual_info_classif


plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)


def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs

def make_mi_scores(X, y):
    mi_scores = mutual_info_classif(X, y)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def scale_features(df, df_test, method):
    if method == 'std':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise Exception("only std, minmax, robust")

    scaled_df = scaler.fit_transform(df)
    scaled_df_test = scaler.transform(df_test)

    return (scaled_df, scaled_df_test)


df_test = pd.read_csv('/kaggle/input/older-dataset-for-dont-overfit-ii-challenge/test.csv')
df = pd.read_csv('/kaggle/input/older-dataset-for-dont-overfit-ii-challenge/train.csv')


indep_cols = [str(i) for i in range(0, 300)]
dep_col = 'target' 


def scale_features(df, df_test, method):
    if method == 'std':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise Exception("only std, minmax, robust")

    scaled_df = scaler.fit_transform(df)
    scaled_df_test = scaler.transform(df_test)

    return (scaled_df, scaled_df_test)


def strat_k_fold(X_df, y_df, model):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=49480182)
    lst_aucroc_strat = []
      
    for train_index, test_index in skf.split(X_df, y_df):
        x_train_fold, x_test_fold = X_df.iloc[train_index, :], X_df.iloc[test_index, :]
        y_train_fold, y_test_fold = y_df[train_index], y_df[test_index]
        model.fit(x_train_fold, y_train_fold)
        y_hat_fold = model.predict(x_test_fold)
        lst_aucroc_strat.append(roc_auc_score(y_test_fold, y_hat_fold))

    return lst_aucroc_strat

models = [
    LogisticRegression(random_state=12349281)
]

def try_train_models(X_df, y_df, models=models):
    model_names = []
    model_scores = []
    
    for pm in models:
        if type(pm) is tuple:
            model = pm[0]
            name = pm[1]
        else:
            model = pm
            name = model.__class__.__name__
            
        print(name)
        aucs = strat_k_fold(X_df, y_df, model)
        score = np.mean(aucs)
        model_names.append(name)
        model_scores.append(score)
    
    model_df = pd.DataFrame({
        'Model': model_names,
        'Model_score': model_scores
    })
    
    return model_df.sort_values(by='Model_score',ascending=False)


# try ANOVA

def feat_select_anova(X, y, k):
    selector = SelectKBest(f_classif, k=20) # k is the number of features to be selected
    X_new = selector.fit_transform(X, y)
    
    features = pd.DataFrame({
        "cols" : X.columns,
        "selected" : selector.get_support(),
        "score": selector.scores_,
        "p_value": selector.pvalues_,
        "abs_score": np.abs(selector.scores_)
    })

    return (features, selector)

scaled_df, scaled_df_test = scale_features(df[indep_cols], df_test[indep_cols], method = 'robust')
scaled_dfXy = pd.DataFrame(scaled_df, columns=indep_cols)
scaled_dfXy['target'] = df[dep_col]

robust_anova_features, robust_anova_selector = feat_select_anova(scaled_dfXy[indep_cols], df[dep_col], k=20)
robust_anova_features_selected = robust_anova_features[robust_anova_features['selected'] == True].sort_values(by = 'abs_score', ascending=False)
robust_anova_features_selected = robust_anova_features_selected['cols'].to_numpy()
print(robust_anova_features_selected)

mi_scores = make_mi_scores(scaled_dfXy[robust_anova_features_selected], scaled_dfXy[dep_col])
mi_scores_df = mi_scores.to_frame()
features_selected_mi = mi_scores_df[mi_scores_df['MI Scores'] > 0].index.to_numpy()

print(features_selected_mi)
try_train_models(scaled_dfXy[features_selected_mi], df[dep_col], [LogisticRegression(random_state=12349281)])


features_selected = features_selected_mi
features_selected


scaled_df, scaled_df_test = scale_features(df[indep_cols], df_test[indep_cols], method = 'robust')
scaled_dfXy = pd.DataFrame(scaled_df, columns=indep_cols)
scaled_dfXy['target'] = df[dep_col]

scaled_df_testXy = pd.DataFrame(scaled_df_test, columns=indep_cols)

# constants taken from grid search
model = LogisticRegression(**{'C': 0.1,
 'class_weight': None,
 'fit_intercept': True,
 'penalty': 'l2',
 'solver': 'sag',
 'tol': 0.1})
model.fit(scaled_dfXy[features_selected], df[dep_col])
y_hat = model.predict(scaled_df_testXy[features_selected])

dtest_submission = pd.DataFrame({
        "id": df_test["id"],
        "target": y_hat.astype(int)
    })
dtest_submission.to_csv('./submission.csv', index=False)


dtest_submission['target'].value_counts()


!head submission.csv

