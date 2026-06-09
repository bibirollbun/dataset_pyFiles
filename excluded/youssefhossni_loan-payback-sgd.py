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


train_path = "/kaggle/input/playground-series-s5e11/train.csv"
test_path = "/kaggle/input/playground-series-s5e11/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


train.head()


train.info()


x = train.drop(columns=['loan_paid_back', 'id'])
y = train['loan_paid_back']


from sklearn.model_selection import train_test_split

xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, test_size=0.2, random_state=42)


xtrain_num = xtrain.select_dtypes(include=['number'])
xtrain_cat = xtrain.select_dtypes(include=['object'])


# Calculating correlation between a single categorical data and a single numerical data (loan_paid_back)

def correlation_ratio(categories, values):
    categories = np.array(categories)
    values = np.array(values)
    cat_values = [values[categories == cat] for cat in np.unique(categories)]
    means = [np.mean(vals) for vals in cat_values]
    overall_mean = np.mean(values)
    num = sum([len(vals) * (m - overall_mean)**2 for vals, m in zip(cat_values, means)])
    den = sum((values - overall_mean)**2)
    return np.sqrt(num / den)



from sklearn.base import BaseEstimator, TransformerMixin

class NumAttributesSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold = 0.1): # no *args or **kargs
        self.threshold = threshold

    def fit(self, X, y):
        corrs = X.apply(lambda col: col.corr(y))
        self.selected_features_ = corrs[abs(corrs) > self.threshold].index.tolist()
        return self
        
    def transform(self, X):
        return X[self.selected_features_]
        


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ("selector", NumAttributesSelector()),
    ("Scaler", StandardScaler()),
])


# Categorical attributes selector

class CatAttributesSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold = 0.1): # no *args or **kargs
        self.threshold = threshold

    def fit(self, X, y):
        self.selected_features_ = []
        for col in X.columns:
            corr = correlation_ratio(X[col], y)
            if abs(corr) > self.threshold:
                self.selected_features_.append(col)
        return self

    def transform(self, X):
        return X[self.selected_features_]
        


from sklearn.preprocessing import OneHotEncoder

cat_pipeline = Pipeline([
    ("selector", CatAttributesSelector()),
    ("cat", OneHotEncoder()),
])


from sklearn.compose import ColumnTransformer


num = list(xtrain_num)
cat = list(xtrain_cat)


full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num),
    ("cat", cat_pipeline, cat)
])


x_train_prep = full_pipeline.fit_transform(xtrain, ytrain)


from sklearn.linear_model import SGDClassifier

sg = SGDClassifier(random_state=42, loss='log_loss')


# from sklearn.model_selection import cross_val_predict

# ytrain_pred = cross_val_predict(sg, x_train_prep, ytrain, cv=3)


# from sklearn.metrics import confusion_matrix

# confusion_matrix(ytrain, ytrain_pred)


# yscores = cross_val_predict(sg, x_train_prep, ytrain, method="decision_function", cv=3)


# from sklearn.metrics import precision_recall_curve

# precisions, recall, threshold = precision_recall_curve(ytrain, yscores)


# import matplotlib.pyplot as plt

# def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, highlight=None):
#     plt.figure(figsize=(8, 5))
    
#     # Precision and recall curves
#     plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
#     plt.plot(thresholds, recalls[:-1], "g-", label="Recall")

#     # Optional: highlight a specific threshold
#     if highlight is not None:
#         # Find the closest threshold index
#         idx = (abs(thresholds - highlight)).argmin()
#         plt.scatter(thresholds[idx], precisions[idx], color="blue")
#         plt.scatter(thresholds[idx], recalls[idx], color="green")
#         plt.axvline(thresholds[idx], color="k", linestyle=":")
#         plt.text(thresholds[idx], 0.5, f"threshold={thresholds[idx]:.2f}", 
#                  rotation=90, va="center")

#     plt.xlabel("Threshold")
#     plt.ylabel("Score")
#     plt.title("Precision/Recall vs Threshold")
#     plt.legend(loc="best")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# plot_precision_recall_vs_threshold(precisions, recall, threshold, 0.5)



# from sklearn.metrics import precision_score, recall_score

# print(precision_score(ytrain, ytrain_pred))
# print(recall_score(ytrain, ytrain_pred))


# from sklearn.metrics import f1_score

# f1_score(ytrain, ytrain_pred)


# from sklearn.metrics import roc_curve

# fpr, tpr, thresholds = roc_curve(ytrain, yscores)


# def plot_roc_curve(fpr, tpr, label=None):
#     plt.plot(fpr, tpr, linewidth=2, label=label)
#     plt.plot([0, 1], [0, 1], 'k--')

# plot_roc_curve(fpr, tpr)


# from sklearn.metrics import roc_auc_score

# roc_auc_score(ytrain, yscores)


sg.fit(x_train_prep, ytrain)


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(random_state=42)


forest.fit(x_train_prep, ytrain)


X_valid_prep = full_pipeline.transform(xvalid)


valid_pred = forest.predict(X_valid_prep)


from sklearn.metrics import f1_score

f1_score(yvalid, valid_pred)


# forest_scores = forest_pred[:, 1]

# fpr, tpr, thresholds = roc_curve(ytrain, forest_scores)


# plt.plot(fpr, tpr, "b:", label="SGD")
# plot_roc_curve(fpr, tpr, "Forest")
# plt.legend(loc="lower right")
# plt.show()


# roc_auc_score(ytrain, tree_scores)


test_copy = test.copy()
test_copy


test_copy.pop('id')
test_copy


x_test_prep = full_pipeline.transform(test_copy)


sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


sample_submission_df['loan_paid_back'] = forest.predict_proba(x_test_prep)[:,1]
sample_submission_df


sample_submission_df.to_csv('/kaggle/working/submission.csv', index=False)
sample_submission_df.head()

