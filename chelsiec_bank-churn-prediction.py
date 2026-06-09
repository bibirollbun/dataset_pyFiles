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
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import randint, uniform

from sklearn.model_selection import train_test_split,StratifiedShuffleSplit

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer


from sklearn.metrics.pairwise import rbf_kernel
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")


train_data = pd.read_csv('/kaggle/input/playground-series-s4e1/train.csv')
print(train_data.shape)
train_data.head()


train_data.describe().T


train_data.info()


#target %
print(f" Exited percent: {len(train_data[train_data['Exited'] == 1])/len(train_data)}")


#plotting distribution of numerical data

num_columns = train_data.select_dtypes(include='number').columns

train_data.hist(bins=50, figsize=(15, 12))
plt.show()


train_data["balance_cat"] = np.where(train_data['Balance'] > 0, 1, 0)

train_data["age_cat"] = pd.cut(train_data["Age"],
                               bins=[17,30,50,70,np.inf],
                               labels=[1, 2, 3, 4])
train_data["salary_cat"] = pd.cut(train_data["EstimatedSalary"],
                               bins=[-np.inf,50000, 100000, 150000, 200000],
                               labels=[1, 2, 3, 4])

bins = np.arange(train_data["CreditScore"].min() - 50,
                 train_data["CreditScore"].max() + 100,
                 100)

train_data["creditscore_cat"] = pd.cut(
    train_data["CreditScore"],
    bins=bins,
    labels=range(1, len(bins))
)
train_data.head()


numerical_cols = ['CustomerId', 'CreditScore','Age', 'Tenure', 'Balance','EstimatedSalary']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

for ax, col in zip(axes.flatten(), numerical_cols):
    sns.boxplot(data=train_data, x='Exited', y=col, ax=ax)
    ax.set_title(f'Exited by {col}')


numerical_data = train_data.select_dtypes(include='number')
correlation_matrix = round(numerical_data.corr(),2)
correlation_matrix

plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.1)
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()


# Category features
cols = ['Geography', 'Gender','NumOfProducts', 'HasCrCard',
        'IsActiveMember','balance_cat','age_cat','salary_cat','Tenure']

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

for ax, col in zip(axes.flatten(), cols):

    # Group by category and 'Exited'
    df = train_data.groupby([col, 'Exited']).size().unstack(fill_value=0)

    # Plot stacked bar first
    df.plot(kind='bar', stacked=True, ax=ax,  legend=False)

    # Add percentage labels inside each stack
    for i, cat in enumerate(df.index):
        total = df.loc[cat].sum()
        bottom = 0
        for j, exited in enumerate(df.columns):
            count = df.loc[cat, exited]
            percent = count / total * 100

            # Place text in the middle of the stack
            ax.text(
                i,                   # x-position (bar index)
                bottom + count/2,    # y-position (middle of stack)
                f'{percent:.1f}%',   # text
                ha='center', va='center', fontsize=8
            )
            bottom += count  # update bottom for next stack

    ax.set_title(f'Exited by {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    ax.set_xticklabels(df.index, rotation=45, ha='right')

plt.tight_layout()
plt.show()



# Given the observable differences in churn rates across tenure levels, I would like to apply tenure binning to better capture these patterns. 
train_data["tenure_cat"] = np.where(train_data['Tenure'] == 1, 1,  
                                    np.where((train_data['Tenure'] > 1) & (train_data['Tenure'] <= 5) , 2, 3)
                                   ).astype('str')
train_data.head()


#check missing value for final dataframe
train_data.isna().sum()


ids = train_data["id"]
features = train_data.drop(columns=["id", "Exited"])
target = train_data["Exited"]


X_full = train_data.drop(columns=["Exited"])  # keep ID inside X
y = train_data["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=train_data["age_cat"]
)

# Extract ID from the splitted sets
train_ids = X_train["id"]
test_ids = X_test["id"]

# Remove ID before feeding into ML model
X_train = X_train.drop(columns=["id"])
X_test = X_test.drop(columns=["id"])

print(y_train.sum()/len(y_train))
print(y_test.sum()/len(y_test))


print(X_train["age_cat"].value_counts() / len(X_train))
print(X_test["age_cat"].value_counts() / len(X_test))
print(X_train.shape)
print(X_test.shape)


print(X_train["balance_cat"].value_counts() / len(X_train))
print(X_test["balance_cat"].value_counts() / len(X_test))
print(X_train.shape)
print(X_test.shape)


# Add new feature1 - Family account features
# surname_df = pd.DataFrame(X_train['Surname'].value_counts()).reset_index()

# mapping = surname_df.set_index('Surname')['count']

# # Map
# X_train['surname_count'] = X_train['Surname'].map(mapping)
# X_train.head()


class SurnameCountTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mapping_ = X.squeeze().value_counts().to_dict()
        return self

    def transform(self, X):
        return pd.DataFrame({
            "surname_count": X.squeeze().map(self.mapping_).fillna(0)
        })

    def get_feature_names_out(self, input_features=None):
        return np.array(["surname_count"])


# Add new feature2 (Version2) - multimodal feature:Customerld

x = X_train["CustomerId"].values

# Estimate density
kde = gaussian_kde(x)
x_grid = np.linspace(x.min(), x.max(), 1000)
density = kde(x_grid)

# Find peaks
peaks, _ = find_peaks(density, distance=50)  # distance to separate peaks
peak_positions = x_grid[peaks]

print("Peak positions:", peak_positions)

# Plot
plt.plot(x_grid, density)
plt.plot(peak_positions, density[peaks], "x", color='red')
plt.show()



sample = np.linspace(X_train["CustomerId"].min(),
                   X_train["CustomerId"].max(),
                   3000).reshape(-1, 1)

# customerids = X_train["CustomerId"].values.reshape(-1, 1)
gamma1 = 0.000000001
gamma2 = 0.0000000005
gamma3 = 0.0000000003
rbf1 = rbf_kernel(sample, [[round(peak_positions[0])]], gamma=gamma1)
rbf2 = rbf_kernel(sample,[[round(sum(peak_positions[1:3])/2)]], gamma=gamma2)
rbf3 = rbf_kernel(sample,[[round(sum(peak_positions[3:])/3)]], gamma=gamma3)


fig, ax1 = plt.subplots()

ax1.set_xlabel("CustomerId")
ax1.set_ylabel("Number of users")
ax1.hist(X_train["CustomerId"], bins=50)

ax2 = ax1.twinx()  # create a twin axis that shares the same x-axis
color = "blue"
ax2.plot(sample, rbf1, color='blue', label=f"gamma = {gamma1}", linestyle="--")
ax2.plot(sample, rbf2, color='red', label=f"gamma = {gamma2}", linestyle="--")
ax2.plot(sample, rbf3, color='green', label=f"gamma = {gamma3}", linestyle="--")

ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel("CustomerId similarity", color=color)

plt.legend(loc="upper left")

plt.show()


# trying to similar the distribution
# peak_positions = [15589974.20620621, 15659540.71471472, 15686316.31331331, 15757384.25725726,
#  15776652.67867868, 15789414.87987988]
# gamma1 = 0.000000001
# gamma2 = 0.0000000005
# gamma3 = 0.0000000003

# def multi_rbf(X, peaks=peak_positions, gamma1=gamma1, gamma2=gamma2, gamma3=gamma3 ):
    
#     rbf1 = rbf_kernel(X, [[round(peak_positions[0])]], gamma=gamma1)
#     rbf2 = rbf_kernel(X,[[round(sum(peak_positions[1:3])/2)]], gamma=gamma2)
#     rbf3 = rbf_kernel(X,[[round(sum(peak_positions[3:])/3)]], gamma=gamma3)
    
#     return np.hstack([rbf1,rbf2,rbf3])

# rbf_transformer = FunctionTransformer(multi_rbf)


#defined class for the transofromer for later use in pipeline
class MultiRBFTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, peak_positions, gamma1, gamma2, gamma3):
        self.peak_positions = peak_positions
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
    
    def fit(self, X, y=None):
        # nothing to learn
        return self
    
    def transform(self, X):
        X = np.asarray(X).reshape(-1, 1)

        # YOUR EXACT RBF LOGIC
        rbf1 = rbf_kernel(X, [[round(self.peak_positions[0])]], gamma=self.gamma1)

        rbf2 = rbf_kernel(
            X,
            [[round(sum(self.peak_positions[1:3]) / 2)]],
            gamma=self.gamma2
        )

        rbf3 = rbf_kernel(
            X,
            [[round(sum(self.peak_positions[3:]) / 3)]],
            gamma=self.gamma3
        )

        # Stack all 3
        return np.hstack([rbf1, rbf2, rbf3])

    def get_feature_names_out(self, input_features=None):
        return ["rbf_peak1", "rbf_peak2", "rbf_peak3"]


peak_positions = [15589974.20620621, 15659540.71471472, 15686316.31331331, 15757384.25725726,
 15776652.67867868, 15789414.87987988]
gamma1 = 0.000000001
gamma2 = 0.0000000005
gamma3 = 0.0000000003

rbf_transformer = MultiRBFTransformer(
    peak_positions=peak_positions,
    gamma1=gamma1,
    gamma2=gamma2,
    gamma3=gamma3
)



# Categorical pipeline
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary"))
])


# Full preprocessing pipeline
preprocessing = ColumnTransformer([
    (
        "surname_count",
        Pipeline([
            ("surname_count_transform", SurnameCountTransformer()),
            ("scale", MinMaxScaler())
        ]),
        ["Surname"]
    ),
    ("custid_simil", rbf_transformer, ["CustomerId"]),     
    ("cat", cat_pipeline, ["Geography","Gender","HasCrCard","IsActiveMember","balance_cat"])
], remainder="passthrough")



feature_prepared = preprocessing.fit_transform(X_train)
feature_names = preprocessing.get_feature_names_out()
print(feature_names)


lr_model = LogisticRegression()
lr_model.fit(feature_prepared, y_train)
cross_val_score(lr_model, feature_prepared, y_train, cv=5, scoring="roc_auc")



dt_model = DecisionTreeClassifier()
dt_model.fit(feature_prepared, y_train)
cross_val_score(dt_model, feature_prepared, y_train, cv=5, scoring="roc_auc")


rf_model = RandomForestClassifier(n_jobs = 3)
rf_model.fit(feature_prepared, y_train)
cross_val_score(rf_model, feature_prepared, y_train, cv=5, scoring="roc_auc")


feature_importances = rf_model.feature_importances_
feature_importances.round(2)
sorted(zip(feature_importances,feature_names))


svm_model = SVC()
svm_model.fit(feature_prepared, y_train)
cross_val_score(svm_model, feature_prepared, y_train, cv=5, scoring="roc_auc")


xgb_model = XGBClassifier(n_jobs = 3)
xgb_model.fit(feature_prepared, y_train)
cross_val_score(xgb_model, feature_prepared, y_train, cv=5, scoring="roc_auc")


xgbf_names = ['f'+str(i) for i in range(len(feature_names))]

importance_dict = xgb_model.get_booster().get_score(importance_type='gain')
importance_list = [importance_dict.get(f, 0) for f in xgbf_names]

pd.DataFrame({
    'feature': feature_names,
    'importance':importance_list
}).sort_values(by='importance', ascending=False)


#from feature importance results, delete less important(<0.015) columns
feature_selected = X_train.drop(columns = ['age_cat'],axis=1)
selected_cols = feature_selected.columns


# Categorical pipeline
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary"))
])

# Full preprocessing
full_preprocessing = ColumnTransformer([
    (
        "surname_count",
        Pipeline([
            ("surname_count_transform", SurnameCountTransformer()),
            ("scale", MinMaxScaler())
        ]),
        ["Surname"]
    ),
    ("custid_simil", rbf_transformer, ["CustomerId"]),  
    ("cat", cat_pipeline, ["Geography","IsActiveMember","creditscore_cat","tenure_cat","creditscore_cat","salary_cat","balance_cat","HasCrCard","Gender"])
], remainder="passthrough")



rf_full_pipeline = Pipeline([
    ("preprocessing", full_preprocessing),
    ("random_forest", RandomForestClassifier(random_state=42)),
])
param_distribs = {'random_forest__max_depth': randint(low=2, high=15),
                  'random_forest__n_estimators': randint(low=2, high=50),
                  'random_forest__criterion': ['gini', 'entropy']}


rf_rnd_search = RandomizedSearchCV(
    rf_full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
    scoring='roc_auc', random_state=42,error_score='raise')

rf_rnd_search.fit(feature_selected, y_train)
print(rf_rnd_search.best_params_)


rf_final_model = rf_rnd_search.best_estimator_
rf_final_model.fit(feature_selected, y_train)


cvs = cross_val_score(rf_final_model, feature_selected, y_train, cv=5, scoring="roc_auc")
print(f"Average ROC-AUC Score {cvs.mean()}")


xgb_full_pipeline = Pipeline([
    ("preprocessing", full_preprocessing),
    ("xgb", XGBClassifier(random_state=42)),
])
param_distribs = {  'xgb__n_estimators': randint(2,50),       # number of trees
                    'xgb__max_depth': randint(2, 30),            # max tree depth
                    'xgb__learning_rate': uniform(0.01, 0.3),    # eta
                    'xgb__reg_alpha': uniform(0, 1),             # L1 regularization
                    'xgb__reg_lambda': uniform(0, 1)             # L2 regularization
                 }


xgb_rnd_search = RandomizedSearchCV(
    xgb_full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
    scoring='roc_auc', random_state=42,error_score='raise')

xgb_rnd_search.fit(feature_selected, y_train)
print(xgb_rnd_search.best_params_)


xgb_final_model = xgb_rnd_search.best_estimator_
xgb_final_model.fit(feature_selected, y_train)
cvs = cross_val_score(xgb_final_model, feature_selected, y_train, cv=5, scoring="roc_auc")
print(f"Average ROC-AUC Score {cvs.mean()}")


selected_cols


final_prediction_prob = xgb_final_model.predict_proba(X_test[selected_cols])[:, 1]  # probability of positive class

roc_auc = roc_auc_score(y_test, final_prediction_prob)
print("ROC-AUC:", roc_auc)


test_data = pd.read_csv("/kaggle/input/playground-series-s4e1/test.csv")
test_data.head()


ids = test_data["id"]


test_data["balance_cat"] = np.where(test_data['Balance'] > 0, 1, 0)

test_data["age_cat"] = pd.cut(test_data["Age"],
                               bins=[17,30,50,70,np.inf],
                               labels=[1, 2, 3, 4])
test_data["salary_cat"] = pd.cut(test_data["EstimatedSalary"],
                               bins=[-np.inf,50000, 100000, 150000, 200000],
                               labels=[1, 2, 3, 4])

# bins = np.arange(test_data["CreditScore"].min() - 50,
#                  test_data["CreditScore"].max() + 100,
#                  100)

test_data["creditscore_cat"] = pd.cut(
    test_data["CreditScore"],
    bins=bins,
    labels=range(1, len(bins))
)
test_data["tenure_cat"] = np.where(test_data['Tenure'] == 1, 1,  
                                    np.where((test_data['Tenure'] > 1) & (test_data['Tenure'] <= 5) , 2, 3)
                                   ).astype('str')



predict_probs = xgb_final_model.predict_proba(test_data[selected_cols])[:, 1]
final_result = pd.DataFrame({
    'id': ids,
    'Exited': predict_probs
})
final_result.head()


final_result.to_csv('submission.csv', index=False)




