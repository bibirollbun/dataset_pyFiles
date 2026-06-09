import numpy as np
import pandas as pd
import warnings
# Data preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Graphing libraries
import matplotlib.pyplot as plt
import seaborn as sns
# Stratified K fold cross validation
from statistics import mean, stdev
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model
from sklearn import datasets
# LightGBM
import lightgbm as lgb # to train a series of decision trees
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
# Decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
# Random forest
from sklearn.ensemble import RandomForestClassifier
# Model visualization
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
from sklearn.tree import export_text
# Model evaluation
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# Ignore runtime warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)


# Load training data
train = pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx")
cate = pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx")
train = pd.merge(train, cate, on='participant_id', how='left') # merge quantitative and categorical data

# Load training solutions
solution = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx")
train = pd.merge(train, solution, on='participant_id', how='left') # merge solutions with training dataset

# Load test data
test = pd.read_excel(f"/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx")
cate = pd.read_excel(f"/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx")
test = pd.merge(test, cate, on='participant_id', how='left')

# Display the first few rows of the training data
train.head()


# Based on the competition's provided Data Dictionary
data = pd.read_excel(f"/kaggle/input/widsdatathon2025/Data Dictionary.xlsx")

df = pd.DataFrame(data)
df.iloc[list(range(2,20)) + list(range(23,32)),1:4].style.set_table_styles( # select specific rows & columns
    [{'selector':'th','props':[('text-align','left')]}]).set_properties( # align column headers left for readability
    subset=pd.IndexSlice[:], **{'text-align':'left'}).hide() # align rows to the left + hide row indexes


# Separate features and target variables
X = train.drop(['participant_id', 'ADHD_Outcome', 'Sex_F'], axis=1, errors='ignore')
y_adhd = train['ADHD_Outcome']
y_sex = train['Sex_F']

# Identify categorical and numerical features
categorical_features = X.iloc[:,18:27].columns.tolist()
numerical_features = X.iloc[:,:18].columns.tolist()

# Create preprocessing pipelines - library in sklearn to create data processing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # fills in missing vals
    ('scaler', StandardScaler()) # scales the vars - z score
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

preprocessor.set_output(transform="pandas")

# Apply preprocessing to training data
X_preprocessed = preprocessor.fit_transform(X)


sns.countplot(data=train, x='ADHD_Outcome', hue='Sex_F')
plt.title('ADHD Outcome Count by Sex')
plt.xlabel('ADHD Outcome (0=Other/None, 1=ADHD)')
plt.ylabel('Count')
plt.legend(title='Sex (0=Male, 1=Female)')
plt.show()


sns.countplot(data=train, x='Sex_F', hue='ADHD_Outcome')
plt.title('Percentage of Genders with ADHD')
plt.xlabel('Sex (0=Male, 1=Female)')
plt.ylabel('Count')
plt.legend(title='ADHD Outcome (0=Other/None, 1=ADHD)')
plt.show()


table = pd.crosstab(train['Sex_F'], train['ADHD_Outcome'], normalize='index')
print(table)


def weighted_f1_score(y_true_adhd, y_pred_adhd, y_true_sex, y_pred_sex):
    weights = [2 if (a == 1 and s == 1) else 1 # places more weight on individuals who are both female and have adhd
            for a, s in zip(y_true_adhd, y_true_sex)]

    def compute_f1(y_true, y_pred, weights):
        TP = sum(w for i, w in enumerate(weights) if y_true.iloc[i] == 1 and y_pred[i] == 1) # target vars are
        FP = sum(w for i, w in enumerate(weights) if y_true.iloc[i] == 0 and y_pred[i] == 1) # pandas objs, pred vals
        FN = sum(w for i, w in enumerate(weights) if y_true.iloc[i] == 1 and y_pred[i] == 0) # are numpy arrays

        if TP + FP == 0 or TP + FN == 0:
            return 0.0

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    f1_adhd = compute_f1(y_true_adhd, y_pred_adhd, weights)
    f1_sex = compute_f1(y_true_sex, y_pred_sex, weights)

    # Final F1 on the leaderboard
    return (f1_adhd + f1_sex) / 2


adhd_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_adhd), y=y_adhd)
sex_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_sex), y=y_sex)

# Define LightGBM models
adhd_model = lgb.LGBMClassifier(
    objective='binary',
    num_leaves=63,
    learning_rate=0.01,
    n_estimators=1000, # specifies num of trees in forest, accuracy stabilizes after certain amount
    scale_pos_weight=adhd_weights[1] / adhd_weights[0], # more adhd than non; lower weight on adhd makes up for bias
    early_stopping_rounds=50,
    verbose=-1
)

sex_model = lgb.LGBMClassifier(
    objective='binary',
    num_leaves=100, # smaller tree better fit for small dataset
    learning_rate=0.1, # how much model adjusts based on new trees/boosting
    n_estimators=10, # num of decision trees
    scale_pos_weight=sex_weights[1] / sex_weights[0], # more males than females; weight on females makes up for bias
    early_stopping_rounds=50, # stop at this num of trees before reaching n_estimators num due to lack of improvement
    verbose=-1 # do not print updates during decision tree process
)

# Apply stratified k fold cross validation using K folds, list to store accuracy scores
skf = StratifiedKFold(n_splits=100, shuffle=True, random_state=80)
list_accu_stratified = []

# Stratify by adhd
for train_index, test_index in skf.split(X, y_adhd):
    X_train_fold, X_test_fold = X_preprocessed.iloc[train_index], X_preprocessed.iloc[test_index] # row indeces
    y_train_adhd_fold, y_test_adhd_fold = y_adhd[train_index], y_adhd[test_index]
    y_train_sex_fold, y_test_sex_fold = y_sex[train_index], y_sex[test_index]
    adhd_model.fit(X_train_fold, y_train_adhd_fold, eval_set=[(X_test_fold, y_test_adhd_fold)])
    sex_model.fit(X_train_fold, y_train_sex_fold, eval_set=[(X_test_fold, y_test_sex_fold)])
    adhd_pred = adhd_model.predict(X_test_fold)
    sex_pred = sex_model.predict(X_test_fold)
    list_accu_stratified.append(weighted_f1_score(y_test_adhd_fold, adhd_pred, y_test_sex_fold, sex_pred))

# print('List of possible accuracies:', list_accu_stratified)
print("-Stratified by ADHD-")
print('Maximum F1 Score:',
	max(list_accu_stratified))
print('Minimum F1 Score:',
	min(list_accu_stratified))
print('Overall F1 Score:',
	mean(list_accu_stratified))
print('Standard Deviation:', stdev(list_accu_stratified))

list_accu_stratified = []

# Stratify by sex
for train_index, test_index in skf.split(X, y_sex):
    X_train_fold, X_test_fold = X_preprocessed.iloc[train_index], X_preprocessed.iloc[test_index] # row indeces
    y_train_adhd_fold, y_test_adhd_fold = y_adhd[train_index], y_adhd[test_index]
    y_train_sex_fold, y_test_sex_fold = y_sex[train_index], y_sex[test_index]
    adhd_model.fit(X_train_fold, y_train_adhd_fold, eval_set=[(X_test_fold, y_test_adhd_fold)])
    sex_model.fit(X_train_fold, y_train_sex_fold, eval_set=[(X_test_fold, y_test_sex_fold)])
    adhd_pred = adhd_model.predict(X_test_fold)
    sex_pred = sex_model.predict(X_test_fold)
    list_accu_stratified.append(weighted_f1_score(y_test_adhd_fold, adhd_pred, y_test_sex_fold, sex_pred))

print("\n-Stratified by Sex-")
print('Maximum F1 Score:',
	max(list_accu_stratified))
print('Minimum F1 Score:',
	min(list_accu_stratified))
print('Overall F1 Score:',
	mean(list_accu_stratified))
print('Standard Deviation:', stdev(list_accu_stratified))


# Optimize best parameter values for decision tree
param_grid = {
    'max_depth': [3, 5, 7],
    'class_weight': ['balanced', None]
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=10), param_grid, cv=10)

# Find best parameters for ADHD model
grid_search.fit(X_preprocessed, y_adhd)
print("Best parameters for ADHD model:", grid_search.best_params_)
print("Best score for ADHD model on training dataset:", grid_search.best_score_)

cv_results = grid_search.cv_results_
df_results = pd.DataFrame(cv_results)
print(df_results[['param_max_depth','mean_test_score']])

clf_adhd = grid_search.best_estimator_

# Find best parameters for Sex model
grid_search.fit(X_preprocessed, y_sex)
print("\nBest parameters for Sex model:", grid_search.best_params_)
print("Best score for Sex model on training dataset:", grid_search.best_score_)

cv_results = grid_search.cv_results_
df_results = pd.DataFrame(cv_results)
print(df_results[['param_max_depth','mean_test_score']])

clf_sex = grid_search.best_estimator_


# Apply stratified k fold cross validation using K folds, list to store accuracy scores
skf = StratifiedKFold(n_splits=100, shuffle=True, random_state=80)
list_accu_stratified = []

# Stratify by adhd
for train_index, test_index in skf.split(X, y_adhd):
    X_train_fold, X_test_fold = X_preprocessed.iloc[train_index], X_preprocessed.iloc[test_index] # row indeces
    y_train_adhd_fold, y_test_adhd_fold = y_adhd[train_index], y_adhd[test_index]
    y_train_sex_fold, y_test_sex_fold = y_sex[train_index], y_sex[test_index]
    clf_adhd.fit(X_train_fold, y_train_adhd_fold)
    clf_sex.fit(X_train_fold, y_train_sex_fold)
    adhd_pred = clf_adhd.predict(X_test_fold)
    sex_pred = clf_sex.predict(X_test_fold)
    list_accu_stratified.append(weighted_f1_score(y_test_adhd_fold, adhd_pred, y_test_sex_fold, sex_pred))

# print('List of possible accuracies:', list_accu_stratified)
print("-Stratified by ADHD-")
print('Maximum F1 Score:',
	max(list_accu_stratified))
print('Minimum F1 Score:',
	min(list_accu_stratified))
print('Overall F1 Score:',
	mean(list_accu_stratified))
print('Standard Deviation:', stdev(list_accu_stratified))

list_accu_stratified = []

# Stratify by sex
for train_index, test_index in skf.split(X, y_sex):
    X_train_fold, X_test_fold = X_preprocessed.iloc[train_index], X_preprocessed.iloc[test_index] # row indeces
    y_train_adhd_fold, y_test_adhd_fold = y_adhd[train_index], y_adhd[test_index]
    y_train_sex_fold, y_test_sex_fold = y_sex[train_index], y_sex[test_index]
    clf_adhd.fit(X_train_fold, y_train_adhd_fold)
    clf_sex.fit(X_train_fold, y_train_sex_fold)
    adhd_pred = clf_adhd.predict(X_test_fold)
    sex_pred = clf_sex.predict(X_test_fold)
    list_accu_stratified.append(weighted_f1_score(y_test_adhd_fold, adhd_pred, y_test_sex_fold, sex_pred))

print("\n-Stratified by Sex-")
print('Maximum F1 Score:',
	max(list_accu_stratified))
print('Minimum F1 Score:',
	min(list_accu_stratified))
print('Overall F1 Score:',
	mean(list_accu_stratified))
print('Standard Deviation:', stdev(list_accu_stratified))


# Optimize best parameter values for random forest
param_grid = {
    'n_estimators': [10, 50, 100], # 150 and 200 produced same or worse results
    'max_depth': [7, 10, 13], # out of [3, 5, 7], 7 was the best-performing for both adhd and sex models (random_state=42)
    'class_weight': ['balanced', None]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=10), param_grid, cv=10)

# Find best parameters for ADHD model
grid_search.fit(X_preprocessed, y_adhd)
print("Best parameters for ADHD model:", grid_search.best_params_)
print("Best score for ADHD model on training dataset:", grid_search.best_score_)

cv_results = grid_search.cv_results_
df_results = pd.DataFrame(cv_results)
print(df_results[['param_max_depth','mean_test_score']])

rf_adhd = grid_search.best_estimator_

# Find best parameters for Sex model
grid_search.fit(X_preprocessed, y_sex)
print("\nBest parameters for Sex model:", grid_search.best_params_)
print("Best score for Sex model on training dataset:", grid_search.best_score_)

cv_results = grid_search.cv_results_
df_results = pd.DataFrame(cv_results)
print(df_results[['param_max_depth','mean_test_score']])

rf_sex = grid_search.best_estimator_


# Apply stratified k fold cross validation using K folds, list to store accuracy scores
skf = StratifiedKFold(n_splits=100, shuffle=True, random_state=80)
list_accu_stratified = []

# Stratify by adhd
for train_index, test_index in skf.split(X, y_adhd):
    X_train_fold, X_test_fold = X_preprocessed.iloc[train_index], X_preprocessed.iloc[test_index] # row indeces
    y_train_adhd_fold, y_test_adhd_fold = y_adhd[train_index], y_adhd[test_index]
    y_train_sex_fold, y_test_sex_fold = y_sex[train_index], y_sex[test_index]
    rf_adhd.fit(X_train_fold, y_train_adhd_fold)
    rf_sex.fit(X_train_fold, y_train_sex_fold)
    adhd_pred = rf_adhd.predict(X_test_fold)
    sex_pred = rf_sex.predict(X_test_fold)
    list_accu_stratified.append(weighted_f1_score(y_test_adhd_fold, adhd_pred, y_test_sex_fold, sex_pred))

# print('List of possible accuracies:', list_accu_stratified)
print("-Stratified by ADHD-")
print('Maximum F1 Score:',
	max(list_accu_stratified))
print('Minimum F1 Score:',
	min(list_accu_stratified))
print('Overall F1 Score:',
	mean(list_accu_stratified))
print('Standard Deviation:', stdev(list_accu_stratified))

list_accu_stratified = []

# Stratify by sex
for train_index, test_index in skf.split(X, y_sex):
    X_train_fold, X_test_fold = X_preprocessed.iloc[train_index], X_preprocessed.iloc[test_index] # row indeces
    y_train_adhd_fold, y_test_adhd_fold = y_adhd[train_index], y_adhd[test_index]
    y_train_sex_fold, y_test_sex_fold = y_sex[train_index], y_sex[test_index]
    rf_adhd.fit(X_train_fold, y_train_adhd_fold)
    rf_sex.fit(X_train_fold, y_train_sex_fold)
    adhd_pred = rf_adhd.predict(X_test_fold)
    sex_pred = rf_sex.predict(X_test_fold)
    list_accu_stratified.append(weighted_f1_score(y_test_adhd_fold, adhd_pred, y_test_sex_fold, sex_pred))

print("\n-Stratified by Sex-")
print('Maximum F1 Score:',
	max(list_accu_stratified))
print('Minimum F1 Score:',
	min(list_accu_stratified))
print('Overall F1 Score:',
	mean(list_accu_stratified))
print('Standard Deviation:', stdev(list_accu_stratified))

