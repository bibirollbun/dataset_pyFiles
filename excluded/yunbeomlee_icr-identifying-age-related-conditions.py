from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
pd.set_option("display.precision", 8)
import pickle


path_to_train = '/kaggle/input/icr-identify-age-related-conditions/'
path_to_test = '/kaggle/input/icr-identify-age-related-conditions/'
path_to_models = '/kaggle/working/'


train, greeks = pd.read_csv(f'{path_to_train}train.csv'), pd.read_csv(f'{path_to_train}greeks.csv')


train.info()


train.describe()


missing = train.isnull().sum()
missing[missing > 0].sort_values(ascending = False)


target_counts = train['Class'].value_counts(normalize=True) * 100
target_counts


import seaborn as sns
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data = train)
plt.show()


train['EJ'] = train['EJ'].map({'A': 0, 'B': 1})


# only numeric (id is not needed)
numeric_df = train.select_dtypes(include=['number'])

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(24,20))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".1f")
plt.title("Correlation Matrix")
plt.show()


train.hist(figsize=(20, 15), bins=30)
plt.suptitle('Feature Distributions')
plt.show()


# missing values are filled recursively using CatBoost Regressor with default hyperparameters
cbr = CatBoostRegressor()

cols_to_fill = train.head(0).drop(['Id', 'Class'], axis = 1).columns

fillna_regressors = {}

for target_col in cols_to_fill:
    null_col_train = train[target_col].isnull()
    except_col = [x for x in cols_to_fill if x != target_col]
    cbr.fit(train.loc[~null_col_train, except_col], train.loc[~null_col_train, target_col], verbose = False)
    fillna_regressors[target_col] = cbr.copy()
    
    if len(train[null_col_train]) > 0:
        print(f'{target_col} has nulls...filled')
        train.loc[null_col_train, target_col] = cbr.predict(train.loc[null_col_train, except_col])
        
with open(f'{path_to_models}fillna_regressors.pkl', 'wb') as handle:
    pickle.dump(fillna_regressors, handle, protocol=pickle.HIGHEST_PROTOCOL)


greeks['Epsilon'] = greeks['Epsilon'].map(lambda x: x.replace("/", ".") if isinstance(x,str) else x)
greeks['Epsilon'] = greeks['Epsilon'].replace('Unknown', np.nan)
greeks['Epsilon'] = pd.to_datetime(greeks['Epsilon'])
greeks['Epsilon'] = greeks['Epsilon'].fillna(greeks['Epsilon'].min())


train_greeks = train.set_index('Id').join(greeks.set_index('Id'))
train_greeks['Alpha'] = train_greeks['Alpha'].map({'A': 0, 'B': 1, 'G': 2, 'D' : 3})
train_greeks['Beta'] = train_greeks['Beta'].map({'A': 0, 'B': 1, 'C': 2})
train_greeks['Gamma'] = train_greeks['Gamma'].map({'A': 0, 'B': 1, 'F': 2, 'G' : 3, 'H': 4, 'E': 5, 'M': 6, 'N': 7})
train_greeks['Delta'] = train_greeks['Delta'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3})
train_greeks = train_greeks.sort_values('Epsilon').reset_index()
train_greeks = train_greeks.reset_index().rename(columns = {'index' : 'row_id'})
train_greeks = train_greeks.set_index('Id')
train_greeks = train_greeks.drop(columns = 'Epsilon')
train_greeks = train_greeks.reset_index(drop=True)


with open(f'{path_to_models}row_id_max.pkl', 'wb') as handle:
    pickle.dump(train_greeks['row_id'].max(), handle, protocol=pickle.HIGHEST_PROTOCOL)


#Alpha, Beta, Gamma, Delta - probability, generated using supplementary models (CatBoost multiclassification)
abgd_classifiers = {}
cv = KFold(n_splits = 5, random_state = 13, shuffle=True)
tmp = train_greeks.drop(['Class', 'Alpha', 'Beta', 'Gamma', 'Delta'], axis = 1)
for col in ['Alpha', 'Beta', 'Gamma', 'Delta']:
    print(col)
    clf_list = []
    for train_index, test_index in cv.split(train_greeks):
        clf = CatBoostClassifier(random_state=42,
                                 iterations=10000,
                                 verbose=100,
                                 learning_rate=0.05,
                                 early_stopping_rounds=500)
        X_train, X_test = tmp.loc[train_index], tmp.loc[test_index]
        y_train, y_test = train_greeks[col].loc[train_index], train_greeks[col].loc[test_index]
        clf.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
        clf_list.append(clf)
        target_cols = [f'{col}_{x}' for x in clf.classes_]
        train_greeks.loc[test_index, target_cols] = clf.predict_proba(tmp.loc[test_index])
    abgd_classifiers[col] = clf_list

with open(f'{path_to_models}abgd_classifiers.pkl', 'wb') as handle:
    pickle.dump(abgd_classifiers, handle, protocol=pickle.HIGHEST_PROTOCOL)



model = CatBoostClassifier(iterations=10000,
                           learning_rate = 0.0005,
                           verbose = 1000)

cols_to_drop = ['Class', 'Alpha', 'Beta','Gamma', 'Delta']


X_full = train_greeks.drop(cols_to_drop, axis=1)
y_full = train_greeks['Class']

class_weight = np.bincount(y_full)/len(y_full)
full_weight = np.zeros_like(y_full, dtype=float)
full_weight[y_full == 0] = [class_weight[1]]
full_weight[y_full == 1] = class_weight[0]

full_pool = Pool(X_full, y_full, weight=full_weight)
model.fit(full_pool)


with open(f'{path_to_models}model.pkl', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


# prediction on test
test = pd.read_csv(f'{path_to_test}test.csv')
test.head()


test['EJ'] = test['EJ'].map({'A': 0, 'B': 1})
test = test.drop(columns = 'Id')


with open(f'{path_to_models}fillna_regressors.pkl', 'rb') as handle:
    fillna_regressors = pickle.load(handle)


null_cols_test = test.columns[test.isnull().any()]
for target_col in null_cols_test:
    print(f'{target_col} has nulls...filled')
    except_col = [x for x in test.columns if target_col != x]
    cbr = fillna_regressors[target_col]
    predicted = cbr.predict(test.loc[test[target_col].isnull(), except_col])
    if target_col == 'EJ':
        predicted = np.round(predicted)
    test.loc[test[target_col].isnull(), target_col] = predicted 


test = test.reset_index().rename(columns={'index' :'row_id'})

with open(f'{path_to_models}row_id_max.pkl', 'rb') as handle:
    row_id_max = pickle.load(handle)
    
test['row_id'] += row_id_max


with open(f'{path_to_models}abgd_classifiers.pkl', 'rb') as handle:
    abgd_classifiers = pickle.load(handle)


for col in ['Alpha', 'Beta', 'Gamma', 'Delta']:
    test_tmp = []
    print(col)
    clf_list = abgd_classifiers[col]
    target_cols = [f'{col}_{x}' for x in clf_list[0].classes_]
    for clf in clf_list:
        expected_features = clf.feature_names_
        test_aligned = test.copy()

        for col_name in expected_features:
            if col_name not in test_aligned.columns:
                test_aligned[col_name] = 0

        test_aligned = test_aligned[expected_features]

        test_tmp.append(clf.predict_proba(test_aligned))
    test[target_cols] = np.stack(test_tmp).mean(axis=0)


with open(f'{path_to_models}model.pkl', 'rb') as handle:
    model = pickle.load(handle)
prediction = model.predict_proba(test)
prediction



sample_submission = pd.read_csv(f'{path_to_test}sample_submission.csv')
sample_submission[['class_0', 'class_1']] = prediction
sample_submission




