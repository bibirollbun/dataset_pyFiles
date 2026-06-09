!pip install -r /kaggle/input/environment-new/requirements_new.txt --upgrade --no-deps


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score
    )

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    VotingClassifier
    )
from imblearn.ensemble import EasyEnsembleClassifier, RUSBoostClassifier, BalancedRandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)


import os
os.environ['PYTHONHASHSEED'] = '0'


df_train_cat = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx')
df_train_num = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx')
df_train_target = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx')
df_matrice = pd.read_csv('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv')


# join the dataframes using participant_id as the key
df = pd.merge(df_train_cat, df_train_num, on='participant_id')
df = pd.merge(df, df_train_target, on='participant_id')
df = pd.merge(df, df_matrice, on='participant_id')
df


df.info()


df_missing = pd.DataFrame(df.isnull().sum(), columns = ['No. of missing values'])
df_missing['% of missing values'] = df_missing['No. of missing values'] / df.shape[0] * 100
df_missing.sort_values(by='No. of missing values', ascending=False)


target1 = 'ADHD_Outcome'
target2 = 'Sex_F'
features = df.columns.drop([target1, target2])


sns.countplot(data=df, x=target2, hue=target1, palette='husl')
plt.show()


cat_cols = df_train_cat.columns.drop(['participant_id'])
cat_cols


num_cols = df_train_num.columns.drop('participant_id')
num_cols


matrice_cols = df_matrice.columns.drop('participant_id')
matrice_cols


print('{} numerical columns in the dataset:\n'.format(len(num_cols)), ', '.join(num_cols))
df[num_cols].describe()


fig, axes = plt.subplots(5, 4, figsize=(16, 20))

for i, col in enumerate(num_cols):
    ax = axes.ravel()[i]
    sns.histplot(data=df, x=col, label=col, color='teal', ax=ax)

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()


df_cat = df[cat_cols].copy()
df_cat[target1] = df[target1]

sns.pairplot(data=df_cat, hue=target1, palette='husl')
plt.show()


num_cat_cols = list(num_cols) + list(cat_cols)
df_num_cat = df[num_cat_cols].copy()
df_num_cat[target1] = df[target1]
df_num_cat[target2] = df[target2]

plt.figure(figsize=(18, 6))
corr = df_num_cat.corr(numeric_only=True)
sns.heatmap(data=corr, annot=True, cmap='viridis')
plt.show()


corr[target1].abs().sort_values(ascending=False)


fea_important_1 = corr[target1].abs().sort_values(ascending=False).index[:7].drop([target1])
fea_important_1


corr[target2].abs().sort_values(ascending=False)


fea_important_2 = corr[target2].abs().sort_values(ascending=False).index[:7].drop([target2])
fea_important_2


target = [target1, target2]
selected_features = list(cat_cols) + list(num_cols)

X = df[selected_features].copy()
y = df[target].copy()


imputer1 = SimpleImputer(missing_values=np.nan, strategy='mean')
X[num_cols] = pd.DataFrame(imputer1.fit_transform(X[num_cols]), columns=num_cols)


imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X[cat_cols] = pd.DataFrame(imputer2.fit_transform(X[cat_cols]), columns=cat_cols)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


adhd_and_female = (y_test['ADHD_Outcome'] == 1) & (y_test['Sex_F'] == 1)
# apply weights 2x to female ADHD cases
weights = np.ones(len(y_test))
weights[adhd_and_female] = 2


selected_target = target1
y_train = y_train[selected_target]
y_test = y_test[selected_target]


# check skewness
skewness = X.skew()
high_skew_threshold = 2
num_skew_cols = skewness[abs(skewness) > high_skew_threshold].index.tolist()
num_noskew_cols = skewness[abs(skewness) <= high_skew_threshold].index.tolist()

skew_cols = num_skew_cols
noskew_cols = num_noskew_cols

print(len(num_skew_cols), len(num_noskew_cols))


scaler_skew = PowerTransformer(method='yeo-johnson', standardize=True)
scaler_skew.fit(X_train[skew_cols])
X_train[skew_cols] = scaler_skew.transform(X_train[skew_cols])
X_test[skew_cols] = scaler_skew.transform(X_test[skew_cols])

scaler_noskew = StandardScaler()
scaler_noskew.fit(X_train[noskew_cols])
X_train[noskew_cols] = scaler_noskew.transform(X_train[noskew_cols])
X_test[noskew_cols] = scaler_noskew.transform(X_test[noskew_cols])

X_train.head()


models = []

models.append(('Bagging', BaggingClassifier(random_state=1)))
models.append(('RandomForest', RandomForestClassifier(random_state=1)))
models.append(('GBC', GradientBoostingClassifier(random_state=1)))
models.append(('Adaboost', AdaBoostClassifier(algorithm="SAMME", random_state=1)))
models.append(('Xgboost', XGBClassifier(seed=1, eval_metric="logloss")))
models.append(('CatBoost', CatBoostClassifier(random_state=1, eval_metric="Logloss", verbose=0)))
models.append(('EasyEnsemble', EasyEnsembleClassifier(random_state=1)))
models.append(('RUSBoost', RUSBoostClassifier(random_state=1)))

results = []
names = []

print('Cross-Validation Performance:' '\n')

for name, model in models:
    scoring = 'f1'
    # scoring = 'roc_auc'
    kfold = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=1
    )
    cv_result = cross_val_score(
        estimator=model, X=X_train, y=y_train, scoring=scoring, cv=kfold
    )
    results.append(cv_result)
    names.append(name)
    print('{}: {}'.format(name, cv_result.mean() * 100))

print('\n' 'Training Performance:' '\n')

for name, model in models:
    model.fit(X_train, y_train)
    scores = f1_score(y_train, model.predict(X_train))
    print("{}: {}".format(name, scores * 100))


# Plotting boxplots for CV scores of all models defined above
fig = plt.figure(figsize=(18, 6))

sns.boxplot(data=results, palette="husl")
plt.xlabel("Models")
plt.xticks(range(len(names)), names)
plt.ylabel("CV scores")

plt.show()


# Initialize a dataframe to store model's performance
models_performance_adhd = pd.DataFrame()
best_models_adhd = {}


# Hyperparameter Tunning for BaggingClassifier using GridSearchCV

model = BaggingClassifier(random_state=1)

param_grid = {
    'n_estimators': np.arange(10, 50, 10),
    'max_samples': np.arange(0.5, 1, 0.1),
    'max_features': np.arange(0.5, 1, 0.1),
    'bootstrap_features': [True, False]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_bagging = grid_search.best_estimator_
y_pred = best_bagging.predict(X_test)
y_pred_prob = best_bagging.predict_proba(X_test)[:, 1]

print('Best Parameters:')
print(grid_search.best_params_)


# check with thresholds

thres = np.linspace(0.1, 0.9, 9)
scores = []
y_preds = []
y_pred_probs = []
for i in thres:
    y_pred_prob = best_bagging.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > i).astype(int)
    score = f1_score(y_test, y_pred, sample_weight=weights)
    scores.append(score)
    y_preds.append(y_pred)
    y_pred_probs.append(y_pred_prob)
best_threshold = thres[np.argmax(scores)]
y_pred = y_preds[np.argmax(scores)]
y_pred_prob = y_pred_probs[np.argmax(scores)]
print(best_threshold, max(scores))

plt.plot(thres, scores)

best_models_adhd['Bagging'] = (best_bagging, best_threshold, max(scores))


# Classification Report
report = pd.DataFrame({'roc_auc': roc_auc_score(y_test, y_pred_prob),
                       'recall': recall_score(y_test, y_pred),
                       'precision': precision_score(y_test, y_pred),
                       'f1_score': f1_score(y_test, y_pred, sample_weight=weights),
                       'accuracy': accuracy_score(y_test, y_pred)},
                       index=['Bagging']).T
print(report)

# Confusion Matrix
cfs_matrix = confusion_matrix(y_test, y_pred)
cfs_matrix = pd.DataFrame(cfs_matrix, index=['Actual No', 'Actual ADHD'], columns=['Predicted No', 'Predicted ADHD'])
sns.heatmap(cfs_matrix, annot=True, fmt='g', cbar=False)
plt.show()

# Save model's performance to dictionary
models_performance_adhd['Bagging'] = report


# Hyperparameter Tunning for Random Forest using GridSearchCV

model = RandomForestClassifier(random_state=1, class_weight='balanced')

param_grid = {
    'n_estimators': np.arange(50, 200, 50),
    'criterion': ['gini', 'entropy'],
    'max_depth': range(3, 5),
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': range(1, 3)
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
y_pred_prob = best_rf.predict_proba(X_test)[:, 1]

print('Best Parameters:')
print(grid_search.best_params_)


# check with thresholds

thres = np.linspace(0.1, 0.9, 9)
scores = []
y_preds = []
y_pred_probs = []
for i in thres:
    y_pred_prob = best_rf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > i).astype(int)
    score = f1_score(y_test, y_pred, sample_weight=weights)
    scores.append(score)
    y_preds.append(y_pred)
    y_pred_probs.append(y_pred_prob)
best_threshold = thres[np.argmax(scores)]
y_pred = y_preds[np.argmax(scores)]
y_pred_prob = y_pred_probs[np.argmax(scores)]
print(best_threshold, max(scores))

plt.plot(thres, scores)

best_models_adhd['RandomForest'] = (best_rf, best_threshold, max(scores))


# Classification Report
report = pd.DataFrame({'roc_auc': roc_auc_score(y_test, y_pred_prob),
                       'recall': recall_score(y_test, y_pred),
                       'precision': precision_score(y_test, y_pred),
                       'f1_score': f1_score(y_test, y_pred, sample_weight=weights),
                       'accuracy': accuracy_score(y_test, y_pred)},
                       index=['RandomForest']).T
print(report)

# Confusion Matrix
cfs_matrix = confusion_matrix(y_test, y_pred)
cfs_matrix = pd.DataFrame(cfs_matrix, index=['Actual No', 'Actual ADHD'], columns=['Predicted No', 'Predicted ADHD'])
sns.heatmap(cfs_matrix, annot=True, fmt='g', cbar=False)
plt.show()

# Add model's performance to dataframe 
models_performance_adhd['RandomForest'] = report


# Hyperparameter Tunning for GBC using GridSearchCV

model = GradientBoostingClassifier(random_state=1)

param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': np.arange(0.05, 0.25, 0.05),
    'max_depth': [3, 4],
    'subsample': [0.8],  # fixed
    'max_features': [0.8]  # fixed as well
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_GBC = grid_search.best_estimator_
y_pred = best_GBC.predict(X_test)
y_pred_prob = best_GBC.predict_proba(X_test)[:, 1]

print('Best Parameters:')
print(grid_search.best_params_)


# check with thresholds

thres = np.linspace(0.1, 0.9, 9)
scores = []
y_preds = []
y_pred_probs = []
for i in thres:
    y_pred_prob = best_GBC.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > i).astype(int)
    score = f1_score(y_test, y_pred, sample_weight=weights)
    scores.append(score)
    y_preds.append(y_pred)
    y_pred_probs.append(y_pred_prob)
best_threshold = thres[np.argmax(scores)]
y_pred = y_preds[np.argmax(scores)]
y_pred_prob = y_pred_probs[np.argmax(scores)]
print(best_threshold, max(scores))

plt.plot(thres, scores)

best_models_adhd['GBC'] = (best_GBC, best_threshold, max(scores))


# Classification Report
report = pd.DataFrame({'roc_auc': roc_auc_score(y_test, y_pred_prob),
                       'recall': recall_score(y_test, y_pred),
                       'precision': precision_score(y_test, y_pred),
                       'f1_score': f1_score(y_test, y_pred, sample_weight=weights),
                       'accuracy': accuracy_score(y_test, y_pred)},
                       index=['GBC']).T
print(report)

# Confusion Matrix
cfs_matrix = confusion_matrix(y_test, y_pred)
cfs_matrix = pd.DataFrame(cfs_matrix, index=['Actual No', 'Actual ADHD'], columns=['Predicted No', 'Predicted ADHD'])
sns.heatmap(cfs_matrix, annot=True, fmt='g', cbar=False)
plt.show()

# Add model's performance to dataframe
models_performance_adhd['GBC'] = report


# Hyperparameter Tunning for AdaBoost using GridSearchCV

model = AdaBoostClassifier(algorithm='SAMME', random_state=1)

param_grid = {
    'n_estimators': np.arange(10, 110, 10),
    'learning_rate': np.arange(0.05, 0.3, 0.05)
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_Adaboost = grid_search.best_estimator_
y_pred = best_Adaboost.predict(X_test)
y_pred_prob = best_Adaboost.predict_proba(X_test)[:, 1]

print('Best Parameters:')
print(grid_search.best_params_)


# check with thresholds

thres = np.linspace(0.1, 0.9, 9)
scores = []
y_preds = []
y_pred_probs = []
for i in thres:
    y_pred_prob = best_Adaboost.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > i).astype(int)
    score = f1_score(y_test, y_pred, sample_weight=weights)
    scores.append(score)
    y_preds.append(y_pred)
    y_pred_probs.append(y_pred_prob)
best_threshold = thres[np.argmax(scores)]
y_pred = y_preds[np.argmax(scores)]
y_pred_prob = y_pred_probs[np.argmax(scores)]
print(best_threshold, max(scores))

plt.plot(thres, scores)

best_models_adhd['AdaBoost'] = (best_Adaboost, best_threshold, max(scores))


# Classification Report
report = pd.DataFrame({'roc_auc': roc_auc_score(y_test, y_pred_prob),
                       'recall': recall_score(y_test, y_pred),
                       'precision': precision_score(y_test, y_pred),
                       'f1_score': f1_score(y_test, y_pred, sample_weight=weights),
                       'accuracy': accuracy_score(y_test, y_pred)},
                       index=['AdaBoost']).T
print(report)

# Confusion Matrix 
cfs_matrix = confusion_matrix(y_test, y_pred)
cfs_matrix = pd.DataFrame(cfs_matrix, index=['Actual No', 'Actual ADHD'], columns=['Predicted No', 'Predicted ADHD'])
sns.heatmap(cfs_matrix, annot=True, fmt='g', cbar=False)
plt.show()

# Add model's performance to dataframe
models_performance_adhd['AdaBoost'] = report


# Hyperparameter Tunning for Xgboost using GridSearchCV

model = XGBClassifier(seed=1, eval_metric="logloss")

param_grid = {
    'n_estimators': np.arange(50, 150, 50),
    'learning_rate': np.arange(0.05, 0.2, 0.05),
    'max_depth': [3, 4],
    'subsample': [0.8],  # fixed to a good default
    'gamma': range(4),
    'scale_pos_weight': range(1, 3)
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_Xgboost = grid_search.best_estimator_
y_pred = best_Xgboost.predict(X_test)
y_pred_prob = best_Xgboost.predict_proba(X_test)[:, 1]

print('Best Parameters:')
print(grid_search.best_params_)


# check with thresholds

thres = np.linspace(0.1, 0.9, 9)
scores = []
y_preds = []
y_pred_probs = []
for i in thres:
    y_pred_prob = best_Xgboost.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > i).astype(int)
    score = f1_score(y_test, y_pred, sample_weight=weights)
    scores.append(score)
    y_preds.append(y_pred)
    y_pred_probs.append(y_pred_prob)
best_threshold = thres[np.argmax(scores)]
y_pred = y_preds[np.argmax(scores)]
y_pred_prob = y_pred_probs[np.argmax(scores)]
print(best_threshold, max(scores))

plt.plot(thres, scores)

best_models_adhd['Xgboost'] = (best_Xgboost, best_threshold, max(scores))


# Classification Report
report = pd.DataFrame({'roc_auc': roc_auc_score(y_test, y_pred_prob),
                       'recall': recall_score(y_test, y_pred),
                       'precision': precision_score(y_test, y_pred),
                       'f1_score': f1_score(y_test, y_pred, sample_weight=weights),
                       'accuracy': accuracy_score(y_test, y_pred)},
                       index=['Xgboost']).T
print(report)

# Confusion Matrix
cfs_matrix = confusion_matrix(y_test, y_pred)
cfs_matrix = pd.DataFrame(cfs_matrix, index=['Actual No', 'Actual ADHD'], columns=['Predicted No', 'Predicted ADHD'])
sns.heatmap(cfs_matrix, annot=True, fmt='g', cbar=False)
plt.show()

# Save model's performance to dictionary
models_performance_adhd['Xgboost'] = report


# Hyperparameter Tunning for Catboost using GridSearchCV

model = CatBoostClassifier(random_state=1, eval_metric="Logloss", verbose=0)

param_grid = {'iterations': np.arange(50, 250, 50),
              'depth': range(2, 6),
              'learning_rate': np.arange(0.05, 0.3, 0.05),
              'l2_leaf_reg': range(1, 4)
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

best_catboost = grid_search.best_estimator_
y_pred = best_catboost.predict(X_test)
y_pred_prob = best_catboost.predict_proba(X_test)[:, 1]

print('Best Parameters:')
print(grid_search.best_params_)


# check with thresholds

thres = np.linspace(0.1, 0.9, 9)
scores = []
y_preds = []
y_pred_probs = []
for i in thres:
    y_pred_prob = best_catboost.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > i).astype(int)
    score = f1_score(y_test, y_pred, sample_weight=weights)
    scores.append(score)
    y_preds.append(y_pred)
    y_pred_probs.append(y_pred_prob)
best_threshold = thres[np.argmax(scores)]
y_pred = y_preds[np.argmax(scores)]
y_pred_prob = y_pred_probs[np.argmax(scores)]
print(best_threshold, max(scores))

plt.plot(thres, scores)

best_models_adhd['CatBoost'] = (best_catboost, best_threshold, max(scores))


# Classification Report
report = pd.DataFrame({'roc_auc': roc_auc_score(y_test, y_pred_prob),
                       'recall': recall_score(y_test, y_pred),
                       'precision': precision_score(y_test, y_pred),
                       'f1_score': f1_score(y_test, y_pred, sample_weight=weights),
                       'accuracy': accuracy_score(y_test, y_pred)},
                       index=['CatBoost']).T
print(report)

# Confusion Matrix
cfs_matrix = confusion_matrix(y_test, y_pred)
cfs_matrix = pd.DataFrame(cfs_matrix, index=['Actual No', 'Actual ADHD'], columns=['Predicted No', 'Predicted ADHD'])
sns.heatmap(cfs_matrix, annot=True, fmt='g', cbar=False)
plt.show()

# Save model's performance to dictionary
models_performance_adhd['CatBoost'] = report


# Hyperparameter Tunning for Easy Ensemble using GridSearchCV

model = EasyEnsembleClassifier(random_state=1)

param_grid = {
    'n_estimators': np.arange(10, 50, 10),
    'sampling_strategy': ['auto', 'not majority', 'not minority']
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_easy = grid_search.best_estimator_
y_pred = best_easy.predict(X_test)
y_pred_prob = best_easy.predict_proba(X_test)[:, 1]

print('Best Parameters:')
print(grid_search.best_params_)


# check with thresholds

thres = np.linspace(0.1, 0.9, 9)
scores = []
y_preds = []
y_pred_probs = []
for i in thres:
    y_pred_prob = best_easy.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > i).astype(int)
    score = f1_score(y_test, y_pred, sample_weight=weights)
    scores.append(score)
    y_preds.append(y_pred)
    y_pred_probs.append(y_pred_prob)
best_threshold = thres[np.argmax(scores)]
y_pred = y_preds[np.argmax(scores)]
y_pred_prob = y_pred_probs[np.argmax(scores)]
print(best_threshold, max(scores))

plt.plot(thres, scores)

best_models_adhd['EasyEnsemble'] = (best_easy, best_threshold, max(scores))


# Classification Report
report = pd.DataFrame({'roc_auc': roc_auc_score(y_test, y_pred_prob),
                       'recall': recall_score(y_test, y_pred),
                       'precision': precision_score(y_test, y_pred),
                       'f1_score': f1_score(y_test, y_pred, sample_weight=weights),
                       'accuracy': accuracy_score(y_test, y_pred)},
                       index=['EasyEnsemble']).T
print(report)

# Confusion Matrix 
cfs_matrix = confusion_matrix(y_test, y_pred)
cfs_matrix = pd.DataFrame(cfs_matrix, index=['Actual No', 'Actual ADHD'], columns=['Predicted No', 'Predicted ADHD'])
sns.heatmap(cfs_matrix, annot=True, fmt='g', cbar=False)
plt.show()

# Save model's performance to dictionary
models_performance_adhd['EasyEnsemble'] = report


# Hyperparameter Tunning for RUS Boost using GridSearchCV

model = RUSBoostClassifier(random_state=1)

param_grid = {
    'n_estimators': np.arange(50, 200, 50),
    'sampling_strategy': ['auto', 'not majority', 'not minority'],
    'learning_rate': np.arange(0.5, 1.1, 0.1)
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_rus = grid_search.best_estimator_
y_pred = best_rus.predict(X_test)
y_pred_prob = best_rus.predict_proba(X_test)[:, 1]

print('Best Parameters:')
print(grid_search.best_params_)


# check with thresholds

thres = np.linspace(0.1, 0.9, 9)
scores = []
y_preds = []
y_pred_probs = []
for i in thres:
    y_pred_prob = best_rus.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > i).astype(int)
    score = f1_score(y_test, y_pred, sample_weight=weights)
    scores.append(score)
    y_preds.append(y_pred)
    y_pred_probs.append(y_pred_prob)
best_threshold = thres[np.argmax(scores)]
y_pred = y_preds[np.argmax(scores)]
y_pred_prob = y_pred_probs[np.argmax(scores)]
print(best_threshold, max(scores))

plt.plot(thres, scores)

best_models_adhd['RUS'] = (best_rus, best_threshold, max(scores))


# Classification Report
report = pd.DataFrame({'roc_auc': roc_auc_score(y_test, y_pred_prob),
                       'recall': recall_score(y_test, y_pred),
                       'precision': precision_score(y_test, y_pred),
                       'f1_score': f1_score(y_test, y_pred, sample_weight=weights),
                       'accuracy': accuracy_score(y_test, y_pred)},
                       index=['RUS']).T
print(report)

# Confusion Matrix 
cfs_matrix = confusion_matrix(y_test, y_pred)
cfs_matrix = pd.DataFrame(cfs_matrix, index=['Actual No', 'Actual ADHD'], columns=['Predicted No', 'Predicted ADHD'])
sns.heatmap(cfs_matrix, annot=True, fmt='g', cbar=False)
plt.show()

# Save model's performance to dictionary
models_performance_adhd['RUS'] = report


models_performance_adhd.T.sort_values(by='f1_score', ascending=False)


df_best_models_adhd = pd.DataFrame.from_dict(best_models_adhd).T
df_best_models_adhd.columns = ['model', 'threshold', 'f1_score']
df_best_models_adhd = df_best_models_adhd.sort_values(by='f1_score', ascending=False)
df_best_models_adhd


# VotingClassifier

estimators = [
    ('Xgboost', best_Xgboost),
    ('CatBoost', best_catboost),
    ('EasyEnsemble', best_easy),
    ('Bagging', best_bagging),
]

model_adhd = VotingClassifier(estimators=estimators, voting='soft')
model_adhd.fit(X_train, y_train)
y_pred_prob = model_adhd.predict_proba(X_test)[:, 1]

threshold = np.linspace(0.1, 0.9, 9)
final_threshold = []
f1_scores = []
for i in threshold:
    y_pred = (y_pred_prob > i).astype(int)
    score = f1_score(y_test, y_pred, sample_weight=weights)
    final_threshold.append(score)
    f1_scores.append(score)

best_threshold_adhd = threshold[np.argmax(final_threshold)]
y_pred = (y_pred_prob > best_threshold_adhd).astype(int)
print(best_threshold_adhd, max(f1_scores))

# Classification Report
report = pd.DataFrame({'roc_auc': roc_auc_score(y_test, y_pred_prob),
                       'recall': recall_score(y_test, y_pred),
                       'precision': precision_score(y_test, y_pred),
                       'f1_score': f1_score(y_test, y_pred, sample_weight=weights),
                       'accuracy': accuracy_score(y_test, y_pred)},
                       index=['VotingClassifier']).T
print(report)

# Confusion Matrix 
cfs_matrix = confusion_matrix(y_test, y_pred)
cfs_matrix = pd.DataFrame(cfs_matrix, index=['Actual No', 'Actual ADHD'], columns=['Predicted No', 'Predicted ADHD'])
sns.heatmap(cfs_matrix, annot=True, fmt='g', cbar=False)
plt.show()


df_test_cat = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx')
df_test_num = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx')
df_test_matrice = pd.read_csv('/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv')


df_test = pd.merge(df_test_cat, df_test_num, on='participant_id')
df_test = pd.merge(df_test, df_test_matrice, on='participant_id')
df_test


# impute missing values

imputer1 = SimpleImputer(missing_values=np.nan, strategy='mean')
df_test[num_cols] = imputer1.fit_transform(df_test[num_cols])

imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df_test[cat_cols] = imputer2.fit_transform(df_test[cat_cols])

df_test.isnull().sum().sum()


X_test_final = df_test[selected_features].copy()


scaler_skew = PowerTransformer(method='yeo-johnson', standardize=True)
scaler_skew.fit(X_test_final[skew_cols])
X_test_final[skew_cols] = scaler_skew.transform(X_test_final[skew_cols])

scaler_noskew = StandardScaler()
scaler_noskew.fit(X_test_final[noskew_cols])
X_test_final[noskew_cols] = scaler_noskew.transform(X_test_final[noskew_cols])


# inference on test data
# final_thres_adhd = best_threshold_adhd
final_thres_adhd = 0.5

y_pred_prob = model_adhd.predict_proba(X_test_final)[:, 1]
y_pred = (y_pred_prob > final_thres_adhd).astype(int)

df_test[selected_target] = y_pred

df_test[selected_target].value_counts()


# save df_test to csv
df_test[['participant_id', selected_target]].to_csv('submission_{}.csv'.format(selected_target), index=False)


target = [target1, target2]
selected_features = list(cat_cols) + list(num_cols) + list(matrice_cols)

X = df[selected_features].copy()
y = df[target].copy()


imputer1 = SimpleImputer(missing_values=np.nan, strategy='mean')
X[num_cols] = pd.DataFrame(imputer1.fit_transform(X[num_cols]), columns=num_cols)


imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X[cat_cols] = pd.DataFrame(imputer2.fit_transform(X[cat_cols]), columns=cat_cols)


# %pip install neuroHarmonize
# %pip install nibabel
# %pip install neuroCombat


from neuroHarmonize import harmonizationLearn

mri_features = X[matrice_cols].values
batch = X['MRI_Track_Scan_Location'].values
covars = pd.DataFrame({'SITE': batch,
                        'age': X['MRI_Track_Age_at_Scan'].values})

model_mri, features_harmonized = harmonizationLearn(mri_features, covars, 'SITE')


X[matrice_cols] = features_harmonized
X


selected_target = target2


# Select top k features based on ANOVA F-value for classification
from sklearn.feature_selection import SelectKBest, f_classif

def select_features(X, y, k):
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_selected, selected_features

X_selected, selected_features = select_features(X[matrice_cols], y[selected_target], k=100)

selected_features = list(cat_cols) + list(num_cols) + list(selected_features)

X = pd.DataFrame(X[selected_features], columns=selected_features)
X


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


adhd_and_female = (y_test['ADHD_Outcome'] == 1) & (y_test['Sex_F'] == 1)
# apply weights 2x to female ADHD cases
weights = np.ones(len(y_test))
weights[adhd_and_female] = 2


y_train = y_train[selected_target]
y_test = y_test[selected_target]


# check skewness
skewness = X.skew()
high_skew_threshold = 2
num_skew_cols = skewness[abs(skewness) > high_skew_threshold].index.tolist()
num_noskew_cols = skewness[abs(skewness) <= high_skew_threshold].index.tolist()

skew_cols = num_skew_cols
noskew_cols = num_noskew_cols

print(len(num_skew_cols), len(num_noskew_cols))


scaler_skew = PowerTransformer(method='yeo-johnson', standardize=True)
scaler_skew.fit(X_train[skew_cols])
X_train[skew_cols] = scaler_skew.transform(X_train[skew_cols])
X_test[skew_cols] = scaler_skew.transform(X_test[skew_cols])

scaler_noskew = StandardScaler()
scaler_noskew.fit(X_train[noskew_cols])
X_train[noskew_cols] = scaler_noskew.transform(X_train[noskew_cols])
X_test[noskew_cols] = scaler_noskew.transform(X_test[noskew_cols])

X_train.head()


models = []

models.append(('Bagging', BaggingClassifier(random_state=1)))
models.append(('RandomForest', RandomForestClassifier(random_state=1)))
models.append(('GBC', GradientBoostingClassifier(random_state=1)))
models.append(('Adaboost', AdaBoostClassifier(algorithm="SAMME", random_state=1)))
models.append(('Xgboost', XGBClassifier(seed=1, eval_metric="logloss")))
models.append(('CatBoost', CatBoostClassifier(random_state=1, eval_metric="Logloss", verbose=0)))
models.append(('EasyEnsemble', EasyEnsembleClassifier(random_state=1)))
models.append(('RUSBoost', RUSBoostClassifier(random_state=1)))

results = []
names = []

print('Cross-Validation Performance:' '\n')

for name, model in models:
    scoring = 'f1'
    # scoring = 'roc_auc'
    kfold = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=1
    )
    cv_result = cross_val_score(
        estimator=model, X=X_train, y=y_train, scoring=scoring, cv=kfold
    )
    results.append(cv_result)
    names.append(name)
    print('{}: {}'.format(name, cv_result.mean() * 100))

print('\n' 'Training Performance:' '\n')

for name, model in models:
    model.fit(X_train, y_train)
    scores = f1_score(y_train, model.predict(X_train))
    print("{}: {}".format(name, scores * 100))


# Plotting boxplots for CV scores of all models defined above
fig = plt.figure(figsize=(18, 6))

sns.boxplot(data=results, palette="husl")
plt.xlabel("Models")
plt.xticks(range(len(names)), names)
plt.ylabel("CV scores")

plt.show()


# Initialize a dataframe to store model's performance
models_performance_gender = pd.DataFrame()
best_models_gender = {}


# Hyperparameter Tunning for BaggingClassifier using GridSearchCV

model = BaggingClassifier(random_state=1)

param_grid = {
    'n_estimators': np.arange(10, 50, 10),
    'max_samples': np.arange(0.5, 1, 0.1),
    'max_features': np.arange(0.5, 1, 0.1),
    'bootstrap_features': [True, False]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_bagging = grid_search.best_estimator_
y_pred = best_bagging.predict(X_test)
y_pred_prob = best_bagging.predict_proba(X_test)[:, 1]

print('Best Parameters:')
print(grid_search.best_params_)


# check with thresholds

thres = np.linspace(0.1, 0.9, 9)
scores = []
y_preds = []
y_pred_probs = []
for i in thres:
    y_pred_prob = best_bagging.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > i).astype(int)
    score = f1_score(y_test, y_pred, sample_weight=weights)
    scores.append(score)
    y_preds.append(y_pred)
    y_pred_probs.append(y_pred_prob)
best_threshold = thres[np.argmax(scores)]
y_pred = y_preds[np.argmax(scores)]
y_pred_prob = y_pred_probs[np.argmax(scores)]
print(best_threshold, max(scores))

plt.plot(thres, scores)

best_models_gender['Bagging'] = (best_bagging, best_threshold, max(scores))


# Classification Report
report = pd.DataFrame({'roc_auc': roc_auc_score(y_test, y_pred_prob),
                       'recall': recall_score(y_test, y_pred),
                       'precision': precision_score(y_test, y_pred),
                       'f1_score': f1_score(y_test, y_pred, sample_weight=weights),
                       'accuracy': accuracy_score(y_test, y_pred)},
                       index=['Bagging']).T
print(report)

# Confusion Matrix
cfs_matrix = confusion_matrix(y_test, y_pred)
cfs_matrix = pd.DataFrame(cfs_matrix, index=['Actual Male', 'Actual Female'], columns=['Predicted Male', 'Predicted Female'])
sns.heatmap(cfs_matrix, annot=True, fmt='g', cbar=False)
plt.show()

# Save model's performance to dictionary
models_performance_gender['Bagging'] = report


# Hyperparameter Tunning for Random Forest using GridSearchCV

model = RandomForestClassifier(random_state=1, class_weight='balanced')

param_grid = {
    'n_estimators': np.arange(50, 200, 50),
    'criterion': ['gini', 'entropy'],
    'max_depth': range(3, 5),
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': range(1, 3)
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
y_pred_prob = best_rf.predict_proba(X_test)[:, 1]

print('Best Parameters:')
print(grid_search.best_params_)


# check with thresholds

thres = np.linspace(0.1, 0.9, 9)
scores = []
y_preds = []
y_pred_probs = []
for i in thres:
    y_pred_prob = best_rf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > i).astype(int)
    score = f1_score(y_test, y_pred, sample_weight=weights)
    scores.append(score)
    y_preds.append(y_pred)
    y_pred_probs.append(y_pred_prob)
best_threshold = thres[np.argmax(scores)]
y_pred = y_preds[np.argmax(scores)]
y_pred_prob = y_pred_probs[np.argmax(scores)]
print(best_threshold, max(scores))

plt.plot(thres, scores)

best_models_gender['RandomForest'] = (best_rf, best_threshold, max(scores))


# Classification Report
report = pd.DataFrame({'roc_auc': roc_auc_score(y_test, y_pred_prob),
                       'recall': recall_score(y_test, y_pred),
                       'precision': precision_score(y_test, y_pred),
                       'f1_score': f1_score(y_test, y_pred, sample_weight=weights),
                       'accuracy': accuracy_score(y_test, y_pred)},
                       index=['RandomForest']).T
print(report)

# Confusion Matrix
cfs_matrix = confusion_matrix(y_test, y_pred)
cfs_matrix = pd.DataFrame(cfs_matrix, index=['Actual Male', 'Actual Female'], columns=['Predicted Male', 'Predicted Female'])
sns.heatmap(cfs_matrix, annot=True, fmt='g', cbar=False)
plt.show()

# Add model's performance to dataframe 
models_performance_gender['RandomForest'] = report


# Hyperparameter Tunning for GBC using GridSearchCV

model = GradientBoostingClassifier(random_state=1)

param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': np.arange(0.05, 0.25, 0.05),
    'max_depth': [3, 4],
    'subsample': [0.8],  # fixed
    'max_features': [0.8]  # fixed as well
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_GBC = grid_search.best_estimator_
y_pred = best_GBC.predict(X_test)
y_pred_prob = best_GBC.predict_proba(X_test)[:, 1]

print('Best Parameters:')
print(grid_search.best_params_)


# check with thresholds

thres = np.linspace(0.1, 0.9, 9)
scores = []
y_preds = []
y_pred_probs = []
for i in thres:
    y_pred_prob = best_GBC.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > i).astype(int)
    score = f1_score(y_test, y_pred, sample_weight=weights)
    scores.append(score)
    y_preds.append(y_pred)
    y_pred_probs.append(y_pred_prob)
best_threshold = thres[np.argmax(scores)]
y_pred = y_preds[np.argmax(scores)]
y_pred_prob = y_pred_probs[np.argmax(scores)]
print(best_threshold, max(scores))

plt.plot(thres, scores)

best_models_gender['GBC'] = (best_GBC, best_threshold, max(scores))


# Classification Report
report = pd.DataFrame({'roc_auc': roc_auc_score(y_test, y_pred_prob),
                       'recall': recall_score(y_test, y_pred),
                       'precision': precision_score(y_test, y_pred),
                       'f1_score': f1_score(y_test, y_pred, sample_weight=weights),
                       'accuracy': accuracy_score(y_test, y_pred)},
                       index=['GBC']).T
print(report)

# Confusion Matrix
cfs_matrix = confusion_matrix(y_test, y_pred)
cfs_matrix = pd.DataFrame(cfs_matrix, index=['Actual Male', 'Actual Female'], columns=['Predicted Male', 'Predicted Female'])
sns.heatmap(cfs_matrix, annot=True, fmt='g', cbar=False)
plt.show()

# Add model's performance to dataframe
models_performance_gender['GBC'] = report


# Hyperparameter Tunning for AdaBoost using GridSearchCV

model = AdaBoostClassifier(algorithm='SAMME', random_state=1)

param_grid = {
    'n_estimators': np.arange(10, 110, 10),
    'learning_rate': np.arange(0.05, 0.3, 0.05)
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_Adaboost = grid_search.best_estimator_
y_pred = best_Adaboost.predict(X_test)
y_pred_prob = best_Adaboost.predict_proba(X_test)[:, 1]

print('Best Parameters:')
print(grid_search.best_params_)


# check with thresholds

thres = np.linspace(0.1, 0.9, 9)
scores = []
y_preds = []
y_pred_probs = []
for i in thres:
    y_pred_prob = best_Adaboost.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > i).astype(int)
    score = f1_score(y_test, y_pred, sample_weight=weights)
    scores.append(score)
    y_preds.append(y_pred)
    y_pred_probs.append(y_pred_prob)
best_threshold = thres[np.argmax(scores)]
y_pred = y_preds[np.argmax(scores)]
y_pred_prob = y_pred_probs[np.argmax(scores)]
print(best_threshold, max(scores))

plt.plot(thres, scores)

best_models_gender['AdaBoost'] = (best_Adaboost, best_threshold, max(scores))


# Classification Report
report = pd.DataFrame({'roc_auc': roc_auc_score(y_test, y_pred_prob),
                       'recall': recall_score(y_test, y_pred),
                       'precision': precision_score(y_test, y_pred),
                       'f1_score': f1_score(y_test, y_pred, sample_weight=weights),
                       'accuracy': accuracy_score(y_test, y_pred)},
                       index=['AdaBoost']).T
print(report)

# Confusion Matrix 
cfs_matrix = confusion_matrix(y_test, y_pred)
cfs_matrix = pd.DataFrame(cfs_matrix, index=['Actual Male', 'Actual Female'], columns=['Predicted Male', 'Predicted Female'])
sns.heatmap(cfs_matrix, annot=True, fmt='g', cbar=False)
plt.show()

# Add model's performance to dataframe
models_performance_gender['AdaBoost'] = report


# Hyperparameter Tunning for Xgboost using GridSearchCV

model = XGBClassifier(seed=1, eval_metric="logloss")

param_grid = {
    'n_estimators': np.arange(50, 150, 50),
    'learning_rate': np.arange(0.05, 0.2, 0.05),
    'max_depth': [3, 4],
    'subsample': [0.8],  # fixed to a good default
    'gamma': range(4),
    'scale_pos_weight': range(1, 3)
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_Xgboost = grid_search.best_estimator_
y_pred = best_Xgboost.predict(X_test)
y_pred_prob = best_Xgboost.predict_proba(X_test)[:, 1]

print('Best Parameters:')
print(grid_search.best_params_)


# check with thresholds

thres = np.linspace(0.1, 0.9, 9)
scores = []
y_preds = []
y_pred_probs = []
for i in thres:
    y_pred_prob = best_Xgboost.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > i).astype(int)
    score = f1_score(y_test, y_pred, sample_weight=weights)
    scores.append(score)
    y_preds.append(y_pred)
    y_pred_probs.append(y_pred_prob)
best_threshold = thres[np.argmax(scores)]
y_pred = y_preds[np.argmax(scores)]
y_pred_prob = y_pred_probs[np.argmax(scores)]
print(best_threshold, max(scores))

plt.plot(thres, scores)

best_models_gender['Xgboost'] = (best_Xgboost, best_threshold, max(scores))


# Classification Report
report = pd.DataFrame({'roc_auc': roc_auc_score(y_test, y_pred_prob),
                       'recall': recall_score(y_test, y_pred),
                       'precision': precision_score(y_test, y_pred),
                       'f1_score': f1_score(y_test, y_pred, sample_weight=weights),
                       'accuracy': accuracy_score(y_test, y_pred)},
                       index=['Xgboost']).T
print(report)

# Confusion Matrix
cfs_matrix = confusion_matrix(y_test, y_pred)
cfs_matrix = pd.DataFrame(cfs_matrix, index=['Actual Male', 'Actual Female'], columns=['Predicted Male', 'Predicted Female'])
sns.heatmap(cfs_matrix, annot=True, fmt='g', cbar=False)
plt.show()

# Save model's performance to dictionary
models_performance_gender['Xgboost'] = report


# Hyperparameter Tunning for Catboost using GridSearchCV

model = CatBoostClassifier(random_state=1, eval_metric="Logloss", verbose=0)

param_grid = {'iterations': np.arange(50, 250, 50),
              'depth': range(2, 6),
              'learning_rate': np.arange(0.05, 0.3, 0.05),
              'l2_leaf_reg': range(1, 4)
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

best_catboost = grid_search.best_estimator_
y_pred = best_catboost.predict(X_test)
y_pred_prob = best_catboost.predict_proba(X_test)[:, 1]

print('Best Parameters:')
print(grid_search.best_params_)


# check with thresholds

thres = np.linspace(0.1, 0.9, 9)
scores = []
y_preds = []
y_pred_probs = []
for i in thres:
    y_pred_prob = best_catboost.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > i).astype(int)
    score = f1_score(y_test, y_pred, sample_weight=weights)
    scores.append(score)
    y_preds.append(y_pred)
    y_pred_probs.append(y_pred_prob)
best_threshold = thres[np.argmax(scores)]
y_pred = y_preds[np.argmax(scores)]
y_pred_prob = y_pred_probs[np.argmax(scores)]
print(best_threshold, max(scores))

plt.plot(thres, scores)

best_models_gender['CatBoost'] = (best_catboost, best_threshold, max(scores))


# Classification Report
report = pd.DataFrame({'roc_auc': roc_auc_score(y_test, y_pred_prob),
                       'recall': recall_score(y_test, y_pred),
                       'precision': precision_score(y_test, y_pred),
                       'f1_score': f1_score(y_test, y_pred, sample_weight=weights),
                       'accuracy': accuracy_score(y_test, y_pred)},
                       index=['CatBoost']).T
print(report)

# Confusion Matrix
cfs_matrix = confusion_matrix(y_test, y_pred)
cfs_matrix = pd.DataFrame(cfs_matrix, index=['Actual Male', 'Actual Female'], columns=['Predicted Male', 'Predicted Female'])
sns.heatmap(cfs_matrix, annot=True, fmt='g', cbar=False)
plt.show()

# Save model's performance to dictionary
models_performance_gender['CatBoost'] = report


# Hyperparameter Tunning for Easy Ensemble using GridSearchCV

model = EasyEnsembleClassifier(random_state=1)

param_grid = {
    'n_estimators': np.arange(10, 50, 10),
    'sampling_strategy': ['auto', 'not majority', 'not minority']
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_easy = grid_search.best_estimator_
y_pred = best_easy.predict(X_test)
y_pred_prob = best_easy.predict_proba(X_test)[:, 1]

print('Best Parameters:')
print(grid_search.best_params_)


# check with thresholds

thres = np.linspace(0.1, 0.9, 9)
scores = []
y_preds = []
y_pred_probs = []
for i in thres:
    y_pred_prob = best_easy.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > i).astype(int)
    score = f1_score(y_test, y_pred, sample_weight=weights)
    scores.append(score)
    y_preds.append(y_pred)
    y_pred_probs.append(y_pred_prob)
best_threshold = thres[np.argmax(scores)]
y_pred = y_preds[np.argmax(scores)]
y_pred_prob = y_pred_probs[np.argmax(scores)]
print(best_threshold, max(scores))

plt.plot(thres, scores)

best_models_gender['EasyEnsemble'] = (best_easy, best_threshold, max(scores))


# Classification Report
report = pd.DataFrame({'roc_auc': roc_auc_score(y_test, y_pred_prob),
                       'recall': recall_score(y_test, y_pred),
                       'precision': precision_score(y_test, y_pred),
                       'f1_score': f1_score(y_test, y_pred, sample_weight=weights),
                       'accuracy': accuracy_score(y_test, y_pred)},
                       index=['EasyEnsemble']).T
print(report)

# Confusion Matrix 
cfs_matrix = confusion_matrix(y_test, y_pred)
cfs_matrix = pd.DataFrame(cfs_matrix, index=['Actual Male', 'Actual Female'], columns=['Predicted Male', 'Predicted Female'])
sns.heatmap(cfs_matrix, annot=True, fmt='g', cbar=False)
plt.show()

# Save model's performance to dictionary
models_performance_gender['EasyEnsemble'] = report


# Hyperparameter Tunning for RUS Boost using GridSearchCV

model = RUSBoostClassifier(random_state=1)

param_grid = {
    'n_estimators': np.arange(50, 200, 50),
    'sampling_strategy': ['auto', 'not majority', 'not minority'],
    'learning_rate': np.arange(0.5, 1.1, 0.1)
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_rus = grid_search.best_estimator_
y_pred = best_rus.predict(X_test)
y_pred_prob = best_rus.predict_proba(X_test)[:, 1]

print('Best Parameters:')
print(grid_search.best_params_)


# check with thresholds

thres = np.linspace(0.1, 0.9, 9)
scores = []
y_preds = []
y_pred_probs = []
for i in thres:
    y_pred_prob = best_rus.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > i).astype(int)
    score = f1_score(y_test, y_pred, sample_weight=weights)
    scores.append(score)
    y_preds.append(y_pred)
    y_pred_probs.append(y_pred_prob)
best_threshold = thres[np.argmax(scores)]
y_pred = y_preds[np.argmax(scores)]
y_pred_prob = y_pred_probs[np.argmax(scores)]
print(best_threshold, max(scores))

plt.plot(thres, scores)

best_models_gender['RUS'] = (best_rus, best_threshold, max(scores))


# Classification Report
report = pd.DataFrame({'roc_auc': roc_auc_score(y_test, y_pred_prob),
                       'recall': recall_score(y_test, y_pred),
                       'precision': precision_score(y_test, y_pred),
                       'f1_score': f1_score(y_test, y_pred, sample_weight=weights),
                       'accuracy': accuracy_score(y_test, y_pred)},
                       index=['RUS']).T
print(report)

# Confusion Matrix 
cfs_matrix = confusion_matrix(y_test, y_pred)
cfs_matrix = pd.DataFrame(cfs_matrix, index=['Actual Male', 'Actual Female'], columns=['Predicted Male', 'Predicted Female'])
sns.heatmap(cfs_matrix, annot=True, fmt='g', cbar=False)
plt.show()

# Save model's performance to dictionary
models_performance_gender['RUS'] = report


models_performance_gender.T.sort_values(by='f1_score', ascending=False)


df_best_models_gender = pd.DataFrame.from_dict(best_models_gender).T
df_best_models_gender.columns = ['model', 'threshold', 'f1_score']
df_best_models_gender = df_best_models_gender.sort_values(by='f1_score', ascending=False)
df_best_models_gender


# VotingClassifier

estimators = [
    ('XGBoost', best_Xgboost),
    ('Bagging', best_bagging),
    ('RandomForest', best_rf),
    ('CatBoost', best_catboost),
    ('EasyEnsemble', best_easy),
    ('RUS', best_rus)
]

model_gender = VotingClassifier(estimators=estimators, voting='soft')
model_gender.fit(X_train, y_train)
y_pred_prob = model_gender.predict_proba(X_test)[:, 1]

threshold = np.linspace(0.1, 0.9, 9)
final_threshold = []
f1_scores = []
for i in threshold:
    y_pred = (y_pred_prob > i).astype(int)
    score = f1_score(y_test, y_pred, sample_weight=weights)
    final_threshold.append(score)
    f1_scores.append(score)

best_threshold_gender = threshold[np.argmax(final_threshold)]
y_pred = (y_pred_prob > best_threshold_gender).astype(int)
print(best_threshold_gender, max(f1_scores))

# Classification Report
report = pd.DataFrame({'roc_auc': roc_auc_score(y_test, y_pred_prob),
                       'recall': recall_score(y_test, y_pred),
                       'precision': precision_score(y_test, y_pred),
                       'f1_score': f1_score(y_test, y_pred, sample_weight=weights),
                       'accuracy': accuracy_score(y_test, y_pred)},
                       index=['VotingClassifier']).T
print(report)

# Confusion Matrix 
cfs_matrix = confusion_matrix(y_test, y_pred)
cfs_matrix = pd.DataFrame(cfs_matrix, index=['Actual Male', 'Actual Female'], columns=['Predicted Male', 'Predicted Female'])
sns.heatmap(cfs_matrix, annot=True, fmt='g', cbar=False)
plt.show()


df_test_cat = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx')
df_test_num = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx')
df_test_matrice = pd.read_csv('/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv')


df_test = pd.merge(df_test_cat, df_test_num, on='participant_id')
df_test = pd.merge(df_test, df_test_matrice, on='participant_id')
df_test


# impute missing values

imputer2 = SimpleImputer(missing_values=np.nan, strategy='mean')
df_test[num_cols] = imputer1.fit_transform(df_test[num_cols])

imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df_test[cat_cols] = imputer2.fit_transform(df_test[cat_cols])

df_test.isnull().sum().sum()


# harmonization

mri_features_test = df_test[matrice_cols].values
batch_test = df_test['MRI_Track_Scan_Location'].values
covars_test = pd.DataFrame({'SITE': batch_test,
                        'age': df_test['MRI_Track_Age_at_Scan'].values})

model_mri_test, features_harmonized_test = harmonizationLearn(mri_features_test, covars_test, 'SITE')

df_test[matrice_cols] = features_harmonized_test


X_test_final = df_test[selected_features]


scaler_skew = PowerTransformer(method='yeo-johnson', standardize=True)
scaler_skew.fit(X_test_final[skew_cols])
X_test_final[skew_cols] = scaler_skew.transform(X_test_final[skew_cols])

scaler_noskew = StandardScaler()
scaler_noskew.fit(X_test_final[noskew_cols])
X_test_final[noskew_cols] = scaler_noskew.transform(X_test_final[noskew_cols])


# inference on test data
final_thres_gender = best_threshold_gender

y_pred_prob = model_gender.predict_proba(X_test_final)[:, 1]
y_pred = (y_pred_prob > final_thres_gender).astype(int)

df_test[selected_target] = y_pred

df_test[selected_target].value_counts()


# save df_test to csv
df_test[['participant_id', selected_target]].to_csv('submission_{}.csv'.format(selected_target), index=False)


sub1 = pd.read_csv('submission_ADHD_Outcome.csv')
sub2 = pd.read_csv('submission_Sex_F.csv')

sub_final = pd.merge(sub1, sub2, on='participant_id')
sub_final


# save final submission
sub_final.to_csv('final_submission.csv', index=False)

