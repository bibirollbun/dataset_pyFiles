pip uninstall scikit-learn imbalanced-learn -y


pip install scikit-learn==1.2.2 imbalanced-learn==0.9.1


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


raw_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
raw2_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


raw_df.head()


raw_df.describe(include = 'all')


raw_df.isna().sum()


raw_df.duplicated().sum()


raw_df.info()


raw2_df.head()


raw2_df.describe(include = 'all')


raw2_df.isna().sum()


raw2_df.duplicated().sum()


raw2_df.info()


train_df = raw_df.copy()
test_df = raw2_df.copy()


train_with_na = len(train_df)
test_with_na = len(test_df)

ttrain_df = train_df.dropna()
ttest_df = test_df.dropna()

train_without_na = len(ttrain_df)
test_without_na = len(ttest_df)

print(f"Length of train_df with NANs: {train_with_na}")
print(f"Length of train_df without NANs: {train_without_na}")
print()
print(f"Length of test_df with NANs: {test_with_na}")
print(f"Length of test_df without NANs: {test_without_na}")


train_df = train_df.drop(columns = 'id')
test_df = test_df.drop(columns = 'id')


train_df['Personality'] = train_df['Personality'].map({'Introvert': 0, 'Extrovert': 1})


from sklearn.preprocessing import LabelEncoder

non_numeric_columns_train = train_df.select_dtypes(include = ['object']).columns
non_numeric_columns_test = test_df.select_dtypes(include = ['object']).columns

le = LabelEncoder()

for column in non_numeric_columns_train:
    temp_data = np.concatenate([train_df[column].values, test_df[column].values])
    le.fit(temp_data)
    train_df[column] = le.transform(train_df[column])
    test_df[column] = le.transform(test_df[column])


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Initialize IterativeImputer (MICE)
imputer = IterativeImputer(random_state=0, max_iter=10)

# Impute the data
imputed_train_df = pd.DataFrame(imputer.fit_transform(train_df), columns=train_df.columns)
imputed_test_df = pd.DataFrame(imputer.fit_transform(test_df), columns=test_df.columns)

print("Imputed DataFrame:")
imputed_train_df



x = imputed_train_df.drop(columns = 'Personality')
y = imputed_train_df['Personality']


train_corr_matrix = imputed_train_df.corr()
plt.figure(figsize = (12, 8))
sns.heatmap(train_corr_matrix, annot = True, fmt = ".2f", cmap = "coolwarm")
plt.title('Correlation Matrix - Train Dataset')
plt.show()


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf.fit(x, y)


importances = rf.feature_importances_
feature_names = x.columns
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Rank features by importance
feature_importance = feature_importance.sort_values(by = 'Importance', ascending=False).reset_index(drop = True)
feature_importance.head(10)


from sklearn.feature_selection import SelectKBest, chi2

best_features = SelectKBest(score_func = chi2, k = 'all')
fit = best_features.fit(x, y)


# Get the scores of the features
feature_scores = pd.DataFrame({'Feature': feature_names, 'Score': fit.scores_})
feature_scores = feature_scores.sort_values(by = 'Score', ascending=False).reset_index(drop = True)
feature_scores.head(10)


fi_columns = feature_importance['Feature'].values
fi_importances = feature_importance['Importance'].values
fi_threshold = feature_importance['Importance'].median()
fi_selected_features = []

for i in range(len(feature_importance)):
    if fi_importances[i] >= fi_threshold:
        fi_selected_features.append(fi_columns[i])

fs_columns = feature_scores['Feature'].values
f_scores = feature_scores['Score'].values
fs_threshold = feature_scores['Score'].median()
fs_selected_features = []

for i in range(len(feature_scores)):
    if f_scores[i] >= fs_threshold:
        fs_selected_features.append(fs_columns[i])

print(len(fi_selected_features))
print(len(fs_selected_features))


# using correlation matrix to find the best features using median as threshold

cor = abs(train_corr_matrix['Personality'].values)
corr_cols = train_corr_matrix.index.to_numpy()
corr_threshold = train_corr_matrix['Personality'].median()
corr_selected_features = []

for i in range(len(cor)):
    if cor[i] >= corr_threshold:
        corr_selected_features.append(corr_cols[i])

len(corr_selected_features)
# corr_selected_features


corr_cols


# selecting the best features for model training

columns_selected = set(fs_selected_features) & set(fi_selected_features) & set(corr_selected_features)
columns_selected = list(columns_selected)
columns_selected


# shuffling the data
from sklearn.utils import shuffle
imputed_train_df = shuffle(imputed_train_df).reset_index(drop = True)
x_train = imputed_train_df.drop(columns = 'Personality')
y_train = imputed_train_df['Personality']

# splitting the train dataset into 2 parts one for model training and other for evaluation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 42)


# using random forest classifier for model training hypertuning included
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

param_grid_rf = {'n_estimators': [200, 300, 400, 500],
                 'max_depth': [10, 15, 20, 25]}

grid_search_rf = GridSearchCV(rf, param_grid = param_grid_rf, scoring = 'accuracy', cv = 5, n_jobs = -1)
grid_search_rf.fit(x_train, y_train)

best_model_rf = grid_search_rf.best_estimator_
best_params_rf = grid_search_rf.best_params_

predictions_rf = best_model_rf.predict(x_val)

best_params_rf


accuracy_rf = metrics.accuracy_score(y_val, predictions_rf)
print(f"Accuracy of Random Forest {accuracy_rf}")


f1_rf = metrics.f1_score(y_val, predictions_rf)
print(f"F1 of Random Forest {f1_rf}")


roc_rf = metrics.roc_auc_score(y_val, predictions_rf)
print(f"ROC AUC score of Random Forest {roc_rf}")


print(metrics.classification_report(y_val, predictions_rf))


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state = 42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", pd.Series(y_train_resampled).value_counts())


# using random forest classifier for model training hypertuning included
from sklearn.ensemble import RandomForestClassifier

rf2 = RandomForestClassifier()

param_grid_rf2 = {'n_estimators': [100, 200, 300, 400],
                 'max_depth': [20, 25, 30, 35]}

grid_search_rf2 = GridSearchCV(rf2, param_grid = param_grid_rf2, scoring = 'accuracy', cv = 5, n_jobs = -1)
grid_search_rf2.fit(x_train_resampled, y_train_resampled)

best_model_rf2 = grid_search_rf2.best_estimator_
best_params_rf2 = grid_search_rf2.best_params_

predictions_rf2 = best_model_rf2.predict(x_val)

best_params_rf2


accuracy_rf2 = metrics.accuracy_score(y_val, predictions_rf2)
print(f"Accuracy of Random Forest {accuracy_rf2}")


f1_rf2 = metrics.f1_score(y_val, predictions_rf2)
print(f"F1 of Random Forest {f1_rf2}")


roc_rf2 = metrics.roc_auc_score(y_val, predictions_rf)
print(f"ROC AUC score of Random Forest {roc_rf2}")


roc_rf = metrics.roc_auc_score(y_val, predictions_rf)
print(f"ROC AUC score of Random Forest {roc_rf}")


if accuracy_rf2 > accuracy_rf:
    best_model = best_model_rf2
    print("RF 2 Selected")
else:
    best_model = best_model_rf
    print("RF Selected")


x_test = test_df


predictions = (best_model.predict(imputed_test_df)).tolist()
ids = raw2_df['id'].values


pred_df = pd.DataFrame()
pred_df['id'] = ids
pred_df['Personality'] = predictions


pred_df['Personality'] = pred_df['Personality'].map({0:'Introvert', 1:'Extrovert'})


pred_df.head()


pred_df.to_csv('predicted.csv', index = False)

