import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _ , filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_connectome=pd.read_csv(f"/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
train_numeric=pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx")
train_categorical=pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx")
train_solution=pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx")

test_connectome=pd.read_csv(f"/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
test_categorical=pd.read_excel(f"/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx")
test_numeric=pd.read_excel(f"/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx")


train_numeric.info()


train_numeric.head()


train_categorical.info()


train_categorical.head()


train_solution.head()


# sort values by participant id
train_numeric.sort_values(by='participant_id', inplace=True)
train_numeric.reset_index(drop=True, inplace=True)

train_categorical.sort_values(by='participant_id', inplace=True)
train_categorical.reset_index(drop=True, inplace=True)

train_connectome.sort_values(by='participant_id', inplace=True)
train_connectome.reset_index(drop=True, inplace=True)

train_solution.sort_values(by='participant_id', inplace=True)
train_solution.reset_index(drop=True, inplace=True)

test_numeric.sort_values(by='participant_id', inplace=True)
test_numeric.reset_index(drop=True, inplace=True)

test_categorical.sort_values(by='participant_id', inplace=True)
test_categorical.reset_index(drop=True, inplace=True)

test_connectome.sort_values(by='participant_id', inplace=True)
test_connectome.reset_index(drop=True, inplace=True)


train_numeric.head()


train_categorical.head()


train_connectome.head()


result_count = pd.DataFrame(train_solution[['ADHD_Outcome', 'Sex_F']].value_counts())
result_count


print(train_numeric.shape[1])
print(train_categorical.shape[1])
print(train_connectome.shape[1])


# concatenate all the TABULAR train-related data (will process the connectome matrices separately)
train_tabular = pd.concat([train_numeric, train_categorical, train_solution], axis=1)
train_tabular = train_tabular.loc[:, ~train_tabular.columns.duplicated()] # remove duplicate columns if there are
train_tabular


# concatenate all the test-related data
test_tabular = pd.concat([test_numeric, test_categorical], axis=1)
test_tabular = test_tabular.loc[:, ~test_tabular.columns.duplicated()] # remove duplicate columns
test_tabular # there's no target columns 


targets = ['ADHD_Outcome','Sex_F']
features = test_tabular.columns


def check_for_nulls(df, df_name):
    """
    Checks for null values in a pandas DataFrame and prints the count of null values per column.

    Args:
        df: The pandas DataFrame to check.

    Returns:
        None
    """
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()

    if total_nulls > 0:
        print(f"\nThe {df_name} dataframe contains {total_nulls} null values.")
        print(null_counts[null_counts > 0])
    else:
        print(f"\nThe {df_name} dataframe does not contain null values.")

check_for_nulls(train_tabular, "train")
check_for_nulls(test_tabular, "test")


X_train_tabular = train_tabular.drop(columns=['participant_id', 'ADHD_Outcome', 'Sex_F'])
y_train_tabular = train_tabular[['ADHD_Outcome', 'Sex_F']]

participant_id_test = test_tabular['participant_id']
X_test_tabular = test_tabular.drop(columns=['participant_id'])


X_train_tabular.head()


X_test_tabular.head()


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X_train_tabular_imputed = pd.DataFrame(imputer.fit_transform(X_train_tabular), columns=X_train_tabular.columns)
X_test_tabular_imputed = pd.DataFrame(imputer.transform(X_test_tabular), columns=X_test_tabular.columns)


train_connectome.shape[1]


train_connectome.head()


train_connectome_copy = train_connectome.drop(columns=['participant_id'])
test_connectome_copy = test_connectome.drop(columns=['participant_id'])


from sklearn.decomposition import PCA

pca = PCA(n_components=50) # dimensionality reduction to 50 components
connectome_train_pca = pca.fit_transform(train_connectome_copy)
connectome_test_pca = pca.transform(test_connectome_copy)


connectome_train_pca_df = pd.DataFrame(connectome_train_pca, columns=[f'conn_{i}' for i in range(connectome_train_pca.shape[1])])
connectome_test_pca_df = pd.DataFrame(connectome_test_pca, columns=[f'conn_{i}' for i in range(connectome_test_pca.shape[1])])


connectome_train_pca_df.head()


from sklearn.feature_selection import SelectKBest, f_classif

# Select top 10 features for ADHD_Outcome
selector_adhd = SelectKBest(f_classif, k=15)
X_train_adhd = selector_adhd.fit_transform(X_train_tabular_imputed, y_train_tabular['ADHD_Outcome'])

# Select top 10 features for Sex_F
selector_gender = SelectKBest(f_classif, k=15)
X_train_sex = selector_gender.fit_transform(X_train_tabular_imputed, y_train_tabular['Sex_F'])

selected_idx = list(set(selector_adhd.get_support(indices=True)) | set(selector_gender.get_support(indices=True)))
X_train_selected = X_train_tabular_imputed.iloc[:, selected_idx]
X_test_selected = X_test_tabular_imputed.iloc[:, selected_idx]


X_train_selected


X_test_selected


# combine tabular features with connectome features
X_train_combined = pd.concat([X_train_selected.reset_index(drop=True), connectome_train_pca_df.reset_index(drop=True)], axis=1)
X_test_combined = pd.concat([X_test_selected.reset_index(drop=True), connectome_test_pca_df.reset_index(drop=True)], axis=1)


from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

base_estimator = RandomForestClassifier(random_state=42, class_weight='balanced')  # class_weight to help with imbalance

multi_clf = MultiOutputClassifier(base_estimator)


from sklearn.model_selection import GridSearchCV

param_grid = {
    'estimator__n_estimators': [100, 200],
    'estimator__max_depth': [10, 20],
    'estimator__min_samples_split': [2, 5],
    'estimator__max_features': ['sqrt', 'log2']
}

# GridSearchCV with MultiOutputClassifier
grid_search = GridSearchCV(multi_clf, param_grid, scoring='f1_weighted', cv=3, verbose=2, n_jobs=-1)

# Fit on your combined features and targets
grid_search.fit(X_train_combined, y_train_tabular)


from sklearn.metrics import classification_report, f1_score

# Get best model
best_model = grid_search.best_estimator_

y_train_pred = best_model.predict(X_train_combined)

y_train_pred_df = pd.DataFrame(y_train_pred, columns=['ADHD_Outcome', 'Sex_F'])

print(classification_report(y_train_tabular, y_train_pred_df))


# Predict on test set
y_test_pred = best_model.predict(X_test_combined)

# Prepare submission
submission_df = pd.DataFrame({
    'participant_id': participant_id_test,
    'ADHD_Outcome': y_test_pred[:, 0],
    'Sex_F': y_test_pred[:, 1]
})

submission_df.to_csv("submission.csv", index=False)




