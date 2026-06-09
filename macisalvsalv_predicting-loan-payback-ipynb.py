import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


import os

os.listdir('/kaggle/input/playground-series-s5e11')


df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


df.head()


df.describe()


df['loan_paid_back'].describe()


df['loan_paid_back'].unique()


df['loan_paid_back'] = df['loan_paid_back'].astype(int)
df['loan_paid_back'].describe()


df['annual_income'].describe()


df['annual_income'].isna().sum()


def view_outlayers(col_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
    sns.histplot(
        df[col_name],
        kde=True,
        ax=ax1,
        color='blue'
    )
    ax1.axvline(df[col_name].mean(), color='red', linewidth=2)
    sns.boxplot(x=df[col_name], ax=ax2)
    plt.tight_layout()
    plt.show()


view_outlayers('annual_income')


# df['credit_score_scaled'] = scaler.fit_transform(df[['credit_score']])
# df_test['credit_score_scaled'] = scaler.transform(df_test[['credit_score']])


view_outlayers('debt_to_income_ratio')


# df['credit_score_scaled'] = scaler.fit_transform(df[['credit_score']])
# df_test['credit_score_scaled'] = scaler.transform(df_test[['credit_score']])


view_outlayers('credit_score')


# df['credit_score_scaled'] = scaler.fit_transform(df[['credit_score']])
# df_test['credit_score_scaled'] = scaler.transform(df_test[['credit_score']])


view_outlayers('loan_amount')


#df['loan_amount_log'] = np.log1p(df['loan_amount'])
#df_test['loan_amount_log'] = np.log1p(df_test['loan_amount'])


view_outlayers('interest_rate')


# df['credit_score_scaled'] = scaler.fit_transform(df[['credit_score']])
# df_test['credit_score_scaled'] = scaler.transform(df_test[['credit_score']])


df['gender'].unique()


def view_categorical_data(col_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

    # Conteo de categorías
    sns.countplot(x=col_name, data=df, ax=ax1)
    ax1.set_title(f"Sum of categories: '{col_name}'")

    # Distribución en porcentaje
    df[col_name].value_counts(normalize=True).mul(100).plot(kind='bar', ax=ax2)
    ax2.tick_params(axis='x', rotation=0)
    ax2.set_title(f"Porcentual distribution: '{col_name}'")
    plt.show()

    print('*** Distribution on train ***')
    print(df[col_name].value_counts())

    print('\n*** Distribution on test ***')
    print(df_test[col_name].value_counts())

view_categorical_data('gender')


def view_categorical_vs_y(categorical_column):
  return_ratio = (
      df.groupby(categorical_column)['loan_paid_back'].value_counts()
        .unstack()
        .assign(ratio=lambda x: x[1.0] / x[0.0])
  )
  print(return_ratio)

view_categorical_vs_y('gender')


view_categorical_data('marital_status')


view_categorical_vs_y('marital_status')


def regroup_marital_status(actual_value):
    if actual_value in ['Single', 'Widowed', 'Divorced']:
        return 'Single'
    return 'Married'


df['marital_status_group'] = df['marital_status'].apply(regroup_marital_status)
df.drop('marital_status', axis=1, inplace=True)

df_test['marital_status_group'] = df_test['marital_status'].apply(regroup_marital_status)
df_test.drop('marital_status', axis=1, inplace=True)
view_categorical_data('marital_status_group')


view_categorical_data('education_level')


view_categorical_vs_y('education_level')


view_categorical_data('employment_status')


view_categorical_vs_y('employment_status')


def regroup_employment_status(actual_value):
    if actual_value in ['Self-employed', 'Employed']:
        return 'Employed'
    elif actual_value in ['Student', 'Unemployed']:
        return 'Unemployed'
    else:
        return 'Retired'


df['employment_status_group'] = df['employment_status'].apply(regroup_employment_status)
df.drop('employment_status', axis=1, inplace=True)

df_test['employment_status_group'] = df_test['employment_status'].apply(regroup_employment_status)
df_test.drop('employment_status', axis=1, inplace=True)


view_categorical_vs_y('employment_status_group')


view_categorical_data('employment_status_group')


view_categorical_data('loan_purpose')


view_categorical_vs_y('loan_purpose')


view_categorical_vs_y('grade_subgrade')


df["grade_group"] = df["grade_subgrade"].str[0]
df_test["grade_group"] = df_test["grade_subgrade"].str[0]

df.drop(['grade_subgrade'], axis=1, inplace=True)
df_test.drop(['grade_subgrade'], axis=1, inplace=True)


view_categorical_vs_y('grade_group')


view_categorical_data('grade_group')


categorical_cols = ['gender', 'marital_status_group', 'education_level', 'employment_status_group', 'loan_purpose', 'grade_group']
continue_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', ]


from sklearn.model_selection import train_test_split

# Primer split: train / (val+test)
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,      # 30% para val y test
    random_state=42,
    shuffle=True
)

# Segundo split: val / test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,      # la mitad del 30% para test → 15%
    random_state=42,
    shuffle=True
)


x_train = train_df.drop(columns=["loan_paid_back"])
y_train = train_df['loan_paid_back']

x_val = val_df.drop(columns = ['loan_paid_back'])
y_val = val_df['loan_paid_back']

x_test = test_df.drop(columns = ['loan_paid_back'])
y_test = test_df['loan_paid_back']

print(f"Train: x_train = {x_train.shape}, y_train = {y_train.shape}")
print(f"Val: x_val = {x_val.shape}, y_val = {y_val.shape}")
print(f"Test: x_test = {x_test.shape}, y_test = {y_test.shape}")


preprocess_linear = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), continue_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

linear_model_pipeline = Pipeline(steps=[
    ("preprocess", preprocess_linear),
    ("model", LogisticRegression(max_iter=1000))
])

linear_model_pipeline.fit(x_train, y_train)


def auc_on_test(model_pipeline, x_test = x_test):
    y_test_predict = model_pipeline.predict_proba(x_test)[:,1]
    return roc_auc_score(y_test, y_test_predict)


kf = KFold(n_splits=5, shuffle=True, random_state=42)

params = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__penalty": ["l2"],
    "model__solver": ["lbfgs"],
}

grid = GridSearchCV(
    estimator=linear_model_pipeline,
    param_grid=params,
    cv=kf,
    scoring="roc_auc",
    n_jobs=-1
)

grid.fit(x_val, y_val)
best_model = grid.best_estimator_

print("Best AUC:", grid.best_score_)
print("Best hiperparams:", grid.best_params_)
print("Best model on Test:", auc_on_test(best_model))


sgd_model = SGDClassifier(loss="log_loss", max_iter=2000)

sgd_pipeline = Pipeline(steps=[
    ("preprocess", preprocess_linear),
    ("model", sgd_model)
])

params_sgd = {
    "model__alpha": [1e-4, 1e-3, 1e-2],     # regularización
    "model__penalty": ["l2", "l1", "elasticnet"],
    "model__l1_ratio": [0.0, 0.5, 1.0]      # solo para elasticnet, los demás lo ignoran
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

grid_sgd = GridSearchCV(
    estimator=sgd_pipeline,
    param_grid=params_sgd,
    cv=kf,
    scoring="roc_auc",
    n_jobs=-1
)

grid_sgd.fit(x_val, y_val)
best_model_sgd = grid_sgd.best_estimator_

print("Best AUC:", grid_sgd.best_score_)
print("Best hiperparams:", grid_sgd.best_params_)
print("Best model on Test:", auc_on_test(best_model_sgd, x_test))


preprocess_tree = ColumnTransformer(
    transformers=[
        ("num", "passthrough", continue_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

tree_model_pipeline = Pipeline(steps=[
    ("preprocess", preprocess_tree),
    ("model", DecisionTreeClassifier())
])

params_tree = {
    "model__max_depth": [3, 5, 7, 10, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__criterion": ["gini", "entropy"]
}

grid_tree= GridSearchCV(
    estimator=tree_model_pipeline,
    param_grid=params_tree,
    cv=kf,
    scoring="roc_auc",
    n_jobs=-1
)

grid_tree.fit(x_val, y_val)
best_model_tree = grid_tree.best_estimator_

print("Best AUC:", grid_tree.best_score_)
print("Best hiperparams:", grid_tree.best_params_)
print("Best model on Test:", auc_on_test(best_model_tree))



params_rf = {
    "model__n_estimators": [100, 200], 
    "model__max_depth": [None, 10],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

rf_pipeline = Pipeline(steps=[
    ("preprocess", preprocess_tree),
    ("model", RandomForestClassifier(random_state=42))
])

grid_rf = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=params_rf,
    scoring="roc_auc",
    cv=kf,
    n_jobs=-1
)

grid_rf.fit(x_val, y_val)

best_model_rf = grid_rf.best_estimator_

print("RF Best AUC:", grid_rf.best_score_)
print("RF Best Params:", grid_rf.best_params_)
print("RF Test AUC:", auc_on_test(best_model_rf, x_test))


ids = df_test['id']
y_final_predict = best_model_tree.predict_proba(df_test)[:,1]

submission = pd.DataFrame({
    "id": ids,
    "loan_paid_back": y_final_predict
})

submission.to_csv("submission.csv", index=False)

