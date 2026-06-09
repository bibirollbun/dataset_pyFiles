import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import chi2
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV




df_train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


df_train.info()


categorical_columns = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status', 'family_history_diabetes', 'hypertension_history', 'cardiovascular_history', 'diagnosed_diabetes', 'alcohol_consumption_per_week']
numeric_columns = [col for col in df_train.columns if col not in categorical_columns]

for col in categorical_columns:
    df_train[col] = df_train[col].astype('category')

numeric_columns = numeric_columns[1:]


df_aux = df_train[numeric_columns]
df_aux


sns.set_theme(style="white")

# Compute the correlation matrix
corr = df_aux.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, center=0, annot=True, fmt=".2f",
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


print("ldl_cholesterol:", df_train['ldl_cholesterol'].isna().sum())
print("cholesterol_total:", df_train['ldl_cholesterol'].isna().sum())


print("waist_to_hip_ratio:", df_train['waist_to_hip_ratio'].isna().sum())
print("bmi:", df_train['bmi'].isna().sum())


df_train.drop(columns=['ldl_cholesterol'], inplace=True)
df_test.drop(columns=['ldl_cholesterol'], inplace=True)

df_train.drop(columns=['waist_to_hip_ratio'], inplace=True)
df_test.drop(columns=['waist_to_hip_ratio'], inplace=True)

to_remove = ['waist_to_hip_ratio', 'ldl_cholesterol']
numeric_columns = [col for col in numeric_columns if col not in to_remove]


def view_outlayers(col_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
    sns.histplot(
        df_train[col_name],
        kde=True,
        ax=ax1,
        color='blue'
    )
    ax1.axvline(df_train[col_name].mean(), color='red', linewidth=2)
    sns.boxplot(x=df_train[col_name], ax=ax2)
    plt.tight_layout()
    plt.show()


for col in numeric_columns:
    view_outlayers(col)


len(numeric_columns)


preprocess = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown='ignore'), categorical_columns)
], remainder='drop')

X_cat = preprocess.fit_transform(df_train[categorical_columns])
y = df_train["diagnosed_diabetes"]

selector = SelectKBest(score_func=chi2, k='all')
selector.fit(X_cat, y)

scores = selector.scores_
features = preprocess.named_transformers_["onehot"].get_feature_names_out(categorical_columns)

df_aux = pd.DataFrame({"feature":features,"chi2":scores}).sort_values("chi2", ascending=False)
df_aux


# categorical_columns = df_aux.loc[df_aux['chi2'] > 1.4, 'feature'].tolist()
selected_categorical_cols_to_drop = df_aux.loc[df_aux['chi2'] < 1.4, 'feature'].tolist()


k = (df_aux['chi2'] > 1.4).sum()


categorical_columns


# Remove target
categorical_columns = [cat_col for cat_col in categorical_columns if cat_col != 'diagnosed_diabetes']
len(categorical_columns)


categorical_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ("chi2_selector", SelectKBest(score_func=chi2, k=k))
])


preprocess_linear = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ]
)


preprocess_tree = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_columns),
        ("cat", "passthrough", categorical_columns)
    ]
)


# Primer split: train / (val+test)
train_df, temp_df = train_test_split(
    df_train,
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


x_train = train_df.drop(columns=["diagnosed_diabetes"])
y_train = train_df['diagnosed_diabetes']

x_val = val_df.drop(columns = ['diagnosed_diabetes'])
y_val = val_df['diagnosed_diabetes']

x_test = test_df.drop(columns = ['diagnosed_diabetes'])
y_test = test_df['diagnosed_diabetes']

print(f"Train: x_train = {x_train.shape}, y_train = {y_train.shape}")
print(f"Val: x_val = {x_val.shape}, y_val = {y_val.shape}")
print(f"Test: x_test = {x_test.shape}, y_test = {y_test.shape}")


def auc_on_test(model_pipeline, x_test = x_test):
    y_test_predict = model_pipeline.predict_proba(x_test)[:,1]
    return roc_auc_score(y_test, y_test_predict)


lr_model_pipeline = Pipeline(steps=[
    ("preprocess", preprocess_linear),
    ("model", LogisticRegression(max_iter=1000))
])

lr_model_pipeline.fit(x_train, y_train)


kf = KFold(n_splits=5, shuffle=True, random_state=42)

params = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__penalty": ["l2"],
    "model__solver": ["lbfgs"],
}

grid = GridSearchCV(
    estimator=lr_model_pipeline,
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
    "model__alpha": [1e-4, 1e-3, 1e-2],
    "model__penalty": ["l2", "l1", "elasticnet"],
    "model__l1_ratio": [0.0, 0.5, 1.0]
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


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf_pipeline = Pipeline(steps=[
    ("preprocess", preprocess_linear),
    ("model", rf_model)
])

params_rf = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 3]
}

grid_rf = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=params_rf,
    cv=kf,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)

grid_rf.fit(x_val, y_val)

best_model_rf = grid_rf.best_estimator_

print("Best AUC:", grid_rf.best_score_)
print("Best hiperparams:", grid_rf.best_params_)
print("Best model on Test:", auc_on_test(best_model_rf, x_test))


ids = df_test['id']
y_final_predict = best_model_rf.predict_proba(df_test)[:,1]

submission = pd.DataFrame({
    "id": ids,
    "diagnosed_diabetes": y_final_predict
})

submission.to_csv("submission.csv", index=False)

