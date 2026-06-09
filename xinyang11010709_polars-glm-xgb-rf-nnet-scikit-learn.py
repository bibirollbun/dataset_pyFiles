import polars as pl 

train_tbl = pl.read_csv("/kaggle/input/playground-series-s5e3/train.csv")



# EDA report in output area

from ydata_profiling import ProfileReport

ProfileReport(
    train_tbl.to_pandas().drop(columns="id"),
    title="Binary Prediction with Rainfall",
    explorative=True
).to_file("Binary Prediction with Rainfall.html")

## Conclusions from EDA

### Cloud, Sunshine and humidity are top 3 most impactful factors
### We can remove the feature "day" which seems irrelevant to target "rainfall"
### We can try PCA to reduce features because features "dewpoint", "maxtemp","mintemp" and "tempreture" are highly correlated to each other.


# Feature engineering

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

X = train_tbl.drop(["id","day","rainfall"]).to_pandas()

y = train_tbl["rainfall"].to_pandas()

pca_columns = ["dewpoint", "maxtemp", "mintemp", "temparature"]
non_pca_columns = [col for col in X.columns if col not in pca_columns]

preprocessor_full = Pipeline([
    ("preprocessor_pca", ColumnTransformer([
        ("pca", PCA(n_components=0.90), pca_columns),
        ("passthrough", "passthrough", non_pca_columns)
    ])),
    ("scaler", StandardScaler())
])

X_prep = preprocessor_full.fit_transform(X)


# Models and grid search

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

models = {
    "rf": RandomForestClassifier(),
    "xgb": XGBClassifier(eval_metric="logloss"),
    "glm": LogisticRegression(max_iter=1000),
    "nnet": MLPClassifier(max_iter=1000)
}

param_grids = {
    "rf": {"n_estimators": [100, 200], "max_features": ["sqrt", "log2"]},
    "xgb": {"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.01, 0.1]},
    "glm": {"C": [0.1, 1, 10], "penalty": ["l1", "l2"], "solver": ["liblinear"]},
    "nnet": {"hidden_layer_sizes": [(50,), (100,)], "alpha": [0.0001, 0.001]}
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

grid_results = {}
for name, model in models.items():
    grid = GridSearchCV(model, param_grids[name], cv=cv, scoring="roc_auc", verbose=0)
    grid.fit(X_prep, y)
    grid_results[name] = grid


## Get the name of the best model
best_model_name = max(grid_results, key=lambda k: grid_results[k].best_score_)

print(best_model_name)
print(grid_results[best_model_name].best_score_)



# Predict with best model

## Impute na in column winddirection and pre-process test dataset

test_tbl = pl.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

test_tbl = test_tbl.with_columns(
    pl.col("winddirection").fill_null(strategy="forward")
)

X_test_prep = preprocessor_full.transform(test_tbl.drop(["id","day"]).to_pandas())

## Get the best model
best_model = grid_results[best_model_name].best_estimator_

preds = best_model.predict(X_test_prep)

submission = pl.DataFrame({
    "id": test_tbl["id"],
    "rainfall": preds
})

submission.write_csv("submission.csv")

