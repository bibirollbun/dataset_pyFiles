import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv", index_col=0)


train.head()


print(f"There is {len(train):,} train data point in total")


train["Sex"].value_counts() / len(train)


def describe_numeric_column(col):
    print(f"Min: {col.min()}")
    print(f"Max: {col.max()}")
    print(f"Median: {col.median()}")
    
    sns.histplot(col)
    plt.show()


numeric_columns = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
for col_name in numeric_columns:
    print(f"Column: {col_name}")
    describe_numeric_column(train[col_name])


sns.pairplot(train[numeric_columns].sample(1000))  


train.isna().sum()


print(f"There is {train.duplicated().sum()} duplicated rows in the train set")


train = train.drop_duplicates()


from sklearn.model_selection import train_test_split

train["Sex"] = (train["Sex"] == "female").astype(bool)

X_train, X_val, y_train, y_val = train_test_split(train.drop("Calories", axis=1), train["Calories"], test_size=0.2, random_state=42)

test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv", index_col=0)
test["Sex"] = (test["Sex"] == "female").astype(bool)


import xgboost as xgb
from sklearn.metrics import mean_squared_log_error


%%time
vanilla_model = xgb.XGBRegressor(objective="reg:squaredlogerror", random_state=42)
vanilla_model.fit(X_train, y_train)


import numpy as np

def predict(model, X_test):
    return np.maximum(model.predict(X_test), 0)
    
def evaluate(model, X=X_val, y=y_val):
    y_pred = predict(model, X)
    score = np.sqrt(mean_squared_log_error(y, y_pred))
    return score


val_score = evaluate(vanilla_model)
print(f"RMSLE on the validation set for the vanilla model is: {val_score:.4f}")


from sklearn.compose import TransformedTargetRegressor


def make_model_with_y_transform(model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {}

    return TransformedTargetRegressor(
        regressor=xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **model_kwargs),
        func=np.log1p,  # Calculates log(1 + x)
        inverse_func=np.expm1
)


vanilla_model3 = make_model_with_y_transform()
vanilla_model3.fit(X_train, y_train)

val_score = evaluate(vanilla_model3)
print(f"RMSLE on the validation set for the model trained with RMSE and log-transformed y: {val_score:.4f}")


xgb.plot_importance(vanilla_model3.regressor_, importance_type='gain', height=0.5, title="Feature Importances")
plt.show()


features_to_remove = [
    ["Height"],
    ["Body_Temp"],
    ["Weight"],
    ["Body_Temp", "Weight"],
    ["Height", "Weight"],
    ["Height", "Body_Temp"],
    ["Height", "Body_Temp", "Weight"]
]

for features in features_to_remove:
    model_less_feature = make_model_with_y_transform()
    model_less_feature.fit(X_train.drop(features, axis=1), y_train)

    val_score = evaluate(model_less_feature, X=X_val.drop(features, axis=1))
    print(f"RMSLE when removing {features}: {val_score:.4f}")


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline


def make_model_pipeline(preprocessor, model_kwargs=None):
    model = make_model_with_y_transform(model_kwargs)
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    return model_pipeline


def divide_columns(X):
    return (X.iloc[:, 0] / X.iloc[:, 1]).to_numpy().reshape(-1, 1)


def calculate_bmi(X):
    return (X.iloc[:, 0] / (X.iloc[:, 1] / 100) ** 2).to_numpy().reshape(-1, 1)

divide_transformer = FunctionTransformer(func=divide_columns, validate=False)
bmi_transformer = FunctionTransformer(func=calculate_bmi, validate=False)


preprocessor = ColumnTransformer(
    transformers=[
        ('pass', 'passthrough', ['Sex', 'Age', 'Weight', 'Duration', 'Heart_Rate']),
        ('BMI', bmi_transformer, ['Weight', 'Height']),
        ('Intensity_heart', divide_transformer, ['Heart_Rate', 'Duration']),
    ],
    remainder='drop'
)

model_feature_eng = make_model_pipeline(preprocessor)
model_feature_eng.fit(X_train, y_train)

val_score = evaluate(model_feature_eng)
print(f"RMSLE on the validation set when trained with BMI and Intensity_heart: {val_score:.4f}")


preprocessor = ColumnTransformer(
    transformers=[
        ('pass', 'passthrough', ['Sex', 'Age', 'Weight', 'Duration', 'Heart_Rate']),
        ('Intensity_heart', divide_transformer, ['Heart_Rate', 'Duration']),
    ],
    remainder='drop'
)

model_feature_eng = make_model_pipeline(preprocessor)
model_feature_eng.fit(X_train, y_train)

val_score = evaluate(model_feature_eng)
print(f"RMSLE on the validation set when trained and Intensity_heart: {val_score:.4f}")


def make_preprocessor_from_trial(trial):
    use_bmi = trial.suggest_categorical("use_bmi", [True, False])
    use_intensity_heart = trial.suggest_categorical("use_intensity_heart", [True, False])
    
    return make_preprocessor(use_bmi, use_intensity_heart)


def make_preprocessor(use_bmi, use_intensity_heart):
    transformers=[('pass', 'passthrough', ['Sex', 'Age', 'Weight', 'Duration', 'Heart_Rate'])]

    if use_bmi:
        transformers.append(('BMI', bmi_transformer, ['Weight', 'Height']))
    if use_intensity_heart:
        transformers.append(('Intensity_heart', divide_transformer, ['Heart_Rate', 'Duration']))
        
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    return preprocessor


from sklearn.base import clone
from sklearn.model_selection import KFold

def run_cross_validation(model, preprocessor, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmsle_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train_fold, y_val_fold = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

        X_train_fold = preprocessor.fit_transform(X_train_fold)
        X_val_fold = preprocessor.transform(X_val_fold)

        # Clone model to reset state for each fold
        fold_model = clone(model)

        fold_model.fit(
            X_train_fold, 
            y_train_fold,
            eval_set=[(X_val_fold, model.func(y_val_fold))],
            verbose=0,
        )

        y_pred = predict(fold_model, X_val_fold)
        fold_rmsle = np.sqrt(mean_squared_log_error(y_val_fold, y_pred))
        rmsle_scores.append(fold_rmsle)

    return np.mean(rmsle_scores)


! pip install optuna-integration --quiet


import optuna
from optuna.integration import XGBoostPruningCallback

def objective(trial):
    pruning_callback = XGBoostPruningCallback(trial, "validation_0-rmsle")

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'eta': trial.suggest_float('eta', 0.001, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.00001, 1.0),
        "lambda": trial.suggest_float("lambda", 0.0, 10),
        'early_stopping_rounds': 100,
        'eval_metric': "rmsle",
        'callbacks': [pruning_callback],
    }

    preprocessor = make_preprocessor_from_trial(trial)
    model = make_model_with_y_transform(params)
    
    score = run_cross_validation(model, preprocessor, X_train, y_train)
    return score


%%time
# greater_is_better flips the sign of the metric, so we want to maximize the negative RMSLE loss
study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
study.optimize(objective, n_trials=25)


from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice


plot_optimization_history(study).show()


plot_slice(study).show()


plot_param_importances(study).show()


best_params = study.best_params
print(best_params)


best_params_model = {k: v for k, v in best_params.items() if k not in {"use_bmi", "use_intensity_heart"}}

preprocessor = make_preprocessor(use_bmi=best_params["use_bmi"], use_intensity_heart=best_params["use_intensity_heart"])


final_model = make_model_pipeline(preprocessor, model_kwargs=best_params_model)
final_model.fit(X_train, y_train)

val_score = evaluate(final_model)
print(f"RMSLE on the validation set after HP search with Optuna: {val_score:.4f}")


final_model = make_model_pipeline(preprocessor, model_kwargs=best_params_model)
final_model.fit(train.drop("Calories", axis=1), train["Calories"])


y_pred = final_model.predict(test)

submission = pd.DataFrame({"Calories": y_pred}, index= test.index)
submission.to_csv("submission.csv")

