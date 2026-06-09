import warnings
from pathlib import Path
import pandas as pd
from typing import Optional, List, Tuple, Callable
from sklearn.model_selection import KFold, train_test_split
from catboost import CatBoostRegressor
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import PredictionErrorDisplay
from sklearn.model_selection import KFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.metrics import mean_squared_log_error
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn import set_config
set_config(transform_output="pandas")


dir_data = Path("/kaggle/input/playground-series-s5e5")

data = pd.read_csv(
    filepath_or_buffer=dir_data / "train.csv",
    dtype={
        'id': 'int32',
        'Sex': 'string',
        'Age': 'int32',
        'Height': 'float32',
        'Weight': 'float32',
        'Duration': 'float32',
        'Heart_Rate': 'float32',
        'Body_Temp': 'float32',
        'Calories': 'float32',
    },
)

data_sub = pd.read_csv(
    filepath_or_buffer=dir_data / "test.csv",
    dtype={
        'id': 'int32',
        'Sex': 'string',
        'Age': 'int32',
        'Height': 'float32',
        'Weight': 'float32',
        'Duration': 'float32',
        'Heart_Rate': 'float32',
        'Body_Temp': 'float32',
    },
)


# drop duplicates
print(data.shape)
data = data.drop_duplicates(subset=data.columns.drop("id"), keep="first")
data.reset_index(drop=True, inplace=True)
print(data.shape)


def multiply(X: pd.DataFrame) -> pd.DataFrame:
    cols = X.columns
    col_name = f"{cols[0]}_x_{cols[1]}"
    col_name = "_x_".join(X.columns)
    X[col_name] = X.product(axis=1)
    return X[[col_name]]


def calculate_bmi_features(X: pd.DataFrame) -> pd.DataFrame:
    """ Calculate BMI based on weight and height and BMI category.
    Weight is in kg, height is in m.
    BMI category is in %.
    BMI is in kg/m^2.
    BMI categories:
    - 0: Underweight (BMI < 18.5)
    - 1: Normal weight (18.5 <= BMI < 25)
    - 2: Overweight (25 <= BMI < 30)
    - 3: Obesity (30 <= BMI < 35)
    - 4: Extreme obesity (BMI >= 35)
    """
    X["BMI"] = X["Weight"] / ( (X["Height"] / 100) ** 2)
    X["BMI_category"] = pd.cut(X["BMI"], bins=[0, 18.5, 25, 30, 35, 100], labels=[0, 1, 2, 3, 4])
    X["BMI_category"] = X["BMI_category"].astype(int)
    return X[["BMI", "BMI_category"]]


def calculate_caloried_burned(X: pd.DataFrame) -> pd.DataFrame:
    """ https://www.omnicalculator.com/sports/calories-burned-by-heart-rate
    """
    X["calories_burned"] = np.where(
        X["Sex"] == "male",
        X["Duration"] * (0.6309 * X["Heart_Rate"] + 0.1988 * X["Weight"] + 0.2017 * X["Age"] - 55.0969) / 4.184,
        X["Duration"] * (0.4472 * X["Heart_Rate"] - 0.1263 * X["Weight"] + 0.074 * X["Age"] - 20.4022) / 4.184
    )
    return X[["calories_burned"]]


def get_age_binned(X: pd.DataFrame) -> pd.DataFrame:
    X["age_binned"] = pd.cut(X["Age"], bins=[0, 20, 30, 40, 50, 60, 70, 100], labels=[0, 1, 2, 3, 4, 5, 6])
    X["age_binned"] = X["age_binned"].astype(int)
    return X[["age_binned"]]

def calculate_bmr(X: pd.DataFrame) -> pd.DataFrame:
    """ Calculate Basal Metabolic Rate (BMR) using the Harris-Benedict equation or Mifflin-St Jeor Equation
    Weight is in kg, height is in cm, age is in years.
    
    """
    # X["BMR"] = np.where(
    #     X["Sex"] == "male",
    #     88.362 + (13.397 * X["Weight"]) + (4.799 * X["Height"]) - (5.677 * X["Age"]),
    #     447.593 + (9.247 * X["Weight"]) + (3.098 * X["Height"]) - (4.330 * X["Age"])
    # )

    X["BMR"] = np.where(
        X["Sex"] == "male",
        10 * X["Weight"] + 6.25 * X["Height"] - 5 * X["Age"] + 5,
        10 * X["Weight"] + 6.25 * X["Height"] - 5 * X["Age"] - 161
    )
    
    return X[["BMR"]]


def calculate_bsa(X: pd.DataFrame) -> pd.DataFrame:
    """ Calculate Body Surface Area (BSA) using the DuBois formula.
    Weight is in kg, height is in cm.
    """
    X["BSA"] = 0.007184 * (X["Weight"] ** 0.425) * (X["Height"] ** 0.725)
    return X[["BSA"]]


def calculate_hr_features(X: pd.DataFrame) -> pd.DataFrame:
    """ Calculate Heart Rate features
    HR_max is in bpm, HR_percentage is in %.
    HR_zone is in %.
    """
    X["HR_max"] = 220 - X["Age"]
    X["HR_percentage"] = X["Heart_Rate"] / X["HR_max"]
    X["HR_zone"] = pd.cut(X["HR_percentage"], bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], labels=[0, 1, 2, 3, 4, 5])
    X["HR_zone"] = X["HR_zone"].astype(int)
    return X[["HR_max", "HR_percentage", "HR_zone"]]


def thermal_features(X: pd.DataFrame) -> pd.DataFrame:
    """ Create temperature-related features """
    result = pd.DataFrame(index=X.index)
    
    # Temperature deviations from normal
    normal_temp = 37.0  # Normal body temp in Celsius
    result["temp_deviation"] = X["Body_Temp"] - normal_temp
    result["temp_deviation_abs"] = np.abs(result["temp_deviation"])
    
    # Temperature categories
    result["temp_category"] = pd.cut(X["Body_Temp"], bins=[35, 36.5, 37.5, 38.5, 42], labels=[0, 1, 2, 3])  # hypothermia, normal, fever, high fever
    result["temp_category"] = result["temp_category"].astype(int)
    
    # Thermoregulation effort (higher temp deviation might indicate more energy expenditure)
    result["thermoregulation_effort"] = result["temp_deviation_abs"] * X["Duration"]
    
    return result


def get_power_to_weight_ratio(X: pd.DataFrame) -> pd.DataFrame:
    X["power_to_weight"] = X["Heart_Rate"] * X["Duration"] / X["Weight"]
    return X[["power_to_weight"]]


# add most features for EDA
data["BMR"] = calculate_bmr(data)
data["BSA"] = calculate_bsa(data)
data["HR_max"] = calculate_hr_features(data).HR_max
data["HR_percentage"] = calculate_hr_features(data).HR_percentage
data["HR_zone"] = calculate_hr_features(data).HR_zone
data["temp_category"] = thermal_features(data).temp_category
data["thermoregulation_effort"] = thermal_features(data).thermoregulation_effort
data["power_to_weight"] = get_power_to_weight_ratio(data).power_to_weight
data["calories_burned"] = calculate_caloried_burned(data).calories_burned
data["age_binned"] = get_age_binned(data).age_binned
data["BMI_category"] = calculate_bmi_features(data).BMI_category
data["BMI"] = calculate_bmi_features(data).BMI


data.info()
data.describe()


CATEGORICAL_FEATURES = ["Sex"]
SECONDARY_CATEGORICAL_FEATURES = ["BMI_category", "age_binned", "HR_zone", "temp_category"]
NUMERICAL_FEATURES = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
SECONDARY_NUMERICAL_FEATURES = ["BMR", "BSA", "HR_max", "HR_percentage", "calories_burned"]
TARGET = "Calories"


# plot categorical feature distribution of categorical features
for feature in CATEGORICAL_FEATURES + SECONDARY_CATEGORICAL_FEATURES:
    sns.countplot(x=feature, data=data)
    plt.show()

# print unique values of categorical features
for feature in CATEGORICAL_FEATURES + SECONDARY_CATEGORICAL_FEATURES:
    print(f"{feature}: {data[feature].unique()}")



for col in NUMERICAL_FEATURES + SECONDARY_NUMERICAL_FEATURES:
    print(f"{col}: n unique = {data[col].nunique()}")
    print(data[col].value_counts())
    print()


# plot kde distribution of numerical features
for feature in NUMERICAL_FEATURES + SECONDARY_NUMERICAL_FEATURES:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        sns.kdeplot(data[feature], fill=True)
    plt.show()

# plot kde distribution of target
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    sns.kdeplot(data[TARGET], fill=True)
plt.show()


# plot boxplot of target vs categorical features
for feature in CATEGORICAL_FEATURES:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=feature, y=TARGET, data=data)
    plt.show()

# plot boxplot of target vs numerical features
for feature in NUMERICAL_FEATURES:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=feature, y=TARGET, data=data)
    plt.xticks(rotation=90)
    plt.show()



# plot heatmap of correlation matrix
# encode categorical features
data_original_encoded = pd.get_dummies(data, columns=CATEGORICAL_FEATURES)
corr_matrix = data_original_encoded.drop(columns=["id"]).corr()
plt.figure(figsize=(16, 16))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()


def multiply(X: pd.DataFrame) -> pd.DataFrame:
    cols = X.columns
    col_name = f"{cols[0]}_x_{cols[1]}"
    col_name = "_x_".join(X.columns)
    X[col_name] = X.product(axis=1)
    return X[[col_name]]


def calculate_bmi_features(X: pd.DataFrame) -> pd.DataFrame:
    """ Calculate BMI based on weight and height and BMI category.
    Weight is in kg, height is in m.
    BMI category is in %.
    BMI is in kg/m^2.
    BMI categories:
    - 0: Underweight (BMI < 18.5)
    - 1: Normal weight (18.5 <= BMI < 25)
    - 2: Overweight (25 <= BMI < 30)
    - 3: Obesity (30 <= BMI < 35)
    - 4: Extreme obesity (BMI >= 35)
    """
    X["BMI"] = X["Weight"] / ( (X["Height"] / 100) ** 2)
    X["BMI_category"] = pd.cut(X["BMI"], bins=[0, 18.5, 25, 30, 35, 100], labels=[0, 1, 2, 3, 4])
    X["BMI_category"] = X["BMI_category"].astype(int)
    return X[["BMI", "BMI_category"]]


def calculate_caloried_burned(X: pd.DataFrame) -> pd.DataFrame:
    """ https://www.omnicalculator.com/sports/calories-burned-by-heart-rate
    """
    X["calories_burned"] = np.where(
        X["Sex"] == "male",
        X["Duration"] * (0.6309 * X["Heart_Rate"] + 0.1988 * X["Weight"] + 0.2017 * X["Age"] - 55.0969) / 4.184,
        X["Duration"] * (0.4472 * X["Heart_Rate"] - 0.1263 * X["Weight"] + 0.074 * X["Age"] - 20.4022) / 4.184
    )
    return X[["calories_burned"]]


def get_age_binned(X: pd.DataFrame) -> pd.DataFrame:
    X["age_binned"] = pd.cut(X["Age"], bins=[0, 20, 30, 40, 50, 60, 70, 100], labels=[0, 1, 2, 3, 4, 5, 6])
    X["age_binned"] = X["age_binned"].astype(int)
    return X[["age_binned"]]

def calculate_bmr(X: pd.DataFrame) -> pd.DataFrame:
    """ Calculate Basal Metabolic Rate (BMR) using the Harris-Benedict equation or Mifflin-St Jeor Equation
    Weight is in kg, height is in cm, age is in years.
    
    """
    # X["BMR"] = np.where(
    #     X["Sex"] == "male",
    #     88.362 + (13.397 * X["Weight"]) + (4.799 * X["Height"]) - (5.677 * X["Age"]),
    #     447.593 + (9.247 * X["Weight"]) + (3.098 * X["Height"]) - (4.330 * X["Age"])
    # )

    X["BMR"] = np.where(
        X["Sex"] == "male",
        10 * X["Weight"] + 6.25 * X["Height"] - 5 * X["Age"] + 5,
        10 * X["Weight"] + 6.25 * X["Height"] - 5 * X["Age"] - 161
    )
    
    return X[["BMR"]]


def calculate_bsa(X: pd.DataFrame) -> pd.DataFrame:
    """ Calculate Body Surface Area (BSA) using the DuBois formula.
    Weight is in kg, height is in cm.
    """
    X["BSA"] = 0.007184 * (X["Weight"] ** 0.425) * (X["Height"] ** 0.725)
    return X[["BSA"]]


def calculate_hr_features(X: pd.DataFrame) -> pd.DataFrame:
    """ Calculate Heart Rate features
    HR_max is in bpm, HR_percentage is in %.
    HR_zone is in %.
    """
    X["HR_max"] = 220 - X["Age"]
    X["HR_percentage"] = X["Heart_Rate"] / X["HR_max"]
    X["HR_zone"] = pd.cut(X["HR_percentage"], bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], labels=[0, 1, 2, 3, 4, 5])
    X["HR_zone"] = X["HR_zone"].astype(int)
    return X[["HR_max", "HR_percentage", "HR_zone"]]


def thermal_features(X: pd.DataFrame) -> pd.DataFrame:
    """ Create temperature-related features """
    result = pd.DataFrame(index=X.index)
    
    # Temperature deviations from normal
    normal_temp = 37.0  # Normal body temp in Celsius
    result["temp_deviation"] = X["Body_Temp"] - normal_temp
    result["temp_deviation_abs"] = np.abs(result["temp_deviation"])
    
    # Temperature categories
    result["temp_category"] = pd.cut(X["Body_Temp"], bins=[35, 36.5, 37.5, 38.5, 42], labels=[0, 1, 2, 3])  # hypothermia, normal, fever, high fever
    result["temp_category"] = result["temp_category"].astype(int)
    
    # Thermoregulation effort (higher temp deviation might indicate more energy expenditure)
    result["thermoregulation_effort"] = result["temp_deviation_abs"] * X["Duration"]
    
    return result


def get_power_to_weight_ratio(X: pd.DataFrame) -> pd.DataFrame:
    X["power_to_weight"] = X["Heart_Rate"] * X["Duration"] / X["Weight"]
    return X[["power_to_weight"]]



def get_preprocessor(random_state: Optional[int] = None) -> Pipeline:
    transformers=[
        ("Sex", OneHotEncoder(sparse_output=False), ["Sex"]),
        
        ("Age", "passthrough", ["Age"]),
        
        ("Height", "passthrough", ["Height"]),
        
        ("Weight", "passthrough", ["Weight"]),
        
        ("Duration", "passthrough", ["Duration"]),
        
        ("Heart_Rate", "passthrough", ["Heart_Rate"]),
        
        ("Body_Temp", "passthrough", ["Body_Temp"]),

        ("BMI_Features", FunctionTransformer(func=calculate_bmi_features), ["Weight", "Height"]),

        ("BMR", FunctionTransformer(func=calculate_bmr), ["Weight", "Height", "Age", "Sex"]),

        ("BSA", FunctionTransformer(func=calculate_bsa), ["Weight", "Height"]),

        ("HR_Features", FunctionTransformer(func=calculate_hr_features), ["Age", "Heart_Rate"]),

        ("Thermal_Features", FunctionTransformer(func=thermal_features), ["Body_Temp", "Duration"]),

        ("Power_to_Weight_Ratio", FunctionTransformer(func=get_power_to_weight_ratio), ["Heart_Rate", "Duration", "Weight"]),

        ("Calories_Burned", FunctionTransformer(func=calculate_caloried_burned), ["Heart_Rate", "Weight", "Age", "Sex", "Duration"]),

        ("Age_Binned", FunctionTransformer(func=get_age_binned), ["Age"]),
    ]

    preprocessor = Pipeline(
        steps=[
            ("preprocessor", ColumnTransformer(transformers=transformers)),
        ]
    )
    return preprocessor


random_state = 123
X = data.drop(columns=["Calories"])
y = data["Calories"]
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=random_state)

preprocessor = get_preprocessor(random_state=random_state)
X_train_val_p = preprocessor.fit_transform(X=X_train_val, y=y_train_val)
X_test_p = preprocessor.transform(X=X_test)
print(X_train_val_p.columns)
X_train_val_p


def get_model(random_state: Optional[int] = None):
    clf = CatBoostRegressor(
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=100,
        random_strength=1,
        loss_function='RMSE',
        task_type='GPU',
        iterations=10_000,
        early_stopping_rounds=100,
        verbose=200,
        random_state=random_state
    )
    return clf


# Log-transform y and optimise RMSE
y_train_val_log = np.log1p(y_train_val)
y_test_log = np.log1p(y_test)
model = get_model(random_state=random_state)
verbose = 100
model.fit(X=X_train_val_p, y=y_train_val_log, eval_set=(X_test_p, y_test_log))


y_pred_log = model.predict(data=X_test_p)
y_pred = np.expm1(y_pred_log)

rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
print(f"RMSE: {rmsle}")


y_pred_log = model.predict(data=X_test_p)
y_pred = np.expm1(y_pred_log)
fig, ax1 = plt.subplots(figsize=(10, 8))
PredictionErrorDisplay.from_predictions(
    y_test,
    y_pred,
    kind="actual_vs_predicted",
    ax=ax1,
    scatter_kwargs={"alpha": 0.5},
)
plt.show()

fig, ax2 = plt.subplots(figsize=(10, 8))
PredictionErrorDisplay.from_predictions(
    y_test,
    y_pred,
    kind="residual_vs_predicted",
    ax=ax2,
)
plt.show()


def k_fold(data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    target: str,
    get_preprocessor_fn: Callable,
    get_model_fn: Callable,
    n_folds: int = 5,
    random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """ K-Fold in log1p space using RMSE """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof = np.zeros(data_train.shape[0])
    y_test_hat_ave = np.zeros(data_test.shape[0])

    X = data_train.drop(columns=[target])
    y = data_train[target]
    X_test = data_test.copy()
    iter = tqdm(enumerate(kf.split(X)), total=n_folds)
    
    for i, (train_index, valid_index) in iter:
        
        X_train: pd.DataFrame = X.loc[train_index]
        y_train: pd.DataFrame = y.loc[train_index]
        y_train_log = np.log1p(y_train)
        X_valid: pd.DataFrame = X.loc[valid_index]
        y_valid: pd.DataFrame = y.loc[valid_index]
        y_valid_log = np.log1p(y_valid)

        preprocessor = get_preprocessor_fn(random_state=random_state)
        X_train_p: pd.DataFrame = preprocessor.fit_transform(X=X_train, y=y_train)
        X_valid_p: pd.DataFrame = preprocessor.transform(X=X_valid)
        X_test_p: pd.DataFrame = preprocessor.transform(X=X_test)

        clf = get_model_fn(random_state=random_state)
        
        clf.fit(X=X_train_p, y=y_train_log, eval_set=(X_valid_p, y_valid_log))
        y_valid_hat_log = clf.predict(data=X_valid_p)
        y_test_hat_log = clf.predict(data=X_test_p)
        # Out-of-fold predictions
        oof[valid_index] = np.expm1(y_valid_hat_log)
        # Test predictions
        y_test_hat_ave += np.expm1(y_test_hat_log)

        current_cv_score = np.sqrt(mean_squared_log_error(y_true=y_valid.to_numpy(), y_pred=np.expm1(y_valid_hat_log)))
        iter.set_description(f" => Fold {i+1} Score = {current_cv_score:.5f}")

    # Average test predictions
    y_test_hat_ave /= n_folds

    return y_test_hat_ave, oof


random_state = 123  

y_test_hat_ave, oof = k_fold(
        data_train=data,
        data_test=data_sub,
        target="Calories",
        n_folds=10,
        get_preprocessor_fn=get_preprocessor,
        get_model_fn=get_model,
        random_state=random_state
    )

# save y_test_hat_ave to csv
data_sub["Calories"] = y_test_hat_ave
data_sub[["id", "Calories"]].to_csv("submission.csv", index=False)

# # save oof to csv
data["Calories_Pred"] = oof
data[["id", "Calories_Pred"]].to_csv("oof.csv", index=False)


local_cv_score = np.sqrt(mean_squared_log_error(y_true=data["Calories"], y_pred=oof))
print(f"Overall CV Score = {local_cv_score:.5f}")

