from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd 
from xgboost import XGBRegressor


# ------------- PANDAS SETUP -------------
pd.set_option("display.max_columns", None)
pd.set_option("future.no_silent_downcasting", True)

# ------------- LOAD DATA -------------
TRAIN = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
TEST = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

# ------------- GLOBAL VARIABLES -------------
ID_LIST = TEST["id"]
DF = pd.DataFrame
SCALER = MinMaxScaler()


def main(train: DF, test: DF) -> None:

    # ------------- Preprocess Data -------------
    train = preprocessing(train, True)
    test = preprocessing(test, False)
    
    # ------------- Features -------------
    X, y = test.columns.tolist(), "loan_paid_back"

    X_train, X_val, y_train, y_val = train_test_split(
        train[X], train[y], test_size=0.2, random_state=42
    )

    # ------------- Fit Model -------------
    model = XGBRegressor(
            eval_metric="auc",
            n_estimators=10000,
            learning_rate=0.01,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=1.5,
            reg_lambda=0.5,
            random_state=42,
            early_stopping_rounds=100
            )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
        )

    # ------------- Make Output File -------------
    final = pd.DataFrame({"id": ID_LIST,
                        "loan_paid_back": model.predict(test[X])})
    final.to_csv("submission.csv", index=False)
    print("----------------------")
    print("Submission is created!")
    print("----------------------")
    print(final.head())


def preprocessing(data: DF, is_train: bool) -> DF:

    # ------------- One-Hot Encoding -------------
    data = pd.get_dummies(data=data, columns=["employment_status"])
    
    # ------------- Binary Encoding -------------
    data.replace([True, False], [1, 0], inplace=True)

    # ------------- Ordinal Encoding -------------
    mode_education = data[data["education_level"] != "Other"]["education_level"].mode()[0]
    data["education_level"] = data["education_level"].replace({"Other": mode_education})
    data["education_level"] = data["education_level"].map({"High School": 0,
                                                        "Bachelor\'s": 1,
                                                        "Master\'s": 2,
                                                        "PhD": 3})

    # ------------- Clear Data -------------
    columns = ["id", "marital_status", "loan_purpose", "grade_subgrade",
            "gender", "employment_status_Retired", "employment_status_Self-employed"]
    data = data.drop(columns=columns)

     # ------------- Scaling Data -------------
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'loan_paid_back' in numeric_columns:
        numeric_columns.remove('loan_paid_back')
    if is_train:
        data[numeric_columns] = SCALER.fit_transform(data[numeric_columns])
    else:
        data[numeric_columns] = SCALER.transform(data[numeric_columns])

    # ------------- Return DataFrame -------------
    return data.astype(float)


if __name__ == "__main__":
    main(TRAIN, TEST)




