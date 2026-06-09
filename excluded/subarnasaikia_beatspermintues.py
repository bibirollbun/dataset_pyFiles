from IPython.display import display
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping


TRAIN_FILE_PATH = "/kaggle/input/playground-series-s5e9/train.csv"
TEST_FILE_PATH = "/kaggle/input/playground-series-s5e9/test.csv"
SAMPLE_SUBMISSION_FILE_PATH = "/kaggle/input/playground-series-s5e9/sample_submission.csv"


train_df = pd.read_csv(TRAIN_FILE_PATH)
test_df = pd.read_csv(TEST_FILE_PATH)
sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_FILE_PATH)


def df_details(df):
    print("------------------------------------------------")
    print("df shape :", df.shape)
    print("------------------------------------------------")

    print("\n------------------------------------------------")
    print("df top 5 data :")
    print("------------------------------------------------")
    display(df.head(5))


    print("\n\n------------------------------------------------")
    print("df info :")
    print("------------------------------------------------")
    display(df.info())


    print("\n\n------------------------------------------------")
    print("df describe numeric data :")
    print("------------------------------------------------")
    display(df.describe())

    obs_cols = df.select_dtypes(include='object').columns
    if len(obs_cols) > 0:
        print("\n\n------------------------------------------------")
        print("df describe object data :")
        print("------------------------------------------------")
        display(df.describe(include=object))
    else:
        print("\n\n------------------------------------------------")
        print("No object data available :")
        print("------------------------------------------------")

    
    print("\n\n------------------------------------------------")
    print("Missing Values :")
    print("------------------------------------------------")
    print( df.isnull().sum()[df.isnull().sum() > 0] )
    
    
    missing_percentage = (df.isnull().sum() / len(df)) * 100 
    print("\n\n------------------------------------------------")
    print("Percentage of Missing values: (%) ")
    print("------------------------------------------------")
    print(missing_percentage[missing_percentage > 0])
    

    
    total_missing_percentage = (df.isnull().sum().sum() / (df.size)) * 100
    print("\n\n------------------------------------------------")
    print(f"Total missing values percentage: {total_missing_percentage:.2f}%")
    print("------------------------------------------------")


print("\n****************************************************")
print("Details of train_df: ")
print("\n****************************************************")
df_details(train_df)

print("\n\n\n\n****************************************************")
print("Details of test_df: ")
print("\n****************************************************")
df_details(test_df)


def encoding(X_train, X_valid, test):
    cat_cols = X_train.select_dtypes(include='object').columns.tolist()
    print("\n------------------------------------------------")
    print(f"Categorical columns: {cat_cols}")
    print("------------------------------------------------")

    # Fill missing values
    for df in [X_train, X_valid, test]:
        df[cat_cols] = df[cat_cols].fillna('missing')

    # Split columns by number of unique values in X_train
    onehot_cols = [col for col in cat_cols if X_train[col].nunique() <= 10]
    ordinal_cols = [col for col in cat_cols if X_train[col].nunique() > 10]

    # Encoded datasets, dropiong the categorical columns as we will encode it
    X_train_encoded = X_train.drop(cat_cols, axis=1).reset_index(drop=True)
    X_valid_encoded = X_valid.drop(cat_cols, axis=1).reset_index(drop=True)
    test_encoded = test.drop(cat_cols, axis=1).reset_index(drop=True)

    # One-Hot Encoding for categorical cols that have less than 11 unique values
    if onehot_cols:
        onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        onehot_encoder.fit(X_train[onehot_cols])

        cols = onehot_encoder.get_feature_names_out(onehot_cols)

        X_train_1hot = pd.DataFrame(onehot_encoder.transform(X_train[onehot_cols]), columns=cols)
        X_valid_1hot = pd.DataFrame(onehot_encoder.transform(X_valid[onehot_cols]), columns=cols)
        test_1hot = pd.DataFrame(onehot_encoder.transform(test[onehot_cols]), columns=cols)

        X_train_encoded = pd.concat([X_train_encoded, X_train_1hot], axis=1)
        X_valid_encoded = pd.concat([X_valid_encoded, X_valid_1hot], axis=1)
        test_encoded = pd.concat([test_encoded, test_1hot], axis=1)

    # Ordinal Encoding for categorical cols that have more than 10 unique values
    if ordinal_cols:
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        ordinal_encoder.fit(X_train[ordinal_cols])

        X_train_ord = pd.DataFrame(ordinal_encoder.transform(X_train[ordinal_cols]), columns=ordinal_cols)
        X_valid_ord = pd.DataFrame(ordinal_encoder.transform(X_valid[ordinal_cols]), columns=ordinal_cols)
        test_ord = pd.DataFrame(ordinal_encoder.transform(test[ordinal_cols]), columns=ordinal_cols)

        X_train_encoded = pd.concat([X_train_encoded, X_train_ord], axis=1)
        X_valid_encoded = pd.concat([X_valid_encoded, X_valid_ord], axis=1)
        test_encoded = pd.concat([test_encoded, test_ord], axis=1)

    return X_train_encoded, X_valid_encoded, test_encoded


def imputation(X_train, X_valid, test, strategy='mean'):
    my_imputer = SimpleImputer(strategy=strategy)
    
    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))
    imputed_test = pd.DataFrame(my_imputer.transform(test))
    
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns
    imputed_test.columns = test.columns
    
    return imputed_X_train, imputed_X_valid, imputed_test


def scaling(X_train, X_valid, test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    test_scaled = scaler.transform(test)

    X_train_data = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_valid_data = pd.DataFrame(X_valid_scaled, columns=X_valid.columns)
    test_data = pd.DataFrame(test_scaled, columns=test.columns)
    
    return X_train_data, X_valid_data, test_data


y = train_df.BeatsPerMinute
X = train_df.drop(['BeatsPerMinute', 'id'], axis=1)
test_df = test_df.drop(['id'], axis=1)

print("X shape : ", X.shape)
print("y shape : ",y.shape)

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)
print("X_train shape : ",X_train.shape)
print("y_train shape : ",y_train.shape)
print("X_valid shape : ",X_valid.shape)
print("y_valid shape : ",y_valid.shape)


print("Categorical Encoding...")
X_train , X_valid, test_df = encoding(X_train, X_valid, test_df)
print("Numerical imputation...")
X_train , X_valid, test_df = imputation(X_train, X_valid, test_df)
print("Feature Scaling...")
X_train , X_valid, test_df = scaling(X_train, X_valid, test_df)

print("X_train shape: ", X_train.shape)
print("X_valid shape: ", X_valid.shape)
print("test_df shape: ", test_df.shape)


print("\n****************************************************")
print("Details of X_train: ")
print("\n****************************************************")
df_details(X_train)

print("\n****************************************************")
print("Details of X_valid: ")
print("\n****************************************************")
df_details(X_valid)

print("\n****************************************************")
print("Details of test_df: ")
print("\n****************************************************")
df_details(test_df)


model = XGBRegressor(
    n_estimators=100000, 
    learning_rate=0.0001,
    max_depth=8,               # Adjust tree depth
    min_child_weight=2,        # Tune to control overfitting
    subsample=0.8,             # Fraction of samples used per tree
    colsample_bytree=0.6,      # Fraction of features used per tree
    reg_alpha=0.1,             # L1 regularization
    reg_lambda=1.0,            # L2 regularization
    objective='reg:squarederror',  # Suitable objective for regression
    random_state=42  
)

model.fit(X_train, y_train, 
        eval_set=[(X_valid, y_valid)],
          eval_metric='rmse',
          callbacks=[EarlyStopping(rounds=10)],
         verbose=True
    )



y_pred = model.predict(X_valid)
mae = mean_absolute_error(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")


final_prediction = model.predict(test_df).flatten()
submission = pd.DataFrame({'id': sample_submission_df['id'], 'BeatsPerMinute': final_prediction})
submission.to_csv('submission.csv', index=False)

