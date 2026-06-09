import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
import time


df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")

df.head()


df.describe()


df.info()


le = LabelEncoder()

df_labeled = df.copy()
categorical_cols = df_labeled.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df_labeled[col] = le.fit_transform(df_labeled[col])

df_labeled.head()


scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

df_scaled = df_labeled.copy()
df_scaled[
    [
        "Age", "Height", "Weight", 
        "Duration", "Heart_Rate", "Body_Temp"
    ]
] = scaler.fit_transform(
    df_labeled[["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]]
)
df_scaled[["Calories"]] = target_scaler.fit_transform(df_labeled[["Calories"]])

df_scaled.head()


x = df_scaled.drop(columns=["Calories", "id"])
y = df_scaled["Calories"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

y_test_unnorm = target_scaler.inverse_transform(y_test.to_numpy().reshape(-1, 1)).flatten()


models_to_evaluate = {
    "LinearRegression": LinearRegression(),
    "SVR_poly": SVR(kernel='poly', max_iter=4000),
    "SVR_rbf": SVR(kernel='rbf', max_iter=2000),
    "SVR_linear": SVR(kernel='linear', max_iter=4000),
    "LinearSVR": LinearSVR(max_iter=4000, tol=1e-3),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=4, algorithm="auto"),
    "MLPRegressor": MLPRegressor(max_iter=1000, early_stopping=True, n_iter_no_change=10, tol=1e-3),
}


total_train_time = time.perf_counter()
for model_name, model in models_to_evaluate.items():
    print(f"Start to train model {model_name}")
    train_time = time.perf_counter()
    model.fit(x_train, y_train)
    print(f"Train took {(time.perf_counter() - train_time):.4f} (s)")
print(f"Total train time {(time.perf_counter() - total_train_time):.4f}")


def rel_l1_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    return np.mean(np.abs(y_pred - y_true) / np.abs(y_true))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    return np.sqrt(mean_squared_error(y_true, y_pred))


for model_name, model in models_to_evaluate.items():
    print(f"Metrics for model {model_name}")
    y_pred = model.predict(x_test)
    y_pred_unnorm = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    print("RMSE:", rmse(y_test_unnorm, y_pred_unnorm))
    print("Relative L1 loss (%):", rel_l1_loss(y_test_unnorm, y_pred_unnorm) * 100)
    print("\n*******\n")

