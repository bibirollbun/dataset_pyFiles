import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from xgboost import XGBRegressor


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")


def rmse(y_obs, y_pred) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) between observed and predicted values.

    Args:
        y_obs: Observed values (numpy array or pandas Series).
        y_pred: Predicted values (numpy array or pandas Series).

    Returns:
        float: The root mean squared error.

    Raises:
        ValueError: If inputs have incompatible shapes.
    """
    y_obs = np.asarray(y_obs)  # Convert to numpy array
    y_pred = np.asarray(y_pred)
    
    if y_obs.shape != y_pred.shape:
        raise ValueError(f"Input shapes must match, got {y_obs.shape} and {y_pred.shape}")
    
    return float(np.sqrt(np.mean((y_obs - y_pred) ** 2)))


print("Train Set Shape: ", train.shape)
print("Test Set Shape: ", test.shape)


print(train.info())


print(test.info())


train.sample(10)


train.describe()


sns.histplot(data = train, x = "RhythmScore", bins = 25)
plt.show()


sns.histplot(data = train, x = "AudioLoudness", bins = 25)
plt.show()


sns.histplot(data = train, x = "VocalContent", bins = 25)
plt.show()


sns.histplot(data = train, x = "AcousticQuality", bins = 25)
plt.show()


sns.histplot(data = train, x = "InstrumentalScore", bins = 25)
plt.show()


sns.histplot(data = train, x = "LivePerformanceLikelihood", bins = 25)
plt.show()


sns.histplot(data = train, x = "MoodScore", bins = 25)
plt.show()


sns.histplot(data = train, x = "TrackDurationMs", bins = 25)
plt.show()


sns.histplot(data = train, x = "Energy", bins = 25)
plt.show()


sns.histplot(data = train, x = "BeatsPerMinute", bins = 25)
plt.show()


X = train.drop(columns = ["id", "BeatsPerMinute"])
y = train["BeatsPerMinute"]


reg = XGBRegressor(random_state = 19)


reg.fit(X, y)


X_test = test.drop(columns = ["id"])


predictions = reg.predict(X_test)
print(predictions)


sample_sub.head()


df_sub = pd.DataFrame({"id": test["id"], "BeatsPerMinute": predictions})
df_sub.head()


df_sub.to_csv("submission.csv", index = False)




