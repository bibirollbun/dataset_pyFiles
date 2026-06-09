import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


train_data = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")


train_data.head(5)


print("Number of rows: ", train_data.shape[0])
print("Number of columns: ", train_data.shape[1])



NaN_values_count = train_data.isna().sum()
print(NaN_values_count)


columns_datatypes = train_data.dtypes
print(columns_datatypes)


ax = sns.heatmap(train_data.corr(), annot=False, cmap='coolwarm', center=0)


fig, axs = plt.subplots(5, 2, figsize=(16, 14))

axs[0][0].boxplot(train_data["RhythmScore"])
axs[1][0].boxplot(train_data["AudioLoudness"])
axs[2][0].boxplot(train_data["VocalContent"])
axs[3][0].boxplot(train_data["AcousticQuality"])
axs[4][0].boxplot(train_data["InstrumentalScore"])

axs[0][1].boxplot(train_data["LivePerformanceLikelihood"])
axs[1][1].boxplot(train_data["MoodScore"])
axs[2][1].boxplot(train_data["TrackDurationMs"])
axs[3][1].boxplot(train_data["Energy"])
axs[4][1].boxplot(train_data["BeatsPerMinute"])

plt.show()


min_vals = []
max_vals = []

for col in train_data.columns:
    min_vals.append(train_data[col].min())
    max_vals.append(train_data[col].max())
    
values_range = pd.DataFrame({"col_name": train_data.columns, "min": min_vals, "max": max_vals})

values_range


def remove_outliers(df):
    for column in df.columns:        
        q_low = df[column].quantile(0.01)
        q_hi  = df[column].quantile(0.99)
        
        df = df[(df[column] <= q_hi) & (df[column] >= q_low)]

    return df


from sklearn.model_selection import train_test_split
from scipy import stats

train_data = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
# Drop outliers
train_data = remove_outliers(train_data)

y = train_data["BeatsPerMinute"]
X = train_data.drop(["BeatsPerMinute", "id"], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, train_size=0.2, random_state=43)


X_test_full = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
X_test = X_test_full.drop(["id"], axis=1)



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

transformer = ColumnTransformer(
    transformers=[
        ('numerical', StandardScaler(), X.select_dtypes(include=['float64', 'int64']).columns),
    ]
)

transformer


from lightgbm import LGBMRegressor

model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6,
    num_leaves=64,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=9,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)


from sklearn.pipeline import Pipeline

pipeline = Pipeline(
    steps=[
        ('transformer', transformer),
        ('lgbm_classifier', model)
    ]
)
pipeline


pipeline.fit(X_train,y_train)
pred = pipeline.predict(X_valid)



from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_valid, pred))
print("RMSE:", rmse)


pred = pipeline.predict(X_test)


submission = pd.DataFrame({
    "id": X_test_full["id"],
    "BeatsPerMinute": pred
})

submission.to_csv("submission.csv", index=False)
submission.head(5)

