# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# pip install --upgrade scikit-learn


print(sklearn.__version__)


TEST_PATH  = "/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv"
TRAIN_PATH  = "/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT.csv"

df_train = pd.read_csv(TRAIN_PATH)
df_test  = pd.read_csv(TEST_PATH)


df_train.head()


df_train.info()


#fixing feature names with a single underscore (_)
df_train.columns = df_train.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
df_test.columns = df_test.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
df_train = df_train.dropna(subset=["CORRUCYSTIC_DENSITY"])


df_train.info()


special_cols = {
    "LOCAL_IDENTIFIER": "id",
    "CORRUCYSTIC_DENSITY": "target",
    "v0rt3X": "vortex1",
    "v1rt3X": "vortex2",
    "r1Ng": "ring1",
    "r2Ng": "ring2",
    "b1oRb13": "blobie1",
    "b2oRb13": "blobie2",
    "MINDSPIKE_VERSION": "mndspke",
    "maT_r": "matter"
}

df_train = df_train.rename(columns=special_cols)
df_test  = df_test.rename(columns=special_cols)


df_train.hist(figsize=(20, 16))
plt.suptitle("histograms", fontsize=22)
plt.show()


#tried to play with the hints given at the end of tutorial notebook

df_train["ring_ratio"] = df_train["ring2"] / df_train["ring1"]
df_train["vortex_diff"]  = abs(df_train["vortex1"] - df_train["vortex2"])
df_train["blobie_sum"]   = df_train["blobie1"] + df_train["blobie2"]

df_test["vortex_diff"]  = abs(df_test["vortex1"] - df_test["vortex2"])
df_test["blobie_sum"]   = df_test["blobie1"] + df_test["blobie2"]
df_test["ring_ratio"] = df_test["ring2"] / df_test["ring1"]


 # from sklearn.preprocessing import LabelEncoder didnt use it as relatively range was too big 

def preprocess_onehot(df, is_train=True, ref_columns=None):
    df = df.copy()
    df.columns = df.columns.astype(str)

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(exclude=["object"]).columns.tolist()

    df[cat_cols] = df[cat_cols].fillna("UNK")
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    if not is_train:
        df = df.reindex(columns=ref_columns, fill_value=0)

    return df



X = df_train.drop(columns=["target", "id","vortex1", "blobie1","vortex2","ring1", "blobie2","ring2"])
X_test_final = df_test.drop(columns=["id","vortex1", "blobie1","vortex2","ring1", "blobie2","ring2"])
y = df_train["target"]


X = preprocess_onehot(X, is_train=True)

X_test_final = preprocess_onehot(X_test_final, is_train=False, ref_columns=X.columns)



from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.1, random_state=42
)

X_train = pd.DataFrame(X_train.values, columns=X.columns.astype(str))
X_valid = pd.DataFrame(X_valid.values, columns=X.columns.astype(str))



from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor



rf_model = RandomForestRegressor(
    n_estimators=500,       
    max_depth=20,           
    max_features="sqrt",    
    n_jobs=-1,
    random_state=42
)

rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
mae  = mean_absolute_error(y_valid, y_pred)

print(f"Validation RMSE: {rmse:.4f}")
print(f"Validation MAE:  {mae:.4f}")

rf_model.fit(X, y)


y_test_pred = rf_model.predict(X_test_final)

submission = pd.DataFrame({
    "LOCAL_IDENTIFIER": df_test["id"].astype(int),
    "CORRUCYSTIC_DENSITY": y_test_pred.astype(float)
})

submission.to_csv("submission.csv", index=False)
print(submission.head())

