# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


TRAIN_PATH = "/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT.csv"
TEST_PATH = "/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv"
SPEC_PATH = "/kaggle/input/recruitment-task-for-gdsc-ml/SPECIMEN.csv"

df = pd.read_csv(TRAIN_PATH)


!pip install --upgrade scikit-learn


import sklearn
print(sklearn.__version__)


df.dropna(subset="CORRUCYSTIC_DENSITY", inplace=True)


cats = df.select_dtypes(include=['object', 'string']).columns.tolist()
nums = df.select_dtypes(include=[np.number]).columns.drop('CORRUCYSTIC_DENSITY').tolist()


target = "CORRUCYSTIC_DENSITY"
preds = df.groupby("maT_r")[target].transform("mean")

rmse = np.sqrt(((df[target] - preds)**2).mean())
print("RMSE just from maT_r category mean:", rmse)


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

km = KMeans(n_clusters=2, random_state=42)
clusters = km.fit_predict(df[[target]])

df['cluster'] = clusters

for col in nums[:5]:  
    plt.scatter(df[col], df[target], c=df['cluster'], alpha=0.3)
    plt.xlabel(col)
    plt.ylabel("CORRUCYSTIC_DENSITY")
    plt.show()


from sklearn.model_selection import train_test_split
X, y = df[nums], df['CORRUCYSTIC_DENSITY']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)


from sklearn.metrics import root_mean_squared_error as rmse
y_pred = model.predict(X_test)
print(f"RMSE = {rmse(y_test, y_pred):.4f}")


model.fit(X, y)

df_unk = pd.read_csv(TEST_PATH)
X_unk = df_unk[nums]
y_unk_pred = model.predict(X_unk)

submission = pd.DataFrame({
    'LOCAL_IDENTIFIER': df_unk['LOCAL_IDENTIFIER'],
    'CORRUCYSTIC_DENSITY': y_unk_pred
})

submission['LOCAL_IDENTIFIER'] = submission['LOCAL_IDENTIFIER'].astype(int)
submission['CORRUCYSTIC_DENSITY'] = submission['CORRUCYSTIC_DENSITY'].astype(float)

submission.to_csv('submission.csv', index=False)
print(submission.head())

