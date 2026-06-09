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


df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")


df.head()


# lets do all of the imports first 

from sklearn.model_selection import train_test_split, cross_val_score ,KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


df.info()


num_col = [col for col in df.columns if np.issubdtype(df[col].dtype,np.number)]


df[num_col]


cat_col = [col for col in df.columns if not np.issubdtype(df[col].dtype, np.number)]


len(cat_col)


df[cat_col]


import matplotlib.pyplot as plt
import seaborn as sns

for i in (cat_col):
    plt.figure(figsize=(6,4))
    sns.countplot(x=i,data=df,palette='pastel',edgecolor='black')
    plt.title(f'Frequency Distribution of {i}')
    plt.show()
    print('-' * 20)
    temp = df[i].value_counts()
    print(temp)
    print('-' * 20)


df.duplicated().sum()


num_col.remove('accident_risk')


print(num_col)


X = df.drop(columns=['accident_risk'])
y = df['accident_risk']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


num_transform = Pipeline(steps=[
    ('num', SimpleImputer(strategy='mean'))
])


preprocessing = ColumnTransformer(transformers=[
    ('first' , num_transform, num_col)
], remainder='passthrough')


from category_encoders import TargetEncoder


model = Pipeline(steps=[
    ('cat', TargetEncoder(cols=cat_col, smoothing=5)),
    ('preprocessing', preprocessing),
    ('model', RandomForestRegressor(n_estimators=100))
])


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


error = mean_squared_error(y_pred,y_test)


print(error)


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

r2 = r2_score(y_test, y_pred)


r2


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(mae, rmse)



scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(scores, scores.mean())


test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


test_y_pred=model.predict(test_df)


test_y_pred


import pandas as pd

# Create submission DataFrame
submission = pd.DataFrame({
    "id": test_df["id"],         # keep id column
    "accident_risk": test_y_pred         # your model predictions
})

# Save to CSV
submission.to_csv("submission.csv", index=False)

print("Submission file created!")





