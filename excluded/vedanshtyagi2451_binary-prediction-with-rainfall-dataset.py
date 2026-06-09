import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df2 = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
df3 = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


print(df.head())


print(df2.head())


print(df3.head())


df.shape


print("Total NaNs in df:", df.isnull().sum().sum())
print("Total NaNs in df2:", df2.isnull().sum().sum())


# The shape is very less so we need to try just ML and nott DL
print(df.dtypes,"\n\n\n")
print(df2.dtypes)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor  # or any other model
import numpy as np


X = df.drop(['id', 'rainfall'], axis=1)
y = df['rainfall']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)


score = model.score(X_test_scaled, y_test)
print("Validation R^2 Score:", score)


X_df2 = df2.drop(['id'], axis=1)
X_df2_scaled = scaler.transform(X_df2)


X_df2 = df2.drop(['id'], axis=1)

# Fill NaNs with column-wise mean (only for numeric columns)
X_df2 = X_df2.fillna(X_df2.mean(numeric_only=True))

# Then scale using the scaler fit on training data
X_df2_scaled = scaler.transform(X_df2)

# Predict
predictions = model.predict(X_df2_scaled)


predictions = model.predict(X_df2_scaled)


submission = pd.DataFrame({
    'id': df2['id'],
    'rainfall': predictions
})
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv saved.")




