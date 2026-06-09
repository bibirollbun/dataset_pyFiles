import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")

train_df.head()


test_df.head()


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np

X = train_df.drop(columns=["id", "Listening_Time_minutes", "Podcast_Name", "Episode_Title"])
y = train_df["Listening_Time_minutes"]
X_test = test_df.drop(columns=["id", "Podcast_Name", "Episode_Title"])

categorical_cols = ['Genre', 'Publication_Day', 'Episode_Sentiment']
numeric_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']

imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

# Train data
X_numeric = X[numeric_cols]
X_numeric_imputed = imputer.fit_transform(X_numeric)
X_numeric_df = pd.DataFrame(X_numeric_imputed, columns=numeric_cols, index=X.index)

X_cat = X[categorical_cols]
X_cat_imputed = cat_imputer.fit_transform(X_cat)
X_cat_encoded = encoder.fit_transform(X_cat_imputed)
encoded_col_names = encoder.get_feature_names_out(categorical_cols)
X_cat_df = pd.DataFrame(X_cat_encoded, columns=encoded_col_names, index=X.index)

# Test data
X_test_numeric = X_test[numeric_cols]
X_test_numeric_imputed = imputer.fit_transform(X_test_numeric)
X_test_numeric_df = pd.DataFrame(X_test_numeric_imputed, columns=numeric_cols, index=X_test.index)

X_test_cat = X_test[categorical_cols]
X_test_cat_imputed = cat_imputer.fit_transform(X_test_cat)
X_test_cat_encoded = encoder.fit_transform(X_test_cat_imputed)
encoded_col_names = encoder.get_feature_names_out(categorical_cols)
X_test_cat_df = pd.DataFrame(X_test_cat_encoded, columns=encoded_col_names, index=X_test.index)

X_test_final = pd.concat([X_test_numeric_df, X_test_cat_df], axis=1)
X_final = pd.concat([X_numeric_df, X_cat_df], axis=1)
X_final.head()


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
# Evaluate the model using RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')


predicted = model.predict(X_test_scaled)


t_id = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")["id"]
predicted = pd.Series(predicted)
submission_df = pd.concat([t_id, predicted], axis=1)
submission_df.columns = ['id', 'Listening_Time_minutes']
submission_df.to_csv("submission.csv", index=False)

