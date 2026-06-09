!pip install --upgrade scikit-learn==1.3
# Necessary to use TargetEncoder 
import sklearn
print(f"scikit-learn version: {sklearn.__version__}")


import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer




df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


# Here we are seperating columns as these will be handy when encoding columns
cat_cols = df_train.select_dtypes(include=['object', 'category']).columns
num_cols = df_train.select_dtypes(exclude=['object', 'category']).columns

cat_cols


df_train.isna().sum()


from sklearn.model_selection import train_test_split

target = "Listening_Time_minutes"

X = df_train.drop(columns=[target], axis=1)
y = df_train[target]

X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.25, random_state=42)


from sklearn.compose import ColumnTransformer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import TargetEncoder
from sklearn.pipeline import Pipeline

imputer = ColumnTransformer([
    ("num_imputer", IterativeImputer(), ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']),
], remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas') # enables ouput as pandas dataframe

encoder = ColumnTransformer([
    ("target_enc", TargetEncoder(), cat_cols)
], remainder='passthrough')

pipe = Pipeline([
    ("imp", imputer),
    ("enc", encoder), 
])

pipe.fit(X_train, y_train)

X_train_processed = pipe.transform(X_train) 
X_validate_processed = pipe.transform(X_validate)
X_test_processed = pipe.transform(df_test)


# View Transformed dataset
X_train_processed


# LightGBM 

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

lg = LGBMRegressor()
lg.fit(X_train_processed, y_train)

y_pred_lg = lg.predict(X_validate_processed)
rmse_lg = np.sqrt(mean_squared_error(y_validate, y_pred_lg))

print("RMSE:", rmse_lg)


# Final Submission

y_test_lg = lg.predict(X_test_processed)

df_submit = pd.DataFrame(data={"id" : df_test['id'], "Listening_Time_minutes" : y_test_lg})
df_submit.to_csv("submission.csv", index=False)

