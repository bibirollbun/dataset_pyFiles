import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv", index_col="id")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv", index_col="id")


X = train_df.drop("accident_risk", axis=1)
y = train_df["accident_risk"]

categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough' # Keep other columns if any (none in this case)
)


from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=10000,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    random_state=0,
                    colsample_bytree=0.9,
                    min_child_weight=3,         
                    reg_alpha=0.1,              
                    reg_lambda=1.5,             
                    gamma=0.0)


my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selector', SelectFromModel(RandomForestRegressor(random_state=42), threshold='median')),
    ('regressor', model)
])


X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42)

my_pipeline.fit(X_train, y_train)
preds = my_pipeline.predict(X_valid)
score = r2_score(y_valid, preds)
print(f"R-squared score on validation data: {score:.4f}")


test_predictions = my_pipeline.predict(test_df)

submission = pd.DataFrame({
    'id': test_df.index,
    'accident_risk': test_predictions
})

submission.to_csv('submission.csv', index=False)




