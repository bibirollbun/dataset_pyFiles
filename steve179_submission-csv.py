from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd


df_train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
df_test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")


features = list(df_train.columns)
features = features[:features.index("efs")]
X_train = df_train[features]
y_train = df_train["efs"]
X_test = df_test[features]
ids = list(df_test.ID)

s = (X_train.dtypes == 'object')
categorical_cols = list(s[s].index)

s = (X_train.dtypes != 'object')
numerical_cols = list(s[s].index)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
])



# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])


# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = list(my_pipeline.predict(X_test))
#print("MAE:", mean_absolute_error(y_test, preds))
#score = cross_val_score(my_pipeline, X_test, y_test, cv=4)
#print("cross validation score:", score.mean())


output = pd.DataFrame({'ID': ids, 'prediction': preds})

output.to_csv('submission.csv', index=False)


