import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



#import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
plt.style.use('ggplot')
pd.set_option('display.max_columns', 200)


# Selección de las variables por tipo
# ==============================================================================
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_selector


#load dataset train and test
df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


df_train.head()


print("the shape of the train dataset:",df_train.shape)
print("the shape of the test dataset:",df_test.shape)


df_train.dtypes


df_train.columns


print("Quantity not unique values Publication_Time :",df_train['Publication_Time'].nunique())
print("Quantity not unique values Genre :",df_train['Genre'].nunique())
print("Quantity not unique values Publication_Day :",df_train['Publication_Day'].nunique())
print("Quantity not unique values Episode_Sentiment :",df_train['Episode_Sentiment'].nunique())


df_train.describe()


# check missing values
print(df_train.isna().sum())
print('======================================')
print(df_test.isna().sum())


#Remove columns 
df_train.drop(['Episode_Title','id'],axis = 1 , inplace = True)
df_test.drop(['Episode_Title','id'],axis = 1 , inplace = True)


df_train['Episode_Sentiment'].unique()


X = df_train.drop(columns="Listening_Time_minutes")
y = df_train["Listening_Time_minutes"]


numeric_train = X.select_dtypes(include=['float64', 'int']).columns.to_list()
cat_train = X.select_dtypes(include=['object', 'category']).columns.to_list()

numeric_test = df_test.select_dtypes(include=['float64', 'int']).columns.to_list()
cat_test = df_test.select_dtypes(include=['object', 'category']).columns.to_list()

# Transformaciones para las variables numéricas
numeric_transformer = Pipeline(
                        steps=[
                            ('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler())
                        ]
                      )


# Transformaciones para las variables categóricas
categorical_transformer = Pipeline(
                            steps=[
                                ('imputer', SimpleImputer(strategy='most_frequent')),
                                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                            ]
                          )

preprocessor_train = ColumnTransformer(
                    transformers=[
                        ('numeric', numeric_transformer, numeric_train),
                        ('cat', categorical_transformer, cat_train)
                    ],
                    remainder='passthrough',
                    verbose_feature_names_out = False
               ).set_output(transform="pandas")

preprocessor_test = ColumnTransformer(
                    transformers=[
                        ('numeric', numeric_transformer, numeric_test),
                        ('cat', categorical_transformer, cat_test)
                    ],
                    remainder='passthrough',
                    verbose_feature_names_out = False
               ).set_output(transform="pandas")



df_train_prep = preprocessor_train.fit_transform(X)
df_test_prep  = preprocessor_test.fit_transform(df_test)


df_train_prep.head(3)


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(df_train_prep, y, test_size=0.2, random_state=42)



from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_valid)

rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print(f"Validation RMSE: {rmse:.4f}")



X_test = df_test_prep.copy()
y_test_pred = model.predict(X_test)


df_subm = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


df_subm['Listening_Time_minutes'] = y_test_pred 
df_subm.to_csv('submission.csv', index=False)


