# Data handling and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# preprocessing and data transformation
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline

# Model selection and evaluation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")


df.head()


df.duplicated().sum()


from sklearn.model_selection import train_test_split

# Separate the features and the target variable
X = df.drop('accident_risk', axis=1)  # Features (all columns except 'accident_risk')
y = df['accident_risk']               # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Verify the dimensions of the resulting splits
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
test_df.shape


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure size and DPI
plt.figure(figsize=(15, 3), dpi=150)

# Create the boxplot
sns.boxplot(x=df['accident_risk'])

# Set the plot orientation
plt.xlabel('accident_risk')

# Show the plot
plt.show()



X_train.head()


X_train.shape


X_train.dtypes.value_counts()


X_train.describe()


X_train.isnull().sum()


missing_values = X_train.isnull().sum()

missing_values = missing_values[missing_values > 0]

missing_percentage = (missing_values / len(X_train)) * 100

print(missing_percentage)


columns_to_drop = missing_percentage[missing_percentage > 30].index

X_train.drop(columns=columns_to_drop, inplace=True)

print(X_train)


columns_to_fill = missing_percentage[missing_percentage < 30].index
columns_to_fill


for column in columns_to_fill:
    if X_train[column].dtype == "float64" or X_train[column].dtype == "int64":
        X_train[column].fillna(X_train[column].mean(), inplace=True)
    else:

        X_train[column].fillna(X_train[column].mode()[0], inplace=True)
X_train.head()


X_train.isnull().sum().sum()


X_train.id.nunique()


X_train.drop(columns=['id'], inplace=True)


num_data = X_train.select_dtypes(include=np.number).columns.tolist()


cat_data = X_train.select_dtypes(exclude=np.number).columns.tolist()


corr = X_train[num_data].corr()
plt.subplots(1,1, figsize=(12,8))
sns.heatmap(data=corr, cmap ='Greens', annot = corr, cbar=None)
plt.title('Correlation between features')
plt.show()


fig, ax = plt.subplots(len(num_data), 1, figsize=(6, 50))
for axs, feature in zip(ax, num_data):
    X_train[[feature]].boxplot(ax=axs)


y_train.value_counts()


X = df.drop('accident_risk',axis=1)
y = df['accident_risk']


num_pipe = make_pipeline(SimpleImputer(strategy='mean'),MinMaxScaler())
cat_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'),
                         OneHotEncoder(handle_unknown='ignore'))
preprocessor = ColumnTransformer([
    ('num_pipe',num_pipe,num_data),
    ('cat_pipe',cat_pipe,cat_data)
])
preprocessor


def prediction(model):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return f"{model}'s RMSE is {np.sqrt(mean_squared_error(y_test, y_pred))}, MAE is {mean_absolute_error(y_test, y_pred)}, R2 is {r2_score(y_test, y_pred)}"


prediction(LinearRegression())


prediction(DecisionTreeRegressor(random_state=42))


prediction(RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=-1))


prediction(GradientBoostingRegressor(random_state=42))


prediction(RidgeCV())


prediction(ElasticNetCV())


prediction(LassoCV())


best_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=4, random_state=42)
prediction(best_model)


test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
id = test_df['id'].copy()
test_df = test_df.drop('id', axis=1)
test_preprocessed = preprocessor.transform(test_df)
predictions = best_model.predict(test_preprocessed)
predictions_df = pd.DataFrame({
    'id': id, 
    'accident_risk': predictions
})
predictions_df.to_csv('/kaggle/working/submission.csv', index=False)

print("Doneï¼Œresults saved into /kaggle/working/submission.csv")


submission= pd.DataFrame(data = {'id': range(517754,690339),
                                'accident_risk': predictions})
submission


predictions_df.head()


predictions_df.tail()




