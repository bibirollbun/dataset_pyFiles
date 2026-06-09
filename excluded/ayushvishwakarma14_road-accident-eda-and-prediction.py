import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
print(train_df.shape)
train_df.head()


train_df.isnull().sum().sum()


train_df.hist(figsize=(10,15))


sns.histplot(train_df['accident_risk'],kde=True)


for col in train_df.columns:
    if col not in ['id','curvature','accident_risk']:
        print(f'---------{col}----------')
        print(train_df.groupby(col)['accident_risk'].mean())
        print(f"Cardinality of {col} is {train_df[col].nunique()}")


# Outlier detection
for col in train_df.select_dtypes(include=['int64','float64']):
    sns.boxplot(train_df[col])
    plt.title(f'{col}')
    plt.show()


def detect_outliers_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr_value = q3 - q1
    lower_bound = q1 - 1.5 * iqr_value
    upper_bound = q3 + 1.5 * iqr_value
    print(lower_bound,upper_bound)
    return data[(data < lower_bound) | (data > upper_bound)]
outlier_index = detect_outliers_iqr(train_df['accident_risk']).index


train_df.iloc[outlier_index]  #  Statistically outliers, but contextually valid.so we don’t remove.


from sklearn.preprocessing import OrdinalEncoder
encoders = {}
categorical_vars = [
    'road_type',
    'lighting',
    'weather',
    'road_signs_present',
    'public_road',
    'time_of_day',
    'holiday',
    'school_season'
]
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train_df[categorical_vars] = encoder.fit_transform(train_df[categorical_vars])


print(encoder.categories_)


train_df


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV



X = train_df.drop(columns=['id','accident_risk'])
y = train_df['accident_risk']

X_train,X_test,y_train,y_test = X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


X_train.shape


#LinearRegression implementation
lr = LinearRegression()
lr.fit(X_train,y_train)
yp = lr.predict(X_test)
mse = mean_squared_error(y_test, yp)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, yp)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


!pip install lightgbm catboost xgboost -q


#using stacked regresser
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


estimators = [
    ('xgb', XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=42)),
    ('lgbm', LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42,verbose=-1)),
    ('cat', CatBoostRegressor(iterations=500, learning_rate=0.05, depth=8, verbose=0, random_state=42))
]

stack_model = StackingRegressor(
    estimators=estimators,
    final_estimator=RidgeCV()
)

stack_model.fit(X_train, y_train)
y_pred = stack_model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Stacking RMSE: {rmse:.4f}")
print(f"Stacking R²: {r2:.4f}")


model = stack_model
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv',on_bad_lines='skip')
test_df[categorical_vars] = encoder.transform(test_df[categorical_vars])
test_df.head()


final_df = test_df.drop(columns=['id'])
ot = model.predict(final_df)
output = pd.DataFrame({
    'id':test_df['id'],
    'accident_risk':ot
})
output.to_csv("s.csv",index=False)
print(output.head())




