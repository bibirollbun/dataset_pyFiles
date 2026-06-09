import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


train_df = pd.read_csv('/kaggle/input/aviachipta-narxini-bashorat-qilish/train_data.csv')
test_df = pd.read_csv('/kaggle/input/aviachipta-narxini-bashorat-qilish/test_data.csv')
sample_solution_df = pd.read_csv('/kaggle/input/aviachipta-narxini-bashorat-qilish/sample_solution.csv')


train_df.head()


test_df.head()


train_df.info()


# Checking unique categories for all categorical columns
categorical_columns = [
    "airline", "flight", "source_city", "departure_time", "stops",
    "arrival_time", "destination_city", "class"
]

for col in categorical_columns:
    print(f"Column: {col}")
    print(train_df[col].value_counts())  # Show category counts
    print("-" * 50)


x = train_df.drop(['id', 'price', 'flight'], axis=1)
y = train_df['price']


numeric_col = ['duration', 'days_left']
str_col = ["airline", "source_city", "departure_time", "stops",
    "arrival_time", "destination_city", "class"]


numeric_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])

full_pipeline = ColumnTransformer([
    ('num', numeric_pipeline, numeric_col),
    ('cat', OneHotEncoder(handle_unknown='ignore'), str_col)
])


train_pre = full_pipeline.fit_transform(x)


RF_model = RandomForestRegressor(random_state=42)
RF_model.fit(train_pre, y)


test_pre = full_pipeline.fit_transform(test_df)


predict = RF_model.predict(test_pre)
predict


np.maximum(predict, 0)


# Saving solution
sample_solution_df = pd.DataFrame({
    'id': test_df['id'],
    'price': predict
})


# Save to CSV
sample_solution_df.to_csv("submission.csv", index=False)




