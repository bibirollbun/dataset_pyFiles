import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer


train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
sam_data = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train_data.head(3)


train_data.isna().sum()


test_data.isna().sum()


Categorical_columns = train_data.select_dtypes(include=['object', 'category']).columns
def mapping_to_num(data, column):
    mapping = {category: idx for idx, category in enumerate(data[column].astype('category').cat.categories)}
    data[column] = data[column].map(mapping)
    return data, mapping
mappings_dict = {}
columns_to_map = Categorical_columns
for col in columns_to_map:
    train_data, mapping = mapping_to_num(train_data, col)
    mappings_dict[col] = mapping


for col in columns_to_map:
    test_data, mapping = mapping_to_num(test_data, col)
    mappings_dict[col] = mapping


train_data.head(3)


train_data.isna().sum()


imputer = KNNImputer(n_neighbors=5)
train_imputed = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)


train_imputed.head(5)


train_imputed.isna().sum()


test_imputed = pd.DataFrame(imputer.fit_transform(test_data), columns=test_data.columns)


test_imputed.isna().sum()


X = train_imputed.drop('Price', axis=1)
y = train_imputed['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


prediction = model.predict(test_imputed)


sam_data['Price']= prediction
sam_data['id'] = test_data['id']
sam_data.to_csv('submission.csv',index=False)


sam_data.head()

