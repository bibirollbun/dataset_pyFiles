import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


train_set = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_set = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


## top 5 rows in data set
train_set.head()


## Shape of Data set
print(f"Shape of Dataset is {train_set.shape}")
print(f"Number of Columns in dataset : {train_set.shape[1]}")
print(f"Number of Rows in dataset : {train_set.shape[0]}")


## Information about dataset
train_set.info()


## Removing ID column
train_set = train_set.drop(columns=['id'])
train_set.head()


## brief description of numerical and statistical structure of Dataset
train_set.describe()


null_values = train_set.isna().sum()
print(null_values)
print(f"Columns : {null_values.index}")
print(f"Values : {null_values.values}")


## splitting the data into features and labels
from sklearn.model_selection import train_test_split

X = train_set.drop(columns='Calories')
y = train_set['Calories'].to_numpy()

## splitting into training and validation dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
print(f"Training data size : {X_train.shape}")
print(f"Validation data size : {X_val.shape}")



X_train.head()


numerical_columns = X_train.select_dtypes(exclude=['object']).columns
categorical_columns = X_train.select_dtypes(include=['object']).columns
print(f"Numerical columns : {numerical_columns}")
print(f"Categorical_columns : {categorical_columns}")


## normalizing column values 
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

## transforing features
ct_features = ColumnTransformer([
    ("standardization", StandardScaler(), numerical_columns),
    ("OHE", OneHotEncoder(), categorical_columns)
])

label_standardizer = StandardScaler()

X_train_new = ct_features.fit_transform(X_train)
y_train_new = label_standardizer.fit_transform(y_train.reshape(y_train.shape[0], -1))
X_val_new = ct_features.transform(X_val)
y_val_new = label_standardizer.transform(y_val.reshape(y_val.shape[0], -1))


test_set.head()


X_test = test_set.drop(columns=['id'])
X_test_new = ct_features.transform(X_test)


from sklearn import tree
decision_tree_model = tree.DecisionTreeRegressor()

## training the model
decision_tree_model.fit(X_train_new, y_train_new)
## checking loss on validation data
val_predictions = decision_tree_model.predict(X_val_new)
val_predictions.shape


## function for scaling validation predictions
def rescaling(val_predictions):
    val_predictions_new = val_predictions.reshape(val_predictions.shape[0], -1)
    val_predictions_scaled = label_standardizer.inverse_transform(val_predictions_new)
    return val_predictions_scaled


val_predictions_scaled = rescaling(val_predictions)


from sklearn.metrics import mean_squared_error
import numpy as np


from sklearn.metrics import mean_squared_error
import numpy as np

# Compute RMSE
rmse = np.sqrt(mean_squared_error(y_val.reshape(y_val.shape[0], -1), val_predictions_scaled))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


from sklearn.ensemble import RandomForestRegressor
rn_forest_model = RandomForestRegressor(n_estimators=100, criterion='squared_error')

rn_forest_model.fit(X_train_new, y_train_new)
val_predictions = rn_forest_model.predict(X_val_new)

val_predictions_scaled = rescaling(val_predictions)
# Compute RMSE ---------------------------------------&------------------------>>>>>>>>>>>>>>>
rmse = np.sqrt(mean_squared_error(y_val.reshape(y_val.shape[0], -1), val_predictions_scaled))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}") #-----


from sklearn.ensemble import GradientBoostingRegressor
gb_model = GradientBoostingRegressor(loss='squared_error', criterion='friedman_mse', learning_rate=0.2, n_estimators=100)

gb_model.fit(X_train_new, y_train_new)
val_predictions = gb_model.predict(X_val_new)


## Making predictions over the Gradient Boosting Model
from sklearn.metrics import mean_squared_error
import numpy as np

val_predictions_scaled = rescaling(val_predictions)
# Compute RMSE -------------------------------------
rmse = np.sqrt(mean_squared_error(y_val.reshape(y_val.shape[0], -1), val_predictions_scaled))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


## submission is like this
test_preds = rn_forest_model.predict(X_test_new)
test_preds_scaled = rescaling(test_preds)
calories = pd.Series(test_preds_scaled.reshape(-1))
ids = pd.Series(test_set['id'])

df = pd.DataFrame({
    'id' : ids,
    'Calories' : calories
})

df.to_csv("submission.csv", index=False)




