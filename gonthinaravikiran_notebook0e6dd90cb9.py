import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


car_dataset= pd.read_csv('/kaggle/input/car-becho-paisa-paao/train.csv')


car_dataset.head()


car_dataset.shape


car_dataset.info()


car_dataset.isnull().sum()


print(car_dataset.fuel_type.value_counts())
print(car_dataset.transmission.value_counts())


import pandas as pd
from sklearn.preprocessing import LabelEncoder

car_dataset = pd.read_csv('/kaggle/input/car-becho-paisa-paao/train.csv')
categorical_cols = car_dataset.select_dtypes(include=['object']).columns

label_encoder = LabelEncoder()
for col in categorical_cols:
    car_dataset[col] = label_encoder.fit_transform(car_dataset[col])

car_dataset.head()


X=car_dataset.drop(['id','price'],axis=1)
Y=car_dataset['price']


print(X)


print(Y)


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.01,random_state=2)


lin_reg_model=LinearRegression()


lin_reg_model.fit(X_train,Y_train)


training_data_prediction=lin_reg_model.predict(X_train)


from sklearn import metrics
import numpy as np

rmse = np.sqrt(metrics.mean_squared_error(Y_train, training_data_prediction))
print("Root Mean Squared Error:", rmse)


import pandas as pd

# Load test data
test_data = pd.read_csv('/kaggle/input/car-becho-paisa-paao/test.csv')

# Preprocess test data (convert categorical columns to numeric using label encoding)
from sklearn.preprocessing import LabelEncoder

categorical_cols = test_data.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()

for col in categorical_cols:
    test_data[col] = label_encoder.fit_transform(test_data[col].astype(str))

# Make predictions
test_predictions = lin_reg_model.predict(test_data.drop(columns=['id']))

# Prepare the submission DataFrame
submission = pd.DataFrame({
    'id': test_data['id'],
    'price': test_predictions
})

# Save to CSV
submission.to_csv('submission.csv', index=False)




