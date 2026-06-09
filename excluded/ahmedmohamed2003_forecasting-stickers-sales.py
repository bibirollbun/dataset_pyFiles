import numpy as np 
import pandas as pd 
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


warnings.filterwarnings("ignore")
train=pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
sample=pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
train


train.info()


train.describe()


test


test.info()


test.describe()


sample


train=train.dropna(subset=['num_sold']) 
train


combined = pd.concat([train.drop(columns=['num_sold']), test], axis=0, ignore_index=True)
combined['year'] = pd.to_datetime(combined['date']).dt.year
combined['month'] = pd.to_datetime(combined['date']).dt.month
combined['day'] = pd.to_datetime(combined['date']).dt.day
combined['day_of_week'] = pd.to_datetime(combined['date']).dt.dayofweek
combined = combined.drop(columns=['date'])
train_correlation=train.copy()


for col in ['country', 'store', 'product']:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])
    train_correlation[col] = le.fit_transform(train_correlation[col])


correlation_data = train_correlation.drop(columns=['id', 'date'])

# Compute correlation matrix
correlation_matrix = correlation_data.corr()

# Display the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


X_train = combined.iloc[:len(train), :]
y_train = train['num_sold']
X_test = combined.iloc[len(train):, :]


# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Evaluate on validation set
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the model on the training split
model.fit(X_train_split, y_train_split)

# Predict on the validation set
val_predictions = model.predict(X_val)

# Calculate Mean Squared Error
mse = mean_squared_error(y_val, val_predictions)
print("Mean Squared Error:", mse)



# Predict on the test set
test_predictions = model.predict(X_test)

# Prepare submission dataframe
submission = pd.DataFrame({'id': test['id'], 'num_sold': test_predictions})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")


