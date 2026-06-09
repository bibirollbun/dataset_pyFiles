import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


data1 = pd.read_csv('../input/rohlik-sales-forecasting-challenge-v2/sales_train.csv')
data2 = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv')
data3 = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv')
data4 = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv')
data5 = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv')
data6 = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')
print("Datasets successfully imported")


print(data1.info())
print(data1.isnull().sum()) #info of train set


unique_ids = [885, 1237, 725, 3778, 5152, 2148, 2424, 3178, 1776, 1689, 612, 2809, 794]
filtered = data1[data1['unique_id'].isin(unique_ids)]
results = []
for unique_id in unique_ids:
    current_filtered = filtered[filtered['unique_id'] == unique_id]
    null_1 = current_filtered['total_orders'].isnull().sum()
    non_null_1 = current_filtered['total_orders'].notnull().sum()
    null_2 = current_filtered['sales'].isnull().sum()
    non_null_2 = current_filtered['sales'].notnull().sum()
    results.append({
        'unique_id': unique_id,
        'Null values in total_orders': null_1,
        'Non-null values in total_orders': non_null_1,
        'Null values in sales': null_2,
        'Non-null values in sales': non_null_2
    })
result_data = pd.DataFrame(results)
print(result_data)


clean1 = data1.dropna()
print(clean1.info())


print(data2.info())
print(data2.isnull().sum()) #info of test set


print(data3.info())
print(data3.isnull().sum()) #info of calender set


print(data4.info())
print(data4.isnull().sum()) #info of inventory set


print(data5.info())
print(data5.isnull().sum()) #info of solution set


print(data6.info())
print(data6.isnull().sum()) #info of test weights set


merge1 = pd.merge(clean1, data6, on='unique_id')
print("Merge successfull")


merge2 = pd.merge(merge1, data3, on=['date', 'warehouse'])
print("Merge successfull")


merge3 = pd.merge(merge2, data4, on=['unique_id', 'warehouse'])
print("Merge successfull")


merge3['date'] = pd.to_datetime(merge3['date'], errors='coerce')
merge3['day'] = merge3['date'].dt.day
merge3['month'] = merge3['date'].dt.month
merge3['year'] = merge3['date'].dt.year
print("Procedure successfull")


print(merge3.info())


merge4 = pd.merge(data2, data3, on=['date', 'warehouse'])
print("Merge successfull")


merge5 = pd.merge(merge4, data4, on=['unique_id', 'warehouse'])
print("Merge successfull")


merge5.insert(4, 'sales', [None] * len(merge5))
print("Column created successfully")


merge5.insert(6, 'availability', [None] * len(merge5))
print("Column created successfully")


merge5.insert(14, 'weight', [None] * len(merge5))
print("Column created successfully")


merge5['date'] = pd.to_datetime(merge5['date'], errors='coerce')
merge5['day'] = merge5['date'].dt.day
merge5['month'] = merge5['date'].dt.month
merge5['year'] = merge5['date'].dt.year
print("Procedure successfull")


print(merge5.info())


merge6 = pd.concat([merge3, merge5], axis=0, ignore_index=True)
print(merge6.info())
print(merge6.notnull().sum())


y =merge3.sales
features = ['warehouse','total_orders','sell_price_main']
X = merge3[features]
X = pd.get_dummies(X)


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


model = RandomForestRegressor(max_depth = 10, n_estimators = 500,random_state=1)
model.fit(train_X, train_y)
value_predictions = model.predict(val_X)
mae = mean_absolute_error(value_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))


print('The accuracy of the model is: ', model.score(val_X, val_y)) 
print('The accuracy of the training model is: ', model.score(train_X, train_y)


full_data = RandomForestRegressor(max_depth = 10, n_estimators = 500,random_state=1)
full_data.fit(X,y)
test_X = data2[features]
test_X = pd.get_dummies(test_X)
final = full_data.predict(test_X)


merge5['date'] = pd.to_datetime(merge5['date'], format='%m-%d-%Y', errors='coerce')
merge5['id'] = merge5['unique_id'].astype(str) + '_' + merge5['date'].dt.strftime('%m-%d-%Y')
output = pd.DataFrame({'id': merge5.id,
                       'sales_hat': test_preds})
output.to_csv('submission.csv', index=False)


min_length = min(len(val_y), len(y_pred))
mae = mean_absolute_error(val_y[:min_length], y_pred[:min_length])
mse = MSE(val_y[:min_length], y_pred[:min_length])
rmse = np.sqrt(mse)
r2 = r2_score(val_y[:min_length], y_pred[:min_length])

print(f"R²: {r2}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")


threshold = 0.5
y_pred_class = np.where(y_pred[:min_length] >= threshold, 1, 0)
val_y_class = np.where(val_y[:min_length] >= threshold, 1, 0)
accuracy = accuracy_score(val_y_class, y_pred_class)
f1 = f1_score(val_y_class, y_pred_class, average='weighted')
precision = precision_score(val_y_class, y_pred_class, average='weighted')
recall = recall_score(val_y_class, y_pred_class, average='weighted')
cm = confusion_matrix(val_y_class, y_pred_class)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Confusion Matrix:\n{cm}")


threshold = 0.5
y_pred_class = np.where(y_pred[:min_length] >= threshold, 1, 0)
val_y_class = np.where(val_y[:min_length] >= threshold, 1, 0)
report = classification_report(val_y_class, y_pred_class)
print(f"Classification Report:\n{report}")

