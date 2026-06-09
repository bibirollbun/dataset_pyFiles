# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_data = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')



train_data['Personality'] = train_data['Personality'].replace({'Extrovert': 1, 'Introvert': 0})
y = train_data.pop('Personality')
y.info()



# train_data = train_data.dropna()
train_data.info()


for cols in train_data.columns:
    print(f"{cols} Contains the values : {train_data[cols].unique()}")


s = (train_data.dtypes=='object')
obj_cols = list(s[s].index)

obj_cols


from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()

encoded_train_data = train_data.copy()
encoded_test_data = test_data.copy()

encoded_train_data[obj_cols] = encoder.fit_transform(train_data[obj_cols])

encoded_test_data[obj_cols] = encoder.transform(test_data[obj_cols])


col_missing= [col for col in train_data.columns if train_data[col].isnull().any()]
print(f"The missing columns are {col_missing}")
    


from sklearn.impute import KNNImputer


impute = KNNImputer(n_neighbors=10)
imputed_train_data = pd.DataFrame(impute.fit_transform(encoded_train_data))
imputed_test_data = pd.DataFrame(impute.transform(encoded_test_data))


imputed_train_data.columns = encoded_train_data.columns
imputed_test_data.columns = encoded_test_data.columns

# Columns to apply threshold logic
binary_cols = ['Stage_fear', 'Drained_after_socializing']

# Apply binary thresholding to selected columns
for col in binary_cols:
    imputed_train_data[col] = imputed_train_data[col].apply(lambda x: 1.0 if x > 0.5 else 0.0)
    imputed_test_data[col] = imputed_test_data[col].apply(lambda x: 1.0 if x > 0.5 else 0.0)



# # Time spent alone vs. friends
# imputed_train_data['Alone_vs_Friends'] = imputed_train_data['Time_spent_Alone'] / (imputed_train_data['Friends_circle_size'] + 1e-5)

# # Social activity level
# imputed_train_data['Social_score'] = (imputed_train_data['Social_event_attendance'] + imputed_train_data['Post_frequency'] + imputed_train_data['Going_outside']) / 3

# # Time spent alone vs. friends (avoid division by zero)
# imputed_test_data['Alone_vs_Friends'] = imputed_test_data['Time_spent_Alone'] / (imputed_test_data['Friends_circle_size'] + 1e-5)

# # Social activity score
# imputed_test_data['Social_score'] = (
#     imputed_test_data['Social_event_attendance'] + 
#     imputed_test_data['Post_frequency'] + 
#     imputed_test_data['Going_outside']
# ) / 3



imputed_train_data.info()


y.nunique()



y.head()


for cols in imputed_train_data.columns:
    print(f"\nBefore Conversion:")
    print(f"{cols} Contains the values : {imputed_train_data[cols].unique()}")
    print(f"\nAfter conversion:")
    imputed_train_data[cols] = np.floor(imputed_train_data[cols]).astype(float)
    imputed_test_data[cols] = np.floor(imputed_test_data[cols]).astype(float)
    print(f"{cols} Contains the values : {imputed_train_data[cols].unique()}")


imputed_train_data.info()


# from sklearn.preprocessing import MinMaxScaler

# num_cols = [
#     'Time_spent_Alone',
#     'Social_event_attendance',
#     'Going_outside',
#     'Friends_circle_size',
#     'Post_frequency',
#     # 'Alone_vs_Friends',
#     # 'Social_score'
# ]

# scaler = MinMaxScaler()
# imputed_train_data[num_cols] = scaler.fit_transform(imputed_train_data[num_cols])
# imputed_test_data[num_cols] = scaler.transform(imputed_test_data[num_cols])




from sklearn.model_selection import train_test_split
trainX,valX,trainY,valY = train_test_split(imputed_train_data,y,train_size=0.9,random_state=2)

print(trainX.shape)
print(valX.shape)




imputed_train_data.head()


from xgboost import XGBClassifier

# model = XGBClassifier()

# n_estimators=1000,max_depth=5,learning_rate=0.04,random_state=1


# from sklearn.metrics import accuracy_score
# from xgboost import XGBClassifier
# import matplotlib.pyplot as plt
# import numpy as np  

# scores = []
# depth_values = list(range(1, 25))  

# for k in depth_values:
#     rfc = XGBClassifier(n_estimators=200, max_depth=k, learning_rate=0.01, use_label_encoder=False, eval_metric='logloss',random_state=1)
#     rfc.fit(trainX, trainY)
#     y_pred = rfc.predict(valX)
#     scores.append(accuracy_score(valY, y_pred))

# x_ticks = np.arange(1, 25, 1)  

# plt.figure(figsize=(10, 6))
# plt.plot(depth_values, scores, marker='o', linestyle='--', color='b')
# plt.xlabel('max_depth for XGBoost Classifier')
# plt.ylabel('Validation Accuracy')
# plt.title('Effect of max_depth on Accuracy')
# plt.grid(True)
# plt.xticks(x_ticks)  
# plt.tight_layout()
# plt.show()



# from sklearn.metrics import accuracy_score

# scores =[]
# for k in range(100, 2000,100):
#     rfc = XGBClassifier(n_estimators=k,max_depth=9,learning_rate=0.01,use_label_encoder=False, eval_metric='logloss',random_state=1)
#     rfc.fit(trainX, trainY)
#     y_pred = rfc.predict(valX)
#     scores.append(accuracy_score(valY, y_pred))

# import matplotlib.pyplot as plt
# %matplotlib inline

# # plot the relationship between K and testing accuracy
# # plt.plot(x_axis, y_axis)
# plt.plot(range(100, 2000, 100), scores)
# plt.xlabel('Value of n_estimators for Random Forest Classifier')
# plt.ylabel('Testing Accuracy')
# x_ticks = np.arange(100, 2000, 100)
# plt.grid(True)
# plt.xticks(x_ticks)  
# plt.tight_layout()
# plt.show()



# from sklearn.metrics import accuracy_score
# from xgboost import XGBClassifier
# import numpy as np
# import matplotlib.pyplot as plt

# scores = []
# learning_rates = np.arange(0.01, 1.01, 0.01)  

# for lr in learning_rates:
#     model = XGBClassifier(n_estimators=200, max_depth=9, learning_rate=lr, use_label_encoder=False, eval_metric='logloss',random_state=1)
#     model.fit(trainX, trainY)
#     y_pred = model.predict(valX)
#     acc = accuracy_score(valY, y_pred)
#     scores.append(acc)

# # Plotting
# plt.figure(figsize=(8, 5))
# plt.plot(learning_rates, scores, color='b')
# plt.xlabel('Learning Rate')
# plt.ylabel('Validation Accuracy')
# plt.title('Effect of Learning Rate on Accuracy')
# plt.grid(True)

# tick_positions = np.arange(0, len(learning_rates), 2)  # every 2th index
# tick_labels = [f"{learning_rates[i]:.2f}" for i in tick_positions]  # format to 2 decimal places

# plt.xticks(ticks=learning_rates[tick_positions], labels=tick_labels, rotation=45)

# plt.tight_layout()

# plt.show()




# from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from xgboost import XGBClassifier

# # Define hyperparameter grid (based on your manual ranges)
# param_grid = {
#     'max_depth': list(range(3, 11, 1)),          # same as your depth_values
#     'n_estimators': list(range(100, 1000, 100)), # sampled from your estimator range
#     'learning_rate': np.arange(0.01, 0.2, 0.02) # narrower LR range for efficiency
# }

# # Initialize model
# xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=1)

# # Cross-validation strategy
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# # Grid search with accuracy scoring
# grid_search = GridSearchCV(
#     estimator=xgb,
#     param_grid=param_grid,
#     scoring='accuracy',
#     cv=cv,
#     verbose=2,
#     n_jobs=-1
# )

# # Fit the grid search
# grid_search.fit(trainX, trainY)





# from sklearn.metrics import accuracy_score

# # Best parameters and score
# print("Best Parameters:", grid_search.best_params_)
# print("Best Cross-Validated Accuracy:", grid_search.best_score_)

# best_model = grid_search.best_estimator_

# # Evaluate on validation set
# val_pred = best_model.predict(valX)
# val_acc = accuracy_score(valY, val_pred)
# print("Validation Accuracy with Best Model:", val_acc)









model = XGBClassifier(n_estimators=100,max_depth=4,learning_rate= 0.2,use_label_encoder=False, eval_metric='logloss',random_state=1)
model.fit(trainX,trainY)


pred = model.predict(valX)


from sklearn.metrics import mean_absolute_error, accuracy_score
 

e=mean_absolute_error(valY,pred)
print(f"percentage aabsolute error: {e*100}")
accuracy_score(valY,pred)*100


# Fit on full training data and predict on test set
model.fit(imputed_train_data, y)
final_predictions = model.predict(imputed_test_data)


# Convert predictions back to labels
final_labels = np.where(final_predictions == 1, 'Extrovert', 'Introvert')



# Prepare submission
output = pd.DataFrame({
    'id': test_data['id'],
    'Personality': final_labels
})
output.to_csv('submission.csv', index=False)


df = pd.read_csv("submission.csv")
df.head()




