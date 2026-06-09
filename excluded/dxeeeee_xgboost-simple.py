import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import matplotlib.pyplot as plt


train = pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')
test = pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')


train.head()


print(list(train.columns))


train.info()


train.shape


train = train.drop_duplicates()


def add_new_features(df):
    df['Total_Distance_To_Hydrology'] = np.sqrt(df['Vertical_Distance_To_Hydrology']**2 + df['Horizontal_Distance_To_Hydrology']**2)
    df['Elevation_Plus_Vertical_Hydrology'] = df['Elevation'] + df['Vertical_Distance_To_Hydrology']
    df['Elevation_Minus_Vertical_Hydrology'] = df['Elevation'] - df['Vertical_Distance_To_Hydrology']
    df['Hydrology_Plus_Fire_Points'] = df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Fire_Points']
    df['Hydrology_Minus_Fire_Points'] = df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Fire_Points']
    df['Hydrology_Plus_Roadways'] = df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Roadways']
    df['Hydrology_Minus_Roadways'] = df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Roadways']
    df['Fire_Points_Plus_Roadways'] = df['Horizontal_Distance_To_Fire_Points'] + df['Horizontal_Distance_To_Roadways']
    df['Fire_Points_Minus_Roadways'] = df['Horizontal_Distance_To_Fire_Points'] - df['Horizontal_Distance_To_Roadways']

    df = df.drop(columns=['Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Hydrology'])

add_new_features(train)
add_new_features(test)


X = train.drop(columns=['Id', 'Cover_Type'])
y = train['Cover_Type']
X_test = test.drop(columns=['Id'])

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 42, test_size = 0.2)

# y_train = y_train-1
# y_val = y_val-1


# param_dist = {
#     'eta': [0.01, 0.05, 0.1, 0.2, 0.3],
#     'max_depth': [4, 5, 6],
#     'subsample': [0.8, 0.9, 1.0],
#     'colsample_bytree': [0.8, 0.9, 1.0]
# }

# xgb_model = xgb.XGBClassifier(
#     objective='multi:softprob',
#     num_class=7,
#     eval_metric='mlogloss',
#     n_estimators=1000,
#     early_stopping_rounds=50
# )

# random_search = RandomizedSearchCV(
#     xgb_model,
#     param_distributions=param_dist,
#     n_iter=20,  
#     cv=3,       
#     scoring='neg_log_loss',
#     verbose=2,
#     random_state=42
# )

# random_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# print("Лучшие параметры:", random_search.best_params_)


dtrain = xgb.DMatrix(X_train, label=y_train-1)
dval = xgb.DMatrix(X_val, label=y_val-1)

param = {
     'max_depth': 6,
    'eta': 0.05,
    'objective': 'multi:softmax',
    'eval_metric': 'mlogloss',
    'num_class': 7,
    'subsample': 0.8,
    'colsample_bytree': 1.0
}

num_round = 1000


model = xgb.train(param, dtrain, num_round)

train_preds = model.predict(dval) + 1


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

accuracy = accuracy_score(y_val, train_preds)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, train_preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, train_preds))


dtest = xgb.DMatrix(X_test)

test_preds = model.predict(dtest)+1

submission = pd.DataFrame({
    'Id': test['Id'],
    'Cover_Type': test_preds.astype(int)
})

submission.to_csv('submission.csv', index=False)

