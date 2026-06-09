import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
original_data = pd.read_csv("/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


print("df shape :",df.shape)
print("test_data shape :",test_data.shape)
print("original_data shape :",original_data.shape)
print("sample_submission shape :",sample_submission.shape)


df.head(5)


df.isnull().sum()


original_data.head()


original_data.isnull().sum()


original_data.dropna(subset = ['Listening_Time_minutes'], inplace = True)
original_data.shape


test_data.isnull().sum()


df = df.drop("id", axis = 1)
train_data = pd.concat([df,original_data],ignore_index = True)
print('Shape of the data: ', train_data.shape)


cols_to_fill = ['Episode_Length_minutes','Guest_Popularity_percentage','Number_of_Ads']

train_data[cols_to_fill] = train_data[cols_to_fill].fillna(0)

test_data[cols_to_fill] = test_data[cols_to_fill].fillna(0)


train_data.isnull().sum()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

cat_cols = train_data.select_dtypes(include = ['object']).columns

for col in cat_cols:
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])


train_data.head()


from sklearn.preprocessing import StandardScaler

num_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage','Guest_Popularity_percentage', 'Number_of_Ads'] 

scaler = StandardScaler()

# Thats not how we do it
# for col in num_cols:
#     train_data[col] = scaler.fit_transform(train_data[col])
#     test_data[col] = scaler.transform(test_data[col])

train_data[num_cols] = scaler.fit_transform(train_data[num_cols])
test_data[num_cols] = scaler.transform(test_data[num_cols])


train_data.head()


from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

X = train_data.drop(columns = ['Listening_Time_minutes'],axis = 1) 
y = train_data['Listening_Time_minutes']


def train(model, X, y):

    #First it is gonna split the data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)

    #Secondly Train the model
    model.fit(X_train, y_train)

    #Predict on test set
    y_pred = model.predict(X_test)

    #calculate rmse
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))

    # Cross-validation (uses negative MSE, so we take sqrt of abs
    cv_scores = cross_val_score(model, X, y, scoring= 'neg_mean_squared_error', cv = 10)
    cv_rmse = np.mean(np.sqrt(np.abs(cv_scores)))  # Convert each fold's MSE to RMSE and average

    #Print result
    print("Results")
    print("Test RMSE",rmse)
    print('CV RMSE', cv_rmse)


from sklearn.linear_model import LinearRegression
model = LinearRegression()
train(model,X, y)
coef = pd.Series(model.coef_,X.columns).sort_values(ascending = False)
coef.plot(kind = 'bar', title = 'Model Coefficients')


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


xgb_params = {
    "n_estimators": 989,
    "max_depth": 15,
    "learning_rate":0.025346012854128856,
    "subsample":0.6869548680199157,
    "colsample_bytree": 0.8595222755385405,
    "gamma":3.507353442211417,
    "reg_alpha": 1.765859825535959,
    "reg_lambda": 2.225795255374737,
    "random_state": 42,
}

cat_params={
    "learning_rate": 0.10533401645239986,
    "depth": 10,
    "l2_leaf_reg": 2.7669873709135895,
    "random_strength": 0.6608914846187091,
    "bagging_temperature": 0.5323443018992894,
    "border_count": 253
}

lgb_params={
  "learning_rate": 0.09691412796423444,
  "num_leaves": 112,
  "max_depth": 14,
  "min_child_samples": 7,
  "subsample": 0.8157612731916237,
  "colsample_bytree": 0.6905973027177817,
  "reg_alpha": 3.679516892720156,
  "reg_lambda": 0.6114371150363473
}


xgb_model = XGBRegressor(**xgb_params)
cat_model = CatBoostRegressor(**cat_params)
lgb_model = LGBMRegressor(**lgb_params)


kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((X.shape[0], 3))
test_preds = np.zeros((test_data.shape[0], 3))

X_test_final = test_data.drop(columns=["id"])

for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
    print(f"\nğŸ”� Fold {fold + 1}")

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # XGBoost
    xgb_model.fit(X_train, y_train)
    oof_preds[valid_idx, 0] = xgb_model.predict(X_valid)
    test_preds[:, 0] += xgb_model.predict(X_test_final) / kf.n_splits

    # CatBoost
    cat_model.fit(X_train, y_train)
    oof_preds[valid_idx, 1] = cat_model.predict(X_valid)
    test_preds[:, 1] += cat_model.predict(X_test_final) / kf.n_splits

    # LightGBM
    lgb_model.fit(X_train, y_train)
    oof_preds[valid_idx, 2] = lgb_model.predict(X_valid)
    test_preds[:, 2] += lgb_model.predict(X_test_final) / kf.n_splits


# ==== Train meta-model ====
meta_model = LinearRegression()
meta_model.fit(oof_preds, y)


test_data = test_data.drop(columns=['id'], axis=1)


# Make test predictions using each base model
test_preds = np.zeros((test_data.shape[0], 3))


test_preds[:, 0] = xgb_model.predict(test_data)
test_preds[:, 1] = cat_model.predict(test_data)
test_preds[:, 2] = lgb_model.predict(test_data)


final_preds = meta_model.predict(test_preds)


submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
submission['Listening_Time_minutes'] = final_preds
submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv!")

