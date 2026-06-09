import pandas as pd


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train.sample(5)


test.sample(5)


for col in train.columns:
    print(col, f"\t\t\t unique: {len(train[col].unique())}")

for col in test.columns:
    print(col, f"\t\t\t unique: {len(test[col].unique())}")


from sklearn.preprocessing import LabelEncoder

for col in train.columns:
    if train[col].dtype == 'object':
        le = LabelEncoder()

        all_values = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(all_values)

        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

        print(f"{col} \t\t\t unique: {len(le.classes_)}")



train


test


target = "Listening_Time_minutes"


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
sns.heatmap(train.corr()[target].drop(target).to_frame(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar=True)
plt.title("Correlation Heatmap of Features")
plt.show()


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

scaler = MinMaxScaler()

X_temp, X_test, y_temp, y_test = train_test_split(
    train.drop(columns=[target]), train[target], test_size=0.2, random_state=42
)

# X_temp = scaler.fit_transform(X_temp)
# X_test = scaler.transform(X_test)

all_features = [col for col in train.columns if col != target]
X_temp = pd.DataFrame(X_temp, columns=all_features)
X_test = pd.DataFrame(X_test, columns=all_features)


# X_train, X_test, y_train, y_test = train_test_split(
#     X_temp, y_train, test_size=0.2, random_state=42  # 20% của 80% = 16%
# )

# print("Shape of X_train:", X_train.shape)
# print("Shape of X_val:", X_val.shape)
# print("Shape of X_test:", X_test.shape)


import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import numpy as np

kf = KFold(n_splits=5, shuffle=True, random_state=42)


rmse_list = []

models = []
cnt = 0


for train_index, val_index in kf.split(X_temp):  # X_temp là dữ liệu đã chuẩn hóa
    X_train_fold, X_val_fold = X_temp.iloc[train_index], X_temp.iloc[val_index]
    y_train_fold, y_val_fold = y_temp.iloc[train_index], y_temp.iloc[val_index]
    
    model = xgb.XGBRegressor(
        tree_method='hist',
        max_depth=11,
        colsample_bytree=0.6,
        subsample=0.8,
        n_estimators=50_000,
        learning_rate=0.04,
        early_stopping_rounds=100,
        min_child_weight=10,
        device="cuda"
    )

    model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=500)

    
    predictions_test = model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, predictions_test))
    print(f"RMSE trên tập test cho fold {cnt}: {rmse_test}")
    
    cnt += 1
    rmse_list.append(rmse_test)  
    models.append(model)


mean_rmse_test = np.mean([rmse for rmse in rmse_list])
print(f"Average RMSE trên tập test: {mean_rmse_test}")


test_predictions = []
for model in models:
    pred = model.predict(test)
    test_predictions.append(pred)

test_predictions = np.mean(test_predictions, axis=0)



test_predictions


test_predictions.shape


submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
submission


submission[target] = test_predictions


submission


test.shape


submission.to_csv("sub1.csv", index=False)




