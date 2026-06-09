import pandas as pd

from config import ROOT_DIR


dir_path = ROOT_DIR.joinpath('data/Kaggle/Predicting Road Accident Risk/')
train = pd.read_csv(dir_path / 'train.csv')
test = pd.read_csv(dir_path / 'test.csv')
sample_submission = pd.read_csv(dir_path / 'sample_submission.csv')


import matplotlib.pyplot as plt
import seaborn as sns


print(train['id'].min(), train['id'].max())
print(test['id'].min(), test['id'].max())


train.isna().mean()


(train == 0).mean()


def clean_labels(data):
    data = data.drop(columns=['id', 'accident_risk'])
    return data

processed_train = clean_labels(train)


processed_train.dtypes


processed_train.describe()


cond = (processed_train.dtypes == int) | (processed_train.dtypes == float)
temp = processed_train.loc[:, cond]

for i in range(len(temp.columns)):
    plt.subplot(2, 2, i+1)
    plt.title(temp.columns[i])
    plt.hist(temp.loc[:, temp.columns[i]])

plt.tight_layout()


temp.nunique()


cond = (processed_train.dtypes == object)
temp = processed_train.loc[:, cond]
temp.nunique()


temp.head()


pd.DataFrame({
    col: temp[col].unique() for col in temp.columns
})


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


raw_train = train.copy()
raw_train, raw_valid = train_test_split(raw_train, test_size=0.15, shuffle=False)
print(raw_train.shape, raw_valid.shape)


road_type_order = ["rural", "urban", "highway"]   # rural < urban < highway
lighting_order = ["daylight", "dim", "night"]     # bright -> darker
weather_order = ["clear", "foggy", "rainy"]       # good -> worse
time_order = ["morning", "afternoon", "evening"]  # time progression

road_type_map = {val: i for i, val in enumerate(road_type_order)}
lighting_map = {val: i for i, val in enumerate(lighting_order)}
weather_map = {val: i for i, val in enumerate(weather_order)}
time_map = {val: i for i, val in enumerate(time_order)}

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def prepare(data, mode):
    data = data.copy()

    drop_columns = ['id']
    if 'accident_risk' in data.columns:
        drop_columns.append('accident_risk')
    data = data.drop(columns=drop_columns)

    data["road_type_ord"] = data["road_type"].map(road_type_map)
    data["lighting_ord"] = data["lighting"].map(lighting_map)
    data["weather_ord"] = data["weather"].map(weather_map)
    data["time_of_day_ord"] = data["time_of_day"].map(time_map)

    categorical_cols = ["road_type", "lighting", "weather", "time_of_day"]
    if mode == "train":
        object_data_enc = encoder.fit_transform(data[categorical_cols])
    else:
        object_data_enc = encoder.transform(data[categorical_cols])
    object_data_enc = pd.DataFrame(object_data_enc, columns=encoder.get_feature_names_out(),
                                   index=data.index)
    data = pd.concat([data.drop(columns=categorical_cols), object_data_enc], axis=1)

    return data

x_train = prepare(raw_train, 'train')
y_train = raw_train['accident_risk'].to_numpy()
x_train


x_valid = prepare(raw_valid, 'valid')
y_valid = raw_valid['accident_risk'].to_numpy()
x_valid


import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from contexttab import ConTextTabRegressor


xgbr_model = xgb.XGBRegressor(
    max_depth=4,
    min_child_weight=5,
    gamma=1.0,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    reg_alpha=0.5,
    learning_rate=0.05,
    n_estimators=2500,
)
xgbr_model.fit(x_train, y_train)

xgbr_pred_train = xgbr_model.predict(x_train)
print("{:.4f} - Train MSE".format(mean_squared_error(y_train, xgbr_pred_train)))

xgbr_pred_valid = xgbr_model.predict(x_valid)
print("{:.4f} - Validation MSE".format(mean_squared_error(y_valid, xgbr_pred_valid)))


# base_model = xgb.XGBRegressor()
# cv = KFold(n_splits=3, shuffle=True)
#
# param_dist = {
#     "n_estimators": np.arange(1000, 3001, 500),
#     "max_depth": np.arange(3, 8, 2),
#     "learning_rate": [0.01, 0.03, 0.05, 0.1],
#     "subsample": np.arange(0.3, 0.8, 0.2),
#     "colsample_bytree": np.arange(0.4, 0.6, 0.1),
#     "min_child_weight": np.arange(3, 8, 2),
#     "gamma": np.arange(0.5, 2.1, 0.5),
#     "reg_alpha": np.arange(0.5, 3.1, 0.5),
#     "reg_lambda": np.arange(0.5, 3.1, 0.5),
# }
# search = RandomizedSearchCV(
#     base_model,
#     param_distributions=param_dist,
#     n_iter=20,
#     scoring="neg_mean_squared_error",
#     cv=cv,
#     verbose=1
# )
# search.fit(x_train, y_train)
#
# cv_score = search.best_score_
# print("{:.4f} - Best CV score".format(cv_score))
# pred_train = search.predict(x_train)
# print("{:.4f} - Train MSE".format(mean_squared_error(y_train, pred_train)))
# pred_valid = search.predict(x_valid)
# print("{:.4f} - Validation MSE".format(mean_squared_error(y_valid, pred_valid)))


cttr_model = ConTextTabRegressor(bagging=1, max_context_size=1024, test_chunk_size=256)
cttr_model.fit(x_train, y_train)

# cttr_pred_train = cttr_model.predict(x_train)
# print("{:.4f} - Train MSE".format(mean_squared_error(y_train, cttr_pred_train)))

cttr_pred_valid = cttr_model.predict(x_valid)
print("{:.4f} - Validation MSE".format(mean_squared_error(y_valid, cttr_pred_valid)))


processed_cttr_pred_valid = cttr_pred_valid.reshape(-1)
avg_pred_valid = (xgbr_pred_valid + processed_cttr_pred_valid) / 2
print("{:.4f} - Validation MSE".format(mean_squared_error(y_valid, avg_pred_valid)))


raw_test = test.copy()
x_test = prepare(raw_test, 'test')
x_test.shape


xgbr_pred_test = xgbr_model.predict(x_test)


cttr_pred_test = cttr_model.predict(x_test)


processed_cttr_pred_test = cttr_pred_test.reshape(-1)
avg_pred_test = (xgbr_pred_test + processed_cttr_pred_test) / 2


avg_pred_test_rounded = np.round(avg_pred_test, 3)
avg_pred_test_rounded


raw_sample_submission = sample_submission.copy()
raw_sample_submission['accident_risk'] = avg_pred_test_rounded
raw_sample_submission


raw_sample_submission.to_csv('submission_v02.csv', index=False)


xgbr_pred_test_rounded = np.round(xgbr_pred_test, 3)
xgbr_pred_test_rounded


raw_sample_submission = sample_submission.copy()
raw_sample_submission['accident_risk'] = xgbr_pred_test_rounded
raw_sample_submission


raw_sample_submission.to_csv('submission_v03_xgb.csv', index=False)


xgbr_model = xgb.XGBRegressor(
    min_child_weight=5,
    subsample=0.8,
    reg_lambda=5.0,
    reg_alpha=0.5,
    learning_rate=0.05,
    n_estimators=500,
)
xgbr_model.fit(x_train, y_train)

xgbr_pred_train = xgbr_model.predict(x_train)
print("{:.4f} - Train MSE".format(root_mean_squared_error(y_train, xgbr_pred_train)))

xgbr_pred_valid = xgbr_model.predict(x_valid)
print("{:.4f} - Validation MSE".format(root_mean_squared_error(y_valid, xgbr_pred_valid)))


raw_test = test.copy()
x_test = prepare(raw_test, 'test')
x_test.shape


xgbr_pred_test = xgbr_model.predict(x_test)


# xgbr_pred_test_rounded = np.round(xgbr_pred_test, 3)
# xgbr_pred_test_rounded


raw_sample_submission = sample_submission.copy()
raw_sample_submission['accident_risk'] = xgbr_pred_test
raw_sample_submission


raw_sample_submission.to_csv('submission_v04.csv', index=False)


xgb.plot_importance(xgbr_model, importance_type='gain')
plt.title('Feature importance (gain)')
plt.show()

xgb.plot_importance(xgbr_model, importance_type='weight')
plt.title('Feature importance (weight)')
plt.show()


path = './SORTED_bquxjob_569d282c_19a19b4b5ab.csv'
df = pd.read_csv(path)
df.rename(columns={'value': 'accident_risk'}, inplace=True)
df


df.to_csv('submission_v05_id_sorted.csv', index=False)

