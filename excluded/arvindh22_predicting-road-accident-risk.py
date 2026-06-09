import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


train_data = pd.read_csv('train.csv')


train_data.shape


train_data.head()


train_data.isnull().sum()


train_data.dtypes


plt.figure(figsize=(5, 3))
sns.histplot(train_data['accident_risk'], kde=True, bins=30)
plt.title("Value Distribution")
plt.show()


results = []
for col in train_data.columns:
  if (train_data[col].dtypes == 'int') | (train_data[col].dtypes == 'float'):
    pearson,_ = stats.pearsonr(train_data[col],train_data['accident_risk'])
    spearman,_ = stats.spearmanr(train_data[col],train_data['accident_risk'])
    mutual_info = mutual_info_regression(train_data[[col]],train_data['accident_risk'])[0]
    results.append([col, pearson, spearman, mutual_info])
corr_data = pd.DataFrame(results, columns = ['Feature','Pearson', 'Spearman', 'Mutual Information Regression'])


corr_data


results = []
for col in train_data.columns:
  if train_data[col].dtypes == 'bool':
    point_biserial,_ = stats.pointbiserialr(train_data[col],train_data['accident_risk'])
    results.append([col, point_biserial])
corr_data = pd.DataFrame(results, columns = ['Feature','Point Biserial'])


corr_data


def correlation_ratio(categories, values):
    categories = pd.Categorical(categories)
    cat_means = values.groupby(categories).mean()
    n = values.groupby(categories).size()
    numerator = sum(n * (cat_means - values.mean())**2)
    denominator = sum((values - values.mean())**2)
    return np.sqrt(numerator / denominator)


results = []
for col in train_data.columns:
  if train_data[col].dtypes == 'object':
    corr_ratio = correlation_ratio(train_data[col],train_data['accident_risk'])
    results.append([col, corr_ratio])
corr_data = pd.DataFrame(results, columns = ['Feature','Correlation Ratio'])


corr_data


filtered_data = train_data[['curvature', 'speed_limit', 'lighting', 'weather', 'num_reported_accidents', 'holiday', 'accident_risk']]


filtered_data.head()


lighting_map = {'daylight':0, 'dim':1,'night':2}
filtered_data['lighting'] = filtered_data['lighting'].map(lighting_map)


weather_map = {'clear':0, 'rainy':1,'foggy':2}
filtered_data['weather'] = filtered_data['weather'].map(weather_map)


holiday_map = {False:0, True:1}
filtered_data['holiday'] = filtered_data['holiday'].map(holiday_map)


filtered_data.head()


X = filtered_data.drop('accident_risk', axis=1)
y = filtered_data['accident_risk']


scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


X.head()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


(X_train.shape, y_train.shape), (X_test.shape, y_test.shape)


algos = {"Decision Tree": {"model": DecisionTreeRegressor(),"params": {"criterion": ["mse", "friedman_mse"],"splitter": ["best", "random"],
            "max_depth": [1, 3, 5, 7, 9, 10, 11, 12, 14, 15, 18, 20, 25, 28, 30, 33, 38, 40],"min_samples_split": [2, 4, 6, 8, 10, 15, 20],
      "min_samples_leaf": [i for i in range(1, 11)],"max_leaf_nodes": [None] + [i for i in range(10, 91, 10)],"max_features": ["auto", "log2", "sqrt", None]}},

         "Random Forest": {"model": RandomForestRegressor(),"params": {"n_estimators": [100, 200, 300],"criterion": ["mse", "friedman_mse"],
            "max_depth": [1, 3, 5, 7, 9, 10, 11, 12, 14, 15, 18, 20, 25, 28, 30, 33, 38, 40],"min_samples_split": [2, 4, 6, 8, 10, 15, 20],
      "min_samples_leaf": [i for i in range(1, 11)],"max_leaf_nodes": [None] + [i for i in range(10, 91, 10)],"max_features": ["auto", "log2", "sqrt", None]}},

         "Ada Boost": {"model": AdaBoostRegressor(),"params": {"n_estimators": [100, 200, 300],"learning_rate": np.arange(0.1, 1, 0.01),
                                                               "loss": ['linear', 'square', 'exponential']}},

         "Gradient Boost": {"model": GradientBoostingRegressor(),"params": {"learning_rate": np.arange(0.1, 1, 0.01),"n_estimators": [100, 200, 300],
            "criterion": ['friedman_mse', 'mse'],"min_samples_split": [2, 4, 6, 8, 10, 15, 20],"min_samples_leaf": [i for i in range(1, 11)],
            "max_depth": [1, 3, 5, 7, 9, 10, 11, 12, 14, 15, 18, 20, 25, 28, 30, 33, 38, 40],"max_features": ["auto", "log2", "sqrt", None],
            "max_leaf_nodes": [None] + [i for i in range(10, 91, 10)],"alpha": np.arange(0.1, 1, 0.01)}}
         }


best_model = {}
best_model_details = []

for model_name, values in algos.items():
    rscv = RandomizedSearchCV(values["model"], values["params"], cv=5, n_iter=5, random_state=42)
    rscv.fit(X_train, y_train)
    best_model[model_name] = rscv
    best_model_details.append({"Model Name": model_name, "Best Score": rscv.best_score_, "Best Parameters": rscv.best_params_})
    print(model_name)


pd.set_option('display.max_colwidth', None)
pd.DataFrame(best_model_details)


test_model = []

for model_name, model in best_model.items():
    test_model.append({"Model Name": model_name, "Test Score": model.score(X_test, y_test)})

pd.DataFrame(test_model)


base_models = {"DecisionTree": DecisionTreeRegressor(splitter="best", min_samples_split=6, min_samples_leaf=2,max_leaf_nodes=10, max_features=None, max_depth=7,
                                                     criterion="friedman_mse", random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_split=15, min_samples_leaf=2,max_leaf_nodes=40, max_features=None, max_depth=25,
                                          criterion="friedman_mse", random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, loss="linear", learning_rate=0.4, random_state=42),
    "GradientBoost": GradientBoostingRegressor(n_estimators=300, min_samples_split=10, min_samples_leaf=10,max_leaf_nodes=10, max_features=None, max_depth=9,learning_rate=0.74,
                                               criterion="friedman_mse", alpha=0.8, random_state=42)}


kf = KFold(n_splits=5, shuffle=True, random_state=42)


oof_train = np.zeros((X_train.shape[0], len(base_models)))
oof_test = np.zeros((X_test.shape[0], len(base_models)))


for i, (name, model) in enumerate(base_models.items()):
    test_preds_folds = []
    print(f"Training {name}...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)
        oof_train[val_idx, i] = model.predict(X_val)
        test_preds_folds.append(model.predict(X_test))

    oof_test[:, i] = np.mean(test_preds_folds, axis=0)

oof_train_df = pd.DataFrame(oof_train, columns=base_models.keys())
oof_test_df = pd.DataFrame(oof_test, columns=base_models.keys())


meta_model = Ridge(alpha=1.0, random_state=42)
meta_model.fit(oof_train_df, y_train)


oof_preds_meta = meta_model.predict(oof_train_df)
stacking_mse = mean_squared_error(y_train, oof_preds_meta)
stacking_rmse = np.sqrt(stacking_mse)
print(f"Stacking CV RMSE: {stacking_rmse:.5f}")


final_preds = meta_model.predict(oof_test_df)


final_preds = np.clip(final_preds, 0, 1)


final_preds


test_data = pd.read_csv('test.csv')


test_data.isnull().sum()


test_data.dtypes


model_test_data = test_data[['curvature', 'speed_limit', 'lighting', 'weather', 'num_reported_accidents', 'holiday']]


lighting_map = {'daylight':0, 'dim':1,'night':2}
model_test_data['lighting'] = model_test_data['lighting'].map(lighting_map)


weather_map = {'clear':0, 'rainy':1,'foggy':2}
model_test_data['weather'] = model_test_data['weather'].map(weather_map)


holiday_map = {False:0, True:1}
model_test_data['holiday'] = model_test_data['holiday'].map(holiday_map)


model_test_data = pd.DataFrame(scaler.transform(model_test_data), columns=model_test_data.columns)


for name, model in base_models.items():
    print(f"Training {name} on full training data...")
    model.fit(X_train, y_train)


oof_test_final = pd.DataFrame()

for name, model in base_models.items():
    oof_test_final[name] = model.predict(model_test_data)


meta_model = Ridge(alpha=1.0, random_state=42)
meta_model.fit(oof_train_df, y_train)


final_preds = meta_model.predict(oof_test_final)


final_preds = np.clip(final_preds, 0, 1)


submission = pd.DataFrame({"id": test_data["id"],"accident_risk": final_preds})


submission.to_csv("submission.csv", index=False)

