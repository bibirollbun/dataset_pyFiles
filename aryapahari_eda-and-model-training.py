# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd 
data_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
data_test  = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


data_train.head()


data_train.columns


plt.figure(figsize=(6, 4))
data_train['Sex'].value_counts().plot(kind='bar', title='Sex Distribution')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.grid(True)
plt.show()


data_train["Age"].plot(kind="hist", bins=20, title="Age Distribution")


data_train.isna().sum()


data_train.isnull().sum()


data_test.isna().sum()


def features_add(data):
    data['BMI'] = data['Weight'] / ((data['Height'] / 100)** 2)
    data["Duration_Heart_Rate"] = data["Duration"] * data["Heart_Rate"]
    data["Duration_Temp"] = data["Duration"] * data["Body_Temp"]
    data["HR_Temp_Interaction"] = data["Heart_Rate"] * data["Body_Temp"]
    data["Weight_Age_Ratio"] = data["Weight"] / data["Age"]
    data["HR_Temp_Ratio"] = data["Heart_Rate"] / data["Body_Temp"]
    data['Weight_sq'] = data['Weight']**2
    data['Height_sq'] = data['Height']**2
    data['Duration_sq'] = data['Duration']**2
    data['Heart_Rate_sq'] = data['Heart_Rate']**2
    
    # data["Temp_per_Minute"] = data["Body_Temp"] / data["Duration"]
    return data
    
    # Heart Rate to Body Temperature Ratio
    # # Temperature per Minute
    
    # # Heart Rate * Temperature (interaction)



data_train = features_add(data_train)


data_train.head()


num_cols= data_train.select_dtypes(exclude=['object']).columns.tolist()


for col in num_cols[1:-1]:
    plt.figure(figsize=(12, 4))
    plt.scatter(data_train[col], data_train['Calories'], alpha=0.5)
    plt.title(f'{col} vs Calories')
    plt.xlabel(col)
    plt.ylabel('Calories')
    plt.grid(True)
    plt.show()


# sns.pairplot(data_train[num_cols[1:]])
# plt.show()


corr_matrix = data_train[num_cols[1:]].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation Heatmap of Numerical Variables')
plt.show()


# Since Weight and height are not much co related so we can remove them 
new_train_data = data_train.drop(columns = ["id"])


new_train_data.head()


gender = {value : key for key , value in enumerate(new_train_data["Sex"].unique())}
gender


new_train_data["Sex"] = new_train_data["Sex"].replace(gender)


new_train_data


new_train_data.iloc[1:,:-1].skew()


eda_num_col = new_train_data.columns[1:-1]
for col in eda_num_col:
    sns.histplot(data_train[col], kde=True, bins=30)
    plt.title(f"Normal Distribution of {col}")
    plt.xlabel(f"{col}")
    plt.ylabel("Frequency")
    plt.show()


fig, axes = plt.subplots(6, 3, figsize=(10, 8))
axes = axes.flatten() 
for i, col in enumerate(eda_num_col):
    sns.boxplot(y=data_train[col], ax=axes[i])
    axes[i].set_title(f'Boxplot of {col}')
    axes[i].set_ylabel(col)
    axes[i].grid(True)

plt.tight_layout()
plt.show()


# def remove_outliers(numerical_col, col:str):
#     numerical_col_25_quantile = numerical_col[col].quantile(0.25) 
#     numerical_col_75_quantile = numerical_col[col].quantile(0.75)
#     iqr = numerical_col_75_quantile - numerical_col_25_quantile 
#     upper_limit = numerical_col_75_quantile + 1.5 * iqr
#     lower_limit = numerical_col_25_quantile - 1.5 * iqr
#     df_filtered = numerical_col[(numerical_col[col] >= lower_limit) & (numerical_col[col] <= upper_limit)]
#     return df_filtered


columns_to_process = [
    "Heart_Rate",
    "Body_Temp",
    "Weight",
    "BMI",
    "Duration_Heart_Rate",
    "Duration_Temp",
    "Weight_Age_Ratio",
    "HR_Temp_Interaction",
    "HR_Temp_Ratio",
    'Heart_Rate_sq',
    'Duration_sq',
    'Height_sq',
    'Weight_sq'
]

processed_data = new_train_data.copy()

for col in columns_to_process:
    # processed_data = remove_outliers(processed_data, col)
    processed_data[col] = np.log1p(processed_data[col])



processed_data.skew()


# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# new_train_data_outlier_removed = new_train_data_outlier_removed.reset_index(drop=True)
# standard_scale_columns = new_train_data_outlier_removed[new_train_data_outlier_removed.columns[1:-1]]
# scaled_col = scaler.fit_transform(standard_scale_columns)
# new_train_data_outlier_removed[new_train_data_outlier_removed.columns[1:-1]] = pd.DataFrame(scaled_col, columns=new_train_data_outlier_removed.columns[1:-1] , index = new_train_data_outlier_removed.index)


# new_train_data_outlier_removed.head()


# new_train_data_outlier_removed.isna().sum()


new_train_data.isna().sum()


# new_train_data_outlier_removed


from sklearn.model_selection import train_test_split
X = processed_data.drop(columns = ["Calories"])
y = processed_data ["Calories"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X


y


from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from catboost import CatBoostRegressor
import numpy as np

catboost_model = CatBoostRegressor(
    iterations=2200,
    learning_rate=0.01,
    depth=12,
    l2_leaf_reg=1.5,
    random_strength=0.5,
    subsample=0.8,
    loss_function='RMSE',
    # task_type='GPU',
    # devices='0',
    bootstrap_type='Bernoulli',
    random_seed=42,
    verbose=0
)


scoring = make_scorer(mean_squared_error, greater_is_better=False)
xgb_param_grid = {
    "n_estimators": [1800 , 2100 , 2200,2300],
    "max_depth" : [None , 12 , 13 , 14],
    "learning_rate" :[0.01 , 0.02 , 0.03 , 0.001]
}

xgb_base = XGBRegressor(
    tree_method="hist",
    # device="cuda",
    predictor='gpu_predictor',
    objective='reg:squarederror',
    random_state=42,
    verbosity=0,
    n_jobs=-1,
    n_estimators= 1800,              
    learning_rate= 0.01,                    
    max_depth= 14,                             
    min_child_weight= 5,                        
    subsample= 0.8,                      
    colsample_bytree= 0.8,               
    reg_alpha= 0.5,                        
    reg_lambda= 1.5 
)


xgb_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=xgb_param_grid,
    n_iter=25,  #
    scoring=make_scorer(mean_squared_error, greater_is_better=False),
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)
lr = LinearRegression()
log_reg = LogisticRegression()


models = {
    "XGBoost_GPU": {"model": xgb_base},
    "CatBoost_GPU": {"model": catboost_model},
    "LinearRegression": {"model":lr},
    "LogisticRegression":{"model":log_reg}
}


cv = KFold(n_splits=15, shuffle=True, random_state=42)
results = []

for name, config in models.items():
    model = config["model"]
    
    r2_scores = cross_val_score(model, X, y, scoring="r2", cv=cv)
    mean_r2 = np.mean(r2_scores)

    rmse_scores = cross_val_score(
        model, X, y,
        scoring=scoring, cv=cv
    )
    mean_rmse = np.sqrt(-np.mean(rmse_scores)) 
    model.fit(X, y)
    y_pred = model.predict(X)
    test_r2 = r2_score(y, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y, y_pred))

    results.append((name, mean_r2, mean_rmse, test_r2, test_rmse))

    print(f"Model: {name}")
    print(f"  CV R2: {mean_r2:.4f}, CV RMSE: {mean_rmse:.4f}")
    print(f"  Test R2: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}\n")






# Model: XGBoost_GPU
#   CV R2: 0.9967, CV RMSE: 3.5809
#   Test R2: 0.9967, Test RMSE: 3.5662


# Model : LinearRegression, R2 : 0.9676492662826234, Root mean square 11.135663987116365
# Model : Ridge, R2 : 0.9676492662683525, Root mean square 11.135663989572507
# Model : Lasso, R2 : 0.9639339435406382, Root mean square 11.757727197651876
# Model : DecisionTree, R2 : 0.9921252519877004, Root mean square 5.494047161409782
# Model : RandomForest, R2 : 0.9941489965264855, Root mean square 4.735755020510084
# Model : GradientBoosting, R2 : 0.9931859458375747, Root mean square 5.110658254638627
# Model : KNN, R2 : 0.9941418839959809, Root mean square 4.738632558697928
# Model : XGBoost, R2 : 0.9949152119364666, Root mean square 4.414794353226736
# Model : AdaBoost, R2 : 0.9437329113828765, Root mean square 14.68592387068887
#               Model  R2_Score       RMSE
# 7           XGBoost  0.994915   4.414794
# 4      RandomForest  0.994149   4.735755
# 6               KNN  0.994142   4.738633
# 5  GradientBoosting  0.993186   5.110658
# 3      DecisionTree  0.992125   5.494047
# 0  LinearRegression  0.967649  11.135664
# 1             Ridge  0.967649  11.135664
# 2             Lasso  0.963934  11.757727
# 8          AdaBoost  0.943733  14.685924





data_test = features_add(data_test)

new_test_data = data_test.drop(columns=["id"])
new_test_data["Sex"] = new_test_data["Sex"].replace(gender)


columns_to_log_transform = [columns_to_process]
for col in columns_to_log_transform:
    new_test_data[col] = np.log1p(new_test_data[col])



new_test_data


y_pred = models["CatBoost_GPU"]["model"].predict(new_test_data)


cat_preds = models["CatBoost_GPU"]["model"].predict(new_test_data)
xgb_preds = models["XGBoost_GPU"]["model"].predict(new_test_data)
ensemble_preds = (0.5 * cat_preds + 0.5 * xgb_preds)


ensemble_preds


y_pred 


(ensemble_preds< 0).sum()





cat_preds = models["CatBoost_GPU"]["model"].predict(new_test_data)
xgb_preds = models["XGBoost_GPU"]["model"].predict(new_test_data)
ensemble_preds = (0.4 * cat_preds + 0.6 * xgb_preds)


ensemble_preds


sub_file = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
sub_file["Calories"] = ensemble_preds
sub_file.to_csv("submission.csv" , index = False)
















