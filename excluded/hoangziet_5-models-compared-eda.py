import os 

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# RMSE 
from sklearn.metrics import mean_squared_error

def root_mean_squared_error(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse


df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df.head()


LE = LabelEncoder()
df["Sex"] = LE.fit_transform(df["Sex"])


df.head()


sex_count = df["Sex"].value_counts()
sex_count


plt.figure(figsize = (6, 4))
sns.boxplot(x = df["Age"])
plt.title("Box plot of Age")
plt.show()


corr = df.corr()

plt.figure(figsize = (8, 7))
sns.heatmap(corr, annot = True)

plt.title("Correlation matrix heatmap of dataset")
plt.show()


fig, axes = plt.subplots(ncols = 3, nrows = 1, figsize = (12, 4))
axes = axes.ravel()
best_features = ['Duration', 'Heart_Rate', 'Body_Temp']

for i, feature in enumerate(best_features):
    sns.regplot(x = df[feature], y = df["Calories"], ax = axes[i])
    axes[i].set_title(f'{feature} vs Calories')

plt.tight_layout()
plt.show()


X, y = df.drop(columns = ["id", "Calories"], axis = 1), df["Calories"]
X.shape, y.shape


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet


model_names = ["LinearRegression", "ElasticNet", "MLPRegressor", "XGBRegressor", "LightBGM"]
instances = [LinearRegression(),
             ElasticNet(),
             MLPRegressor(tol = 1e-2, early_stopping = True), 
             XGBRegressor(),
             LGBMRegressor()]
RMSEs = []


for i in range(5):
    instance = instances[i]
    name = model_names[i]
    print(f"Fitting model {name}...")

    # Fit the dataset
    instance.fit(X_train, y_train)

    # Get predictions
    y_pred = instance.predict(X_test)

    # Get RMSE and append to list
    rmse = root_mean_squared_error(y_test, y_pred)
    RMSEs.append(rmse)
    
    print(f"RMSE: {rmse}")
    print()


plt.figure(figsize = (8, 6))
sns.set(style="whitegrid")

sns.lineplot(x=range(len(RMSEs)), y=RMSEs, marker="o", linewidth=2, markersize=8)
for i, rmse in enumerate(RMSEs):
    plt.text(i, rmse + 0.2, model_names[i], ha='center', fontsize=10, fontweight='bold')

plt.title("RMSE Comparison Between Models")



my_xgb = instances[3]


test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
test_df.head()


test_df["Sex"] = LE.fit_transform(test_df["Sex"])


submission_dict = {}

# get id
submission_dict["id"] = test_df["id"]

# get features
test_X = test_df.drop(["id"], axis = 1)

# get prediction
test_pred = my_xgb.predict(test_X)
submission_dict["Calories"] = test_pred

print(submission_dict)


submission_df = pd.DataFrame(submission_dict)
submission_df.to_csv('submission.csv', index=False)

submission_df.head()

