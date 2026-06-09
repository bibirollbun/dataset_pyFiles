import numpy as np 
import pandas as pd 
from scipy.stats import iqr 
from scipy import stats
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import  mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/predicting-the-price-of-diamond/train.csv")
test = pd.read_csv("/kaggle/input/predicting-the-price-of-diamond/test.csv")
tr_df = pd.read_csv("/kaggle/input/predicting-the-price-of-diamond/train.csv")


train["volume"] = train["x"] * train["y"] * train["z"]
test["volume"] = test["x"] * test["y"] * test["z"]
train = train.drop(columns=(["id", "x", "y", "z"]))
test = test.drop(columns=(["id", "x", "y", "z"]))

train.head()


train.head()


train.describe()


test.head()


test.describe()


print(f"{' Sum Null Training Dataset ':=^50}")
print(train.isna().sum())
print("="*50)
print(f"{' Sum Null Testing Dataset ':=^50}")
print(test.isna().sum())


from sklearn.preprocessing import OrdinalEncoder

# Define orders
cut_order = [["Fair", "Good", "Very Good", "Premium", "Ideal"]]
color_order = [["D", "E", "F", "G", "H", "I", "J"]]
clarity_order = [["I3","I2","I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF","FL"]]

# Initialize encoders (fit only on train!)
cut_encoder = OrdinalEncoder(categories=cut_order, handle_unknown="use_encoded_value", unknown_value=-1)
color_encoder = OrdinalEncoder(categories=color_order, handle_unknown="use_encoded_value", unknown_value=-1)
clarity_encoder = OrdinalEncoder(categories=clarity_order, handle_unknown="use_encoded_value", unknown_value=-1)

# Fit on train, transform both train and test
train["cut_encoded"] = cut_encoder.fit_transform(train[["cut"]])
test["cut_encoded"] = cut_encoder.transform(test[["cut"]])

train["color_encoded"] = color_encoder.fit_transform(train[["color"]])
test["color_encoded"] = color_encoder.transform(test[["color"]])

train["clarity_encoded"] = clarity_encoder.fit_transform(train[["clarity"]])
test["clarity_encoded"] = clarity_encoder.transform(test[["clarity"]])

# Drop original columns
train = train.drop(columns=["cut","color","clarity"])
test = test.drop(columns=["cut","color","clarity"])

print(f"{' TRAIN DATA ':=^50}")
print(train.head())
print()
print(f"{' TEST DATA ':=^50}")
print(test.head())



test.describe()


for i, col in enumerate(train.select_dtypes(include=[np.number]).columns):
    Q1 = np.quantile(train[col], 0.25)
    Q3 = np.quantile(train[col], 0.75)
    iqr = Q3 - Q1
    
    lower_threshold = Q1 - 1.5 * iqr
    upper_threshold = Q3 + 1.5 * iqr
    
    outliers = train[(train[col] < lower_threshold) | (train[col] > upper_threshold)]
    inliers = train[(train[col] >= lower_threshold) & (train[col] <= upper_threshold)]
    print("")
    print(f"{i+1}: {col} ")
    print(f"The inlier shape of {col}: {inliers.shape[0]}")
    print(f"The Range of {col} : {lower_threshold, upper_threshold}")
    print(f"The outlier shape of {col} : {outliers.shape[0]}")
    print("")
    
    ## Visulation the handilling
    plt.figure(figsize=(6,3))
    sns.boxplot(x=train[col], data= train)
    plt.title(f"Boxplot of {col}")
    plt.show()

    plt.figure(figsize=(5.23,3))
    sns.histplot(x=train[col], data= train)
    plt.title(f"Boxplot of {col}")
    plt.show()
    
    print("="*50)


# xyz_cols = train[["x","y","z"]]

for col in train:
    Q1 = np.quantile(train[col], 0.25)
    Q3 = np.quantile(train[col], 0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Outliers
    outliers= train[(train[col] < lower) | (train[col] > upper)][col]
    train = train[(train[col] >= lower) & (train[col] <= upper)]

    print(f"Outliers in {col}:")
    print(outliers.values)  
    print(outliers.shape[0])    
    print(train.shape[0])   
    print("-"*40)


train.describe()


test.describe()


test.shape


train.shape


train.corr()


plt.figure(figsize=(12, 8))
sns.heatmap(train.corr(), linewidths=1, annot= True)
plt.show()


corr_with_price = train.corr()["price"].drop("price")  # نحذف السعر نفسه

# Bar plot
plt.figure(figsize=(8,5))
sns.barplot(x=corr_with_price.index, y=corr_with_price.values)
plt.title("Correlation of each feature with Price")
plt.ylabel("Correlation coefficient")
plt.xticks(rotation=45)
plt.show()


sns.pairplot(data= train, vars= ["carat", "depth", "table"])


# sns.pairplot(data= train, vars= ["x", "y", "z"])


sns.pairplot(data= train, vars= ['cut_encoded','color_encoded', 'clarity_encoded'])


cols = ['cut', 'color', 'clarity']
fig, axes = plt.subplots(len(cols), 4, figsize=(14, 12))

for i, col in enumerate(cols):
    # Histogram
    sns.histplot(
        data=tr_df,
        x="price",
        hue=col,
        multiple="stack",
        ax=axes[i, 0]
    )
    axes[i, 0].set_title(f"{col} - Histogram")

    # KDE plot
    sns.kdeplot(
        data=tr_df,
        x="price",
        hue=col,
        common_norm=False,
        ax=axes[i, 1]
    )
    axes[i, 1].set_title(f"{col} - KDE")
    
    # KDE 2 plot
    sns.kdeplot(
        data=tr_df,
        x="price",
        hue=col,
        common_norm=False,
        cut=0,
        ax=axes[i, 2]
    )
    axes[i, 2].set_title(f"{col} - KDE2")

    # KDE 3 plot
    sns.kdeplot(
        data=tr_df,
        x="price",
        hue=col,
        common_norm=False,
        cut=0,
        cumulative=True,
        ax=axes[i, 3]
    )
    axes[i, 3].set_title(f"{col} - KDE3")
plt.tight_layout()
plt.show()



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

target_col = "price"  
X = train.drop(columns=[target_col])
y = train[target_col]

# validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# List of Parameters hyperparameters 
param_grids = {
    "Ridge": {"alpha": [0.1, 1.0, 10]},
    "Lasso": {"alpha": [0.001, 0.01, 0.1]},
    "RandomForest": {"n_estimators": [100, 200], "max_depth": [None, 5, 10]},
    "AdaBoost": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1, 0.5]},
    "GradientBoosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [1, 3, 5]},
    "XGBoost": {"n_estimators": [100,200], "learning_rate": [0.05,0.1], "max_depth": [3,5]},
    "LightGBM": {"n_estimators": [100,200], "learning_rate": [0.05,0.1], "max_depth": [-1,5]},
    "CatBoost": {"iterations": [100,200], "learning_rate": [0.05,0.1], "depth": [4,6]}
}

# The Models
base_models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, verbosity=0),
    "LightGBM": LGBMRegressor(random_state=42),
    "CatBoost": CatBoostRegressor(random_seed=42, verbose=0)
}

best_models = {}

# Train a models with GridSearchCV
for name, model in base_models.items():
    print(f"Training {name}...")
    if name in param_grids:
        grid = GridSearchCV(model, param_grids[name], cv=3, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print(f"Best params for {name}: {grid.best_params_}")
    else:
        model.fit(X_train, y_train)
        best_model = model
    
    best_models[name] = best_model
    
    # validation
    val_preds = best_model.predict(X_val)
    mse = mean_squared_error(y_val, val_preds)
    r2 = r2_score(y_val, val_preds)
    print(f"{name} => Validation MSE: {mse:.2f}, R2: {r2:.4f}\n")

# Stacking Regressor
stacking_reg = StackingRegressor(
    estimators=[(name, mdl) for name, mdl in best_models.items()],
    final_estimator=RandomForestRegressor(n_estimators=100, random_state=42),
    n_jobs=-1
)
stacking_reg.fit(X, y)
stacking_preds = stacking_reg.predict(test)

# Voting Regressor
voting_reg = VotingRegressor(
    estimators=[(name, mdl) for name, mdl in best_models.items()],
    n_jobs=-1
)
voting_reg.fit(X, y)
voting_preds = voting_reg.predict(test)

# # Save
# submission = pd.DataFrame({
#     "id": range(20000, 20000 + len(stacking_preds)),  
#     "price": stacking_preds  
# })

# submission.to_csv("submission.csv", index=False)
# print("✅ Submission file saved correctly as submission.csv (Ids start from 20000)")



from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Choose a Best Model
strong_models = {name: mdl for name, mdl in best_models.items() 
                 if name in ["RandomForest", "GradientBoosting", "XGBoost", "LightGBM", "CatBoost"]}

# Improved Stacking
stacking_reg = StackingRegressor(
    estimators=[(name, mdl) for name, mdl in strong_models.items()],
    final_estimator=GradientBoostingRegressor(n_estimators=100, random_state=42),
    n_jobs=-1
)


stacking_reg.fit(X, y)

stacking_preds = stacking_reg.predict(test)

# Validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
stacking_reg.fit(X_train, y_train)
stacking_val_preds = stacking_reg.predict(X_val)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_val, stacking_val_preds)
r2 = r2_score(y_val, stacking_val_preds)
print(f"Improved Stacking => Validation MSE: {mse:.2f}, R2: {r2:.4f}")

# Save submission
submission = pd.DataFrame({
    "id": range(20000, 20000 + len(stacking_preds)),
    "price": stacking_preds
})
submission.to_csv("submission.csv", index=False)
print("✅ Improved Stacking submission saved as submission_improved.csv")



submission

