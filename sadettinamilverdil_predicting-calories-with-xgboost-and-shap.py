import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.interpolate import griddata

import shap
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
sample=pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")

print(train.shape)
print(test.shape)
print(sample.shape)


print("Train Missing Values")
print(train.isnull().sum())
print("Test Missing Values")
print(test.isnull().sum())


plt.figure(figsize=(10, 5))
sns.histplot(train["Calories"], bins=50, kde=True, color="Black")
plt.title("Calories Distribution")
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(x=train["Calories"], y=train["Sex"])
plt.title("Calories Distribution by Sex")
plt.ylabel("Sex")
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(train['Age'], kde=True, bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(data=train, x='Height', y='Weight', hue='Sex')
plt.title('Height vs Weight by Sex')
plt.show()


x = train['Age']
y = train['Duration']
z = train['Calories']

xi = np.linspace(x.min(), x.max(), 50)
yi = np.linspace(y.min(), y.max(), 50)
xi, yi = np.meshgrid(xi, yi)

zi = griddata((x, y), z, (xi, yi), method='linear')

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none')

ax.set_xlabel('Age')
ax.set_ylabel('Duration')
ax.set_zlabel('Calories')
ax.set_title('3D Surface Plot: Age & Duration vs Calories')
plt.colorbar(surf)
plt.show()


x = train['Height']
y = train['Weight']
z = train['Calories']
xi = np.linspace(x.min(), x.max(), 50)
yi = np.linspace(y.min(), y.max(), 50)
xi, yi = np.meshgrid(xi, yi)
zi = griddata((x, y), z, (xi, yi), method='linear')

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xi, yi, zi, cmap='plasma', edgecolor='none')

ax.set_xlabel('Height (cm)')
ax.set_ylabel('Weight (kg)')
ax.set_zlabel('Calories')
ax.set_title('3D Surface Plot: Height & Weight vs Calories')
plt.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


x = train['Heart_Rate']
y = train['Body_Temp']
z = train['Calories']
xi = np.linspace(x.min(), x.max(), 50)
yi = np.linspace(y.min(), y.max(), 50)
xi, yi = np.meshgrid(xi, yi)
zi = griddata((x, y), z, (xi, yi), method='linear')

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xi, yi, zi, cmap='plasma', edgecolor='none')

ax.set_xlabel('Heart Rate')
ax.set_ylabel('Body Temp')
ax.set_zlabel('Calories')
ax.set_title('3D Surface Plot: Heart Rate & Body Temp vs Calories')
plt.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


train.head()


def feature_engineering(df):
    df = df.copy()
    
    #df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Duration_per_kg'] = df['Duration'] / df['Weight']
    df['HR_Age_Ratio'] = df['Heart_Rate'] / df['Age']
    #df["Heart_Rate/Duration"]=df['Heart_Rate']/df['Duration']
    #df['HW_Ratio'] = df['Height'] / df['Weight']
    #df['HR_zscore'] = (df['Heart_Rate'] - df['Heart_Rate'].mean()) / df['Heart_Rate'].std()

    return df

train=feature_engineering(train)
test=feature_engineering(test)


RMV = ["id","Calories"]
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


combined = pd.concat([train,test],axis=0,ignore_index=True)

print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
for c in FEATURES:
    if c in CATS:
        print(f"{c}, ",end="")
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")
        combined[c] = combined[c].astype("category")
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
    
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#" * 25)
    print(f"### Fold {i+1}")
    print("#" * 25)

    x_train = train.loc[train_index, FEATURES].copy()
    y_train = np.log1p(train.loc[train_index, "Calories"])
    x_valid = train.loc[test_index, FEATURES].copy()
    y_valid = np.log1p(train.loc[test_index, "Calories"])
    x_test = test[FEATURES].copy()

    model_xgb = XGBRegressor(
        device="cuda",
        early_stopping_rounds=25,
        n_estimators=2000,
        max_depth=9,
        learning_rate=0.0097845,
        eval_metric="rmse",
        enable_categorical=True
    )

    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=200
    )

    oof_xgb[test_index] = np.expm1(model_xgb.predict(x_valid))
    pred_xgb += np.expm1(model_xgb.predict(x_test))
rmsle = np.sqrt(mean_squared_log_error(train['Calories'], oof_xgb))
print("Final RMSLE Score = ",rmsle)
pred_xgb /= FOLDS


explainer = shap.TreeExplainer(model_xgb, feature_perturbation="tree_path_dependent", model_output="raw")
shap_values = explainer.shap_values(x_test)

shap.summary_plot(shap_values, x_test)


sample["Calories"] = pred_xgb
sample.to_csv("submission.csv", index=False)
sample.head()




