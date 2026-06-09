import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder

from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

blue_shade = plt.cm.Blues(0.8)



df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df.head()


df = df.drop(columns=['id'])

cols = list(df.columns)
cols.insert(0, cols.pop(cols.index('Calories')))
df = df[cols]


df.duplicated().sum()


df.isnull().sum()


df['Height_m'] = df['Height'] / 100
df['BMI'] = df['Weight'] / (df['Height_m'] ** 2)
df['BMI_zscore'] = df.groupby('Sex')['BMI'].transform(lambda x: (x - x.mean()) / x.std())
df.drop(['Height_m','BMI'], axis=1, inplace=True)
df["Intensity"] = df["Duration"] * df["Heart_Rate"]


corr = df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


df.describe()


features = ['Age', 'Duration', 'Heart_Rate', 'Body_Temp','BMI_zscore','Intensity']



plt.figure(figsize=(8, 6))
df['Sex'].value_counts().plot(
    kind='bar',
    color=blue_shade,
    edgecolor=None
)
plt.xlabel('Sex')
plt.ylabel('Count')
plt.title('Sex')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


for feature in features:
    plt.figure(figsize=(8, 5))
    
    sns.histplot(df[feature], kde=True, bins=30, color=blue_shade)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()


for feature in features:
    plt.figure(figsize=(8, 6))
    plt.hexbin(df[feature], df['Calories'], gridsize=40, cmap='Blues', bins='log')
    plt.colorbar(label='log10(N)')
    plt.xlabel(feature)
    plt.ylabel('Calories')
    plt.title(f'Hexbin Plot of {feature} vs Calories')
    plt.tight_layout()
    plt.show()


SEED = 42
FOLDS = 5

train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


train["Height_m"] = train["Height"] / 100
test["Height_m"] = test["Height"] / 100
train["BMI"] = train["Weight"] / (train["Height_m"]**2)
test["BMI"] = test["Weight"] / (test["Height_m"]**2)

mean_std = train.groupby("Sex")["BMI"].agg(["mean", "std"]).reset_index()
train = train.merge(mean_std, on="Sex", how="left")
test = test.merge(mean_std, on="Sex", how="left")
train["BMI_zscore"] = (train["BMI"] - train["mean"]) / train["std"]
test["BMI_zscore"] = (test["BMI"] - test["mean"]) / test["std"]

train["Intensity"] = train["Duration"] * train["Heart_Rate"]
test["Intensity"] = test["Duration"] * test["Heart_Rate"]


X = train[["Sex", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "BMI_zscore", "Intensity"]].copy()
y = np.log1p(train["Calories"])
X_test = test[["Sex", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "BMI_zscore", "Intensity"]].copy()


oof = np.zeros(len(X))
preds = np.zeros(len(X_test))


kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    te = TargetEncoder(cols=["Sex"])
    X_train["Sex"] = te.fit_transform(X_train["Sex"], y_train)
    X_val["Sex"] = te.transform(X_val["Sex"])
    X_test_enc = X_test.copy()
    X_test_enc["Sex"] = te.transform(X_test_enc["Sex"])

    model_lgb = LGBMRegressor(
        learning_rate=0.0998,
        num_leaves=85,
        max_depth=9,
        min_child_samples=10,
        subsample=0.607,
        colsample_bytree=0.684,
        reg_alpha=2.95,
        reg_lambda=4.11,
        n_estimators=1000,
        random_state=SEED,
        verbosity=-1
    )

    model_lgb.fit(X_train, y_train)
    pred_val = model_lgb.predict(X_val)
    oof[val_idx] = pred_val
    preds += model_lgb.predict(X_test_enc) / FOLDS

final_preds = np.expm1(preds)


score = np.sqrt(mean_squared_log_error(np.expm1(y), np.maximum(np.expm1(oof), 0)))
print(f"\nRMSLE: {score:.5f}")

sub["Calories"] = final_preds
sub.to_csv("submission.csv", index=False)


sub["Calories"] = final_preds
sub.to_csv("submission.csv", index=False)


importance = model_lgb.feature_importances_

features = X_train.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importance, y=features, palette="Blues_d")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

