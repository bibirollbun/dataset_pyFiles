# This Python 3 environment comes with many helpful analytics libraries installed


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
sns.set(style="whitegrid", palette="flare")
warnings.filterwarnings("ignore", category=FutureWarning)


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from cuml import RandomForestRegressor as GRF


train_path="/kaggle/input/playground-series-s5e9/train.csv"
test_path="/kaggle/input/playground-series-s5e9/test.csv"


train_df=pd.read_csv(train_path,index_col="id")
test_df=pd.read_csv(test_path,index_col="id")


train_df.head(100)


train_df.info()


test_df.info()


train_df["TrackDurationMs"]=train_df["TrackDurationMs"]/1000


columns=train_df.columns.tolist()
target_col="BeatsPerMinute"


columns.remove(target_col)
columns






palette = sns.color_palette("flare", n_colors=10)
# Create figure with 3x3 grid
fig, ax = plt.subplots(5, 2, figsize=(8, 17.5))

# Loop over first 9 columns
for i, col in enumerate(train_df.columns.tolist()):
    r, c = divmod(i, 2)
    sns.histplot(
        data=train_df,
        x=col,
        kde=True,
        stat="percent",
        bins=30,           # fixed number of bins
        alpha=0.6,         # transparency
        ax=ax[r, c],
        color=palette[i]
    )
    ax[r, c].set_title(col, fontsize=14)
    ax[r, c].set_xlabel('')
    ax[r, c].set_ylabel('Density')

plt.tight_layout()
plt.show()



corr=train_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))  # Upper triangle mask

# ---------------------------
# 4️⃣ High-quality heatmap plot
# ---------------------------
plt.figure(figsize=(12, 8))
sns.set_theme(style="white")  # Clean white background

# Custom diverging colormap centered at 0
cmap = sns.diverging_palette(220, 20, as_cmap=True)
palette = sns.color_palette("flare", n_colors=1)
heatmap = sns.heatmap(
    corr,
    cmap=cmap,
    vmin=-1, vmax=1, center=0,
    annot=True, fmt=".2f",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
    color=palette
)

plt.title("Correlation Matrix", fontsize=16, weight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


## feature engineering 
f1=["Energy","Energy","Energy","AcousticQuality","AudioLoudness","TrackDurationMs","TrackDurationMs"]
f2=["MoodScore","LivePerformanceLikelihood","AudioLoudness","InstrumentalScore","TrackDurationMs","RhythmScore","LivePerformanceLikelihood"]
df=train_df.copy()
for i,(x,y) in enumerate(zip(f1,f2)):
    df[f"f{i}"]=df[x]/(df[y]+1)
    


df.head()


corr=df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))  # Upper triangle mask

# ---------------------------
# 4️⃣ High-quality heatmap plot
# ---------------------------
plt.figure(figsize=(15, 10))
sns.set_theme(style="white")  # Clean white background

# Custom diverging colormap centered at 0
cmap = sns.diverging_palette(220, 20, as_cmap=True)
palette = sns.color_palette("flare", n_colors=1)
heatmap = sns.heatmap(
    corr,
    cmap=cmap,
    vmin=-1, vmax=1, center=0,
    annot=True, fmt=".2f",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
    color=palette
)

plt.title("Correlation Matrix", fontsize=16, weight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


columns=df.columns.tolist()
columns.remove("BeatsPerMinute")


X=df[columns].values
y=train_df[target_col].values


skfld=KFold(n_splits=5,random_state=42,shuffle=True)


for i,(train_idx,test_idx) in enumerate(skfld.split(X,y)):
    print(f"Fold {i}")
    scaler=StandardScaler()
    X_train,X_test=X[train_idx],X[test_idx]
    y_train,y_test=y[train_idx],y[test_idx]
    scaler.fit(X_train)
    X_scaled_train=scaler.transform(X_train)
    X_scaled_test=scaler.transform(X_test)

    model=LinearRegression()
    model.fit(X_scaled_train,y_train)
    y_train_pred=model.predict(X_scaled_train)
    y_test_pred=model.predict(X_scaled_test)

    l1=mean_squared_error(y_train,y_train_pred)
    l2=mean_squared_error(y_test,y_test_pred)
    print(f"the mse in the trainng :{l1}\n the mse in test: {l2}")


for i,(train_idx,test_idx) in enumerate(skfld.split(X,y)):
    print(f"Fold {i}")
    scaler=StandardScaler()
    X_train,X_test=X[train_idx],X[test_idx]
    y_train,y_test=y[train_idx],y[test_idx]
    scaler.fit(X_train)
    X_scaled_train=scaler.transform(X_train)
    X_scaled_test=scaler.transform(X_test)

    model=GRF(n_estimators=1000,min_samples_split=3)
    model.fit(X_scaled_train,y_train)
    y_train_pred=model.predict(X_scaled_train)
    y_test_pred=model.predict(X_scaled_test)

    l1=mean_squared_error(y_train,y_train_pred)
    l2=mean_squared_error(y_test,y_test_pred)

    del X_scaled_train,X_scaled_test,y_train_pred,y_test_pred,scaler,model
    print(f"the mse in the trainng :{l1}\n the mse in test: {l2}")





import lightgbm as lgb
import numpy as np
for i,(train_idx,test_idx) in enumerate(skfld.split(X,y)):
    print(f"Fold {i}")
    scaler=StandardScaler()
    X_train,X_test=X[train_idx],X[test_idx]
    y_train,y_test=y[train_idx],y[test_idx]
    scaler.fit(X_train)
    X_scaled_train=scaler.transform(X_train)
    X_scaled_test=scaler.transform(X_test)

    model=lgb.LGBMRegressor(random_state=42)
    model.fit(X_scaled_train,y_train)
    y_train_pred=model.predict(X_scaled_train)
    y_test_pred=model.predict(X_scaled_test)

    l1=mean_squared_error(y_train,y_train_pred)
    l2=mean_squared_error(y_test,y_test_pred)
    del X_scaled_train,X_scaled_test,y_train_pred,y_test_pred,scaler,model
    print(f"the mse in the trainng :{l1}\n the mse in test: {l2}")


from xgboost import XGBRegressor

for i,(train_idx,test_idx) in enumerate(skfld.split(X,y)):
    print(f"Fold {i}")
    scaler=StandardScaler()
    X_train,X_test=X[train_idx],X[test_idx]
    y_train,y_test=y[train_idx],y[test_idx]
    scaler.fit(X_train)
    X_scaled_train=scaler.transform(X_train)
    X_scaled_test=scaler.transform(X_test)

    model = XGBRegressor(
    n_estimators=5000,
    max_depth=8,
    learning_rate=0.001,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",# <─ GPU-accelerated training
    device="gpu",
    random_state=42
    )
    model.fit(X_scaled_train,y_train)
    y_train_pred=model.predict(X_scaled_train)
    y_test_pred=model.predict(X_scaled_test)

    l1=mean_squared_error(y_train,y_train_pred)
    l2=mean_squared_error(y_test,y_test_pred)
    del X_scaled_train,X_scaled_test,y_train_pred,y_test_pred,scaler,model
    print(f"the mse in the trainng :{l1}\n the mse in test: {l2}")


model = XGBRegressor(
    n_estimators=5000,
    max_depth=8,
    learning_rate=0.001,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",# <─ GPU-accelerated training
    device="gpu",
    random_state=42
    )
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
model.fit(X,y)
print(f"loss is {mean_squared_error(model.predict(X),y)}")


## feature engineering 
f1=["Energy","Energy","Energy","AcousticQuality","AudioLoudness","TrackDurationMs","TrackDurationMs"]
f2=["MoodScore","LivePerformanceLikelihood","AudioLoudness","InstrumentalScore","TrackDurationMs","RhythmScore","LivePerformanceLikelihood"]
df1=test_df.copy()
for i,(x,y) in enumerate(zip(f1,f2)):
    df1[f"f{i}"]=df1[x]/(df1[y]+1)


df1.info()


X_test=df1[columns].values
X_test_scaled=scaler.transform(X_test)


X_test_scaled.shape


y_test_pred=model.predict(X_test_scaled)


y_test_pred.reshape(-1,1)


submission_df=pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv",index_col="id")
submission_df.info()


submission_df["BeatsPerMinute"]=y_test_pred


submission_df.head()


submission_df.to_csv("/kaggle/working/submission.csv")




