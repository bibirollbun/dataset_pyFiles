import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
print("Train shape:", df_train.shape )


df_train


df = df_train
sns.set(style="whitegrid", palette="muted")

# 1️⃣ Distribution of accident risk
plt.figure(figsize=(6,4))
sns.histplot(df["accident_risk"], kde=True, bins=5, color='skyblue')
plt.title("Distribution of Accident Risk")
plt.show()

# 2️⃣ Accident risk vs. road type
plt.figure(figsize=(6,4))
sns.barplot(x="road_type", y="accident_risk", data=df)
plt.title("Average Accident Risk by Road Type")
plt.show()

# 3️⃣ Accident risk vs. speed limit
plt.figure(figsize=(6,4))
sns.scatterplot(x="speed_limit", y="accident_risk", hue="road_type", style="weather", data=df, s=10)
plt.title("Accident Risk vs. Speed Limit")
plt.show()

# 4️⃣ Accident risk vs. weather
plt.figure(figsize=(6,4))
sns.boxplot(x="weather", y="accident_risk", data=df)
plt.title("Accident Risk by Weather")
plt.show()

# 5️⃣ Accident risk vs. lighting conditions
plt.figure(figsize=(6,4))
sns.barplot(x="lighting", y="accident_risk", data=df)
plt.title("Accident Risk by Lighting Condition")
plt.show()

# 6️⃣ Correlation heatmap (for numeric features)
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# 7️⃣ Relationship between curvature and accident risk
plt.figure(figsize=(6,4))
sns.scatterplot(x="curvature", y="accident_risk", hue="weather", data=df, s=10)
plt.title("Curvature vs. Accident Risk")
plt.show()


df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
df_test['accident_risk'] = 0.5
print("Test shape:", df_test.shape )


df_test


orig = []
for k in [2,10,100]:
    df = pd.read_csv(f"/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_{k}k.csv")
    orig.append(df)
orig = pd.concat(orig,axis=0)
orig['id'] = np.arange(len(orig))+df_test['id'].max()+1
orig = orig[ df_test.columns ] 
print("Original data shape:", orig.shape )
orig.head()


combine = pd.concat([df_train,df_test,orig],axis=0,ignore_index=True)
print("Combine shape:", combine.shape )
combine.head()


FEATURES = list( orig.columns[1:-1] )
TARGET = orig.columns[-1]
print(f"Features: {FEATURES}, Target: '{TARGET}'")


import scipy

def f(X):
    return \
    0.3 * X["curvature"] + \
    0.2 * (X["lighting"] == "night").astype(int) + \
    0.1 * (X["weather"] != "clear").astype(int) + \
    0.2 * (X["speed_limit"] >= 60).astype(int) + \
    0.1 * (X["num_reported_accidents"] > 2).astype(int)

def clip(f):
    def clip_f(X):
        sigma = 0.05
        mu = f(X)
        a, b = -mu/sigma, (1-mu)/sigma
        Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
        phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
        return mu*(Phi_b-Phi_a)+sigma*(phi_a-phi_b)+1-Phi_b
    return clip_f

z = clip(f)(combine)
combine["y"] = z.values
FEATURES.append("y")


CATS = []
NUMS = []
for c in FEATURES:
    t = "CAT"
    if combine[c].dtype=='object':
        CATS.append(c)
    else:
        NUMS.append(c)
        t = "NUM"
    n = combine[c].nunique()
    na = combine[c].isna().sum()
    print(f"[{t}] {c} has {n} unique and {na} NA")
print("CATS:", CATS )
print("NUMS:", NUMS )


SIZES = {}
for c in CATS:
    combine[c],_ = combine[c].factorize()
    SIZES[c] = combine[c].max()+1
    combine[c] = combine[c].astype('int32')
    combine[c] = combine[c].astype('int32')
print("Cardinality of all CATS:", SIZES )


df_train = combine.iloc[:len(df_train)]
df_test = combine.iloc[len(df_train):len(df_train)+len(df_test)]
orig = combine.iloc[-len(orig):]
print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}, Original data shape: {orig.shape}")


TE = []
for c in FEATURES:
    tmp = orig.groupby(c)[TARGET].mean()
    n = f"TE_{c}"
    print(f"{n}, ",end="")
    tmp.name = n
    df_train = df_train.merge(tmp, on=c, how='left')
    df_test = df_test.merge(tmp, on=c, how='left')
    TE.append(n)


from sklearn.model_selection import KFold
import xgboost as xgb


FOLDS = 7
SEED = 42

params = {
    "objective": "reg:squarederror",   
    "eval_metric": "rmse",             
    "learning_rate": 0.01,
    "max_depth": 6,                    
    "subsample": 0.9,
    "colsample_bytree": 0.6,
    "seed": SEED,
    "device": "cuda",
}


oof_preds = np.zeros(len(df_train))
test_preds = np.zeros(len(df_test))

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(df_train)):
    print("#"*25)
    print(f"### Fold {fold+1} ###")
    print("#"*25)

    X_train = df_train.iloc[train_idx][FEATURES+TE].copy()
    y_train = df_train.iloc[train_idx][TARGET] - df_train.iloc[train_idx]['y']
    
    X_valid = df_train.iloc[val_idx][FEATURES+TE].copy()
    y_valid = df_train.iloc[val_idx][TARGET] - df_train.iloc[val_idx]['y']
    y_valid2 = df_train.iloc[val_idx]['y'].values
    
    X_test = df_test[FEATURES+TE].copy()
    y_test2 = df_test['y'].values
        
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval   = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
    dtest  = xgb.DMatrix(X_test, enable_categorical=True)

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100_000,
        evals=[(dtrain, "df_train"), (dval, "valid")],
        early_stopping_rounds=200,
        verbose_eval=200
    )

    oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1)) +y_valid2
    test_preds += (model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) +y_test2)/ FOLDS


m = np.sqrt( np.mean( (oof_preds - df_train[TARGET].values)**2. ) )
print(f" Overall CV RMSE = {m}")
np.save(f"oof",oof_preds)


m = np.sqrt( np.mean( (df_train.y.values - df_train[TARGET].values)**2. ) )
print(f" Baseline CV RMSE = {m}")


import matplotlib.pyplot as plt

plt.scatter(df_train[TARGET].values,oof_preds,s=0.25)
plt.plot([0,1],[0,1],'--',color='black')
plt.title("True vs Predicted")
plt.xlabel("True Target")
plt.ylabel("Predicted Target")
plt.show()


plt.rcParams["figure.dpi"] = 160      
fig, ax = plt.subplots(figsize=(15, 12))

xgb.plot_importance(
    model,
    max_num_features=100,
    importance_type="gain",
    ax=ax,
    show_values=False,                
    grid=False
)

ax.set_title("XGB Feature Importances", fontsize=18)
ax.tick_params(axis="both", labelsize=12)
fig.tight_layout()
plt.show()


sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
sub['accident_risk'] = test_preds
sub.to_csv("submission.csv",index=False)
sub.head()

