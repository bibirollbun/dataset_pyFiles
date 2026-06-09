import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
print("Train shape", train.shape )
train.head()


train2 = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
print("Train extra shape", train2.shape )
train2.head()


train = pd.concat([train,train2],axis=0,ignore_index=True)
print("Train combined shape",train.shape)


test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
print("Test shape", test.shape )


orig = pd.read_csv("/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv")
print("Original data shape", orig.shape )
orig.head()


orig = orig.groupby("Weight Capacity (kg)").Price.mean()
orig.name = "orig_Price"
orig.head()


train = train.merge(orig, on="Weight Capacity (kg)", how="left")
test = test.merge(orig, on="Weight Capacity (kg)", how="left")
train.head()


tmp = train.groupby("Weight Capacity (kg)")[['Price','orig_Price']].agg(["mean","count"])
tmp = tmp.iloc[:,:-1]
tmp.columns = ['Price','count','orig_Price']
tmp = tmp.loc[(tmp['count']>100)&(~tmp.orig_Price.isna())]
print( tmp.shape )
tmp.head()


plt.scatter(tmp.orig_Price,tmp.Price,s=1)
a,b = np.polyfit(tmp.loc[~tmp.orig_Price.isna()].orig_Price,tmp.loc[~tmp.orig_Price.isna()].Price,deg=1)
x = np.arange(15,150)
y = b+a*x
plt.plot(x,y,'--',color='black',linewidth=3)
r = np.corrcoef(tmp.Price,tmp.orig_Price)[0,1]
plt.xlabel("Original Dataset Price")
plt.ylabel("Synthetic Dataset Price")
plt.title(
    f"Relationship between Original Dataset Price and Synthetic Dataset Price\n"
    f"Correlation r={r:.2f} with equation Synth_Price = {a:.3f}*Orig_Price + {b:.3f}"
)
plt.show()


plt.hist(tmp.loc[~tmp.orig_Price.isna()].Price,bins=100)
plt.title("Train data Price histogram")
plt.show()


plt.hist(tmp.loc[~tmp.orig_Price.isna()].orig_Price,bins=100)
plt.title("Original data Price histogram")
plt.show()


orig = pd.read_csv("/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv")
orig = orig.loc[(orig["Weight Capacity (kg)"]>5)&(orig["Weight Capacity (kg)"]<30)]
orig.columns = [f"orig_{c}" for c in orig.columns]
train = train.merge(orig.iloc[:,:-1], left_on="Weight Capacity (kg)", right_on="orig_Weight Capacity (kg)", how="left")
train = train.drop("id",axis=1)
test = test.merge(orig.iloc[:,:-1], left_on="Weight Capacity (kg)", right_on="orig_Weight Capacity (kg)", how="left")
train.head()


CATS = []
for c in train.columns:
    if train[c].dtype=='object':
        CATS.append(c)
print(f"There are {len(CATS)} categorical columns:")
print( CATS )
NUMS = ['Weight Capacity (kg)','orig_Price']
print(f"There are {len(NUMS)} numerical columns:")
print( NUMS )
FEATURES = CATS + NUMS


from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import xgboost as xgb
print(f"XGBoost version",xgb.__version__)


%%time

FOLDS = 7
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros((len(train)))
pred = np.zeros((len(test)))

# OUTER K FOLD
for i, (train_index, test_index) in enumerate(kf.split(train)):
    print(f"### Fold {i+1} ###")

    X_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,'Price']

    X_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,'Price']

    X_test = test[FEATURES].copy()

    # CONVERT TO CATS SO XGBOOST RECOGNIZES THEM
    X_train[CATS] = X_train[CATS].astype("category")
    X_valid[CATS] = X_valid[CATS].astype("category")
    X_test[CATS] = X_test[CATS].astype("category")

    # BUILD MODEL
    model = XGBRegressor(
        device="cuda",
        max_depth=6,  
        colsample_bytree=0.5, 
        subsample=0.8,  
        n_estimators=10_000,  
        learning_rate=0.2,  
        enable_categorical=True,
        min_child_weight=10,
        early_stopping_rounds=100,
    )
    
    # TRAIN MODEL
    COLS = X_train.columns
    model.fit(
        X_train[COLS], y_train,
        eval_set=[(X_valid[COLS], y_valid)],  
        verbose=100,
    )

    # PREDICT OOF AND TEST
    oof[test_index] = model.predict(X_valid[COLS])
    pred += model.predict(X_test[COLS])

pred /= FOLDS


# COMPUTE OVERALL CV SCORE
true = train.Price.values
s = np.sqrt(np.mean( (oof-true)**2.0 ) )
print(f"=> Overall CV Score = {s}")


# SAVE OOF TO DISK FOR ENSEMBLES
np.save(f"oof",oof)
print("Saved oof to disk")


print(f"\nIn total, we used {len(COLS)} features, Wow!\n")
print( list(COLS) )


import xgboost as xgb
fig, ax = plt.subplots(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=100, importance_type='gain',ax=ax)
plt.title("Top 100 Feature Importances (XGBoost)")
plt.show()


sub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
sub.Price = pred
sub.to_csv(f"submission.csv",index=False)
sub.head(10)


sub1 = pd.read_csv("/kaggle/input/testing-more/submission.csv")
sub2 = pd.read_csv("/kaggle/input/feature-engineering-with-rapids-lb-38-847/submission_v1.csv")
sub3 = pd.read_csv("PS-S5E2 | Dividing attention v43")
sub1['Price'] = 0.04* sub['Price'] + 0.96*sub3['Price']
sub1.to_csv(f"submission.csv",index=False)
sub1.head(10)

