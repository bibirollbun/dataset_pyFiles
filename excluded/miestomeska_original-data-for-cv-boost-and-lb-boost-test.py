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


from sklearn.preprocessing import LabelEncoder

def rank_features_no_price_leakage(train_df, test_df):
    """
    Adds leakage-free rank-based features using non-target features.
    No price or price-derived statistics are used.
    Ranks computed based on non-target features (e.g., weight capacity).
    """

    # --- Position in Brand-Size Group based on Weight Capacity ---
    train_df['position_in_brand_size'] = train_df.groupby(['Brand', 'Size'])['Weight Capacity (kg)'].rank(method='dense')
    position_map_brand_size = train_df.groupby(['Brand', 'Size'])['position_in_brand_size'].mean().to_dict()
    test_df['position_in_brand_size'] = test_df.apply(
        lambda x: position_map_brand_size.get((x['Brand'], x['Size']), np.nan), axis=1
    )

    # --- Frequency Rank of Brand-Material Combinations ---
    combo_freq = train_df.groupby(['Brand', 'Material']).size().reset_index(name='brand_material_freq')
    train_df = train_df.merge(combo_freq, on=['Brand', 'Material'], how='left')
    test_df = test_df.merge(combo_freq, on=['Brand', 'Material'], how='left')

    # --- Brand weight mean & std ---
    brand_weight_mean = train_df.groupby('Brand')['Weight Capacity (kg)'].transform('mean')
    train_df['brand_weight_mean'] = brand_weight_mean
    test_df['brand_weight_mean'] = test_df.groupby('Brand')['Weight Capacity (kg)'].transform('mean')

    brand_weight_std = train_df.groupby('Brand')['Weight Capacity (kg)'].transform('std').fillna(0)
    train_df['brand_weight_std'] = brand_weight_std
    test_df['brand_weight_std'] = test_df.groupby('Brand')['Weight Capacity (kg)'].transform('std').fillna(0)

    # --- Label Encoding for all categorical features ---
    cat_features = [col for col in train_df.columns if train_df[col].dtype == 'object']
    le = LabelEncoder()
    for col in cat_features:
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))

    return train_df, test_df


# âœ… Apply the updated leakage-free feature engineering
train, test = rank_features_no_price_leakage(train, test)

# ğŸ”� Check datasets
print(f"âœ… Train shape after updated feature engineering: {train.shape}")
print(f"âœ… Test shape after updated feature engineering: {test.shape}")


# âœ… Automatically detect available engineered features based on the actual dataset

# Base categorical columns
CATS = ['Brand', 'Material', 'Size', 'Style', 'Color',
        'orig_Brand', 'orig_Material', 'orig_Size',
        'orig_Style', 'orig_Color']

# Potentially engineered categorical features (if they exist)
engineered_cats = ['price_rank_brand_size', 'price_rank_material']
CATS += [col for col in engineered_cats if col in train.columns]

# Numerical columns (original + engineered)
NUMS = ['Weight Capacity (kg)', 'orig_Price', 'weight_capacity_ratio',
        'brand_weight_mean', 'brand_weight_std']

# Potentially engineered numerical features
engineered_nums = ['brand_size_avg_price', 'price_vs_brand_size_avg',
                   'material_median_price', 'price_vs_material_median']
NUMS += [col for col in engineered_nums if col in train.columns]

# âœ… Final filtered features that exist in the dataset
CATS = [col for col in CATS if col in train.columns]
NUMS = [col for col in NUMS if col in train.columns]
FEATURES = CATS + NUMS

# ğŸ”� Print final results
print(f"âœ… There are {len(CATS)} categorical columns:\n{CATS}")
print(f"âœ… There are {len(NUMS)} numerical columns:\n{NUMS}")
print(f"âœ… Total number of features selected: {len(FEATURES)}")



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

