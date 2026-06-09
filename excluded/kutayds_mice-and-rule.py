!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
print("Test shape:", test.shape )

train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
print("Train shape:",train.shape)
train.head()


plt.hist(train.loc[train.efs==1,"efs_time"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"efs_time"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Time of Observation, efs_time")
plt.ylabel("Density")
plt.title("Times of Observation. Either time to event, or time observed without event.")
plt.legend()
plt.show()


from lifelines import KaplanMeierFitter
def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col].apply(round), df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y
train["y"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')

plt.hist(train.loc[train.efs==1,"y"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"y"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Transformed Target y")
plt.ylabel("Density")
plt.title("KaplanMeier Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()


from lifelines import WeibullFitter#LogLogisticFitter
def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    llf = WeibullFitter()
    llf.fit(df[time_col], event_observed=df[event_col])
    y = llf.survival_function_at_times(df[time_col]).values
    return y
train["y"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')

plt.hist(train.loc[train.efs==1,"y"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"y"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Transformed Target y")
plt.ylabel("Density")
plt.title("KaplanMeier Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()


RMV = ["ID","efs","efs_time","y"]
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


# First pass do not fill NANs yet
CATS = []
NUMS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
    else:
        NUMS.append(c)

print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


# The variety in numerical data
'''variety = {}
for c in NUMS:
    variety[c] = set(train[c].dropna())

for c, vals in variety.items():
    print(f"Column {c} contains {len(vals)} number of unique numerical values.")
'''


#for c, vals in variety.items():
#    print(f"Column {c} first {min(10,len(vals))} values: {sorted(vals,reverse=True)[:10]}")



# Count the NaN fields column wise
'''nan_counts = train[CATS].isna().sum().sort_values(ascending=False)
nan_counts_num = train[NUMS].isna().sum().sort_values(ascending=False)
print("NaN Counts per Column in catagorical data:")
print(nan_counts)
train_cols_sorted_by_nan = train[nan_counts.keys()].copy()
print(train_cols_sorted_by_nan.columns)

print("NaN Counts per Column in numerical data:")
print(nan_counts_num)
train_cols_sorted_by_nan_num = train[nan_counts_num.keys()].copy()
print(train_cols_sorted_by_nan_num.columns)
'''


from collections import Counter
from math import ceil


'''
dist = {}
for c in NUMS:
    dist[c] = train[c].dropna()
for c, vals in dist.items():
    cnt = Counter(vals)
    if len(cnt) < 15:
        print(f"Column {c} first {min(10,len(vals))} value distribution: {cnt}")
'''


# This added code is to compute Thelis U score for predicting one column given
# another.

from scipy.stats import entropy

def conditional_entropy(x, y):
    """Computes H(X|Y), the conditional entropy of X given Y."""
    y_counter = Counter(y)
    xy_counter = Counter(zip(x, y))
    
    total_occurrences = sum(y_counter.values())
    entropy_conditional = 0.0

    for (x_val, y_val), count in xy_counter.items():
        p_xy = count / total_occurrences
        p_y = y_counter[y_val] / total_occurrences
        entropy_conditional += p_xy * np.log2(p_y / p_xy)

    return entropy_conditional

def theils_u(x, y):
    """Computes Theil's U statistic (Asymmetric Predictive Power)."""
    h_x = entropy(np.array(list(Counter(x).values()), dtype=float), base=2)
    h_x_given_y = conditional_entropy(x, y)
    
    if h_x == 0:
        return 1
    return (h_x - h_x_given_y) / h_x



# Set a seed for reproducibility
np.random.seed(42)

def impute_missing(row):
            a_value = row[best_A]
            if pd.isna(row[best_B]) and a_value in prob_dist and prob_dist[a_value]:
                possible_values = list(prob_dist[a_value].keys())
                probabilities = list(prob_dist[a_value].values())
                return np.random.choice(possible_values, p=probabilities)
            return row[best_B]

def learn_imputation_rules(df_train, no_of_loops=10, threshold=0.1):
    """ Learns the best (A -> B) imputation rules based on Theil’s U """
    rules = {}  

    columns = df_train.columns  # Already sorted by NaN count

    i = 0
    while i < no_of_loops:
        best_A, best_B, best_U, max_nan_diff = None, None, -1, 0

        for b_idx in range(len(columns)):
            B = columns[b_idx]
            for a_idx in range(b_idx + 1, len(columns)):
                A = columns[a_idx]

                U_A_to_B = theils_u(df_train[A].dropna(), df_train[B].dropna())

                nan_diff = df_train[B].isna().sum() - df_train[A].isna().sum()
                if U_A_to_B > threshold and nan_diff > max_nan_diff:
                    best_A, best_B, best_U, max_nan_diff = A, B, U_A_to_B, nan_diff

        if best_A is None or best_B is None:
            print("No more imputations possible with the given threshold.")
            break

        print(f"Learning imputation rule: {best_A} -> {best_B} (U={best_U:.4f})")

        # P(B | A)
        prob_dist_full = df_train.groupby(best_A)[best_B].apply(lambda x: x.dropna().value_counts(normalize=True)).to_dict()

        # Transform prob_dist to map from A -> {B: probability}
        prob_dist = {}
        for (a_value, b_value), prob in prob_dist_full.items():
            if a_value not in prob_dist:
                prob_dist[a_value] = {}
            prob_dist[a_value][b_value] = prob


        rules[best_B] = (best_A, prob_dist)

        df_train[best_B] = df_train.apply(impute_missing, axis=1)
        i += 1

    return rules

#rules = learn_imputation_rules(train_cols_sorted_by_nan)
#print(train_cols_sorted_by_nan['tce_match'].head())


def apply_imputation_rules(df_test, rules):
    for B, (A, prob_dist) in rules.items():
        print(f"Applying imputation rule: {B} → {A}")

        df_test[B] = df_test.apply(impute_missing, axis=1)
    
    return df_test

#apply_imputation_rules(test, rules)


def restore_column_order(df_imputed, original_df):
    """ Merges imputed categorical data back into the original dataframe while preserving all columns. """
    # Get original column order
    original_columns = original_df.columns

    # Replace only the imputed categorical columns
    for col in df_imputed.columns:
        original_df[col] = df_imputed[col]

    # Restore the original column order
    return original_df[original_columns]

#train = restore_column_order(train_cols_sorted_by_nan, train)


#DISC_NUMS = NUMS.copy()
#DISC_NUMS.remove("age_at_hct")
#DISC_NUMS.remove("donor_age")

#for c in DISC_NUMS:
#    train[c] = train[c].astype(pd.Int32Dtype())
#    test[c] = test[c].astype(pd.Int32Dtype())


# MICE method euqivalent program
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#imputer = IterativeImputer(max_iter=35, random_state=42)
#imputer.fit(train[DISC_NUMS])


#train[DISC_NUMS] = imputer.transform(train[DISC_NUMS])
#test[DISC_NUMS] = imputer.transform(test[DISC_NUMS])


for c in CATS:
    train[c] = train[c].fillna("NAN")
    test[c] = test[c].fillna("NAN")


combined = pd.concat([train,test],axis=0,ignore_index=True)
#print("Combined data shape:", combined.shape )

# LABEL ENCODE CATEGORICAL FEATURES
print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
for c in FEATURES:

    # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
    if c in CATS:
        print(f"{c}, ",end="")
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")
        combined[c] = combined[c].astype("category")
        
    # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
    
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
print("Using XGBoost version",xgb.__version__)


%%time
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"y"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y"]
    x_test = test[FEATURES].copy()

    model_xgb = XGBRegressor(
        device="cuda",
        max_depth=3,  
        colsample_bytree=0.5,  
        subsample=0.8,  
        n_estimators=2000,  
        learning_rate=0.02,  
        enable_categorical=True,
        min_child_weight=80,
        objective="reg:tweedie",
        #eval_metric="logloss"
        #early_stopping_rounds=25,
    )
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=500 
    )

    # INFER OOF
    oof_xgb[test_index] = model_xgb.predict(x_valid)
    # INFER TEST
    pred_xgb += model_xgb.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb /= FOLDS


from metric import score

y_true = train[["ID","efs","efs_time","race_group"]].copy()
print(y_true[:10])
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_xgb
print(y_pred[:10])
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost KaplanMeier =",m)


feature_importance = model_xgb.feature_importances_
importance_df = pd.DataFrame({
    "Feature": FEATURES,  # Replace FEATURES with your list of feature names
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost KaplanMeier Feature Importance")
plt.gca().invert_yaxis()  # Flip features for better readability
plt.show()


from catboost import CatBoostRegressor, CatBoostClassifier
import catboost as cb
print("Using CatBoost version",cb.__version__)


%%time
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_cat = np.zeros(len(train))
pred_cat = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"y"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y"]
    x_test = test[FEATURES].copy()

    model_cat = CatBoostRegressor(
        task_type="GPU",  
        learning_rate=0.1,    
        grow_policy='Lossguide',
        loss_function="RMSE",
        #early_stopping_rounds=25,
    )
    model_cat.fit(x_train,y_train,
              eval_set=(x_valid, y_valid),
              cat_features=CATS,
              verbose=250)

    # INFER OOF
    oof_cat[test_index] = model_cat.predict(x_valid)
    # INFER TEST
    pred_cat += model_cat.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_cat /= FOLDS


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_cat
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for CatBoost KaplanMeier =",m)


feature_importance = model_cat.get_feature_importance()
importance_df = pd.DataFrame({
    "Feature": FEATURES, 
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("CatBoost KaplanMeier Feature Importance")
plt.gca().invert_yaxis()  # Flip features for better readability
plt.show()


from lightgbm import LGBMRegressor
import lightgbm as lgb
print("Using LightGBM version",lgb.__version__)


FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_lgb = np.zeros(len(train))
pred_lgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"y"]    
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y"]
    x_test = test[FEATURES].copy()

    model_lgb = LGBMRegressor(
        device="gpu", 
        max_depth=3, 
        colsample_bytree=0.4,  
        #subsample=0.9, 
        n_estimators=2500, 
        learning_rate=0.02, 
        objective="tweedie", 
        verbose=-1
        #early_stopping_rounds=25,
    )
    model_lgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
    )
    
    # INFER OOF
    oof_lgb[test_index] = model_lgb.predict(x_valid)
    # INFER TEST
    pred_lgb += model_lgb.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_lgb /= FOLDS


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_lgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for LightGBM KaplanMeier =",m)


feature_importance = model_lgb.feature_importances_ 
importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"], color='skyblue')
plt.xlabel("Importance (Gain)")
plt.ylabel("Feature")
plt.title("LightGBM KaplanMeier Feature Importance")
plt.gca().invert_yaxis()  # Flip features for better readability
plt.show()


# SURVIVAL COX NEEDS THIS TARGET (TO DIGEST EFS AND EFS_TIME)
train["efs_time2"] = train.efs_time.copy()
train.loc[train.efs==0,"efs_time2"] *= -1


train["efs_time_low"] = train.efs_time.copy()
train["efs_time_up"] = train.efs_time.copy()
train.loc[train.efs==0,"efs_time_up"] = train["efs_time"].max() * 2
train.head(10)


FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb_cox = np.zeros(len(train))
pred_xgb_cox = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
   # x_train = train.loc[train_index,FEATURES].copy()
   # y_train = train.loc[train_index,"efs_time2"]    
   # x_valid = train.loc[test_index,FEATURES].copy()
   # y_valid = train.loc[test_index,"efs_time2"]
   # x_test = test[FEATURES].copy()

    x_train = train.loc[train_index,FEATURES].copy()
    y_train_low = train.loc[train_index,"efs_time_low"].apply(round)
    y_train_up = train.loc[train_index,"efs_time_up"].apply(round)
    #y_train = pd.concat([y_train_low, y_train_up], axis=1).to_numpy()
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid_low = train.loc[test_index,"efs_time_low"].apply(round)
    y_valid_up = train.loc[test_index,"efs_time_up"].apply(round)
    #y_valid = pd.concat([y_valid_low, y_valid_up], axis=1).to_numpy()
    x_test = test[FEATURES].copy()

    dtrain = xgb.DMatrix(x_train, enable_categorical=True)
    dtrain.set_float_info("label_lower_bound", y_train_low)
    dtrain.set_float_info("label_upper_bound", y_train_up)
    
    dvalid = xgb.DMatrix(x_valid, enable_categorical=True)
    dvalid.set_float_info("label_lower_bound", y_valid_low)
    dvalid.set_float_info("label_upper_bound", y_valid_up)
    
    dtest = xgb.DMatrix(x_test, enable_categorical=True)

    '''model_xgb_cox = XGBRegressor(
        device="cuda",
        max_depth=3,  
        colsample_bytree=0.5,  
        subsample=0.8,  
        n_estimators=2000,  
        learning_rate=0.02,  
        enable_categorical=True,
        min_child_weight=80,
        objective='survival:aft',
        eval_metric='aft-nloglik',
    )'''
    
    '''model_xgb_cox.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=500  
    )'''

    aft_params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "max_depth": 3,
        "colsample_bytree": 0.5,
        "subsample": 0.8,
        "learning_rate": 0.02,
        "min_child_weight": 80,
        "aft_loss_distribution": "normal",  # Can be 'normal', 'logistic', 'extreme'
        "aft_loss_distribution_scale": 20.0,
        "tree_method": "hist",
    }

    model_xgb_cox = xgb.train(
        aft_params,
        dtrain,
        num_boost_round=2000,
        evals=[(dvalid, "validation")],
        verbose_eval=500
    )    
    # INFER OOF
    oof_xgb_cox[test_index] = model_xgb_cox.predict(dvalid)
    # INFER TEST
    pred_xgb_cox += model_xgb_cox.predict(dtest)

# COMPUTE AVERAGE TEST PREDS
pred_xgb_cox /= FOLDS


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_xgb_cox
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost Survival:Cox =",m)


feature_importance = model_xgb_cox.feature_importances_
importance_df = pd.DataFrame({
    "Feature": FEATURES,  # Replace FEATURES with your list of feature names
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Survival:Cox Feature Importance")
plt.gca().invert_yaxis()  # Flip features for better readability
plt.show()


FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_cat_cox = np.zeros(len(train))
pred_cat_cox = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"efs_time2"]  
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"efs_time2"]
    x_test = test[FEATURES].copy()

    model_cat_cox = CatBoostRegressor(
        loss_function="Cox",
        #task_type="GPU",   
        iterations=400,     
        learning_rate=0.1,  
        grow_policy='Lossguide',
        use_best_model=False,
    )
    model_cat_cox.fit(x_train,y_train,
              eval_set=(x_valid, y_valid),
              cat_features=CATS,
              verbose=100)
    
    # INFER OOF
    oof_cat_cox[test_index] = model_cat_cox.predict(x_valid)
    # INFER TEST
    pred_cat_cox += model_cat_cox.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_cat_cox /= FOLDS


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_cat_cox
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for CatBoost Survival:Cox =",m)


feature_importance = model_cat_cox.get_feature_importance()
importance_df = pd.DataFrame({
    "Feature": FEATURES, 
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("CatBoost Survival:Cox Feature Importance")
plt.gca().invert_yaxis()  # Flip features for better readability
plt.show()


from scipy.stats import rankdata 

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = rankdata(oof_xgb) + rankdata(oof_cat) + rankdata(oof_lgb)\
                     + rankdata(oof_xgb_cox) + rankdata(oof_cat_cox)
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for Ensemble =",m)


sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction = rankdata(pred_xgb) + rankdata(pred_cat) + rankdata(pred_lgb)\
                     + rankdata(pred_xgb_cox) + rankdata(pred_cat_cox)
sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()

