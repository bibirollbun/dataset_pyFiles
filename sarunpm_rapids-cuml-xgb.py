# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
#https://www.kaggle.com/cdeotte
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 

VER=1


import cudf, numpy as np
from cuml import Lasso
from xgboost import XGBRegressor
from cuml.preprocessing.TargetEncoder import TargetEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


train_df = cudf.read_csv("/kaggle/input/playground-series-s5e2/train.csv") 
train_ex = cudf.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv") 


train = cudf.concat([train_df, train_ex], axis=0, ignore_index=True)


sub_df = cudf.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv") 


train.head()


train = cudf.read_csv("/kaggle/input/playground-series-s5e2/train.csv") 
train["Weight Capacity (kg)"] = (train["Weight Capacity (kg)"]//10)*10 # TRANSFORM TO BINS
print("Train shape:", train.shape )
train.head(1)


test = cudf.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
test["Weight Capacity (kg)"] = (test["Weight Capacity (kg)"]//1000)*1000 # TRANSFORM TO BINS
print("Test shape:", test.shape )
test.head(1)


COLS = list( train.columns[1:-1] )
print( COLS )
print(len(COLS),"uni-grams exist")


new_columns = {}
new_columns2 = {}
COLS2 = []
for i, c1 in enumerate(COLS[:-1]):
    for j, c2 in enumerate(COLS[i+1:]):
        name = f"{c1}-{c2}"
        new_columns[name] = train[c1].astype("str") + "_" + train[c2].astype("str")
        new_columns2[name] = test[c1].astype("str") + "_" + test[c2].astype("str")
        COLS2.append(name)
        print(f"{i}-{i+j+1}, ", end='')
train = cudf.concat([train, cudf.DataFrame(new_columns)], axis=1)
test = cudf.concat([test, cudf.DataFrame(new_columns2)], axis=1)
print()
print(len(COLS2),"bi-grams generated")


new_columns = {}
new_columns2 = {}
COLS3 = []
for i, c1 in enumerate(COLS[:-2]):
    for j, c2 in enumerate(COLS[i+1:-1]):
        for k, c3 in enumerate(COLS[i+j+2:]):
            name = f"{c1}-{c2}-{c3}"
            new_columns[name] = train[c1].astype("str") + "_" + train[c2].astype("str") + "_" + train[c3].astype("str")
            new_columns2[name] = test[c1].astype("str") + "_" + test[c2].astype("str") + "_" + test[c3].astype("str")
            COLS3.append(name)
            print(f"{i}-{i+j+1}-{i+j+k+2}, ", end='')
train = cudf.concat([train, cudf.DataFrame(new_columns)], axis=1)
test = cudf.concat([test, cudf.DataFrame(new_columns2)], axis=1)
print()
print(len(COLS3),"tri-grams generated")


TARGET = "Price"
TARGET_ENCODE = [f"{c}-TE" for c in COLS+COLS2+COLS3]
more_train = cudf.DataFrame(data=np.zeros( (len(train),len(TARGET_ENCODE)) ), columns=TARGET_ENCODE)
train = cudf.concat([train,more_train],axis=1)
more_test = cudf.DataFrame(data=np.zeros( (len(test),len(TARGET_ENCODE)) ), columns=TARGET_ENCODE)
test = cudf.concat([test,more_test],axis=1)

FEATURES = TARGET_ENCODE
print(f"Here are all our {len(FEATURES)} features:")
print( FEATURES )


FOLDS = 10
cv_scores, cv_test_preds, cv_best_scores, cv_best_test_preds = list(), list(), list(), list()
kf = KFold(n_splits=FOLDS, random_state=42, shuffle=True)

# SAVE OOF AND TEST PREDS
oof = np.zeros( len(train) )
pred = np.zeros( len(test) )

# TRAIN/INFER K-FOLD
for i, (train_index, test_index) in enumerate(kf.split(train)):
    
    # PRINT FOLD NUMBER
    if i%FOLDS==0: print()
    print("#"*25)
    print(f"### Fold {i+1} ###")
    print("#"*25)
    
    # GET TRAIN, VALID, TEST
    X_train = train.iloc[train_index,].copy()
    y_train = train[TARGET].iloc[train_index]
    
    X_valid = train.iloc[test_index,].copy()
    y_valid = train[TARGET].iloc[test_index]
    
    X_test = test.copy()
    
    # TARGET ENCODE AND STANDARD ENCODE
    print(f"TE for {len(TARGET_ENCODE)} features...")
    for j,c in enumerate(TARGET_ENCODE):
        c = c.replace("-TE","")
        print(f"{j}, ",end="") 
        
        # TARGET ENCODE 
        enc_auto = TargetEncoder(smooth=5,
                                 split_method="random", 
                                 seed=42, 
                                 stat="mean",
                                 n_folds=5)
        X_train[f"{c}-TE"] = enc_auto.fit_transform(X_train[[c]], y_train)
        X_valid[f"{c}-TE"] = enc_auto.transform(X_valid[[c]])
        X_test[f"{c}-TE"] = enc_auto.transform(X_test[[c]])
        
        # STANDARD ENCODE
        m = X_train[f"{c}-TE"].mean()
        s = X_train[f"{c}-TE"].std()
        X_train[f"{c}-TE"] = (X_train[f"{c}-TE"]-m)/s
        X_valid[f"{c}-TE"] = (X_valid[f"{c}-TE"]-m)/s
        X_test[f"{c}-TE"] = (X_test[f"{c}-TE"]-m)/s
        
    print()    
    X_train = X_train[FEATURES]
    X_valid = X_valid[FEATURES]
    X_test = X_test[FEATURES]
        
    # FIT LASSO MODEL
    model = XGBRegressor(random_state=42, n_estimators=1166, 
                             learning_rate=0.01, 
                             objective= 'reg:squarederror', 
                             max_depth = 6, min_child_weight = 8,
                             subsample = 0.6, colsample_bytree=0.6, 
                             sampling_method = 'uniform', 
                             reg_alpha = 1e-3, reg_lambda = 1e-3, 
                             tree_method='hist', device='cuda')
    #model = Lasso(alpha=1e2) 
    model.fit(X_train, y_train) 
    cv_preds = model.predict(X_valid)
    cv_test_preds.append(model.predict(X_test))

    cv_score = mean_squared_error(y_valid.to_numpy(), cv_preds, squared=False)

    if cv_score < 38.85:
            cv_best_scores.append(cv_score)
            cv_best_test_preds.append(model.predict(X_test))
        
    print(f'Fold{i+1} = RMSE: {round(cv_score,8)} ')
    cv_scores.append(cv_score)


    
    # INFER OOF AND TEST
    oof[test_index] = model.predict(X_valid)#.to_numpy()
    
    print(oof)
    if i==0: 
        pred = model.predict(X_test)#.to_numpy()
        print(pred)
    else: 
        pred += model.predict(X_test)#.to_numpy()
        print(pred)
    
pred /= FOLDS

cv_mean_score = np.mean(cv_scores)  
cv_mean_best_score = np.mean(cv_best_scores)
cv_std_score = np.std(cv_scores)
cv_std_best_score = np.std(cv_best_scores)

print(f"The 5-fold average RMSE score (OOF) : {cv_mean_score}")
print(f"The 5-fold average RMSE Best score (OOF) : {cv_mean_best_score}")
print(f"The 5-fold std oof RMSE score (OOF) : {cv_std_score}")
print(f"The 5-fold std oof RMSE score (OOF) : {cv_std_best_score}")

sumbission_filename = 'submission_' + str(VER) + '_' + str(cv_mean_score) + '_cuML_CV.csv'
#xgb_output = pd.DataFrame({'id': test.id, 'Price': np.mean(cv_test_preds.to_numpy(), axis=0)})
xgb_output = sub_df
xgb_output['Price']=np.mean(cv_test_preds)
xgb_output.to_csv(sumbission_filename, index=False)

sumbission_best_filename = 'submission_' + str(VER) + '_' + str(cv_mean_best_score) + '_cuML_CVB.csv'
xgb_best_output = sub_df
xgb_best_output['Price']=np.mean(cv_best_test_preds, axis=0)
xgb_best_output.to_csv(sumbission_best_filename, index=False)


if False:
    sumbission_filename = 'submission_' + str(VER) + '_' + str(cv_mean_score) + '_cuML_CV.csv'
    #xgb_output = pd.DataFrame({'id': test.id, 'Price': np.mean(cv_test_preds.to_numpy(), axis=0)})
    xgb_output = sub_df
    xgb_output['Price']=np.mean(cv_test_preds, axis=0)
    xgb_output.to_csv(sumbission_filename, index=False)

    sumbission_best_filename = 'submission_' + str(VER) + '_' + str(cv_mean_best_score) + '_cuML_CVB.csv'
    xgb_best_output = sub_df
    xgb_best_output['Price']=np.mean(cv_best_test_preds, axis=0)
    xgb_best_output.to_csv(sumbission_best_filename, index=False)


joined_df = sub_df.drop('Price', axis=1)
joined_df['CV']  = np.mean(cv_test_preds, axis=0)
#joined_df['CVB'] = np.mean(cv_best_test_preds, axis=0)


def chistplot(df, columns, nrows=1, ncols=1):
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Only take the first nrows * ncols columns if there are more than that
    num_cols = nrows * ncols
    columns = columns[:num_cols]
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))

    for i, cname in enumerate(columns):
        if nrows == 1 and ncols == 1:
            ax = axes
        elif nrows == 1 or ncols == 1:
            ax = axes[i]
        else:
            ax = axes[i // ncols, i % ncols]
        
        sns.histplot(data=df, x=cname, kde=True, palette='viridis', ax=ax)
        
        ax.set_xlabel(cname)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {cname}')
        
    # Hide any unused subplots
    for j in range(num_cols, nrows * ncols):
        if nrows == 1 or ncols == 1:
            fig.delaxes(axes[j])
        else:
            fig.delaxes(axes[j // ncols, j % ncols])

    plt.tight_layout()
    #plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.show()


if False:
    chistplot(joined_df, joined_df.columns[1:],1,2)


# COMPUTE AND DISPLAY CV RSME SCORE
rmse = np.sqrt(np.mean( (oof-train.Price.to_numpy())**2 ))
print("Overall CV RSME =",rmse)

# SAVE OOF 
oof_df = train[["id"]].copy()
oof_df["pred"] = oof
oof_df.to_csv(f"oof_v{VER}.csv",index=False)


import pandas as pd
import matplotlib.pyplot as plt
M_IMP=True

if M_IMP:
    import xgboost as xgb
    fig, ax = plt.subplots(figsize=(15,25))
    xgb.plot_importance(model, max_num_features=100, importance_type='gain',ax=ax)
    plt.title("Top 10 Feature Importances (XGBoost)")
    plt.show()




M_IMP = False

if M_IMP:
    # Create a horizontal bar plot
    model.plot_importance(x='Names', y='Numbers', kind='barh', legend=False, figsize=(10, 20))
    # Show the plot
    plt.show()


sub = cudf.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
sub.Price = pred
print("Submission shape:",sub.shape)
sub.to_csv(f"submission_cuML{rmse}.csv",index=False)
sub.head()

