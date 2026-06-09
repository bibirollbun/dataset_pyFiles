import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder,OrdinalEncoder
from sklearn.linear_model import LinearRegression #çº¿æ€§å›�å½’
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")

print("ok")


import numpy as np
import pandas as pd
import re
import time
import math


df_train = pd.read_csv('/kaggle/input/flight-delays-fall-2018/flight_delays_train.csv.zip')  
df_test = pd.read_csv('/kaggle/input/flight-delays-fall-2018/flight_delays_test.csv.zip')  

print("trainï¼š")
#print(df_test.isnull().sum()) 
print("testï¼š")
#print(df_test.isnull().sum()) 

#print(df.nunique())
#print(df.info())
print(df_train.columns)


df_train["DepTime_hours"] = df_train["DepTime"] // 100 + (df_train["DepTime"] % 100) / 60
df_test["DepTime_hours"] = df_test["DepTime"] // 100 + (df_test["DepTime"] % 100) / 60
#print("ok")

all_data = pd.concat([df_train, df_test], ignore_index=True)


all_data['Route'] = all_data['Origin'] + all_data['Dest']
all_data['UniqueCarrier_Origin'] = all_data['UniqueCarrier'] + "_" + all_data['Origin']
all_data['UniqueCarrier_Dest'] = all_data['UniqueCarrier'] + "_" + all_data['Dest']
all_data['is_weekend'] = (all_data['DayOfWeek'] == 6) | (all_data['DayOfWeek'] == 7)

# Hour and minute
all_data['hour'] = all_data['DepTime'] // 100
all_data.loc[all_data['hour'] == 24, 'hour'] = 0
all_data.loc[all_data['hour'] == 25, 'hour'] = 1
all_data['minute'] = all_data['DepTime'] % 100

# give more importance to hour variable
all_data['hour_sq'] = all_data['hour'] ** 2
all_data['hour_sq2'] = all_data['hour'] ** 4

#2- Binning
#Season
all_data['summer'] = (all_data['Month'].isin([6, 7, 8]))
all_data['autumn'] = (all_data['Month'].isin([9, 10, 11]))
all_data['winter'] = (all_data['Month'].isin([12, 1, 2]))
all_data['spring'] = (all_data['Month'].isin([3, 4, 5]))

#Departure Time
#all_data['DayTime'] = 0
all_data.loc[all_data.DepTime <= 600 , 'DepTime_bin'] = 'Night'
all_data.loc[(all_data.DepTime > 600) & (all_data.DepTime <= 1200), 'DepTime_bin'] = 'Morning'
all_data.loc[(all_data.DepTime > 1200) & (all_data.DepTime <= 1800), 'DepTime_bin'] = 'Afternoon'
all_data.loc[(all_data.DepTime > 1800) & (all_data.DepTime <= 2600), 'DepTime_bin'] = 'Evening'


#all_data['DepTime_bin'] = 0
all_data.loc[all_data.DepTime <= 600 , 'DepTime_bin'] = 'vem'
all_data.loc[(all_data.DepTime > 600) & (all_data.DepTime <= 900), 'DepTime_bin'] = 'm'
all_data.loc[(all_data.DepTime > 900) & (all_data.DepTime <= 1200), 'DepTime_bin'] = 'mm'
all_data.loc[(all_data.DepTime > 1200) & (all_data.DepTime <= 1500), 'DepTime_bin'] = 'maf'
all_data.loc[(all_data.DepTime > 1500) & (all_data.DepTime <= 1800), 'DepTime_bin'] = 'af'
all_data.loc[(all_data.DepTime > 1800) & (all_data.DepTime <= 2100), 'DepTime_bin'] = 'n'
all_data.loc[(all_data.DepTime > 2100) & (all_data.DepTime <= 2400), 'DepTime_bin'] = 'nn'
all_data.loc[all_data.DepTime > 2400, 'DepTime_bin'] = 'lm'
#all_data = all_data.drop(['DepTime'], axis=1)




#Distance
#all_data['Dist_bin'] = 0
all_data.loc[all_data.Distance <= 500 , 'Dist_bin'] = 'vshort'
all_data.loc[(all_data.Distance > 500) & (all_data.Distance <= 1000), 'Dist_bin'] = 'short'
all_data.loc[(all_data.Distance > 1000) & (all_data.Distance <= 1500), 'Dist_bin'] = 'mid'
all_data.loc[(all_data.Distance > 1500) & (all_data.Distance <= 2000), 'Dist_bin'] = 'midlong'
all_data.loc[(all_data.Distance > 2000) & (all_data.Distance <= 2500), 'Dist_bin'] = 'long'
all_data.loc[all_data.Distance > 2500, 'Dist_bin'] = 'vlong'
#all_data = all_data.drop(['Distance'], axis=1)


df_train = all_data.iloc[:100000]
df_test = all_data.iloc[100000:]

print("ok")


# ç­›é€‰æ•°æ�®ç±»å�‹ä¸ºé��æ•°å€¼å�‹ï¼ˆobjectã€�datetimeã€�bool ç­‰ï¼‰çš„åˆ—
non_numeric_cols = df_train.select_dtypes(exclude=['number']).columns.tolist()
 
# æ‰“å�°é��æ•°å€¼åˆ—å��
print("é��æ•°å€¼å�‹æ•°æ�®åˆ—å��ï¼š", non_numeric_cols)


print(df_train.info())
le = LabelEncoder()
df_train['dep_delayed_15min'] = le.fit_transform(df_train['dep_delayed_15min'])

cat_cols=['Month', 'DayofMonth', 'DayOfWeek', 'UniqueCarrier',
       'Origin', 'Dest','Route', 'UniqueCarrier_Origin', 'UniqueCarrier_Dest', 
          'DepTime_bin','Dist_bin','dep_delayed_15min',
         'is_weekend', 'summer', 'autumn', 'winter', 'spring']


#cat_cols=['Month', 'DayofMonth', 'DayOfWeek', 'UniqueCarrier',
#       'Origin', 'Dest']


df_all = pd.concat([df_train, df_test], ignore_index=True)  

for i in cat_cols:
    label_enc = LabelEncoder()
    df_all[i] = label_enc.fit_transform(df_all[i])
    df_train[i] = label_enc.transform(df_train[i])
    df_test[i] = label_enc.transform(df_test[i])

table_feature=['Month', 'DayofMonth', 'DayOfWeek', 'UniqueCarrier', 'Origin', 'Dest',
        'Route', 'UniqueCarrier_Origin',
       'UniqueCarrier_Dest', 'is_weekend', 'hour', 'minute', 'hour_sq',
       'hour_sq2', 'summer', 'autumn', 'winter', 'spring', 'DepTime_bin',
       'Dist_bin','DepTime_hours','DepTime','Distance']

#table_feature=['Month', 'DayofMonth', 'DayOfWeek','DepTime','UniqueCarrier',
#       'Origin', 'Dest', 'Distance','DepTime_hours']

X = df_train[table_feature]
y = df_train['dep_delayed_15min']
df_test=df_test[table_feature]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
print("ok")


import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, classification_report
import time

FOLDS = 10
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

fold_auc_scores=[]
# åˆ�å§‹åŒ–å­˜å‚¨æ•°ç»„
n_classes =y.nunique()  # ç±»åˆ«æ•°é‡�
oof_proba = np.zeros((len(df_train), n_classes))  # OOF æ¦‚ç�‡é¢„æµ‹
oof_preds = np.zeros(len(X), dtype=int)  # OOF ç±»åˆ«é¢„æµ‹ï¼ˆå�¯é€‰ï¼‰
test_proba = np.zeros((len(df_test), n_classes))  # æµ‹è¯•é›†æ¦‚ç�‡é¢„æµ‹ï¼ˆå¹³å�‡å��ï¼‰

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\nğŸ”� Fold {fold}/{FOLDS}")
    
    # åˆ’åˆ†è®­ç»ƒé›†å’ŒéªŒè¯�é›†
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # æ•°æ�®é¢„å¤„ç�†ï¼ˆå¦‚æ�œéœ€è¦�ï¼‰
    # X_train_scaled = preprocessor.fit_transform(X_train)
    # X_val_scaled = preprocessor.transform(X_val)
    # test_scaled = preprocessor.transform(df_test)
    X_train_scaled, X_val_scaled, test_scaled = X_train, X_val, df_test  # æ— ç¼©æ”¾æ—¶ç›´æ�¥ä½¿ç”¨å�Ÿæ•°æ�®
    
    # å®šä¹‰ XGBoost æ¨¡å�‹ï¼ˆå¤šåˆ†ç±»ï¼‰
    model = XGBClassifier(
        max_depth=16,
        colsample_bytree=0.4,
        subsample=0.86,
        n_estimators=6000,
        learning_rate=0.01,
        gamma=0.26,
        max_delta_step=5,
        reg_alpha=3,
        reg_lambda=1.4,
        min_child_weight=5,
        #objective='multi:softprob',
        objective='binary:logistic',  # äºŒåˆ†ç±»é»˜è®¤ç›®æ ‡
        eval_metric='logloss',       # äºŒåˆ†ç±»è¯„ä¼°æŒ‡æ ‡

        #eval_metric='mlogloss',
        random_state=42,
        enable_categorical=True,  # å¦‚æ�œä½¿ç”¨ç±»åˆ«ç‰¹å¾�
        device='cuda',  # ä½¿ç”¨ GPU
        n_jobs=-1
    )
    
    # è®­ç»ƒæ¨¡å�‹ï¼ˆå¸¦æ—©å�œï¼‰
    start_time = time.time()
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        early_stopping_rounds=400,
        verbose=500
    )
    print(f"â�±ï¸� Train Time: {time.time() - start_time:.2f}s")
    
    # OOF é¢„æµ‹ï¼ˆæ¦‚ç�‡ + ç±»åˆ«ï¼‰
    val_proba = model.predict_proba(X_val_scaled)
    oof_proba[val_idx] = val_proba
    oof_preds[val_idx] = model.predict(X_val_scaled)  # å�¯é€‰
    
    # æµ‹è¯•é›†é¢„æµ‹ï¼ˆç´¯åŠ å��å�–å¹³å�‡ï¼‰
    test_proba += model.predict_proba(test_scaled) / FOLDS
    
    # è®¡ç®—å½“å‰�æŠ˜çš„ Log Loss å’Œ MAP@3
    fold_logloss = log_loss(y_val, val_proba)
    fold_auc = roc_auc_score(y_val, val_proba[:, 1])  # äºŒåˆ†ç±» AUCï¼ˆå�–æ­£ç±»çš„æ¦‚ç�‡ï¼‰
    fold_auc_scores.append(fold_auc)

    # è®¡ç®—å½“å‰�æŠ˜çš„ Log Loss å’Œ MAP@3
    fold_logloss = log_loss(y_val, val_proba)
    print(f"âœ… Fold {fold} Log Loss: {fold_logloss:.4f}")
    print(f"âœ… Fold {fold} AUC: {fold_auc:.4f}")

    
  
# --- æœ€ç»ˆè¯„ä¼° ---
# è®¡ç®—æ•´ä½“ OOF Log Loss
oof_logloss = log_loss(y, oof_proba)
print(f"\nğŸ“Š Overall OOF Log Loss: {oof_logloss:.4f}")
oof_auc = roc_auc_score(y, oof_proba[:, 1])  # äºŒåˆ†ç±» AUC
print(f"ğŸ“Š Overall OOF AUC: {oof_auc:.4f}")


# OOF åˆ†ç±»æŠ¥å‘Š
oof_pred_category = np.argmax(oof_proba, axis=1)
print("\n=== OOF Classification Report ===")
print(classification_report(
    y, oof_pred_category,
    target_names=[str(c) for c in np.unique(y)],
    digits=4
))


# æµ‹è¯•é›†æœ€ç»ˆé¢„æµ‹ï¼ˆæ¦‚ç�‡ï¼‰
test_pred_category = np.argmax(test_proba, axis=1)  # å¦‚æ�œéœ€è¦�ç±»åˆ«



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# åˆ›å»ºç”»å¸ƒï¼Œæ°´å¹³æ�’åˆ— 2 ä¸ªå­�å›¾
plt.figure(figsize=(16, 6))  # å®½åº¦æ˜¯ 2 å€�å�•å›¾

# ------------------- å­�å›¾1ï¼šæ··æ·†çŸ©é˜µ -------------------
plt.subplot(1, 2, 1)  # 1è¡Œ2åˆ—ï¼Œç¬¬1ä¸ªå›¾
cm = confusion_matrix(y, oof_pred_category)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[str(c) for c in np.unique(y)],
            yticklabels=[str(c) for c in np.unique(y)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# ------------------- å­�å›¾2ï¼šROC æ›²çº¿ -------------------
plt.subplot(1, 2, 2)  # 1è¡Œ2åˆ—ï¼Œç¬¬2ä¸ªå›¾
fpr, tpr, thresholds = roc_curve(y, oof_proba[:, 1])  # å�–æ­£ç±»ï¼ˆ1ï¼‰çš„æ¦‚ç�‡
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # å¯¹è§’çº¿
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

# è°ƒæ•´å¸ƒå±€å¹¶æ˜¾ç¤º
plt.tight_layout()
plt.show()


df = pd.read_csv('/kaggle/input/flight-delays-fall-2018/sample_submission.csv.zip')
df['dep_delayed_15min']=1-test_proba
df.to_csv('version_xgboost_cv_10_1.csv',index=False)
#print()
#print('successfully save!')
df.head(10)

