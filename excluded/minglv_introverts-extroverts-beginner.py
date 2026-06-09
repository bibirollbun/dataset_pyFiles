import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# å¿½ç•¥ä¸�å¿…è¦�çš„è­¦å‘Šä¿¡æ�¯
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn æ¨¡å�‹å’Œå·¥å…·
from skopt import BayesSearchCV 
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# æ¨¡å�‹
import xgboost as xgb

# è®¾ç½®ç»˜å›¾æ ·å¼�
sns.set_style('whitegrid')


try:
    # åœ¨Kaggleç�¯å¢ƒä¸­è¿�è¡Œ
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    extra_df = pd.read_csv('/kaggle/input/dataset/personality_datasert.csv')
    sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
except FileNotFoundError:
    # åœ¨æœ¬åœ°æˆ–å…¶ä»–ç�¯å¢ƒä¸­è¿�è¡Œï¼ˆå¦‚æ�œæ•°æ�®åœ¨ä¸�å�Œè·¯å¾„ï¼‰
    # è¯·æ ¹æ�®ä½ çš„å®�é™…è·¯å¾„ä¿®æ”¹
    print("Could not find data in Kaggle default path. You may need to adjust file paths.")
    # ç¤ºä¾‹: 
    # train_df = pd.read_csv('path/to/your/train.csv')
    # test_df = pd.read_csv('path/to/your/test.csv')
    # ...
    # ä¸ºé˜²æ­¢æŠ¥é”™ï¼Œæ­¤å¤„åˆ›å»ºç©ºDataFrame
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    extra_df = pd.DataFrame()
    sample_submission = pd.DataFrame()


print("--- è®­ç»ƒæ•°æ�® ---")
print(f"å½¢çŠ¶: {train_df.shape}")
print("\nå‰�5è¡Œ:")
display(train_df.head())

print("\n--- æµ‹è¯•æ•°æ�® ---")
print(f"å½¢çŠ¶: {test_df.shape}")
print("\nå‰�5è¡Œ:")
display(test_df.head())


print("è®­ç»ƒæ•°æ�®ä¿¡æ�¯:")
train_df.info()


plt.figure(figsize=(8, 6))
sns.countplot(data=train_df, x='Personality', palette=['#ff7f50', '#6495ed'])
plt.title('Distribution of Personality', fontsize=16)
plt.xlabel('Personality', fontsize=12)
plt.ylabel('Counts', fontsize=12)
plt.show()


missing_values = train_df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

plt.figure(figsize=(10, 6))
missing_values.plot(kind='bar')
plt.title('Counts of missing data', fontsize=16)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Counts of missing', fontsize=12)
plt.xticks(rotation=45)
plt.show()


extra_df = extra_df.rename(columns={'Personality': 'known_personality'})
key_cols = [col for col in extra_df.columns if col != 'known_personality']
extra_df = extra_df.drop_duplicates(subset=key_cols)

train_df = train_df.merge(extra_df, how='left', on=key_cols)
test_df = test_df.merge(extra_df, how='left', on=key_cols)

print("å�ˆå¹¶å��è®­ç»ƒé›†å¤§å°�:", train_df.shape)
print("å�ˆå¹¶å��æµ‹è¯•é›†å¤§å°�:", test_df.shape)


train_df


target_encoder = LabelEncoder()
train_df['Personality_encoded'] = target_encoder.fit_transform(train_df['Personality'])

# æŸ¥çœ‹ç¼–ç �ç»“æ�œ
print(f"'Extrovert' ç¼–ç �ä¸º: {target_encoder.transform(['Extrovert'])[0]}")
print(f"'Introvert' ç¼–ç �ä¸º: {target_encoder.transform(['Introvert'])[0]}")


# ä¸ºäº†æ–¹ä¾¿å¤„ç�†ï¼Œæˆ‘ä»¬å°†è®­ç»ƒé›†å’Œæµ‹è¯•é›†å�ˆå¹¶
# å…ˆä¿�å­˜IDå’Œç›®æ ‡å�˜é‡�ï¼Œå› ä¸ºå®ƒä»¬ä¸�å�‚ä¸�è®­ç»ƒ
train_ids = train_df['id']
test_ids = test_df['id']
y = train_df['Personality_encoded']

# åˆ é™¤ä¸�å¿…è¦�çš„åˆ—
train_df = train_df.drop(columns=['id', 'Personality', 'Personality_encoded'])
test_df = test_df.drop(columns=['id'])

# å�ˆå¹¶
all_data = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

print("å�ˆå¹¶å��çš„æ€»æ•°æ�®é›†å½¢çŠ¶:", all_data.shape)


def create_features(df):
    # 1. åº�æ•°ç‰¹å¾�ç¼–ç �
    # å¯¹äº�æœ‰æ˜�æ˜¾é¡ºåº�å…³ç³»çš„åˆ†ç±»ç‰¹å¾�ï¼Œæˆ‘ä»¬å�¯ä»¥æ‰‹åŠ¨ç¼–ç �
    stage_fear_map = {'No': 0, 'Maybe': 1, 'Yes': 2}
    drained_map = {'No': 0, 'Maybe': 1, 'Yes': 2}
    df['Stage_fear_ordinal'] = df['Stage_fear'].map(stage_fear_map)
    df['Drained_ordinal'] = df['Drained_after_socializing'].map(drained_map)

    # 2. åˆ›å»ºä¸€ä¸ªâ€œå¤–å�‘æ€§â€�æŒ‡æ•°
    # è¿™æ˜¯ä¸€ä¸ªåŸºäº�é¢†åŸŸçŸ¥è¯†çš„ç‰¹å¾�ï¼Œé«˜åˆ†å�¯èƒ½æ„�å‘³ç�€æ›´å¤–å�‘
    df['extroversion_index'] = (
        df['Social_event_attendance'] * 0.4 +
        df['Going_outside'] * 0.3 +
        df['Friends_circle_size'] * 0.2 +
        df['Post_frequency'] * 0.1 -
        df['Time_spent_Alone'] * 0.2
    )
    
    # 3. ç¤¾äº¤èƒ½é‡�å¹³è¡¡
    df['social_energy_balance'] = (
        (df['Social_event_attendance'] + df['Going_outside']) / (df['Time_spent_Alone'] + 1) # +1é�¿å…�é™¤ä»¥0
    )
    
    # 4. å¤„ç�†å�ˆå¹¶è¿›æ�¥çš„å¤–éƒ¨æ•°æ�®
    df['known_personality'] = df['known_personality'].fillna('Unknown')
    df = pd.get_dummies(df, columns=['known_personality'], prefix='known')
    
    # åˆ é™¤å�Ÿå§‹åˆ†ç±»åˆ—
    df = df.drop(columns=['Stage_fear', 'Drained_after_socializing'])
    
    return df

all_data_featured = create_features(all_data)
print("ç‰¹å¾�å·¥ç¨‹å��æ•°æ�®å½¢çŠ¶:", all_data_featured.shape)
display(all_data_featured.head())


# ä½¿ç”¨ IterativeImputer
imputer = IterativeImputer(max_iter=10, random_state=42)

# Imputerå�ªèƒ½å¤„ç�†æ•°å€¼å�‹æ•°æ�®
numeric_cols = all_data_featured.select_dtypes(include=np.number).columns
all_data_imputed_array = imputer.fit_transform(all_data_featured[numeric_cols])

# è½¬æ�¢å›�DataFrame
all_data_imputed = pd.DataFrame(all_data_imputed_array, columns=numeric_cols)

print("å¡«å……å��æ˜¯å�¦è¿˜æœ‰ç¼ºå¤±å€¼?", all_data_imputed.isnull().sum().sum() == 0)


scaler = StandardScaler()
all_data_scaled_array = scaler.fit_transform(all_data_imputed)

all_data_final = pd.DataFrame(all_data_scaled_array, columns=all_data_imputed.columns)

display(all_data_final.head())


# å°†å¤„ç�†å¥½çš„æ•°æ�®åˆ†ç¦»å›�è®­ç»ƒé›†å’Œæµ‹è¯•é›†
X = all_data_final.iloc[:len(train_df)]
X_test = all_data_final.iloc[len(train_df):]

print("æœ€ç»ˆè®­ç»ƒé›†å½¢çŠ¶:", X.shape)
print("æœ€ç»ˆæµ‹è¯•é›†å½¢çŠ¶:", X_test.shape)


NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(X.shape[0])
sub_preds = np.zeros(X_test.shape[0])

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    # å®šä¹‰æ¨¡å�‹
    # æˆ‘ä»¬ä½¿ç”¨XGBoostï¼Œè¿™æ˜¯ä¸€ä¸ªå¼ºå¤§ä¸”æµ�è¡Œçš„æ¢¯åº¦æ��å�‡æ¨¡å�‹
    model = xgb.XGBClassifier(
        n_estimators=1000,       # æ ‘çš„æ•°é‡�
        learning_rate=0.05,      # å­¦ä¹ ç�‡
        max_depth=5,             # æ ‘çš„æœ€å¤§æ·±åº¦
        subsample=0.8,           # è®­ç»ƒæ¯�æ£µæ ‘æ—¶ä½¿ç”¨çš„æ ·æœ¬æ¯”ä¾‹
        colsample_bytree=0.8,    # è®­ç»ƒæ¯�æ£µæ ‘æ—¶ä½¿ç”¨çš„ç‰¹å¾�æ¯”ä¾‹
        use_label_encoder=False, # ç¦�ç”¨æ—§çš„æ ‡ç­¾ç¼–ç �å™¨
        eval_metric='logloss',   # è¯„ä¼°æŒ‡æ ‡
        random_state=42,
        n_jobs=-1                # ä½¿ç”¨æ‰€æœ‰CPUæ ¸å¿ƒ
    )

    # è®­ç»ƒæ¨¡å�‹
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)], # æ—©å�œæ³•é˜²æ­¢è¿‡æ‹Ÿå�ˆ
              verbose=False)

    # é¢„æµ‹
    oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
    sub_preds += model.predict_proba(X_test)[:, 1] / folds.n_splits

    print(f"Fold {n_fold+1} éªŒè¯�é›†å‡†ç¡®ç�‡: {accuracy_score(y_valid, model.predict(X_valid))}")

print(f"\næ•´ä½“ OOF å‡†ç¡®ç�‡: {accuracy_score(y, (oof_preds > 0.5).astype(int))}")


best_threshold = 0.5
best_score = 0.0

for threshold in np.arange(0.3, 0.7, 0.01):
    predictions = (oof_preds > threshold).astype(int)
    score = accuracy_score(y, predictions)
    if score > best_score:
        best_score = score
        best_threshold = threshold

print(f"æœ€ä½³é˜ˆå€¼: {best_threshold:.4f}")
print(f"ä½¿ç”¨æœ€ä½³é˜ˆå€¼å��çš„å‡†ç¡®ç�‡: {best_score:.4f}")


final_predictions_encoded = (sub_preds > best_threshold).astype(int)
final_predictions = target_encoder.inverse_transform(final_predictions_encoded)

submission = pd.DataFrame({'id': test_ids, 'Personality': final_predictions})
submission.to_csv('submission.csv', index=False)

print("æ��äº¤æ–‡ä»¶ 'submission.csv' å·²æˆ�åŠŸç”Ÿæˆ�.")
display(submission.head())
submission

