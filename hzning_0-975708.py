import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# è®¾ç½®æ ·å¼�
plt.style.use("seaborn-v0_8-darkgrid")
warnings.filterwarnings("ignore")
plt.rc("font",family="SimHei",size="15")  #è§£å†³ä¸­æ–‡ä¹±ç �é—®é¢˜
# import csv
train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")




train_df.info()
train_df.describe()


train_df.head()


numeric_df = train_df.select_dtypes(include='number').drop(columns=['id'])
numeric_df.corr()


plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(),annot=True,cmap='coolwarm',fmt='.2f', vmin=-1, vmax=1)
plt.title('Heatmap')
plt.show()


train_df.info()


train_ID = train_df['id']
test_ID = test_df['id']

#Now drop the  'id' colum since it's unnecessary for  the prediction process.
train_df.drop("id", axis = 1, inplace = True)
test_df.drop("id", axis = 1, inplace = True)

ntrain = train_df.shape[0] # è®­ç»ƒé›†æ•°ç›®
ntest = test_df.shape[0] # æµ‹è¯•é›†æ•°ç›®
y_train = train_df['Personality'].map({'Extrovert': 1, 'Introvert': 0}).values # è®­ç»ƒé›†çš„Y

all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
all_data.drop(['Personality'], axis=1, inplace=True)


all_data.info()


# 1. åˆ›å»ºæ–°çš„åˆ†ç»„åˆ—ï¼ŒåŸºäº� Social_event_attendance çš„å››åˆ†ä½�
all_data['social_attend_bin'] = pd.qcut(
    all_data['Social_event_attendance'], 
    q=[0, 0.25, 0.5, 0.75, 1.0], 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)


# 2. å®šä¹‰ä¸€ä¸ªå‡½æ•°ï¼šåœ¨æ¯�ä¸ªåˆ†ç»„å†…ç”¨ä¸­ä½�æ•°å¡«å…… Time_spent_Alone çš„ç¼ºå¤±å€¼
def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

# 3. æ‰§è¡Œåˆ†ç»„å¡«å……
all_data['Time_spent_Alone'] = fill_by_group_median(
    all_data, group_col='social_attend_bin', target_col='Time_spent_Alone'
)

# 4. åˆ é™¤ä¸´æ—¶åˆ†ç»„åˆ—
all_data.drop(columns=['social_attend_bin'], inplace=True)

all_data.info()


# 1. åˆ›å»ºæ–°çš„åˆ†ç»„åˆ—ï¼ŒåŸºäº� Social_event_attendance çš„å››åˆ†ä½�
all_data['Going_outside_bin'] = pd.qcut(
    all_data['Going_outside'], 
    q=[0, 0.25, 0.5, 0.75, 1.0], 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)


# 2. å®šä¹‰ä¸€ä¸ªå‡½æ•°ï¼šåœ¨æ¯�ä¸ªåˆ†ç»„å†…ç”¨ä¸­ä½�æ•°å¡«å…… Time_spent_Alone çš„ç¼ºå¤±å€¼
def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

# 3. æ‰§è¡Œåˆ†ç»„å¡«å……
all_data['Time_spent_Alone'] = fill_by_group_median(
    all_data, group_col='Going_outside_bin', target_col='Time_spent_Alone'
)

# 4. åˆ é™¤ä¸´æ—¶åˆ†ç»„åˆ—
all_data.drop(columns=['Going_outside_bin'], inplace=True)

all_data.info()


# 1. åˆ›å»ºæ–°çš„åˆ†ç»„åˆ—ï¼ŒåŸºäº� Social_event_attendance çš„å››åˆ†ä½�
all_data['Going_outside_bin'] = pd.qcut(
    all_data['Going_outside'], 
    q=[0, 0.25, 0.5, 0.75, 1.0], 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)


# 2. å®šä¹‰ä¸€ä¸ªå‡½æ•°ï¼šåœ¨æ¯�ä¸ªåˆ†ç»„å†…ç”¨ä¸­ä½�æ•°å¡«å…… Time_spent_Alone çš„ç¼ºå¤±å€¼
def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

# 3. æ‰§è¡Œåˆ†ç»„å¡«å……
all_data['Social_event_attendance'] = fill_by_group_median(
    all_data, group_col='Going_outside_bin', target_col='Social_event_attendance'
)

# 4. åˆ é™¤ä¸´æ—¶åˆ†ç»„åˆ—
all_data.drop(columns=['Going_outside_bin'], inplace=True)

all_data.info()


# 1. åˆ›å»ºæ–°çš„åˆ†ç»„åˆ—ï¼ŒåŸºäº� Social_event_attendance çš„å››åˆ†ä½�
all_data['Friends_circle_bin'] = pd.qcut(
    all_data['Friends_circle_size'], 
    q=[0, 0.25, 0.5, 0.75, 1.0], 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)


# 2. å®šä¹‰ä¸€ä¸ªå‡½æ•°ï¼šåœ¨æ¯�ä¸ªåˆ†ç»„å†…ç”¨ä¸­ä½�æ•°å¡«å…… Time_spent_Alone çš„ç¼ºå¤±å€¼
def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

# 3. æ‰§è¡Œåˆ†ç»„å¡«å……
all_data['Social_event_attendance'] = fill_by_group_median(
    all_data, group_col='Friends_circle_bin', target_col='Social_event_attendance'
)

# 4. åˆ é™¤ä¸´æ—¶åˆ†ç»„åˆ—
all_data.drop(columns=['Friends_circle_bin'], inplace=True)

all_data.info()


# 1. åˆ›å»ºæ–°çš„åˆ†ç»„åˆ—ï¼ŒåŸºäº� Social_event_attendance çš„å››åˆ†ä½�
all_data['Post_frequency_bin'] = pd.qcut(
    all_data['Post_frequency'], 
    q=[0, 0.25, 0.5, 0.75, 1.0], 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)


# 2. å®šä¹‰ä¸€ä¸ªå‡½æ•°ï¼šåœ¨æ¯�ä¸ªåˆ†ç»„å†…ç”¨ä¸­ä½�æ•°å¡«å…… Time_spent_Alone çš„ç¼ºå¤±å€¼
def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

# 3. æ‰§è¡Œåˆ†ç»„å¡«å……
all_data['Social_event_attendance'] = fill_by_group_median(
    all_data, group_col='Post_frequency_bin', target_col='Social_event_attendance'
)

# 4. åˆ é™¤ä¸´æ—¶åˆ†ç»„åˆ—
all_data.drop(columns=['Post_frequency_bin'], inplace=True)

all_data.info()


def fill_missing_by_quantile_group(df, group_source_col, target_col, quantiles=[0, 0.25, 0.5, 0.75, 1.0], labels=None):
    """
    æ ¹æ�® group_source_col çš„åˆ†ä½�æ•°å¯¹ target_col åˆ†ç»„ï¼Œå¹¶åœ¨æ¯�ç»„ä¸­ç”¨ç»„å†…ä¸­ä½�æ•°å¡«å…… target_col çš„ç¼ºå¤±å€¼ã€‚
    
    å�‚æ•°ï¼š
        df (pd.DataFrame): å�Ÿå§‹æ•°æ�®
        group_source_col (str): ç”¨äº�åˆ†ç»„çš„åˆ—ï¼ˆæ•°å€¼å�‹ï¼‰
        target_col (str): è¦�å¡«è¡¥ç¼ºå¤±å€¼çš„ç›®æ ‡åˆ—
        quantiles (list): åˆ†ç»„åˆ†ä½�ç‚¹ï¼ˆé»˜è®¤æ˜¯å››åˆ†ä½�ï¼‰
        labels (list): æ¯�ç»„å¯¹åº”çš„æ ‡ç­¾å��ï¼ˆé»˜è®¤è‡ªåŠ¨ç”Ÿæˆ� Q1/Q2/...ï¼‰
        
    è¿”å›�ï¼š
        pd.DataFrame: è¿”å›�å¡«å……å��çš„ DataFrameï¼ˆå�Ÿåœ°ä¿®æ”¹ï¼‰
    """
    # è‡ªåŠ¨ç”Ÿæˆ�åˆ†ç»„æ ‡ç­¾
    if labels is None:
        labels = [f'Q{i+1}' for i in range(len(quantiles)-1)]

    temp_bin_col = f'{group_source_col}_bin'

    # æ­¥éª¤1ï¼šåˆ›å»ºåˆ†ç»„åˆ—
    df[temp_bin_col] = pd.qcut(df[group_source_col], q=quantiles, labels=labels)

    # æ­¥éª¤2ï¼šç»„å†…ç”¨ä¸­ä½�æ•°å¡«å……ç¼ºå¤±å€¼
    df[target_col] = df[target_col].fillna(df.groupby(temp_bin_col)[target_col].transform('median'))

    # æ­¥éª¤3ï¼šåˆ é™¤ä¸´æ—¶åˆ—
    df.drop(columns=[temp_bin_col], inplace=True)

    return df

all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Social_event_attendance',
    target_col='Going_outside'
)

all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Post_frequency',
    target_col='Friends_circle_size'
)
all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Going_outside',
    target_col='Friends_circle_size'
)
all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Friends_circle_size',
    target_col='Post_frequency'
)
all_data.info()


all_data.fillna({
    'Stage_fear': 'UnKnow',
    'Drained_after_socializing': 'UnKnow'
}, inplace=True)
all_data.info()


numeric_all_data = all_data.select_dtypes(include='number')
numeric_all_data.plot(kind='box', title='Boxplot', figsize=(12, 5))


plt.figure(figsize=(8, 5))
sns.histplot(train_df['Time_spent_Alone'], bins=30, kde=True)
plt.title("Time_spent_Alone")
plt.xlabel("Time_spent_Alone")
plt.show()


# å¯¹ä¸¤ä¸ªæŒ‡å®šçš„åˆ—è¿›è¡Œç‹¬çƒ­ç¼–ç �ï¼Œå¹¶è¿�æ�¥å›�å�Ÿå§‹æ•°æ�®ä¸­
all_data = pd.get_dummies(all_data, columns=['Stage_fear', 'Drained_after_socializing'], prefix=['Stage', 'Drained'])
all_data.info()


import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

X_train = all_data[:ntrain]
X_test = all_data[ntrain:]
# å�‚æ•°ç½‘æ ¼
xgb_params = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 150],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# å®�ä¾‹åŒ– XGBoost æ¨¡å�‹
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# ç½‘æ ¼æ�œç´¢äº¤å�‰éªŒè¯�
xgb_cv = GridSearchCV(xgb_model, xgb_params, cv=5, n_jobs=-1, verbose=1)
xgb_cv.fit(X_train, y_train)

# é¢„æµ‹
xgb_pred = xgb_cv.predict(X_test)

# åˆ›å»º DataFrame ä¿�å­˜ç»“æ�œ
kaggle = pd.DataFrame({'id': test_ID, 'Personality': xgb_pred})
kaggle['Personality'] = kaggle['Personality'].map({1: 'Extrovert', 0: 'Introvert'})

# ä¿�å­˜ä¸º CSV æ–‡ä»¶
kaggle.to_csv('submission.csv', index=False)
print("Submitted successfully with XGBoost")


