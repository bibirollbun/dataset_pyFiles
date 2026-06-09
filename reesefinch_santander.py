# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

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


# æ•°æ�®å¤„ç�†
import numpy as np
import pandas as pd
import random
import itertools
from scipy import stats
from scipy.sparse import hstack

# æ•°æ�®å�¯è§†åŒ–
import matplotlib.pyplot as plt
# åœ¨å¯¼å…¥matplotlibå��æ·»åŠ å­—ä½“è®¾ç½®

import seaborn as sns
import matplotlib.font_manager as fm

# ç‰¹å¾�å·¥ç¨‹
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# æ¨¡å�‹
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_curve, roc_auc_score, log_loss

# æ�‚è´¨
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
%matplotlib inline
sns.set(palette='muted', style='whitegrid')
np.random.seed(13154)


train = pd.read_csv('/kaggle/input/santander-customer-satisfaction/train.csv')
test = pd.read_csv('/kaggle/input/santander-customer-satisfaction/test.csv')

print("è®­ç»ƒé›†æ ·æœ¬æ•°ä¸º %iï¼Œå�˜é‡�æ•°ä¸º %i" % (train.shape[0], train.shape[1]))
print("æµ‹è¯•é›†æ ·æœ¬æ•°ä¸º %iï¼Œå�˜é‡�æ•°ä¸º %i" % (test.shape[0], test.shape[1]))


train.head()


test.head()


import os

# åˆ›å»ºoutputç›®å½•ï¼ˆå¦‚æ�œä¸�å­˜åœ¨ï¼‰
os.makedirs('./output', exist_ok=True)

# ç�°åœ¨å�¯ä»¥å®‰å…¨ä¿�å­˜
train.describe().round(3).T.to_csv('./output/train_describe.csv')


# è¿‡æ»¤é›¶æ–¹å·®ç‰¹å¾�
i = 0
for col in train.columns:
    if train[col].var() == 0:
        i += 1
        del train[col]
        del test[col]
print("%i ä¸ªç‰¹å¾�å…·æœ‰é›¶æ–¹å·®å¹¶ä¸”å·²è¢«åˆ é™¤" % (i))


# è¿‡æ»¤ç¨€ç–�ç‰¹å¾�
i = 0

for col in train.columns:
    if np.percentile(train[col], 99) == 0:
        i += 1
        del train[col]
        del test[col]

print("%i ä¸ªç‰¹å¾�æ˜¯ç¨€ç–�çš„å¹¶ä¸”å·²è¢«åˆ é™¤" % (i))


combinations = list(itertools.combinations(train.columns, 2))
print(combinations[:20])
len(combinations)


# åˆ é™¤é‡�å¤�ç‰¹å¾�ï¼Œä¿�ç•™å…¶ä¸€
import itertools

combinations = list(itertools.combinations(train.columns, 2))
remove = []
keep = []

for f1, f2 in combinations:
    if (f1 not in remove) & (f2 not in remove):
        if train[f1].equals(train[f2]):
            remove.append(f1)
            keep.append(f2)

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)
print("%i ä¸ªç‰¹å¾�æ˜¯é‡�å¤�çš„ï¼Œå¹¶ä¸” %i ä¸ªç‰¹å¾�å·²è¢«åˆ é™¤" % (len(remove)*2, len(remove)))
print("å…¶ä¸­ç‰¹å¾� %s è¢«åˆ é™¤\nç‰¹å¾� %s è¢«ä¿�ç•™ä¸‹æ�¥" % (remove, keep))

del remove
del keep
del combinations


train.shape, test.shape


train.isnull().sum().sum()


test.isnull().sum().sum()


# å®šä¹‰ç»˜å›¾å‡½æ•°countplot_target
def countplot_target(df, h=500):
    """
    ç»˜åˆ¶ç›®æ ‡å�˜é‡�çš„é¢‘ç�‡åˆ†å¸ƒï¼Œå¹¶è¾“å‡ºæ»¡æ„�å®¢æˆ·å’Œä¸�æ»¡æ„�å®¢æˆ·çš„æ•°é‡�
    h: æ•°æ�®æ ‡ç­¾çš„é™„åŠ é«˜åº¦
    """
    plt.figure(figsize=(5, 5))
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    ax = sns.countplot(x='TARGET', data=df)
    
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + h,
                '{:1.2f}%'.format(height*100/df.shape[0]), ha="center")
    
    plt.title("TARGET å�˜é‡�çš„é¢‘ç�‡åˆ†å¸ƒå›¾")
    print("æ»¡æ„�å®¢æˆ·çš„æ•°é‡�ä¸º %iï¼Œä¸�æ»¡æ„�å®¢æˆ·çš„æ•°é‡�ä¸º %i" %
          (df[df['TARGET'] == 0].shape[0], df[df['TARGET'] == 1].shape[0]))
    plt.show()


# å®šä¹‰ç»˜å›¾å‡½æ•°histplot_comb
def histplot_comb(col, train=train, test=test, size=(20, 5), bins=20):
    """  
ç»˜åˆ¶è®­ç»ƒé›†å’Œæµ‹è¯•é›†æŸ�ç‰¹å¾�çš„ç›´æ–¹å›¾  
    """  
    plt.subplots(1, 2, figsize=size)  
    
    # è®­ç»ƒé›†ç›´æ–¹å›¾
    plt.subplot(121)  
    plt.title("è®­ç»ƒé›†ç‰¹å¾� {} å€¼çš„åˆ†å¸ƒ".format(col))  
    plt.ylabel('é¢‘æ•°')  
    plt.xlabel(col)  
    plt.hist(train[col], bins=bins, alpha=0.7, color='blue', edgecolor='black')
    
    # æµ‹è¯•é›†ç›´æ–¹å›¾  
    plt.subplot(122)  
    plt.title("æµ‹è¯•é›†ç‰¹å¾� {} å€¼çš„åˆ†å¸ƒ".format(col))  
    plt.ylabel('é¢‘æ•°')
    plt.xlabel(col)
    plt.hist(test[col], bins=bins, alpha=0.7, color='red', edgecolor='black')
    
    plt.tight_layout()
    plt.show()


# å®šä¹‰ç»˜å›¾å‡½æ•°valuecounts_plot
def valuecounts_plot(col, train=train, test=test):
    """  
    ç»˜åˆ¶è®­ç»ƒé›†å’Œæµ‹è¯•é›†ç‰¹å®šåˆ—çš„é¢‘æ•°åˆ†å¸ƒæŠ˜çº¿å›¾ï¼Œå¹¶è¾“å‡ºå‡ºç�°ç™¾åˆ†æ¯”æœ€é«˜çš„å‰� 5 ä¸ªå€¼å’Œæœ€ä½�çš„å‰� 5 ä¸ªå€¼
    """  
    # æ£€æŸ¥åˆ—æ˜¯å�¦å­˜åœ¨
    if col not in train.columns:
        print(f"é”™è¯¯ï¼šè®­ç»ƒé›†ä¸­ä¸�å­˜åœ¨åˆ— '{col}'")
        return
    if col not in test.columns:
        print(f"è­¦å‘Šï¼šæµ‹è¯•é›†ä¸­ä¸�å­˜åœ¨åˆ— '{col}'")
    
    # åˆ›å»ºå›¾å½¢
    plt.subplots(1, 2, figsize=(15, 6))
    
    # è®­ç»ƒé›†é¢‘æ•°åˆ†å¸ƒ
    plt.subplot(121)
    df_train = train[col].value_counts().sort_index()
    sns.lineplot(x=df_train.index, y=df_train.values)
    plt.title("{} çš„é¢‘æ•°åˆ†å¸ƒæŠ˜çº¿å›¾ï¼ˆè®­ç»ƒé›†ï¼‰".format(col))
    plt.xlabel(col)
    plt.ylabel('é¢‘æ•°')
    
    # æµ‹è¯•é›†é¢‘æ•°åˆ†å¸ƒ
    plt.subplot(122)
    df_test = test[col].value_counts().sort_index()
    sns.lineplot(x=df_test.index, y=df_test.values)
    plt.title("{} çš„é¢‘æ•°åˆ†å¸ƒæŠ˜çº¿å›¾ï¼ˆæµ‹è¯•é›†ï¼‰".format(col))
    plt.xlabel(col)
    plt.ylabel('é¢‘æ•°')
    
    plt.tight_layout()
    plt.show()
    
    # è¾“å‡ºç»Ÿè®¡åˆ†æ��
    print("*" * 100)
    print("è®­ç»ƒé›†ç‰¹å¾� '%s' å…¶å€¼å� æ¯”ï¼ˆtop 5ï¼‰ï¼š" % (col))
    print("å€¼\tå� æ¯”%")
    print((train[col].value_counts() * 100 / train.shape[0]).iloc[:5])
    
    print("*" * 100)
    print("è®­ç»ƒé›†ç‰¹å¾� '%s' å…¶å€¼å� æ¯”ï¼ˆbottom 5ï¼‰ï¼š" % (col))
    print("å€¼\tå� æ¯”%")
    print((train[col].value_counts() * 100 / train.shape[0]).iloc[-5:])
    
    print("*" * 100)
    print("æµ‹è¯•é›†ç‰¹å¾� '%s' å…¶å€¼å� æ¯”ï¼ˆtop 5ï¼‰ï¼š" % (col))
    print("å€¼\tå� æ¯”%")
    print((test[col].value_counts() * 100 / test.shape[0]).iloc[:5])
    
    print("*" * 100)
    print("æµ‹è¯•é›†ç‰¹å¾� '%s' å…¶å€¼å� æ¯”ï¼ˆbottom 5ï¼‰ï¼š" % (col))
    print("å€¼\tå� æ¯”%")
    print((test[col].value_counts() * 100 / test.shape[0]).iloc[-5:])


# å®šä¹‰ç»˜å›¾å‡½æ•°histplot_target
def histplot_target(col, df=train, height=6, bins=20):
    """
    ç»˜åˆ¶æ•°æ�®é›†ç‰¹å¾�åˆ—åœ¨ä¸�å�Œç›®æ ‡å�˜é‡�å€¼ä¸‹çš„é¢‘æ•°åˆ†å¸ƒå›¾
    """
    sns.FacetGrid(data=df, hue='TARGET', height=height).map(plt.hist, col, bins=bins).add_legend()
    plt.title("ç‰¹å¾�%såœ¨ä¸�å�Œç›®æ ‡å�˜é‡�å€¼ä¸‹çš„é¢‘æ•°åˆ†å¸ƒ" % (col))
    plt.ylabel("é¢‘æ•°")
    plt.show()


# è°ƒç”¨å‡½æ•°åˆ†æ��è®­ç»ƒé›†çš„TARGETåˆ†å¸ƒ
countplot_target(train)


##æŸ¥çœ‹VAR3è¿™ä¸ªç‰¹å¾�
np.array(sorted(train['var3'].unique()))


print("var3å”¯ä¸€å€¼çš„æ•°é‡�ä¸º: %i" % (len(np.array(sorted(train['var3'].unique())))))


print("å€¼\tè®¡æ•°")
print(train['var3'].value_counts()[:5])
print("\nå€¼\tå� æ¯”%")
print(train['var3'].value_counts()[:5] / train.shape[0] * 100)


print("å€¼\tè®¡æ•°")
print(test['var3'].value_counts()[:5])
print("\nå€¼\tå� æ¯”%")
print(test['var3'].value_counts()[:5] / test.shape[0] * 100)


train['var3'].replace(-999999, 2, inplace=True)
test['var3'].replace(-999999, 2, inplace=True)


countplot_target(train[train['var3'] == 2], h=20)
countplot_target(train[train['var3'] != 2], h=10)


histplot_comb('var15')
print("è®­ç»ƒé›†ä¸­å¹´é¾„åœ¨30å²�ä»¥ä¸‹çš„å®¢æˆ·çº¦å� æ‰€æœ‰æ•°æ�®çš„ %.2f%%" % (stats.percentileofscore(train['var15'].values, 30)))
print("æµ‹è¯•é›†ä¸­å¹´é¾„åœ¨30å²�ä»¥ä¸‹çš„å®¢æˆ·çº¦å� æ‰€æœ‰æ•°æ�®çš„ %.2f%%" % (stats.percentileofscore(test['var15'].values, 30)))


ax = histplot_target('var15', bins=40)

plt.figure(figsize=(6, 6))
mask = train[train['TARGET'] == 1]
plt.hist(mask['var15'], color='orange')
plt.title("ç‰¹å¾� var15 åœ¨ target=1 ä¸‹çš„é¢‘æ•°åˆ†å¸ƒ")
plt.xlabel('var15')
plt.show()

print("ä¸�æ»¡æ„�å®¢æˆ·çš„var15æœ€å°�å€¼ä¸ºï¼š%iï¼Œä¸�æ»¡æ„�å®¢æˆ·çš„var15æœ€å¤§å€¼ä¸ºï¼š%i" % (mask['var15'].min(), mask['var15'].max()))


# ç”±å‰�é�¢å·²ç»�çŸ¥é�“23æ˜¯ä¸ªåˆ†æ°´å²­ï¼Œåˆ›å»ºæ–°ç‰¹å¾�ç”¨æ�¥åˆ¤æ–­å®¢æˆ·æ˜¯å�¦å°�äº�23å²�.æ˜¯1å�¦0
for df in [train, test]:
    df['var15_below_23'] = np.zeros(df.shape[0], dtype=int)
    df.loc[df['var15'] < 23, 'var15_below_23'] = 1


#å†�è¯•è¯•çœ‹23æ˜¯ä¸�æ˜¯çœŸçš„åˆ†æ°´å²­ï¼Œç­‰è·�åˆ†ç®±ä¸º5æ®µåˆ†ç®±å��ä¼šç¦»æ•£åŒ–æ›´å®¹æ˜“äºŒåˆ†ç±»
_, bins = pd.cut(train['var15'].values, 5, retbins=True)
print(_)


train['var15'] = pd.cut(train['var15'].values, bins, labels=False)
test['var15'] = pd.cut(test['var15'].values, bins, labels=False)
histplot_target('var15')


print("var38 æœ€å°�å€¼ä¸º: %.3f, æœ€å¤§å€¼ä¸º: %.3f" % (train['var38'].min(), train['var38'].max()))


train['var38'].value_counts()


for i in np.arange(0, 1.1, 0.1):
    print("%i percentile: %i" % (i*100, np.quantile(train['var38'].values, i)))


mask = train[train['var38'] <= np.quantile(train['var38'].values, 0.975)]
histplot_target('var38', df=mask, bins=20)


mask['var38'] = np.log(mask['var38'].values)
histplot_target('var38', df=mask, bins=20)


for df in [train, test]:
    df['var38'] = np.log(df['var38'].values)
histplot_target('var38', bins=20)


import re
[col for col in train.columns if col[:3] == 'var']


f_keywords = {col.split('_')[0] for col in train.columns if (len(col.split('_')) > 1) & ('var15' not in col)}
f_keywords


# è®¡ç®—æ¯�ç§�å…³é”®è¯�å‰�ç¼€ç‰¹å¾�çš„æ•°é‡�
f_keywords = dict(zip(f_keywords, np.zeros(len(f_keywords), dtype=int)))
for key in f_keywords.keys():
    for col in train.columns:
        if key in col:
            f_keywords[key] += 1
f_keywords


# å°†ç»Ÿè®¡ç»“æ�œè½¬æ�¢ä¸º pandas Series
k = pd.Series(f_keywords)

# ä½¿ç”¨ seaborn ç»˜åˆ¶æ�¡å½¢å›¾
ax = sns.barplot(x=k.index, y=k.values)

# è®¾ç½®å›¾è¡¨æ ‡é¢˜å’Œå��æ ‡è½´æ ‡ç­¾
plt.title("ç‰¹å¾�å…³é”®è¯�å‰�ç¼€çš„é¢‘æ•°åˆ†å¸ƒ")
plt.ylabel('é¢‘æ•°')
plt.xlabel('å…³é”®è¯�å‰�ç¼€')  # ä¿®æ­£ï¼šåˆ é™¤äº†å¤šä½™çš„å�³æ‹¬å�·

# æ˜¾ç¤ºå›¾è¡¨
plt.show()


imp = [col for col in train.columns if 'imp' in col]
print('å…³é”®è¯�å‰�ç¼€impç‰¹å¾�å…±æœ‰ %d ä¸ª' % (len(imp)))
imp


import random
random.seed(a=0)
print('è¢«éš�æœºé€‰æ‹©ä¸­çš„impç‰¹å¾�ï¼š%s' % (random.sample(imp, 1)[0]))


col = 'imp_trans_var37_ult1'
print('è®­ç»ƒé›†ä¸­ %s æœ€å°�å€¼ä¸ºï¼š%iï¼Œæœ€å¤§å€¼ä¸ºï¼š%i' % (col, train[col].min(), train[col].max()))
print('æµ‹è¯•é›†ä¸­ %s æœ€å°�å€¼ä¸ºï¼š%iï¼Œæœ€å¤§å€¼ä¸ºï¼š%i' % (col, test[col].min(), test[col].max()))


valuecounts_plot(train=train, test=test, col=col)


valuecounts_plot(train=train[train[col] != 0], test=test[test[col] != 0], col=col)


df = train[train[col] != 0]
df1 = test[test[col] != 0]
for data in [df, df1]:
    data.loc[data[col] != 0, col] = np.log(data.loc[data[col] != 0, col])

histplot_comb(col, train=df, test=df1)


for df in [train, test]:
    df.loc[df[col] != 0, col] = np.log(df.loc[df[col] != 0, col])
histplot_comb(col, train=train, test=test)


import numpy as np

# å¯¹æ‰€æœ‰impç‰¹å¾�è¿›è¡Œå¤„ç�†
for col in imp:
    # è®­ç»ƒé›†
    train[col] = np.where(train[col] != 0, np.log(train[col]), train[col])
    # æµ‹è¯•é›†  
    test[col] = np.where(test[col] != 0, np.log(test[col]), test[col])


saldo = [col for col in train.columns if 'saldo' in col]
print("å…³é”®è¯�å‰�ç¼€saldoç‰¹å¾�å…±æœ‰ %i ä¸ª" % (len(saldo)))
saldo


# å¯¹saldoç‰¹å¾�è¿›è¡Œå¯¹æ•°å�˜æ�¢ï¼Œ0å€¼ä¿�æŒ�ä¸�å�˜
import numpy as np

for col in saldo:
    # è®­ç»ƒé›†
    train[col] = np.where(train[col] != 0, np.log(train[col]), train[col])
    # æµ‹è¯•é›†  
    test[col] = np.where(test[col] != 0, np.log(test[col]), test[col])


num = [col for col in train.columns if 'num' in col]
print('å…³é”®è¯�å‰�ç¼€numç‰¹å¾�å…±æœ‰ %i ä¸ª' % (len(num)))
num[:10]


train.shape, test.shape


#Processing other features with â€�numâ€œ

THRESHOLD = 10

num = [
    col for col in train.columns 
    if col.startswith('num') and 
    max(train[col].nunique(), test[col].nunique()) <= THRESHOLD
]



train.to_pickle('output/train.pkl')
test.to_pickle('output/test.pkl')


train.describe()



train = pd.read_pickle('output/train.pkl')
test = pd.read_pickle('output/test.pkl')
X_train = train.copy()
X_test = test.copy()
X_train.shape, X_test.shape


def add_feature_no_zeros(train=X_train, test=X_test):
  """
  æ�„é€ æ–°ç‰¹å¾�ï¼Œè¡¨ç¤ºæ¯�è¡Œæ ·æœ¬ä¸­143ä¸ªç‰¹å¾�å�–å€¼ä¸ºé›¶æˆ–é��é›¶çš„å‡ºç�°æ¬¡æ•°
  """
  col = [k for k in train.columns if k != 'TARGET']
  for df in [train, test]:
    df['no_zeros'] = (df.loc[:, col] == 0).sum(axis=1).values
    df['no_nonzeros'] = (df.loc[:, col] != 0).sum(axis=1).values


def add_feature_no_zeros_keyword(keyword, train=X_train, test=X_test):
    """
æ�„é€ æ–°ç‰¹å¾�ï¼Œè¡¨ç¤ºæ¯�è¡Œæ ·æœ¬ä¸­å¯¹äº�æ¯�ä¸€ç§�å…³é”®è¯�å�¯å�šçš„ç‰¹å¾�å�–å€¼ä¸ºé›¶æˆ–é��é›¶çš„å‡ºç�°æ¬¡æ•°
    """
    col = [k for k in train.columns if keyword in k]
    for df in [train, test]:
        df['no_zeros_' + keyword] = (df.loc[:, col] == 0).sum(axis=1).values
        df['no_nonzeros_' + keyword] = (df.loc[:, col] != 0).sum(axis=1).values

add_feature_no_zeros()
keywords = ['imp', 'saldo', 'num', 'ind']
for k in keywords:
    add_feature_no_zeros_keyword(k)


X_train.shape, X_test.shape


def average_col(col, features, train=X_train, test=X_test):
    """
è�·å�– 'col' ç‰¹å¾�ä¸­å�–æ¯�ä¸€ç§�å”¯ä¸€å€¼çš„æƒ…å†µä¸‹featureç‰¹å¾�çš„å�‡å€¼ï¼Œå¹¶ä»¤å…¶ä¸ºæ–°ç‰¹å¾�
    """

    for df in [train, test]:
        unique_values = df[col].unique()

        for feature in features:
            avg_value = []
            for value in unique_values:
                # å¯¹äº�æ¯�ä¸€ä¸ªç‰¹å¾�åˆ—colï¼Œæ±‚å…¶æ¯�ä¸€ç§�å”¯ä¸€å€¼çš„æƒ…å†µä¸‹featureç‰¹å¾�çš„å�‡å€¼
                avg = df.loc[df[col] == value, feature].mean()
                avg_value.append(avg)
            avg_dict = dict(zip(unique_values, avg_value))
            new_col = 'avg_' + col + '_' + feature
            df[new_col] = np.zeros(df.shape[0])
            for value in unique_values:
                df.loc[df[col] == value, new_col] = avg_dict[value]


features = [i for i in X_train.columns if (('imp' in i) or ('saldo' in i)) & ('no_zeros' not in i)]

# æŸ¥æ‰¾åˆ—ä¸­å”¯ä¸€å€¼æ•°é‡�åœ¨ 50 åˆ° 210 ä¹‹é—´çš„åˆ—
columns = [i for i in X_train.columns if (X_train[i].nunique() <= 210) & (X_train[i].nunique() > 50)]

len(features), len(columns)


%%time
for col in tqdm(columns):
    average_col(col,features)


X_train.shape, X_test.shape


def remove_corr_var(train=X_train, test=X_test, target_threshold=10**-3, within_threshold=0.95):
    """
    åˆ é™¤ä¸�ç›®æ ‡å�˜é‡�ç›¸å…³æ€§ä½�çš„ç‰¹å¾�ï¼Œåˆ é™¤å½¼æ­¤ä¹‹é—´ç›¸å…³æ€§é«˜çš„ç‰¹å¾�ï¼ˆä¿�ç•™ä¸€ä¸ªï¼‰
    """
    # åˆ é™¤ä¸�ç›®æ ‡å�˜é‡�ç›¸å…³æ€§ä½�çš„ç‰¹å¾�
    initial_feature = train.shape[1]
    corr = train.drop("ID", axis=1).corr().abs().T
    corr_target = pd.DataFrame(corr['TARGET'].sort_values())  # ç§»é™¤ by='TARGET' å�‚æ•°
    feat_df = corr_target[(corr_target['TARGET']) <= target_threshold]
    print("æœ‰ %i ä¸ªç‰¹å¾�å› ä¸ºä¸�ç›®æ ‡å�˜é‡�TARGETçš„ç›¸å…³ç³»æ•°ç»�å¯¹å€¼å°�äº� %.3f è€Œè¢«åˆ é™¤" % (feat_df.shape[0], target_threshold))
    print("åˆ é™¤ä¸­......")
    for df in [train, test]:
        df.drop(feat_df.index, axis=1, inplace=True)
    print("å·²åˆ é™¤ï¼�")
    
    # åˆ é™¤å½¼æ­¤ä¹‹é—´ç›¸å…³æ€§é«˜çš„ç‰¹å¾�ï¼ˆä¿�ç•™ä¸€ä¸ªä¸�TARGETç›¸å…³æ€§æœ€é«˜çš„ç‰¹å¾�ï¼‰
    corr.sort_values(by='TARGET', ascending=False, inplace=True) # å°†ç›¸å…³çŸ©é˜µæ¯�ä¸€è¡Œå…ˆæŒ‰TARGETåˆ—é™�åº�æ�’åº�
    corr = corr.reindex(columns=corr.index) # å†�å°†æ¯�ä¸€åˆ—æŒ‰ç…§è¡Œç´¢å¼•é‡�æ�’åº�
    corr.drop('TARGET', axis=1, inplace=True) # åˆ é™¤TARGETåˆ—
    corr.drop('TARGET', axis=0, inplace=True)
    corr.drop(feat_df.index, axis=1, inplace=True) # åˆ é™¤feat_dfä¸­ç‰¹å¾�åœ¨corrè¡¨é‡Œçš„åˆ—
    corr.drop(feat_df.index, inplace=True)
    
    # ä¿®å¤�ï¼šä½¿ç”¨ bool æ›¿ä»£ np.bool
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)) # è�·å�–ç›¸å…³çŸ©é˜µçš„ä¸Šä¸‰è§’
    
    column = [col for col in upper.columns if any(upper[col] > within_threshold)] # è�·å�–ä¸�ç‰¹å¾�ä¹‹ä¸€é«˜åº¦ç›¸å…³çš„æ‰€æœ‰åˆ—
    print("æœ‰ %i ä¸ªç‰¹å¾�ä¸�å�¦ä¸€ä¸ªç‰¹å¾�é«˜åº¦ç›¸å…³ä¸”ç›¸å…³ç³»æ•°ä¸º %.3f å�Šä»¥ä¸Šè€Œè¢«åˆ é™¤" % (len(column), within_threshold))
    print("åˆ é™¤ä¸­......")
    for df in [train, test]:
        df.drop(column, axis=1, inplace=True)
    print("å·²åˆ é™¤ï¼�")
    print("ç‰¹å¾�æ•°ä»�%iä¸ªï¼Œå�˜æˆ�%iä¸ªï¼Œå…¶ä¸­%iä¸ªç‰¹å¾�å·²åˆ é™¤"%
          (initial_feature, test.shape[1], initial_feature - test.shape[1]))



%%time
remove_corr_var()


X_train.shape


# ä¿�å­˜ä¸ºPæ–‡ä»¶ï¼Œæ–¹ä¾¿å��ç»­è°ƒç”¨
X_train.to_pickle('./output/X_train.pkl')
X_test.to_pickle('./output/X_test.pkl')

# è¯»å�–ä¸Šè¿°Pæ–‡ä»¶
X_train = pd.read_pickle('./output/X_train.pkl')
X_test = pd.read_pickle('./output/X_test.pkl')
X_train.shape, X_test.shape


def apply_loglp(column, train=X_train, test=X_test):
    """
    å¯¹æ•°å�˜æ�¢æ‰€æœ‰åˆ—ç‰¹å¾�
    """
    tr = train.copy()
    te = test.copy()
    for df in [tr, te]:
        for col in column:
            df.loc[df[col] >= 0, col] = np.log1p(df.loc[df[col] >= 0, col].values)
    return tr, te



# å¯¹æ‰€æœ‰æœ€å°�å€¼å¤§äº�ç­‰äº�0çš„impå’Œsaldoç‰¹å¾�è¿›è¡Œå¯¹æ•°å�˜æ�¢ï¼ˆvar38åœ¨EDAä¸­å·²ç»�å¯¹æ•°åŒ–ï¼Œè¿™é‡Œä¸�å†�æ“�ä½œï¼‰
features = [i for i in X_train.columns if (('saldo' in i) | ('imp' in i)) & ((X_train[i].values >= 0).all())]
X_train_1, X_test_1 = apply_loglp(features)


X_train_1.to_pickle('./output/X_train_1.pkl')
X_test_1.to_pickle('./output/X_test_1.pkl')


# é€‰æ‹©å”¯ä¸€å€¼åœ¨ (2, 10]èŒƒå›´ä¸­çš„ç‰¹å¾�
cat_col = []
for col in X_train.columns:
    if (X_train[col].nunique() <= 10) & (col != 'TARGET') & (X_train[col].nunique() > 2):
        cat_col.append(col)

print("æœ‰ %i ä¸ªç‰¹å¾�å…¶å”¯ä¸€å€¼æ•°é‡�(2, 10] å¹¶ä½¿ç”¨å®ƒä»¬åˆ›å»ºç‹¬çƒ­ç¼–ç �å’Œå“�åº”ç¼–ç �å�˜é‡�ï¼Œå�Œæ—¶åˆ é™¤å�Ÿå§‹ç‰¹å¾�" % (len(cat_col)))


def one_hot_encoding(col, train=X_train, test=X_test):
    """å¯¹è®­ç»ƒé›†å’Œæµ‹è¯•é›†ä¸­çš„ç‰¹å¾�è¿›è¡Œç‹¬çƒ­ç¼–ç �
    """ 
    ohe = OneHotEncoder(sparse=True, handle_unknown='ignore')
    ohe.fit(train[col])
    feature_names = list(ohe.get_feature_names_out(input_features=col))
    features = list(train.drop(col, axis=1).columns)
    features.extend(feature_names)

    # train
    df = train.copy()
    temp = ohe.transform(df[col])
    df.drop(col, axis=1, inplace=True)
    train = pd.DataFrame(hstack([df.values, temp]).toarray(), columns=features)
    train = train.loc[:, ~train.columns.duplicated(keep='first')] # åˆ é™¤é‡�å¤�åˆ—
    
    # test
    df = test.copy()
    temp = ohe.transform(df[col])
    df.drop(col, axis=1, inplace=True)
    features.remove('TARGET')
    test = pd.DataFrame(hstack([df.values, temp]).toarray(), columns=features)
    test = test.loc[:, ~test.columns.duplicated(keep='first')]

    return train, test


X_train_ohe, X_test_ohe = one_hot_encoding(cat_col)
X_train_1_ohe, X_test_1_ohe = one_hot_encoding(cat_col, X_train_1, X_test_1)
X_train_ohe.shape, X_test_ohe.shape, X_train_1_ohe.shape, X_test_1_ohe.shape


def response_encoding_return(df, column, target, alpha=5000):
    """
    ä½¿ç”¨å¸¦æœ‰æ‹‰æ™®æ‹‰æ–¯å¹³æ»‘çš„å“�åº”ç¼–ç �åˆ°åˆ†ç±»åˆ—columnï¼Œå¹¶åœ¨è®­ç»ƒã€�æµ‹è¯•ã€�éªŒè¯�æ•°æ�®é›†ä¸­è½¬æ�¢ç›¸åº”çš„åˆ—ã€‚
    æ­¤å‡½æ•°ç”¨æ�¥è®­ç»ƒå‡ºæœ€ä¼˜çš„å�‚æ•°alpha
    """
    unique_values = set(df[column].values)
    dict_values = {}
    for value in unique_values:
        total = len(df[df[column] == value])
        sum_promoted = len(df[(df[column] == value) & (df[target] == 1)]) 
        dict_values[value] = np.round((sum_promoted + alpha) / (total + alpha * len(unique_values)), 2)
    return dict_values


# å¯»æ‰¾æœ€å¥½çš„alpha
def find_alpha(seed):
    random.seed(seed)
    ran_in = random.randint(0, 9) # éš�æœºç”Ÿæˆ�0-9çš„æ•´æ•°
    col = [col for col in cat_col if X_train[col].nunique() > 3][ran_in]
    print('Feature: "%s"' % (col))
    for alpha in [100, 500, 1000, 2500, 5000, 10000]:
        print('for alpha %i: %s' % (alpha, response_encoding_return(X_train, col, "TARGET", alpha=alpha)))


find_alpha(seed=100)


find_alpha(seed=1000)


def response_encoding(df, test_df, column, target='TARGET', alpha=5000):
    """
    åœ¨è¿™é‡Œï¼Œæˆ‘ä»¬ä½¿ç”¨å¸¦æœ‰æ‹‰æ™®æ‹‰æ–¯å¹³æ»‘çš„å“�åº”ç¼–ç �åˆ°åˆ†ç±»åˆ—ï¼Œå¹¶åœ¨è®­ç»ƒã€�æµ‹è¯•ã€�éªŒè¯�æ•°æ�®é›†ä¸­è½¬æ�¢ç›¸åº”çš„åˆ—ã€‚
    åœ¨è¿™é‡Œï¼Œæˆ‘ä»¬å°†é‡�å¤�æ¯�ä¸ªç±»åˆ«çš„å€¼ alpha æ—¶é—´ã€‚
    """ 
    feature = column + '_1'
    feature_ = column + '_0'
    unique_values = set(df[column].values)
    dict_values = {} # å­˜å‚¨target=1çš„å“�åº”ç¼–ç �å€¼
    dict_values_ = {} # å­˜å‚¨target=0çš„å“�åº”ç¼–ç �å€¼

    for value in unique_values:
        total = len(df[df[column] == value]) # æ­¤ç±»åˆ«å€¼åœ¨dfä¸­çš„æ€»ä¸ªæ•°
        # ç±»åˆ«ä¸ºæŸ�'value'å€¼ä¸”ç›®æ ‡å�˜é‡�å�–1æ—¶åœ¨dfä¸­çš„æ€»ä¸ªæ•°
        sum_promoted = len(df[(df[column] == value) & (df[target] == 1)])
        sum_unpromoted = total - sum_promoted # ç±»åˆ«ä¸ºæŸ�'value'å€¼ä¸”ç›®æ ‡å�˜é‡�å�–0æ—¶åœ¨dfä¸­çš„æ€»ä¸ªæ•°
        dict_values[value] = np.round((sum_promoted + alpha) / (total + alpha * len(unique_values)), 2) # æ‹‰æ™®æ‹‰æ–¯å¹³æ»‘
        dict_values_[value] = np.round((sum_unpromoted + alpha) / (total + alpha * len(unique_values)), 2)
    dict_values['unknown'] = 0.5 # åœ¨è®­ç»ƒé›†ä¸Šè§‚æµ‹ä¸�åˆ°çš„æœªçŸ¥ç±»åˆ«å°†è¢«åˆ†é…�ä¸º0.5
    dict_values_['unknown'] = 0.5
    df[feature] = (df[column].map(dict_values)).values
    df[feature_] = (df[column].map(dict_values_)).values
    df.drop(column, axis=1, inplace=True)

    unique_values_test = set(test_df[column].values)
    # æ‰¾å‡ºä¸¤setä¸­ä¸�å�Œå…ƒç´ å¹¶å°†å…¶èµ‹å€¼ä¸ºunknown
    test_df[column] = test_df[column].apply(lambda x: 'unknown' if x in (unique_values_test - unique_values) else x)
    test_df[feature] = (test_df[column].map(dict_values)).values
    test_df[feature_] = (test_df[column].map(dict_values_)).values
    test_df.drop(column, axis=1, inplace=True)


alpha = 100
X_train_re = X_train.copy()
X_test_re = X_test.copy()
X_train_1_re = X_train_1.copy()
X_test_1_re = X_test_1.copy()
for col in tqdm(cat_col):
    response_encoding(X_train_re, X_test_re, col, alpha=alpha)
    response_encoding(X_train_1_re, X_test_1_re, col, alpha=alpha)

X_train_re.shape, X_test_re.shape, X_train_1_re.shape, X_test_1_re.shape


def stdzation(train, test):
    """
    å¯¹ç‰¹å¾�è¿›è¡Œæ ‡å‡†åŒ–
    """
    col = [i for i in train.columns if (i != 'TARGET') & (i != 'ID')]
    scaler = StandardScaler()
    train[col] = scaler.fit_transform(train[col])
    test[col] = scaler.transform(test[col])


datasets = [
    (X_train, X_test), 
    (X_train_re, X_test_re), 
    (X_train_ohe, X_test_ohe),
    (X_train_1, X_test_1), 
    (X_train_1_re, X_test_1_re), 
    (X_train_1_ohe, X_test_1_ohe)
]

for train, test in datasets:
    stdzation(train, test)


datasets_labels = ['normal', 'normal_re', "normal_ohe", "log", 'log_re', "log_ohe"]
print("ä¸�å�Œæ•°æ�®é›†æœ€ç»ˆçš„ç‰¹å¾�æ•°æ˜¯ï¼š")
for i, (train, test) in enumerate(datasets):
    print("%s:\t%i" % (datasets_labels[i], test.shape[1]))


for i, (train, test) in enumerate(datasets):
    file = datasets_labels[i] + '.pkl'
    train.to_pickle('./output/train_' + file)
    test.to_pickle('./output/test_' + file)


# åŠ è½½æ•°æ�®é›†
dataset = 'Normal'
train = pd.read_pickle('./output/train_normal.pkl')
test = pd.read_pickle('./output/test_normal.pkl')
X_train = train.drop(['ID', 'TARGET'], axis=1)
y_train = train['TARGET'].values
X_test = test.drop('ID', axis=1)
test_id = test['ID']
del train, test

# åˆ’åˆ†æ•°æ�®é›†
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.15)
X_train.shape, X_val.shape, X_test.shape


global i
i = 0

def plot_auc(y_true, y_pred, label, dataset=dataset):
    """ 
    ç»™å‡º y_true å’Œ y_pred æ—¶ç»˜åˆ¶ROCæ›²çº¿
    dataset: å‘Šè¯‰æˆ‘ä»¬ä½¿ç”¨äº†å“ªä¸ªæ•°æ�®é›†
    label: å‘Šè¯‰æˆ‘ä»¬ä½¿ç”¨äº†å“ªä¸ªæ¨¡å�‹ï¼Œè‹¥labelæ˜¯ä¸€ä¸ªåˆ—è¡¨ï¼Œåˆ™ç»˜åˆ¶æ‰€æœ‰æ ‡ç­¾çš„æ‰€æœ‰ROCæ›²çº¿
    """
    from sklearn.metrics import roc_auc_score, roc_curve, log_loss
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    if (type(label) != list) and (type(label) != np.ndarray):
        print("\t' %s on %s dataset \t\t \n" % (label, dataset))
        auc = roc_auc_score(y_true, y_pred)
        logloss = log_loss(y_true, y_pred)
        label_1 = label + ' AUC=%.3f' % (auc)

        # ç»˜åˆ¶ROCæ›²çº¿
        fpr, tpr, threshold = roc_curve(y_true, y_pred)
        sns.lineplot(x=fpr, y=tpr, label=label_1)
        x = np.arange(0, 1.1, 0.1)  # ç»˜åˆ¶AUC=0.5çš„ç›´çº¿
        sns.lineplot(x=x, y=x, label="AUC=0.5")
        plt.title("ROC on %s dataset" % (dataset))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  # è®¾ç½®å›¾ä¾‹åœ¨å›¾å½¢å¤–
        plt.show()
        print("åœ¨ %s æ•°æ�®é›†ä¸Š %s æ¨¡å�‹çš„ logloss = %.3f AUC = %.3f" % (dataset, label, logloss, auc))
        
        # åˆ›å»ºç»“æ�œæ•°æ�®æ¡†
        result_dict = {
            "Model": label,
            'Dataset': dataset,
            'log_loss': logloss,
            'AUC': auc
        }
        return pd.DataFrame(result_dict, index=[i])
        
    else:
        # ç»˜åˆ¶å¤šä¸ªæ¨¡å�‹çš„ROCæ›²çº¿
        plt.figure(figsize=(12, 8))
        for k, y in enumerate(y_pred):
            fpr, tpr, threshold = roc_curve(y_true, y)
            auc = roc_auc_score(y_true, y)
            label_ = label[k] + ' AUC=%.3f' % (auc)
            sns.lineplot(x=fpr, y=tpr, label=label_)

        x = np.arange(0, 1.1, 0.1)
        sns.lineplot(x=x, y=x, label="AUC=0.5")
        plt.title("Combined ROC")
        plt.xlabel('False Positive Rate')
        plt.ylabel("True Positive Rate")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()  


def find_best_params(model, params, cv=3, n_jobs=-1, X_train=None):
    """ä¿®æ”¹å��çš„å‡½æ•°ï¼Œç¡®ä¿�å¤„ç�† NaN"""
    from sklearn.model_selection import RandomizedSearchCV
    
    # ç¡®ä¿�æ•°æ�®æ²¡æœ‰ NaN
    if X_train.isna().any().any():
        print("æ£€æµ‹åˆ° NaNï¼Œæ­£åœ¨è‡ªåŠ¨å¤„ç�†...")
        X_train = safe_fillna(X_train)
    
    random_cv = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        cv=cv,
        scoring='roc_auc',
        n_jobs=n_jobs,
        n_iter=100,
        random_state=42,
        verbose=2
    )
    
    try:
        random_cv.fit(X_train, y_train)
        print("æœ€ä½³çš„AUCå¾—åˆ†ä¸ºï¼š%.3f" % (random_cv.best_score_))
        print("æœ€ä½³çš„å�‚æ•°ä¸ºï¼š%s" % (random_cv.best_params_))
    except Exception as e:
        print(f"è®­ç»ƒå¤±è´¥ï¼Œé”™è¯¯ä¿¡æ�¯: {e}")
        # è¿›ä¸€æ­¥è°ƒè¯•
        print(f"æ•°æ�®å½¢çŠ¶: {X_train.shape}")
        print(f"NaN æ•°é‡�: {X_train.isna().sum().sum()}")
        print(f"æ— ç©·å¤§å€¼æ•°é‡�: {np.isinf(X_train.values).sum()}")
    
    return random_cv


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# æ£€æŸ¥ NaN æƒ…å†µ
print("è®­ç»ƒé›† NaN ç»Ÿè®¡:")
print(f"æ€»è¡Œæ•°: {X_train.shape[0]}, æ€»åˆ—æ•°: {X_train.shape[1]}")
print(f"NaN æ€»æ•°: {X_train.isna().sum().sum()}")
print(f"åŒ…å�« NaN çš„è¡Œæ•°: {X_train.isna().any(axis=1).sum()}")
print(f"åŒ…å�« NaN çš„åˆ—æ•°: {X_train.isna().any(axis=0).sum()}")

# æŸ¥çœ‹å‰�å‡ åˆ—çš„ NaN æƒ…å†µ
nan_columns = X_train.columns[X_train.isna().any()].tolist()
if nan_columns:
    print(f"\nåŒ…å�« NaN çš„åˆ—ï¼ˆå‰�10ä¸ªï¼‰: {nan_columns[:10]}")


# ====== Nullå€¼å¤„ç�† ======
def fill_missing(X):
    """ç”¨ä¸­ä½�æ•°å¡«å……ç¼ºå¤±å€¼"""
    X_filled = X.copy()
    for col in X_filled.columns:
        if X_filled[col].isna().any():
            median_val = X_filled[col].median()
            X_filled[col] = X_filled[col].fillna(median_val)
    return X_filled

# å¡«å……æ•°æ�®
X_train_filled = fill_missing(X_train)
X_val_filled = fill_missing(X_val) if 'X_val' in locals() else None


print("\n=== è®­ç»ƒå†³ç­–æ ‘æ¨¡å�‹ ===")

# 1. è®­ç»ƒå†³ç­–æ ‘
model_dt = DecisionTreeClassifier(
    class_weight='balanced', 
    max_depth=10, 
    max_leaf_nodes=500,
    min_samples_leaf=10, 
    min_samples_split=5,
    random_state=42
)
model_dt.fit(X_train_filled, y_train)
print("âœ… å†³ç­–æ ‘è®­ç»ƒå®Œæˆ�")

# 2. æ¦‚ç�‡æ ¡å‡†
print("\n=== æ¦‚ç�‡æ ¡å‡† ===")
cc_model_dt = CalibratedClassifierCV(model_dt, cv='prefit')
cc_model_dt.fit(X_train_filled, y_train)
print("âœ… æ¦‚ç�‡æ ¡å‡†å®Œæˆ�")

# 3. é¢„æµ‹
print("\n=== éªŒè¯�é›†é¢„æµ‹ ===")
y_pred = cc_model_dt.predict_proba(X_val_filled)[:, 1]

# 4. è®¡ç®—AUC
dt_auc = roc_auc_score(y_val, y_pred)
print(f"éªŒè¯�é›† AUC: {dt_auc:.4f}")

# 5. ä¿�å­˜ç»“æ�œåˆ°result_df
if 'result_df' not in locals():
    result_df = pd.DataFrame()

dt_result = pd.DataFrame({
    'model': ['DecisionTree'],
    'train_auc': [roc_auc_score(y_train, cc_model_dt.predict_proba(X_train_filled)[:, 1])],
    'val_auc': [dt_auc],
    'n_features': [X_train_filled.shape[1]],
    'n_samples': [X_train_filled.shape[0]]
})

result_df = pd.concat([result_df, dt_result], ignore_index=True)

print("\n=== å½“å‰�æ‰€æœ‰æ¨¡å�‹ç»“æ�œ ===")
print(result_df.round(4))

# 6. å�¯é€‰ï¼šç»˜åˆ¶ROCæ›²çº¿
fpr, tpr, _ = roc_curve(y_val, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'g-', lw=2, label=f'Decision Tree (AUC={dt_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("\nâœ… å†³ç­–æ ‘è®­ç»ƒå®Œæˆ�")


import pandas as pd
from sklearn.metrics import roc_auc_score

# æ–¹æ³•äºŒï¼šç›´æ�¥ä¿®å¤�å·²æœ‰çš„ result_df
if 'result_df' in locals() or 'result_df' in globals():
    # æ£€æŸ¥é€»è¾‘å›�å½’è¡Œæ˜¯å�¦æœ‰NaN
    lr_idx = result_df[result_df['model'] == 'LogisticRegression'].index
    if len(lr_idx) > 0:
        try:
            # è®¡ç®—éªŒè¯�é›†AUC
            y_val_pred = model_lr.predict_proba(X_val_filled)[:, 1]
            val_auc = roc_auc_score(y_val, y_val_pred)
            
            # æ›´æ–°NaNå€¼
            result_df.loc[lr_idx, 'val_auc'] = val_auc
            print(f"âœ… å·²ä¿®å¤�é€»è¾‘å›�å½’çš„ val_auc: {val_auc:.4f}")
        except Exception as e:
            print(f"â�Œ ä¿®å¤�è¿‡ç¨‹ä¸­å‡ºç�°é”™è¯¯: {e}")
    
    print("\næ›´æ–°å��çš„ç»“æ�œ:")
    print(result_df.round(4))
    
    # æ˜¾ç¤ºè¯¦ç»†ç»Ÿè®¡ä¿¡æ�¯
    print(f"\nğŸ“Š æ•°æ�®å½¢çŠ¶:")
    print(f"è®­ç»ƒé›†: {X_train_filled.shape}")
    print(f"éªŒè¯�é›†: {X_val_filled.shape}")
    print(f"ç»“æ�œæ•°æ�®æ¡†: {result_df.shape}")
    
    # ä¿�å­˜ä¿®å¤�å��çš„ç»“æ�œ
    result_df_fixed = result_df.copy()
    
else:
    print("â�Œ æœªæ‰¾åˆ° result_dfï¼Œåˆ›å»ºæ–°çš„ç»“æ�œæ•°æ�®æ¡†...")
    try:
        # å¦‚æ�œæ²¡æœ‰result_dfï¼Œåˆ›å»ºæ–°çš„
        result_df = pd.DataFrame({
            'model': ['LogisticRegression'],
            'train_auc': [train_auc],
            'val_auc': [roc_auc_score(y_val, model_lr.predict_proba(X_val_filled)[:, 1])],
            'n_features': [X_train_filled.shape[1]],
            'n_samples': [X_train_filled.shape[0]]
        })
        print("âœ… å·²åˆ›å»ºæ–°çš„ result_df")
        print(result_df.round(4))
    except Exception as e:
        print(f"â�Œ åˆ›å»ºæ–°æ•°æ�®æ¡†æ—¶å‡ºç�°é”™è¯¯: {e}")

# ç¡®ä¿� result_df å�¯ä»¥æ­£å¸¸è®¿é—®
try:
    if 'result_df' in locals() or 'result_df' in globals():
        print("\nğŸ�¯ å½“å‰� result_df å†…å®¹:")
        print(result_df)
    else:
        print("âš ï¸� result_df ä¸�å­˜åœ¨")
except Exception as e:
    print(f"âš ï¸� è®¿é—® result_df æ—¶å‡ºé”™: {e}")


result_df


print("=== æ•°æ�®é¢„å¤„ç�† ===")

# 1. é¢„å¤„ç�†æ•°æ�®
X_train_filled = X_train.fillna(X_train.median())
train_median = X_train.median()
X_val_filled = X_val.fillna(train_median)

print(f"è®­ç»ƒé›† NaN æ•°é‡�: {X_train_filled.isna().sum().sum()}")
print(f"éªŒè¯�é›† NaN æ•°é‡�: {X_val_filled.isna().sum().sum()}")

# 2. è®­ç»ƒéš�æœºæ£®æ�—
print("\n=== è®­ç»ƒéš�æœºæ£®æ�—æ¨¡å�‹ ===")
model_rf = RandomForestClassifier(
    class_weight='balanced',
    n_estimators=100,
    max_depth=50,
    max_leaf_nodes=100,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model_rf.fit(X_train_filled, y_train)
print("âœ… éš�æœºæ£®æ�—è®­ç»ƒå®Œæˆ�")

# 3. æ¦‚ç�‡æ ¡å‡†
cc_model_rf = CalibratedClassifierCV(model_rf, cv='prefit')
cc_model_rf.fit(X_train_filled, y_train)

# 4. é¢„æµ‹å’Œè¯„ä¼°
y_pred = cc_model_rf.predict_proba(X_val_filled)[:, 1]
rf_auc = roc_auc_score(y_val, y_pred)
print(f"éªŒè¯�é›† AUC: {rf_auc:.4f}")

# 5. ä¿�å­˜ç»“æ�œåˆ°result_df
rf_result = pd.DataFrame({
    'model': ['RandomForest'],
    'train_auc': [roc_auc_score(y_train, cc_model_rf.predict_proba(X_train_filled)[:, 1])],
    'val_auc': [rf_auc],
    'n_features': [X_train_filled.shape[1]],
    'n_samples': [X_train_filled.shape[0]]
})

if 'result_df' in locals():
    result_df = pd.concat([result_df, rf_result], ignore_index=True)
else:
    result_df = rf_result

print("\n=== å½“å‰�æ‰€æœ‰æ¨¡å�‹ç»“æ�œ ===")
print(result_df.round(4))

# 6. å�¯é€‰ï¼šç»˜åˆ¶ROCæ›²çº¿
fpr, tpr, _ = roc_curve(y_val, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'r-', lw=2, label=f'Random Forest (AUC={rf_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("\nâœ… éš�æœºæ£®æ�—è®­ç»ƒå®Œæˆ�")



# å®šä¹‰XGBoostæ¨¡å�‹ï¼ˆä¿®æ­£ç‰ˆæœ¬ï¼‰
model_xgb = xgb.XGBClassifier(
    n_jobs=-1,  # ä½¿ç”¨æ‰€æœ‰å�¯ç”¨çš„CPUæ ¸å¿ƒ
    scale_pos_weight=1.,  
    learning_rate=0.01,  
    colsample_bytree=0.5,  
    subsample=0.9,  
    objective='binary:logistic',  
    n_estimators=1000,  
    reg_alpha=0.3,  
    max_depth=5,  
    gamma=5,  
    random_state=42
)

print("âœ… XGBoostæ¨¡å�‹å®šä¹‰å®Œæˆ�")
print(f"æ¨¡å�‹å�‚æ•°: n_estimators={1000}, learning_rate={0.01}, max_depth={5}")


%%time
# è®­ç»ƒXGBoostæ¨¡å�‹
# å¯¹äº�äºŒå…ƒåˆ†ç±»ï¼Œä½¿ç”¨æ­£ç¡®çš„è¯„ä¼°æŒ‡æ ‡
eval_set = [(X_train, y_train), (X_val, y_val)]

model_xgb.fit(
    X_train, y_train, 
    eval_set=eval_set,
    eval_metric=['error', 'auc'],  # äºŒå…ƒåˆ†ç±»çš„è¯„ä¼°æŒ‡æ ‡
    early_stopping_rounds=50, 
    verbose=20
)

print("\n" + "="*50)
print("ğŸ�¯ è®­ç»ƒå®Œæˆ�ï¼�æœ€ä½³æ¨¡å�‹ä¿¡æ�¯ï¼š")
print(f"æœ€ä½³è¿­ä»£æ¬¡æ•°: {model_xgb.best_iteration}")
print(f"æœ€ä½³å¾—åˆ†: {model_xgb.best_score}")
print("="*50)


model_xgb.best_score,model_xgb.best_iteration


# ç»˜åˆ¶AUCè®­ç»ƒæ›²çº¿
results = model_xgb.evals_result_  
auc_train = results['validation_0']['auc']  
auc_val = results['validation_1']['auc']  

fig, ax = plt.subplots(figsize=(10, 6))  
epochs = len(auc_val)  
ax.plot(range(0, epochs), auc_train, label='Train AUC')  
ax.plot(range(0, epochs), auc_val, label='Validation AUC')  
ax.legend()  
plt.title('XGBoost - AUC Training History')  
plt.xlabel('Iteration')
plt.ylabel('AUC')  
plt.grid(True, alpha=0.3)
plt.show()  

# è¾“å‡ºæœ€ä½³æ€§èƒ½
print("ğŸ“Š æ¨¡å�‹æ€§èƒ½æ€»ç»“ï¼š")
print(f"éªŒè¯�é›†ä¸Šæœ€å¤§AUCï¼š{max(auc_val):.4f}")  
print(f"æœ€ä¼˜è¿­ä»£æ¬¡æ•°ï¼š{auc_val.index(max(auc_val))}")  

# è®¡ç®—å½“å‰�è®­ç»ƒé›†å’ŒéªŒè¯�é›†çš„AUC
train_auc_xgb = roc_auc_score(y_train, model_xgb.predict_proba(X_train)[:, 1])
val_auc_xgb = roc_auc_score(y_val, model_xgb.predict_proba(X_val)[:, 1])
print(f"æœ€ç»ˆæ¨¡å�‹è®­ç»ƒé›†AUCï¼š{train_auc_xgb:.4f}")
print(f"æœ€ç»ˆæ¨¡å�‹éªŒè¯�é›†AUCï¼š{val_auc_xgb:.4f}")

# æ›´æ–°result_df
print("\n" + "="*50)
print("ğŸ“‹ æ›´æ–°æ¨¡å�‹ç»“æ�œæ•°æ�®æ¡† (result_df)")

# æ£€æŸ¥å¹¶åˆ›å»ºæˆ–æ›´æ–°result_df
if 'result_df' not in locals() and 'result_df' not in globals():
    result_df = pd.DataFrame(columns=['model', 'train_auc', 'val_auc', 'n_features', 'n_samples', 'best_iteration'])
    print("âœ… åˆ›å»ºæ–°çš„result_df")

# æ£€æŸ¥æ˜¯å�¦å·²å­˜åœ¨XGBoostè®°å½•
xgb_exists = 'result_df' in locals() or 'result_df' in globals()
if xgb_exists and 'model' in result_df.columns:
    existing_idx = result_df[result_df['model'] == 'XGBoost'].index
else:
    existing_idx = []

if len(existing_idx) > 0:
    # æ›´æ–°ç�°æœ‰è®°å½•
    result_df.loc[existing_idx, 'train_auc'] = train_auc_xgb
    result_df.loc[existing_idx, 'val_auc'] = val_auc_xgb
    result_df.loc[existing_idx, 'best_iteration'] = model_xgb.best_iteration
    print("âœ… æ›´æ–°äº†XGBoostæ¨¡å�‹è®°å½•")
else:
    # æ·»åŠ æ–°è®°å½•
    new_row = pd.DataFrame({
        'model': ['XGBoost'],
        'train_auc': [train_auc_xgb],
        'val_auc': [val_auc_xgb],
        'n_features': [X_train.shape[1]],
        'n_samples': [X_train.shape[0]],
        'best_iteration': [model_xgb.best_iteration]
    })
    result_df = pd.concat([result_df, new_row], ignore_index=True)
    print("âœ… æ·»åŠ äº†XGBoostæ¨¡å�‹æ–°è®°å½•")

# æ˜¾ç¤ºæ›´æ–°å��çš„ç»“æ�œ
print("\nğŸ“„ å½“å‰�result_dfå†…å®¹ï¼š")
print(result_df.round(4))


result_df


# ==================== ç®€åŒ–ç‰ˆLightGBMæ—©å�œ =====================
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

print("=== LightGBMæ—©å�œä¼˜åŒ– ===")

# æ•°æ�®é¢„å¤„ç�†
X_train_filled = X_train.fillna(X_train.median())
X_val_filled = X_val.fillna(X_train.median())

# ç®€åŒ–å�‚æ•°
model_lgb_simple = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# è®­ç»ƒï¼ˆé�™é»˜æ¨¡å¼�ï¼‰
model_lgb_simple.fit(
    X_train_filled, 
    y_train,
    eval_set=[(X_val_filled, y_val)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

# è¯„ä¼°
train_auc = roc_auc_score(y_train, model_lgb_simple.predict_proba(X_train_filled)[:, 1])
val_auc = roc_auc_score(y_val, model_lgb_simple.predict_proba(X_val_filled)[:, 1])

print(f"è®­ç»ƒé›†AUC: {train_auc:.4f}")
print(f"éªŒè¯�é›†AUC: {val_auc:.4f}")
print(f"æœ€ä½³è¿­ä»£: {model_lgb_simple.best_iteration_}")

# æ›´æ–°result_df
result_df.loc[result_df['model'] == 'LightGBM', ['train_auc', 'val_auc', 'best_iteration']] = [
    train_auc, val_auc, model_lgb_simple.best_iteration_
]

print("\nğŸ“Š æ›´æ–°å��çš„ç»“æ�œ:")
print(result_df.round(4))


# ================== FINAL FIX FOR result_df ==================

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ---------- 1. å�»é‡�ï¼šæŒ‰ model ä¿�ç•™æœ€å��ä¸€æ¬¡ç»“æ�œ ----------
if 'result_df' in globals():
    result_df = (
        result_df
        .drop_duplicates(subset=['model'], keep='last')
        .reset_index(drop=True)
    )
else:
    raise ValueError("â�Œ result_df ä¸�å­˜åœ¨")

# ---------- 2. è¡¥å…… LightGBMï¼ˆå¦‚æ�œæ¨¡å�‹å·²è®­ç»ƒä½†è¡¨ä¸­æ²¡æœ‰ï¼‰ ----------
if 'model_lgb_simple' in globals():
    if 'LightGBM' not in result_df['model'].values:
        lgb_train_auc = roc_auc_score(
            y_train, model_lgb_simple.predict_proba(X_train_filled)[:, 1]
        )
        lgb_val_auc = roc_auc_score(
            y_val, model_lgb_simple.predict_proba(X_val_filled)[:, 1]
        )

        lgb_row = pd.DataFrame([{
            'model': 'LightGBM',
            'train_auc': lgb_train_auc,
            'val_auc': lgb_val_auc,
            'n_features': X_train_filled.shape[1],
            'n_samples': X_train_filled.shape[0],
            'best_iteration': model_lgb_simple.best_iteration_,
            'calibrated': 'No'
        }])

        result_df = pd.concat([result_df, lgb_row], ignore_index=True)

# ---------- 3. æ�’åº� & å±•ç¤ºæœ€ç»ˆç»“æ�œ ----------
result_df = result_df.sort_values('val_auc', ascending=False).reset_index(drop=True)

print("\nâœ… æœ€ç»ˆå¹²å‡€ç‰ˆ result_dfï¼š")
print(result_df.round(4))



# ç‰¹å¾�é‡�è¦�æ€§æ�’åº�å��çš„ç‰¹å¾�å��
best_model = model_lgb_simple
feat_imp = best_model.feature_importances_
feat_indices = np.argsort(feat_imp)[::-1]
important_feat = X_train.columns[feat_indices]
important_feat


# ä¿�å­˜ç‰¹å¾�é‡�è¦�æ€§ç»“æ�œ
important_feat_df = pd.DataFrame({'feat_name': important_feat, 'feat_imp': feat_imp[feat_indices]})
important_feat_df.to_csv('./output/'+dataset+'_feat_imp.csv', index=False, encoding='utf-8')


# ç»˜åˆ¶å‰�50ä¸ªé‡�è¦�æ€§å¾—åˆ†æœ€é«˜çš„ç‰¹å¾�æ�’åº�å›¾
top = 50
top_indices = feat_indices[:top]
most_important_feat = X_train.columns[top_indices]
plt.figure(figsize=(7, 12))
sns.barplot(x=feat_imp[top_indices], y=most_important_feat)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel("Feature names")
plt.show()


# Generate predictions
y_pred = best_model.predict_proba(X_test)[:, 1]

# Create submission file
submission = pd.DataFrame({'ID': test_id, 'TARGET': y_pred})
submission.to_csv('submission.csv', index=False)

print(f"âœ… Predictions saved: submission.csv ({len(submission)} samples)")
print(f"ğŸ“Š Range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")




