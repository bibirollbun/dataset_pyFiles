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


from sklearn.linear_model import Ridge , LogisticRegression
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error 
from lightgbm import LGBMRegressor
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neural_network import MLPRegressor

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from numba import njit


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
%matplotlib inline

import warnings
warnings.filterwarnings(action="ignore")


@njit
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@njit
def ftrl_update(w, z, n, g, alpha, beta, L1, L2):
    """FTRL-Proximalã‚¢ãƒ«ã‚´ãƒªã‚ºãƒ ã�«ã‚ˆã‚‹é‡�ã�¿æ›´æ–°"""
    sigma = (np.sqrt(n + g * g) - np.sqrt(n)) / alpha
    z += g - sigma * w
    n += g * g
    
    sign_z = np.sign(z)
    abs_z = np.abs(z)
    
    if abs_z <= L1:
        w = 0.0
    else:
        w = (sign_z * L1 - z) / ((beta + np.sqrt(n)) / alpha + L2)
    return w, z, n

@njit
def train_fm_epoch(indices, indptr, data, target, 
                   w, z, n, w_fm, z_fm, n_fm,
                   alpha, beta, L1, L2,
                   alpha_fm, beta_fm, L1_fm, L2_fm,
                   D_fm):
    
    for i in range(len(target)):
        start_idx = indptr[i]
        end_idx = indptr[i+1]
        
        if start_idx == end_idx: continue
        
        feature_indices = indices[start_idx:end_idx]
        feature_values = data[start_idx:end_idx]
        y = target[i]
        
        # 1. äºˆæ¸¬è¨ˆç®—
        w_dot_x = 0.0
        for j in range(len(feature_indices)):
            idx = feature_indices[j]
            val = feature_values[j]
            w_dot_x += w[idx] * val
            
        # FMäº¤äº’ä½œç”¨é …ã�®è¨ˆç®—
        vx = np.zeros(D_fm)
        v2x2 = np.zeros(D_fm)
        
        for j in range(len(feature_indices)):
            idx = feature_indices[j]
            val = feature_values[j]
            for f in range(D_fm):
                v_val = w_fm[idx, f]
                vx[f] += v_val * val
                v2x2[f] += (v_val * val) ** 2
                
        interaction = 0.5 * np.sum(vx**2 - v2x2)
        prediction = w_dot_x + interaction
        
        # 2. å‹¾é…�è¨ˆç®—
        grad = prediction - y
        
        # 3. ç·šå½¢é …ã�®æ›´æ–°
        for j in range(len(feature_indices)):
            idx = feature_indices[j]
            val = feature_values[j]
            g = grad * val
            w[idx], z[idx], n[idx] = ftrl_update(w[idx], z[idx], n[idx], g, alpha, beta, L1, L2)
            
        # 4. å› å­�é …(FM)ã�®æ›´æ–°
        for j in range(len(feature_indices)):
            idx = feature_indices[j]
            val = feature_values[j]
            for f in range(D_fm):
                g_fm = grad * (vx[f] * val - w_fm[idx, f] * val**2)
                w_fm[idx, f], z_fm[idx, f], n_fm[idx, f] = ftrl_update(
                    w_fm[idx, f], z_fm[idx, f], n_fm[idx, f], g_fm, 
                    alpha_fm, beta_fm, L1_fm, L2_fm
                )

@njit
def predict_fm(indices, indptr, data, w, w_fm, D_fm):
    n_samples = len(indptr) - 1
    preds = np.zeros(n_samples)
    
    for i in range(n_samples):
        start_idx = indptr[i]
        end_idx = indptr[i+1]
        
        if start_idx == end_idx: continue
        
        feature_indices = indices[start_idx:end_idx]
        feature_values = data[start_idx:end_idx]
        
        # ç·šå½¢é …
        pred = 0.0
        for j in range(len(feature_indices)):
            pred += w[feature_indices[j]] * feature_values[j]
            
        # FMé …
        vx = np.zeros(D_fm)
        v2x2 = np.zeros(D_fm)
        for j in range(len(feature_indices)):
            idx = feature_indices[j]
            val = feature_values[j]
            for f in range(D_fm):
                v_val = w_fm[idx, f]
                vx[f] += v_val * val
                v2x2[f] += (v_val * val) ** 2
        
        interaction = 0.5 * np.sum(vx**2 - v2x2)
        preds[i] = pred + interaction
        
    return preds

class FM_FTRL_Wrapper:
    def __init__(self, D, D_fm=8, alpha=0.1, alpha_fm=0.05, iters=3):
        self.D = D
        self.D_fm = D_fm
        self.alpha = alpha
        self.alpha_fm = alpha_fm
        self.iters = iters
        
    def fit(self, X, y):
        self.w = np.zeros(self.D)
        self.z = np.zeros(self.D)
        self.n = np.zeros(self.D)
        self.w_fm = np.random.normal(0, 0.01, (self.D, self.D_fm))
        self.z_fm = np.zeros((self.D, self.D_fm))
        self.n_fm = np.zeros((self.D, self.D_fm))
        
        for i in range(self.iters):
            print(f"FM Epoch {i+1}/{self.iters}")
            train_fm_epoch(X.indices, X.indptr, X.data, y.values, 
                           self.w, self.z, self.n, self.w_fm, self.z_fm, self.n_fm,
                           self.alpha, 1.0, 0.1, 1.0, 
                           self.alpha_fm, 1.0, 0.1, 1.0, self.D_fm)
            
    def predict(self, X):
        return predict_fm(X.indices, X.indptr, X.data, self.w, self.w_fm, self.D_fm)


!apt-get install p7zip
!p7zip -d -f -k /kaggle/input/mercari-price-suggestion-challenge/train.tsv.7z
!p7zip -d -f -k /kaggle/input/mercari-price-suggestion-challenge/sample_submission.csv.7z
!unzip -o /kaggle/input/mercari-price-suggestion-challenge/test_stg2.tsv.zip


try:
    # --- ãƒ‡ãƒ¼ã‚¿ã�®èª­ã�¿è¾¼ã�¿ ---
    # (Kaggleç’°å¢ƒã�§ã�¯ã€�'/kaggle/input/...' ã�ªã�©ã�®ãƒ‘ã‚¹ã�‹ã‚‰èª­ã�¿è¾¼ã�¿ã�¾ã�™)
    train_df = pd.read_csv("/kaggle/working/train.tsv", sep='\t')
    test_df = pd.read_csv("/kaggle/working/test_stg2.tsv", sep='\t')
    sample_df = pd.read_csv("/kaggle/working/sample_submission.csv")
    
    print("âœ… 'train.csv', 'test.csv', 'sample_submission.csv' ã�®èª­ã�¿è¾¼ã�¿ã�«æˆ�åŠŸã�—ã�¾ã�—ã�Ÿã€‚\n")

except FileNotFoundError as e:
    print(f"â�Œ ã‚¨ãƒ©ãƒ¼: ãƒ•ã‚¡ã‚¤ãƒ«ã�Œè¦‹ã�¤ã�‹ã‚Šã�¾ã�›ã‚“ã€‚{e}")
    print("å¿…è¦�ã�ªCSVãƒ•ã‚¡ã‚¤ãƒ«ã�ŒColabç’°å¢ƒã�«ã‚¢ãƒƒãƒ—ãƒ­ãƒ¼ãƒ‰ã�•ã‚Œã�¦ã�„ã‚‹ã�‹ç¢ºèª�ã�—ã�¦ã��ã� ã�•ã�„ã€‚")
except Exception as e:
    print(f"äºˆæœŸã�›ã�¬ã‚¨ãƒ©ãƒ¼ã�Œç™ºç”Ÿã�—ã�¾ã�—ã�Ÿ: {e}")


# print(f"âœ… è¨“ç·´ãƒ‡ãƒ¼ã‚¿è¡Œæ•°: {len(train_df)}")  # 1482535
# print(f"âœ… ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿è¡Œæ•°: {len(test_df)}")  # 3460725
# print(f"âœ… å�ˆè¨ˆ: {len(train_df) + len(test_df)}")  # 4943260


# train_df.head()


# print(train_df.shape)


y_train_df = train_df['price']
plt.figure()
sns.histplot(y_train_df, element = 'poly', color = 'red')
plt.show()
# right skewed


y_train_df = np.log1p(y_train_df) # log transformation
sns.histplot(y_train_df, element = 'poly', color = 'red')
plt.show


train_df['price'] = np.log1p(train_df['price'])
train_df.head()


# print('[train] shipping value types:\n',train_df['shipping'].value_counts())
# print('[train] item_condition_id value types:\n',train_df['item_condition_id'].value_counts())
# print('')
# print('[test] shipping value types:\n',test_df['shipping'].value_counts())
# print('[test] item_condition_id value types:\n',test_df['item_condition_id'].value_counts())


# 1. NaN (æ¬ æ��å€¤) ã‚’ 'Null/Null/Null' ã�¨ã�„ã�†æ–‡å­—åˆ—ã�§å…ˆã�«åŸ‹ã‚�ã�¾ã�™
safe_categories = train_df['category_name'].fillna('Null/Null/Null')

# 2. .str.split() ã�§ä¸€æ‹¬åˆ†å‰²ã�—ã�¾ã�™
# n=2: æœ€å¤§2å›�ã�¾ã�§åˆ†å‰²ï¼ˆï¼�3ã�¤ã�®éƒ¨åˆ†ã�«åˆ†ã�‘ã‚‹ï¼‰
# expand=True: åˆ†å‰²ã�—ã�Ÿãƒªã‚¹ãƒˆã‚’DataFrameã�®æ–°ã�—ã�„åˆ—ã�¨ã�—ã�¦å±•é–‹ã�™ã‚‹
split_df = safe_categories.str.split('/', n=2, expand=True)

# 3. æ–°ã�—ã�„åˆ—ã‚’å…ƒã�®DataFrameã�«ä¸€æ‹¬ã�§å‰²ã‚Šå½“ã�¦ã‚‹
# éš�å±¤ã�Œè¶³ã‚Šã�ªã�„å ´å�ˆã€�è¶³ã‚Šã�ªã�„éƒ¨åˆ†ã�¯è‡ªå‹•çš„ã�« `None` (NaN) ã�§åŸ‹ã‚�ã‚‰ã‚Œã�¾ã�™
train_df['category_1'] = split_df[0]
train_df['category_2'] = split_df[1]
train_df['category_3'] = split_df[2]

# --- ä»¥ä¸‹ã�¯å�Œã�˜ ---
print('1st Category :', train_df['category_1'].value_counts())
print('2nd Category :', train_df['category_2'].nunique())
print('3rd Category :', train_df['category_3'].nunique())


# 1. NaN ã‚’ 'Null/Null/Null' ã�§åŸ‹ã‚�ã�¾ã�™
safe_categories_test = test_df['category_name'].fillna('Null/Null/Null')

# 2. .str.split() ã�§ä¸€æ‹¬åˆ†å‰²ã�—ã�¾ã�™
split_df_test = safe_categories_test.str.split('/', n=2, expand=True)

# 3. æ–°ã�—ã�„åˆ—ã‚’ test_df ã�«ä¸€æ‹¬ã�§å‰²ã‚Šå½“ã�¦ã�¾ã�™
test_df['category_1'] = split_df_test[0]
test_df['category_2'] = split_df_test[1]
test_df['category_3'] = split_df_test[2]

# --- çµ�æ�œã�®å‡ºåŠ›ï¼ˆã�“ã‚Œã�¯å�Œã�˜ï¼‰ ---
print('1st Category :', test_df['category_1'].value_counts())
print('2nd Category :', test_df['category_2'].nunique())
print('3rd Category :', test_df['category_3'].nunique())


#æ¬ æ��å€¤ã�®æ•°ã‚’ç¢ºèª�
train_df.isnull().sum()



#æ¬ æ��å€¤ (NaN) ã‚’ã€�æ–‡å­—åˆ—ã�® 'Null' ã�«ç½®ã��æ�›ã�ˆã�¦ã�„ã‚‹
train_df['brand_name'] = train_df['brand_name'].fillna(value='Null')
train_df['category_name'] = train_df['category_name'].fillna(value='Null')
train_df['item_description'] = train_df['item_description'].fillna(value='Null')\

train_df.isnull().sum()


train_df



test_df.isnull().sum()



test_df['brand_name'] = test_df['brand_name'].fillna(value='Null')
test_df['category_name'] = test_df['category_name'].fillna(value='Null')
test_df['item_description'] = test_df['item_description'].fillna(value='Null')

test_df.isnull().sum()


test_df



# çµ�å�ˆ train dataset & test dataset
train_df_target = train_df['price']
train_df.drop(['price'],axis=1,inplace=True)

mercari_df = pd.concat([train_df, test_df])
mercari_df = mercari_df.drop(columns=['train_id', 'test_id'], axis = 1)

mercari_df.reset_index(drop=True, inplace=True)




text_cols = ['name', 'item_description']

print("ğŸ”§ ãƒ†ã‚­ã‚¹ãƒˆã‚¯ãƒªãƒ¼ãƒ‹ãƒ³ã‚°ä¸­...")

for col in text_cols:
    # 1. å°�æ–‡å­—åŒ– 
    mercari_df[col] = mercari_df[col].astype(str).str.lower()
    
    # 2. URLã�®é™¤å�» (http, https, wwwã�§å§‹ã�¾ã‚‹æ–‡å­—åˆ—)
    mercari_df[col] = mercari_df[col].str.replace(r'http\S+|www.\S+', ' ', regex=True)
    
    # 3. [rm] (RetailMeNotã�ªã�©ä¾¡æ ¼é–¢é€£ã�®å®šå�‹çš„ã�ªãƒ—ãƒ¬ãƒ¼ã‚¹ãƒ›ãƒ«ãƒ€ãƒ¼) ã�®é™¤å�»
    mercari_df[col] = mercari_df[col].str.replace(r'\[rm\]', ' ', regex=True)
    
    # # 4. æ•°å­—ã�®ç½®ã��æ�›ã�ˆï¼ˆé‡�è¦�ï¼šç‰¹å®šã�®æ•°å­—ã‚’æ„�å‘³ã�®ã�ªã�„ãƒˆãƒ¼ã‚¯ãƒ³ 'NUM' ã�«å¤‰æ�›ï¼‰
    # # å¤§é‡�ã�®æ•°å­—ï¼ˆä¾‹: 100, 2017, 10000ï¼‰ã‚’ 'NUM' ãƒˆãƒ¼ã‚¯ãƒ³ã�«å¤‰æ�›ã�™ã‚‹ã�“ã�¨ã�§ã€�
    # # æ•°å­—è‡ªä½“ã�® TF-IDF é‡�ã�¿ã�Œä¸�å¿…è¦�ã�«é«˜ã��ã�¤ã��ã�®ã‚’é˜²ã��ã€�æ•°å­—ã�®ã€Œå­˜åœ¨ã€�ã�®ã�¿ã‚’ã‚·ã‚°ãƒŠãƒ«åŒ–ã�—ã�¾ã�™ã€‚
    # mercari_df[col] = mercari_df[col].str.replace(r'\b\d+\b', ' NUM ', regex=True) 
    
    # 5. ç‰¹æ®Šè¨˜å�·ã‚„æ”¹è¡Œã�®é™¤å�»/ã‚¹ãƒšãƒ¼ã‚¹ç½®æ�›ï¼ˆæ”¹è¡Œã‚’ã‚¹ãƒšãƒ¼ã‚¹ã�«çµ±ä¸€ã�—ã€�è¨˜å�·ã‚’åŒºåˆ‡ã‚Šæ–‡å­—åŒ–ï¼‰
    # è‹±æ•°å­—ã�¨ã‚¹ãƒšãƒ¼ã‚¹ä»¥å¤–ã�¯ã�™ã�¹ã�¦ã‚¹ãƒšãƒ¼ã‚¹ã�«ç½®æ�›ã�—ã�¾ã�™
    mercari_df[col] = mercari_df[col].str.replace(r'[^a-z0-9\s]+', ' ', regex=True)
    
    # 6. è¤‡æ•°ã‚¹ãƒšãƒ¼ã‚¹ã‚’å�˜ä¸€ã‚¹ãƒšãƒ¼ã‚¹ã�«ç½®æ�›
    mercari_df[col] = mercari_df[col].str.strip().str.replace(r'\s+', ' ', regex=True)

print("âœ… ã‚¯ãƒªãƒ¼ãƒ‹ãƒ³ã‚°å®Œäº†ã€‚")


# ===== ã�“ã�“ã�‹ã‚‰è¿½åŠ  =====
print("ç‰¹å¾´é‡�ä½œæˆ�ä¸­...")

# ãƒ†ã‚­ã‚¹ãƒˆé•·
mercari_df['name_len'] = mercari_df['name'].str.len()
mercari_df['desc_len'] = mercari_df['item_description'].str.len()

# å�˜èª�æ•°
mercari_df['desc_word_count'] = mercari_df['item_description'].str.split().str.len()

# ãƒ–ãƒ©ãƒ³ãƒ‰ã�®æœ‰ç„¡
mercari_df['has_brand'] = (mercari_df['brand_name'] != 'Null').astype(int)

# ã‚«ãƒ†ã‚´ãƒªã�®æœ‰ç„¡
mercari_df['has_category'] = (mercari_df['category_name'] != 'Null').astype(int)

print("âœ… æ–°ç‰¹å¾´é‡�:")
print(mercari_df[['name_len', 'desc_len', 'desc_word_count', 'has_brand', 'has_category']].head())
# ===== ã�“ã�“ã�¾ã�§è¿½åŠ  =====


vectorizer_name = TfidfVectorizer(
    max_features=300000,     #250000 30000
    #stop_words='english',
    #analyzer='char',           # â†� æ–‡å­—ãƒ¬ãƒ™ãƒ«ã�«å¤‰æ›´
    #ngram_range=(2, 5),        # â†� 2æ–‡å­—ã€œ5æ–‡å­—ã�®n-gram
    ngram_range=(1, 2),
    min_df=3,                # 3å›�æœªæº€ã�®å�˜èª�ã�¯ç„¡è¦–
    max_df=0.8, # 80%ä»¥ä¸Šã�®æ–‡æ›¸ã�«å‡ºç�¾ã�™ã‚‹å�˜èª�ã�¯ç„¡è¦–
    dtype=np.float32
)
X_name = vectorizer_name.fit_transform(mercari_df['name'])
print('name vectorization shape:',X_name.shape)


import time  # æ™‚é–“è¨ˆæ¸¬ç”¨ã�®ãƒ¢ã‚¸ãƒ¥ãƒ¼ãƒ«ã‚’è¿½åŠ 
import numpy as np
from sklearn.decomposition import TruncatedSVD
import gc # ã‚¬ãƒ™ãƒ¼ã‚¸ã‚³ãƒ¬ã‚¯ã‚·ãƒ§ãƒ³ï¼ˆãƒ¡ãƒ¢ãƒªè§£æ”¾ç”¨ï¼‰

# ---------------------------------------------------------
# 1. ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿è¨­å®š
# ---------------------------------------------------------
n_comp = 150

svd = TruncatedSVD(
    n_components=n_comp, 
    algorithm='randomized', 
    random_state=42         
)

# ---------------------------------------------------------
# 2. æ¬¡å…ƒå‰Šæ¸›ã�®å®Ÿè¡Œï¼ˆæ™‚é–“è¨ˆæ¸¬ä»˜ã��ï¼‰
# ---------------------------------------------------------
print(f'Starting Truncated SVD to reduce dimensions to {n_comp}...')

# è¨ˆæ¸¬é–‹å§‹
start_time = time.time()

# å®Ÿè¡Œ
X_name_svd = svd.fit_transform(X_name)
X_name_svd = X_name_svd.astype(np.float32)

# è¨ˆæ¸¬çµ‚äº†
elapsed_time = time.time() - start_time

# çµ�æ�œè¡¨ç¤º
print(f'Done. Execution Time: {elapsed_time:.2f} seconds')
print('X_name shape:', X_name_svd.shape)

# ---------------------------------------------------------
# 3. æƒ…å ±ã�®æ��å¤±ã‚’ç¢ºèª�
# ---------------------------------------------------------
print(f'Explained Variance Ratio (Total): {np.sum(svd.explained_variance_ratio_):.4f}')



#20åˆ†ã��ã‚‰ã�„ã�‹ã�‹ã‚‹ã€€ãƒ¡ãƒ¢ãƒª20Gã��ã‚‰ã�„å¿…è¦�
tfidf_descp = TfidfVectorizer(
    max_features=500000,     #500000    75000
    ngram_range=(1, 2) ,
    #stop_words='english',
    #analyzer='char',           # â†� æ–‡å­—ãƒ¬ãƒ™ãƒ«ã�«å¤‰æ›´
    #ngram_range=(2, 5),        # â†� 2æ–‡å­—ã€œ5æ–‡å­—ã�®n-gram
    min_df=5,                # 3å›�æœªæº€ã�®å�˜èª�ã�¯ç„¡è¦–
    max_df=0.8,         # 80%ä»¥ä¸Šã�®æ–‡æ›¸ã�«å‡ºç�¾ã�™ã‚‹å�˜èª�ã�¯ç„¡è¦–
    dtype=np.float32
)

# â­� name ã�¨ item_description ã‚’çµ�å�ˆï¼ˆã‚¹ãƒšãƒ¼ã‚¹åŒºåˆ‡ã‚Šï¼‰
print('ğŸ”§ Combining name and item_description...')
#combined_text = mercari_df['name'] + ' ' + mercari_df['item_description']

combined_text = (
    mercari_df['name'] + ' ' + 
    mercari_df['brand_name'] + ' ' +
    mercari_df['item_description']
)

# â­� çµ�å�ˆã�—ã�Ÿãƒ†ã‚­ã‚¹ãƒˆã�§TF-IDFåŒ–
X_descp = tfidf_descp.fit_transform(combined_text)

# â­� ä¸€æ™‚å¤‰æ•°ã‚’å�³å‰Šé™¤
del combined_text
gc.collect()

# X_descp = tfidf_descp.fit_transform(mercari_df['item_description'])
# print('item_description vectorization shape:',X_descp.shape)


# ---------------------------------------------------------
# 1. ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿è¨­å®š
# ---------------------------------------------------------
n_comp = 150

svd = TruncatedSVD(
    n_components=n_comp, 
    algorithm='randomized', 
    random_state=42         
)

# ---------------------------------------------------------
# 2. æ¬¡å…ƒå‰Šæ¸›ã�®å®Ÿè¡Œï¼ˆæ™‚é–“è¨ˆæ¸¬ä»˜ã��ï¼‰
# ---------------------------------------------------------
print(f'Starting Truncated SVD to reduce dimensions to {n_comp}...')

# è¨ˆæ¸¬é–‹å§‹
start_time = time.time()

# å®Ÿè¡Œ
X_descp_svd = svd.fit_transform(X_descp)
X_descp_svd = X_descp_svd.astype(np.float32)

# è¨ˆæ¸¬çµ‚äº†
elapsed_time = time.time() - start_time

# çµ�æ�œè¡¨ç¤º
print(f'Done. Execution Time: {elapsed_time:.2f} seconds')
print('X_descp shape:', X_descp_svd.shape)

# ---------------------------------------------------------
# 3. æƒ…å ±ã�®æ��å¤±ã‚’ç¢ºèª�
# ---------------------------------------------------------
print(f'Explained Variance Ratio (Total): {np.sum(svd.explained_variance_ratio_):.4f}')



ohe = OneHotEncoder()

# why reshape()? series type change to two dimension
X_brand_name = ohe.fit_transform(mercari_df['brand_name'].values.reshape(-1, 1))
X_item_condition_id = ohe.fit_transform(mercari_df['item_condition_id'].values.reshape(-1, 1))
X_shipping = ohe.fit_transform(mercari_df['shipping'].values.reshape(-1, 1))
X_category_1 = ohe.fit_transform(mercari_df['category_1'].values.reshape(-1, 1))
X_category_2 = ohe.fit_transform(mercari_df['category_2'].values.reshape(-1, 1))
X_category_3 = ohe.fit_transform(mercari_df['category_3'].values.reshape(-1, 1))


print('brand encoding shape:', X_brand_name.shape)
print('item condition id encoding shape:', X_item_condition_id.shape)
print('shipping encoding shape:', X_shipping.shape)
print('category_1 encoding shape:', X_category_1.shape)
print('category_2 encoding shape:', X_category_2.shape)
print('category_3 encoding shape:', X_category_3.shape)


# --- 1. ã‚µã‚¤ã‚ºãƒ»å®¹é‡�ãƒ»ç´ æ��æƒ…å ±ã�®æŠ½å‡ºã�¨Target Encoding (å®Œå…¨ç‰ˆ) ---
import re
from sklearn.model_selection import KFold
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler
from lightgbm import early_stopping
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

print("\n" + "="*60)
print("ğŸ�¯ Stage 4-1 & 5-1: Stackingã�®ã�Ÿã‚�ã�®ç‰¹å¾´é‡�ç”Ÿæˆ� (æ‹¡å¼µç‰ˆ)")
print("="*60)

# 1-0. ãƒ†ã‚­ã‚¹ãƒˆã�®çµ�å�ˆï¼ˆã�“ã�“ã�‹ã‚‰å…¨ã�¦ã�®æƒ…å ±ã‚’æ�¢ã�—ã�¾ã�™ï¼‰
mercari_df['text_combined'] = mercari_df['name'] + ' ' + mercari_df['item_description']

# ==========================================
# 1-1. ã€�ã‚µã‚¤ã‚ºã€‘æƒ…å ±ã�®æŠ½å‡ºã�¨æ­£è¦�åŒ– (æ—¢å­˜)
# ==========================================
# æœ�ã‚„é�´ã�®ã‚µã‚¤ã‚º (S, M, L, XL, 24.5cmã�ªã�©) ã‚’æŠ½å‡º
size_pattern = r'(\b(?:xs|s|m|l|xl|xxl|os|one size|free size)\b|\b\d{1,2}(?:t|y|m|x|ddd|dd|c|b|a)?\b)'
mercari_df['raw_size'] = mercari_df['text_combined'].str.extract(size_pattern, flags=re.IGNORECASE, expand=False)
mercari_df['raw_size'] = mercari_df['raw_size'].fillna('No_Size_Info').str.lower().str.strip()

def normalize_size(raw_size):
    size_map = {'os': 'OS', 'one size': 'OS', 'free size': 'OS', 'xs': 'XS_S', 'xxs': 'XS_S', '0': 'XS_S', '2': 'XS_S', 's': 'S', '4': 'S', 'm': 'M', '6': 'M', 'l': 'L', '8': 'L', '10': 'L', 'xl': 'XL', 'xxl': 'XL', '12': 'XL', '1x': 'PLUS', '2x': 'PLUS', '3x': 'PLUS', 'no_size_info': 'OTHER'}
    norm_size = size_map.get(raw_size, raw_size)
    if 'ddd' in norm_size or 'dd' in norm_size or re.match(r'\d{2,3}[a-d]', norm_size): return 'BRA'
    elif 'y' in norm_size or 't' in norm_size: return 'KIDS'
    elif re.match(r'\d+(\.\d)?', norm_size): return 'NUM_SIZE'
    return norm_size

mercari_df['standardized_size'] = mercari_df['raw_size'].apply(normalize_size)
print("âœ… ã‚µã‚¤ã‚ºæƒ…å ±ã�®æŠ½å‡ºå®Œäº†")


# ==========================================
# 1-2. ã€�å®¹é‡�ã€‘æƒ…å ±ã�®æŠ½å‡º (æ–°è¦�è¿½åŠ : é›»å­�æ©Ÿå™¨å�‘ã�‘)
# ==========================================
# iPhoneã‚„PCã�ªã�©ã�®ä¾¡æ ¼ã�«ç›´çµ�ã�™ã‚‹ãƒ¡ãƒ¢ãƒªå®¹é‡�(16GB, 256GBã�ªã�©)ã‚’æŠ½å‡º
def extract_memory(text):
    # æ•°å­—ã�®å¾Œã�« gb, tb ã�Œç¶šã��ãƒ‘ã‚¿ãƒ¼ãƒ³ã‚’æ�¢ã�™ (ä¾‹: 64gb, 128 GB)
    match = re.search(r'\b(\d{1,4})\s*(?:gb|tb)\b', text, re.IGNORECASE)
    if match:
        # ç©ºç™½ã‚’å‰Šé™¤ã�—ã�¦å°�æ–‡å­—ã�«çµ±ä¸€ (ä¾‹: "128 GB" -> "128gb")
        return match.group(0).lower().replace(' ', '')
    return 'No_Memory'

mercari_df['extracted_memory'] = mercari_df['text_combined'].apply(extract_memory)
print("âœ… ãƒ¡ãƒ¢ãƒªå®¹é‡�(GB/TB)ã�®æŠ½å‡ºå®Œäº†")


# ==========================================
# 1-3. ã€�ç´ æ��ã€‘æƒ…å ±ã�®æŠ½å‡º (æ–°è¦�è¿½åŠ : ã‚¸ãƒ¥ã‚¨ãƒªãƒ¼å�‘ã�‘)
# ==========================================
# ãƒ�ãƒƒã‚¯ãƒ¬ã‚¹ã‚„ãƒªãƒ³ã‚°ã�®ä¾¡æ ¼ã�«ç›´çµ�ã�™ã‚‹è²´é‡‘å±�ã�®ç¨®é¡�ã‚’åˆ¤å®š
def extract_material(text):
    text = text.lower() # å°�æ–‡å­—åŒ–ã�—ã�¦æ¤œç´¢
    # ã‚´ãƒ¼ãƒ«ãƒ‰ (K18, 14k, Goldã�ªã�©)
    if re.search(r'\b(?:10k|14k|18k|22k|24k|gold)\b', text):
        return 'GOLD'
    # ãƒ—ãƒ©ãƒ�ãƒŠ (Platinum, Pt900ã�ªã�©)
    elif re.search(r'\b(?:platinum|pt900|pt950)\b', text):
        return 'PLATINUM'
    # ã‚·ãƒ«ãƒ�ãƒ¼ (Silver, 925, Sterlingã�ªã�©)
    elif re.search(r'\b(?:925|sterling|silver)\b', text):
        return 'SILVER'
    return 'OTHER_MAT'

mercari_df['extracted_material'] = mercari_df['text_combined'].apply(extract_material)
print("âœ… ç´ æ��æƒ…å ±(Gold/Silver)ã�®æŠ½å‡ºå®Œäº†")


# --- 4. ã€�å¹´å¼�ã€‘æƒ…å ±ã�®æŠ½å‡º (æ–°è¦�è¿½åŠ ) ---
# å•†å“�èª¬æ˜�ã�«ã�‚ã‚‹ã€Œ2017ã€�ã‚„ã€Œ1998ã€�ã�ªã�©ã�®è¥¿æš¦ã‚’æŠ½å‡ºã�—ã�¾ã�™
def extract_year(text):
    # 1950å¹´ã€œ2019å¹´ã�®é–“ã�®4æ¡�ã�®æ•°å­—ã‚’æ�¢ã�™
    # \b ã�¯å�˜èª�ã�®åŒºåˆ‡ã‚Šï¼ˆä¾¡æ ¼ã�®2000ãƒ‰ãƒ«ã�ªã�©ã‚’èª¤æ¤œå‡ºã�—ã�ªã�„ã‚ˆã�†ã�«é…�æ…®ï¼‰
    match = re.search(r'\b(19[5-9]\d|200\d|201[0-9])\b', text)
    if match:
        return match.group(0) # è¦‹ã�¤ã�‹ã�£ã�Ÿæ•°å­—ï¼ˆä¾‹: "2017"ï¼‰ã‚’è¿”ã�™
    return 'No_Year'

mercari_df['extracted_year'] = mercari_df['text_combined'].apply(extract_year)
print("âœ… å¹´å¼�æƒ…å ±(Year)ã�®æŠ½å‡ºå®Œäº†")


import numpy as np
import pandas as pd

# 1. åˆ†æ��ç”¨ã�«å­¦ç¿’ãƒ‡ãƒ¼ã‚¿ã�¨ä¾¡æ ¼ã‚’çµ�å�ˆ
y_train_full = train_df_target 
n_train = len(y_train_full)
analysis_df = mercari_df.iloc[:n_train].copy()
analysis_df['price'] = y_train_full.values

# åˆ†æ��ã�—ã�Ÿã�„3ã�¤ã�®åˆ—
target_cols = ['brand_name','category_3','standardized_size','extracted_memory', 'extracted_material','extracted_year']

for col in target_cols:
    print(f"\n" + "="*50)
    print(f"ğŸ“Š ç‰¹å¾´é‡�: {col} ã�®ä¾¡æ ¼ãƒ©ãƒ³ã‚­ãƒ³ã‚°")
    print("="*50)
    
    # é›†è¨ˆ: å¹³å�‡ä¾¡æ ¼(log)ã�¨ä»¶æ•°
    agg_df = analysis_df.groupby(col)['price'].agg(['mean', 'count'])
    
    # ãƒ‰ãƒ«æ�›ç®—
    agg_df['mean_price_dollar'] = np.expm1(agg_df['mean'])
    
    # ä»¶æ•°ã�Œæ¥µç«¯ã�«å°‘ã�ªã�„ãƒ�ã‚¤ã‚ºï¼ˆ10ä»¶æœªæº€ï¼‰ã‚’é™¤å¤–ã�—ã�¦ã€�é«˜ã�„é †ã�«ã‚½ãƒ¼ãƒˆ
    # â€» ãƒ�ã‚¤ã‚ºã‚‚è¦‹ã�Ÿã�„å ´å�ˆã�¯ [agg_df['count'] > 10] ã‚’å‰Šé™¤ã�—ã�¦ã��ã� ã�•ã�„
    result = agg_df[agg_df['count'] > 10].sort_values(by='mean_price_dollar', ascending=False)
    
    print(result[['mean_price_dollar', 'count']].head(15))


# ==========================================
# 1-4. Target Encodingã�®å®Ÿè¡Œ (å¯¾è±¡ãƒªã‚¹ãƒˆã‚’æ›´æ–°)
# ==========================================
# ã�“ã�“ã�«æ–°ã�—ã�„ç‰¹å¾´é‡� 'extracted_memory' ã�¨ 'extracted_material' ã‚’è¿½åŠ ã�—ã�¾ã�—ã�Ÿ
TARGET_ENCODE_COLS = ['brand_name', 'category_3', 'standardized_size', 
                      'extracted_memory', 'extracted_material','extracted_year']

NFOLDS = 5  
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
y_train_full = train_df_target 

# OOF Target Encoding ã‚’é�©ç”¨
for col in TARGET_ENCODE_COLS:
    # æ–°ã�—ã�„TEç‰¹å¾´é‡�åˆ—ã‚’ä½œæˆ�
    mercari_df[f'{col}_te'] = 0.0
    
    # è¨“ç·´ãƒ‡ãƒ¼ã‚¿ã�®ã�¿ã‚’ä¸€æ™‚çš„ã�ªDFã�¨ã�—ã�¦æŠ½å‡º
    train_df_encoded = mercari_df.iloc[:len(y_train_full)].copy()
    train_df_encoded['price'] = y_train_full

    # K-Foldã‚’å›�ã�—ã�¦æƒ…å ±ãƒªãƒ¼ã‚¯ã‚’é˜²ã�� (ã‚«ãƒ³ãƒ‹ãƒ³ã‚°é˜²æ­¢)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df_encoded)):
        target_mean = train_df_encoded.iloc[train_idx].groupby(col)['price'].mean()
        
        # æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿ã�¸ã�®å‰²ã‚Šå½“ã�¦
        mercari_df.loc[valid_idx, f'{col}_te'] = \
            mercari_df.iloc[valid_idx][col].map(target_mean).fillna(y_train_full.mean())

    # ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�¸ã�®å‰²ã‚Šå½“ã�¦ (å…¨è¨“ç·´ãƒ‡ãƒ¼ã‚¿ã�®å¹³å�‡ã‚’ä½¿ç”¨)
    full_target_mean = train_df_encoded.groupby(col)['price'].mean()
    test_start_idx = len(y_train_full)
    mercari_df.loc[test_start_idx:, f'{col}_te'] = \
        mercari_df.iloc[test_start_idx:][col].map(full_target_mean).fillna(y_train_full.mean())

    print(f"âœ… {col} ã�® CV Target Encoding å®Œäº†ã€‚")


# --- 2. Target Encodingå¾Œã�®ç‰¹å¾´é‡�ã�®å†�çµ�å�ˆã�¨Stackingå®Ÿè¡Œ (æœ€çµ‚ç‰ˆ) ---
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack, csr_matrix

print("\n" + "="*60)
print("ğŸ�¯ Stage 4-2: ç‰¹å¾´é‡�ã�®æœ€çµ‚çµ�å�ˆ (æ–°ç‰¹å¾´é‡�è¾¼ã�¿)")
print("="*60)

# 1. æ–°ã�—ã�„ã‚«ãƒ†ã‚´ãƒªç‰¹å¾´é‡�ã�® OneHotEncoding (OHE)
# ã‚µã‚¤ã‚ºã€�å®¹é‡�ã€�ç´ æ��ã€�å¹´å¼�ã�®4ã�¤ã‚’ã�¾ã�¨ã‚�ã�¦OHEåŒ–ã�—ã�¾ã�™
new_cat_cols = ['standardized_size', 'extracted_memory', 'extracted_material', 'extracted_year']
print(f"OHEå‡¦ç�†ä¸­: {new_cat_cols} ...")

# handle_unknown='ignore' ã�¯ã€�ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�«æœªçŸ¥ã�®ã‚«ãƒ†ã‚´ãƒªã�Œã�‚ã�£ã�¦ã‚‚ã‚¨ãƒ©ãƒ¼ã�«ã�—ã�ªã�„è¨­å®šã�§ã�™
ohe = OneHotEncoder(handle_unknown='ignore')
X_new_cats_ohe = ohe.fit_transform(mercari_df[new_cat_cols])

print(f"âœ… æ–°ã‚«ãƒ†ã‚´ãƒªOHEå®Œäº† shape: {X_new_cats_ohe.shape}")


# 2. æ•°å€¤ç‰¹å¾´é‡�ã�®å†�æ§‹ç¯‰ã�¨æ¨™æº–åŒ–
all_numeric_features = ['name_len', 'desc_len', 'desc_word_count', 'has_brand', 'has_category']

# â˜…ã�“ã�“é‡�è¦�: æ–°ã�—ã��ä½œã�£ã�ŸTEç‰¹å¾´é‡�ã‚’ã�™ã�¹ã�¦ãƒªã‚¹ãƒˆã�«å…¥ã‚Œã�¾ã�™
all_te_features = [
    'brand_name_te', 'category_3_te', 
    'standardized_size_te', 
    'extracted_memory_te', 
    'extracted_material_te', 
    'extracted_year_te'
]

print(f"æ•°å€¤ç‰¹å¾´é‡�æ§‹ç¯‰ä¸­... å�ˆè¨ˆ {len(all_numeric_features) + len(all_te_features)} åˆ—")

numeric_features_raw = mercari_df[all_numeric_features + all_te_features].values
scaler = StandardScaler()
numeric_features = scaler.fit_transform(numeric_features_raw)


# 3. æœ€çµ‚çµ�å�ˆè¡Œåˆ—ã�®ä½œæˆ�
# å…¨ã�¦ã�®ç–�è¡Œåˆ—ã�¨æ•°å€¤ç‰¹å¾´é‡�ã‚’æ°´å¹³æ–¹å�‘(hstack)ã�«çµ�å�ˆã�—ã�¾ã�™ã€‚
n_train = len(y_train_full)

combined_matrix_train = (
    X_name[:n_train],                # Name TF-IDF
    X_descp[:n_train],               # Description TF-IDF
    X_brand_name[:n_train],          # Brand OHE
    X_item_condition_id[:n_train],   # Condition OHE
    X_shipping[:n_train],            # Shipping OHE
    X_category_1[:n_train],          # Category1 OHE
    X_category_2[:n_train],          # Category2 OHE
    X_category_3[:n_train],          # Category3 OHE
    X_new_cats_ohe[:n_train],        # â˜…æ–°ç‰¹å¾´é‡� (Size, Memory, Material, Year) OHE
    csr_matrix(numeric_features[:n_train]) # â˜…å…¨æ•°å€¤ç‰¹å¾´é‡� (TEå�«ã‚€)
)
X_train_full = hstack(combined_matrix_train).tocsr()

combined_matrix_test = (
    X_name[n_train:],
    X_descp[n_train:], 
    X_brand_name[n_train:], 
    X_item_condition_id[n_train:], 
    X_shipping[n_train:],
    X_category_1[n_train:], 
    X_category_2[n_train:], 
    X_category_3[n_train:], 
    X_new_cats_ohe[n_train:],        # â˜…æ–°ç‰¹å¾´é‡� (Size, Memory, Material, Year) OHE
    csr_matrix(numeric_features[n_train:]) # â˜…å…¨æ•°å€¤ç‰¹å¾´é‡� (TEå�«ã‚€)
)
X_test_full = hstack(combined_matrix_test).tocsr()


# å¤‰æ•°å��ã‚’çµ±ä¸€
X_train = X_train_full
#X_test = X_test_full

print(f"âœ… æœ€çµ‚ç‰¹å¾´é‡�çµ�å�ˆå®Œäº†ã€‚")
print(f"   X_train.shape: {X_train.shape}")


import gc
from sklearn.neural_network import MLPRegressor # â†� è¿½åŠ 

# --- 2. Stackingã�®å®Ÿè¡Œï¼ˆãƒ™ãƒ¼ã‚¹ãƒ¢ãƒ‡ãƒ« OOFäºˆæ¸¬ç”Ÿæˆ�ï¼‰ ---

# ğŸ’¡ TF-IDFã�®ç‰¹å¾´é‡�æ¬¡å…ƒã‚’å�–å¾—
name_dim = X_name.shape[1]
desc_dim = X_descp.shape[1]

# â˜… ãƒ¡ãƒ¢ãƒªç¯€ç´„ã�®ã�Ÿã‚�ã€�ã�“ã�“ã�§ç¢ºå®Ÿã�«ä¸�è¦�å¤‰æ•°ã‚’æ¶ˆã�™
del X_name, X_descp, train_df_encoded, analysis_df#,numeric_features, numeric_features_raw, safe_categories_test, mercari_df, X_new_cats_ohe, safe_categories
del numeric_features_raw, safe_categories_test, mercari_df, X_new_cats_ohe, safe_categories
gc.collect() 

X_train_full = X_train
y_train_full = y_train_full

D_features = X_train_full.shape[1]  # å…¨ç‰¹å¾´é‡�ã�®æ¬¡å…ƒæ•°ã‚’å�–å¾—

# OOFäºˆæ¸¬ã‚’æ ¼ç´�ã�™ã‚‹é…�åˆ—ã‚’åˆ�æœŸåŒ–
oof_preds = {
    'all': np.zeros(n_train),
    'name': np.zeros(n_train),
    'desc': np.zeros(n_train),
    'sgd_all': np.zeros(n_train),
    'sgd_name': np.zeros(n_train),
    'sgd_desc': np.zeros(n_train),
    'fm_all': np.zeros(n_train),
    'mlp_all': np.zeros(n_train) # â†� è¿½åŠ : MLPç”¨
}

test_preds = {
    'all': np.zeros(X_test_full.shape[0]), 
    'name': np.zeros(X_test_full.shape[0]),
    'desc': np.zeros(X_test_full.shape[0]),
    'sgd_all': np.zeros(X_test_full.shape[0]),
    'sgd_name': np.zeros(X_test_full.shape[0]),
    'sgd_desc': np.zeros(X_test_full.shape[0]),
    'fm_all': np.zeros(X_test_full.shape[0]),
    'mlp_all': np.zeros(X_test_full.shape[0]) # â†� è¿½åŠ : MLPç”¨
}

RIDGE_ALPHA = 4.3 

print("\n" + "="*60)
print("ğŸ�¯ Stacking Level 1: Ridge / SGD / FM / MLP (5-Fold)")
print("="*60)

# X_train_fullã�‹ã‚‰ãƒ†ã‚­ã‚¹ãƒˆç‰¹å¾´é‡�ã�®ã�¿ã‚’æŠ½å‡º
X_train_name_only = X_train_full[:, :name_dim]
X_train_desc_only = X_train_full[:, name_dim: name_dim + desc_dim]

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train_full)):
    print(f"--- Fold {fold+1}/{NFOLDS} ---")
    
    X_tr, X_val = X_train_full[train_idx], X_train_full[valid_idx]
    y_tr, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[valid_idx]

    # --- 1. Ridge Models ---
    print(f"   Training Ridge...")
    ridge_all = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", fit_intercept=False)
    ridge_all.fit(X_tr, y_tr)
    oof_preds['all'][valid_idx] = ridge_all.predict(X_val)
    test_preds['all'] += ridge_all.predict(X_test_full) / NFOLDS
    
    ridge_name = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", fit_intercept=False)
    ridge_name.fit(X_train_name_only[train_idx], y_tr)
    oof_preds['name'][valid_idx] = ridge_name.predict(X_train_name_only[valid_idx])
    test_preds['name'] += ridge_name.predict(X_test_full[:, :name_dim]) / NFOLDS
    
    ridge_desc = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", fit_intercept=False)
    ridge_desc.fit(X_train_desc_only[train_idx], y_tr)
    oof_preds['desc'][valid_idx] = ridge_desc.predict(X_train_desc_only[valid_idx])
    test_preds['desc'] += ridge_desc.predict(X_test_full[:, name_dim: name_dim + desc_dim]) / NFOLDS

    del ridge_all, ridge_name, ridge_desc
    
    # --- 2. SGD Models ---
    print(f"   Training SGD...")
    # SGDè¨­å®š: Huber Lossã�§å¤–ã‚Œå€¤ã�«å¼·ã��ã�™ã‚‹
    sgd_params = {
        'loss': 'huber', 'penalty': 'l2', 'alpha': 1e-4, 
        'random_state': 42, 'max_iter': 1000, 'tol': 1e-3
    }
    
    sgd_all = SGDRegressor(**sgd_params)
    sgd_all.fit(X_tr, y_tr)
    oof_preds['sgd_all'][valid_idx] = sgd_all.predict(X_val)
    test_preds['sgd_all'] += sgd_all.predict(X_test_full) / NFOLDS
    
    sgd_name = SGDRegressor(**sgd_params)
    sgd_name.fit(X_train_name_only[train_idx], y_tr)
    oof_preds['sgd_name'][valid_idx] = sgd_name.predict(X_train_name_only[valid_idx])
    test_preds['sgd_name'] += sgd_name.predict(X_test_full[:, :name_dim]) / NFOLDS

    sgd_desc = SGDRegressor(**sgd_params)
    sgd_desc.fit(X_train_desc_only[train_idx], y_tr)
    oof_preds['sgd_desc'][valid_idx] = sgd_desc.predict(X_train_desc_only[valid_idx])
    test_preds['sgd_desc'] += sgd_desc.predict(X_test_full[:, name_dim: name_dim + desc_dim]) / NFOLDS
    
    del sgd_all, sgd_name, sgd_desc
    gc.collect()
    
    # --- 3. FM Model ---
    print(f"   Training FM...")
    fm = FM_FTRL_Wrapper(
        D=D_features, D_fm=8, alpha=0.05, alpha_fm=0.01, iters=2
    )
    fm.fit(X_tr, y_tr)
    oof_preds['fm_all'][valid_idx] = fm.predict(X_val)
    test_preds['fm_all'] += fm.predict(X_test_full) / NFOLDS
    
    del fm
    gc.collect()

    # ==========================
    # 4. MLP Regressor (çœ�ãƒ¡ãƒ¢ãƒªæˆ¦ç•¥)
    # ==========================
    print(f"   Training MLP (SVD On-the-fly)...")
    
    # â˜… ã�“ã�“ã�§ã€Œå¿…è¦�ã�ªåˆ†ã� ã�‘ã€�çµ�å�ˆã�—ã�¾ã�™ (Train foldåˆ†ã�®ã�¿)
    # ãƒ¡ãƒ¢ãƒªæ¶ˆè²»: å…¨ä½“ã�®1/5ã�ªã�®ã�§ ç´„400MB ç¨‹åº¦ã�§æ¸ˆã�¿ã�¾ã�™ï¼�
    X_tr_mlp = np.hstack([
        X_name_svd[train_idx], 
        X_descp_svd[train_idx], 
        numeric_features[train_idx]
    ]).astype(np.float32)
    
    # mlp = MLPRegressor(
    #     hidden_layer_sizes=(256,128, 64), # å°‘ã�—ãƒªãƒƒãƒ�ã�«
    #     activation='relu', solver='adam', batch_size=512,
    #     early_stopping=True, max_iter=100, random_state=42, verbose=False
    # )
    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),  # å±¤ã‚’æ·±ã��ã€�å°‘ã�—åºƒã��
        activation='relu', 
        solver='adam', 
        batch_size=512,      # ãƒ�ãƒƒãƒ�ã‚µã‚¤ã‚ºã�¯ç¶­æŒ�ï¼ˆé€Ÿåº¦ã�¨ãƒ¡ãƒ¢ãƒªã�®ãƒ�ãƒ©ãƒ³ã‚¹ï¼‰
        alpha=0.01,          # æ­£å‰‡åŒ–ã‚’å¼·åŒ– (é��å­¦ç¿’æŠ‘åˆ¶)
        learning_rate='adaptive', # å­¦ç¿’ç�‡ã‚’å¾�ã€…ã�«ä¸‹ã�’ã‚‹
        learning_rate_init=0.001, # åˆ�æœŸå­¦ç¿’ç�‡
        early_stopping=True, 
        validation_fraction=0.1, # early_stoppingç”¨ã�®ãƒ‡ãƒ¼ã‚¿å‰²å�ˆ
        n_iter_no_change=10,     # 10å›�æ”¹å–„ã�ªã�‘ã‚Œã�°æ­¢ã‚�ã‚‹
        max_iter=200, 
        random_state=42, 
        verbose=False
    )
    mlp.fit(X_tr_mlp, y_tr)
    del X_tr_mlp # å­¦ç¿’çµ‚ã‚�ã�£ã�Ÿã‚‰å�³è§£æ”¾
    gc.collect()
    
    # OOFäºˆæ¸¬ (Validation foldåˆ†ã�®ã�¿ä½œæˆ�)
    X_val_mlp = np.hstack([
        X_name_svd[valid_idx], 
        X_descp_svd[valid_idx], 
        numeric_features[valid_idx]
    ]).astype(np.float32)
    oof_preds['mlp_all'][valid_idx] = mlp.predict(X_val_mlp)
    del X_val_mlp # å�³è§£æ”¾
    
    # ãƒ†ã‚¹ãƒˆäºˆæ¸¬ (ã�“ã�“ã�Œæœ€é›£é–¢)
    # 346ä¸‡è¡Œã‚’ä¸€æ°—ã�«ä½œã‚‹ã�¨æ­»ã�¬ã�®ã�§ã€�50ä¸‡è¡Œã�šã�¤äºˆæ¸¬ã�—ã�¦è¶³ã�—ã�¾ã�™
    print("      Predicting Test in chunks...", end="")
    chunk_size = 500000
    test_ids_len = X_test_full.shape[0]
    n_test_chunks = (test_ids_len + chunk_size - 1) // chunk_size
    
    current_fold_preds = []
    
    for i in range(n_test_chunks):
        s_idx = i * chunk_size
        e_idx = min((i + 1) * chunk_size, test_ids_len)
        
        # SVDã�®ãƒ†ã‚¹ãƒˆéƒ¨åˆ†ã�¯ X_name_svd[n_train:] ã�«ã�‚ã‚‹ã�“ã�¨ã�«æ³¨æ„�
        # indexèª¿æ•´: ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�®é–‹å§‹ä½�ç½®ã�¯ã�šã‚‰ã�™å¿…è¦�ã�‚ã‚Š
        test_s = n_train + s_idx
        test_e = n_train + e_idx
        
        # ãƒ�ãƒ£ãƒ³ã‚¯ã� ã�‘çµ�å�ˆ
        X_test_chunk = np.hstack([
            X_name_svd[test_s : test_e], 
            X_descp_svd[test_s : test_e], 
            numeric_features[test_s : test_e]
        ]).astype(np.float32)
        
        # äºˆæ¸¬
        chunk_pred = mlp.predict(X_test_chunk)
        current_fold_preds.append(chunk_pred)
        
        del X_test_chunk
    
    # çµ�å�ˆã�—ã�¦åŠ ç®—
    test_preds['mlp_all'] += np.concatenate(current_fold_preds) / NFOLDS
    del current_fold_preds, mlp
    print(" Done.")
    gc.collect()

# ============================================================
# ã‚¹ã‚³ã‚¢ç¢ºèª�
# ============================================================
rmse_ridge = np.sqrt(mean_squared_error(y_train_full, oof_preds['all']))
rmse_sgd = np.sqrt(mean_squared_error(y_train_full, oof_preds['sgd_all']))
rmse_fm = np.sqrt(mean_squared_error(y_train_full, oof_preds['fm_all']))
rmse_mlp = np.sqrt(mean_squared_error(y_train_full, oof_preds['mlp_all']))

print("\n" + "="*30)
print(f"âœ… OOF RMSLE (Ridge All): {rmse_ridge:.5f}")
print(f"âœ… OOF RMSLE (SGD All):   {rmse_sgd:.5f}")
print(f"âœ… OOF RMSLE (FM All):    {rmse_fm:.5f}")
print(f"âœ… OOF RMSLE (MLP All):   {rmse_mlp:.5f}")
print("="*30)

# =======================================================
print("ğŸ”§ å¤§å®¹é‡�ç–�è¡Œåˆ—ã�Šã‚ˆã�³ä¸�è¦�ã�ªãƒ¢ãƒ‡ãƒ«ã‚’è§£æ”¾ä¸­...")

# å­¦ç¿’æ¸ˆã�¿ãƒ¢ãƒ‡ãƒ«ã‚„ä¸€æ™‚å¤‰æ•°ã‚’è§£æ”¾
del X_train_name_only, X_train_desc_only
gc.collect() 

print("âœ… ãƒ¡ãƒ¢ãƒªè§£æ”¾å®Œäº†ã€‚LightGBMå­¦ç¿’ã�¸ç§»è¡Œã�—ã�¾ã�™ã€‚")


# import gc
# # --- 2. Stackingã�®å®Ÿè¡Œï¼ˆãƒ™ãƒ¼ã‚¹ãƒ¢ãƒ‡ãƒ« OOFäºˆæ¸¬ç”Ÿæˆ�ï¼‰ ---

# # ğŸ’¡ TF-IDFã�®ç‰¹å¾´é‡�æ¬¡å…ƒã‚’å�–å¾—
# name_dim = X_name.shape[1]
# desc_dim = X_descp.shape[1]

# del X_name, X_descp,train_df_encoded,analysis_df,numeric_features,numeric_features_raw,safe_categories_test,mercari_df,X_new_cats_ohe,safe_categories

# gc.collect() 

# X_train_full = X_train
# y_train_full = y_train_full

# D_features = X_train_full.shape[1]  # å…¨ç‰¹å¾´é‡�ã�®æ¬¡å…ƒæ•°ã‚’å�–å¾—

# # OOFäºˆæ¸¬ã‚’æ ¼ç´�ã�™ã‚‹é…�åˆ—ã‚’åˆ�æœŸåŒ–
# oof_preds = {
#     'all': np.zeros(n_train),
#     'name': np.zeros(n_train),
#     'desc': np.zeros(n_train),
#     'sgd_all': np.zeros(n_train),      # â†� è¿½åŠ 
#     'sgd_name': np.zeros(n_train),     # â†� è¿½åŠ 
#     'sgd_desc': np.zeros(n_train),
#     # FMç”¨ã�®äºˆæ¸¬å€¤é…�åˆ—ã‚’åˆ�æœŸåŒ–
#     'fm_all': np.zeros(n_train)
# }

# test_preds = {
#     'all': np.zeros(X_test_full.shape[0]), 
#     'name': np.zeros(X_test_full.shape[0]),
#     'desc': np.zeros(X_test_full.shape[0]),
#     'sgd_all': np.zeros(X_test_full.shape[0]),   # â†� è¿½åŠ 
#     'sgd_name': np.zeros(X_test_full.shape[0]),  # â†� è¿½åŠ 
#     'sgd_desc': np.zeros(X_test_full.shape[0]),
#     'fm_all': np.zeros(X_test_full.shape[0])
# }

# RIDGE_ALPHA = 4.3 

# print("\n" + "="*60)
# print("ğŸ�¯ Ridge OOF/ãƒ†ã‚¹ãƒˆäºˆæ¸¬ã�®ç”Ÿæˆ� (5-Fold) - Stackingãƒ™ãƒ¼ã‚¹")
# print("="*60)

# # X_train_fullã�‹ã‚‰ãƒ†ã‚­ã‚¹ãƒˆç‰¹å¾´é‡�ã�®ã�¿ã‚’æŠ½å‡º
# X_train_name_only = X_train_full[:, :name_dim]
# X_train_desc_only = X_train_full[:, name_dim: name_dim + desc_dim]

# for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train_full)):
#     print(f"--- Fold {fold+1}/{NFOLDS} ---")
    
#     X_tr, X_val = X_train_full[train_idx], X_train_full[valid_idx]
#     y_tr, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[valid_idx]

#     print(f"   Training Ridge...")

#     # --- 1. å…¨ç‰¹å¾´é‡� Ridge ãƒ¢ãƒ‡ãƒ« ---
#     ridge_all = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", fit_intercept=False)
#     ridge_all.fit(X_tr, y_tr)
#     oof_preds['all'][valid_idx] = ridge_all.predict(X_val)
#     test_preds['all'] += ridge_all.predict(X_test_full) / NFOLDS
    
#     # --- 2. Name ã�®ã�¿ Ridge ãƒ¢ãƒ‡ãƒ« ---
#     ridge_name = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", fit_intercept=False)
#     ridge_name.fit(X_train_name_only[train_idx], y_tr)
#     oof_preds['name'][valid_idx] = ridge_name.predict(X_train_name_only[valid_idx])
#     test_preds['name'] += ridge_name.predict(X_test_full[:, :name_dim]) / NFOLDS
    
#     # --- 3. Description ã�®ã�¿ Ridge ãƒ¢ãƒ‡ãƒ« ---
#     ridge_desc = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", fit_intercept=False)
#     ridge_desc.fit(X_train_desc_only[train_idx], y_tr)
#     oof_preds['desc'][valid_idx] = ridge_desc.predict(X_train_desc_only[valid_idx])
#     test_preds['desc'] += ridge_desc.predict(X_test_full[:, name_dim: name_dim + desc_dim]) / NFOLDS


#     del ridge_all, ridge_name, ridge_desc
#     # ============================================================
#     # SGD (Huber Loss) ãƒ¢ãƒ‡ãƒ«ï¼ˆæ–°è¦�è¿½åŠ ï¼‰
#     # ============================================================
    
#     # --- 4. å…¨ç‰¹å¾´é‡� SGD ãƒ¢ãƒ‡ãƒ« ---
#     print(f"   Training SGD...")
    
#     sgd_all = SGDRegressor(
#         loss='huber',          # å¤–ã‚Œå€¤ã�«å¼·ã�„
#         penalty='l2',
#         alpha=1e-4,
#         random_state=42,
#         max_iter=1000,
#         tol=1e-3
#     )
#     sgd_all.fit(X_tr, y_tr)
#     oof_preds['sgd_all'][valid_idx] = sgd_all.predict(X_val)
#     test_preds['sgd_all'] += sgd_all.predict(X_test_full) / NFOLDS
    
#     # --- 5. Name ã�®ã�¿ SGD ãƒ¢ãƒ‡ãƒ« ---
#     sgd_name = SGDRegressor(
#         loss='huber',
#         penalty='l2',
#         alpha=1e-4,
#         random_state=42,
#         max_iter=1000,
#         tol=1e-3
#     )
#     sgd_name.fit(X_train_name_only[train_idx], y_tr)
#     oof_preds['sgd_name'][valid_idx] = sgd_name.predict(X_train_name_only[valid_idx])
#     test_preds['sgd_name'] += sgd_name.predict(X_test_full[:, :name_dim]) / NFOLDS

#     # --- 6. Description ã�®ã�¿ SGD ãƒ¢ãƒ‡ãƒ« ---
#     sgd_desc = SGDRegressor(
#         loss='huber',
#         penalty='l2',
#         alpha=1e-4,
#         random_state=42,
#         max_iter=1000,
#         tol=1e-3
#     )
#     sgd_desc.fit(X_train_desc_only[train_idx], y_tr)
#     oof_preds['sgd_desc'][valid_idx] = sgd_desc.predict(X_train_desc_only[valid_idx])
#     test_preds['sgd_desc'] += sgd_desc.predict(X_test_full[:, name_dim: name_dim + desc_dim]) / NFOLDS
    
#     # Foldå†…ã�®ãƒ¢ãƒ‡ãƒ«ã‚’å‰Šé™¤ï¼ˆãƒ¡ãƒ¢ãƒªç¯€ç´„ï¼‰
#     del sgd_all, sgd_name, sgd_desc
#     gc.collect()
    
#     print(f"   Training FM...")
#     # FMãƒ¢ãƒ‡ãƒ«ã�®åˆ�æœŸåŒ–ã�¨å­¦ç¿’
#     fm = FM_FTRL_Wrapper(
#         D=D_features,
#         D_fm=8,           # å› å­�æ•°ï¼ˆãƒ¡ãƒ¢ãƒªåˆ¶ç´„ã�«å¿œã�˜ã�¦èª¿æ•´ï¼‰
#         alpha=0.05,       # å­¦ç¿’ç�‡
#         alpha_fm=0.01,    # FMå­¦ç¿’ç�‡
#         iters=2           # ã‚¨ãƒ�ãƒƒã‚¯æ•°ï¼ˆæ™‚é–“åˆ¶ç´„ã�«å¿œã�˜ã�¦èª¿æ•´ï¼‰
#     )
#     fm.fit(X_tr, y_tr)
    
#     # OOFäºˆæ¸¬
#     oof_preds['fm_all'][valid_idx] = fm.predict(X_val)
    
#     # ãƒ†ã‚¹ãƒˆäºˆæ¸¬ï¼ˆå¹³å�‡åŒ–ï¼‰
#     test_preds['fm_all'] += fm.predict(X_test_full) / NFOLDS
    
#     # ãƒ¡ãƒ¢ãƒªè§£æ”¾
#     del fm, X_tr, X_val, y_tr, y_val
#     gc.collect()

# # ============================================================
# # ã‚¹ã‚³ã‚¢ç¢ºèª�
# # ============================================================
# rmse_ridge = np.sqrt(mean_squared_error(y_train_full, oof_preds['all']))
# rmse_sgd = np.sqrt(mean_squared_error(y_train_full, oof_preds['sgd_all']))
# rmse_fm = np.sqrt(mean_squared_error(y_train_full, oof_preds['fm_all']))


# print("\n" + "="*30)
# print(f"âœ… OOF RMSLE (Ridge All): {rmse_ridge:.5f}")
# print(f"âœ… OOF RMSLE (SGD All):   {rmse_sgd:.5f}")
# print(f"âœ… OOF RMSLE (FM All):   {rmse_fm:.5f}")

# print("="*30)
# # ğŸš¨ ãƒ¡ãƒ¢ãƒªè§£æ”¾ã�®è¿½åŠ ã‚³ãƒ¼ãƒ‰ã�¯ã�“ã�“ã�§ã�™ ğŸš¨
# # =======================================================
# import gc

# print("ğŸ”§ å¤§å®¹é‡�ç–�è¡Œåˆ—ã�Šã‚ˆã�³ä¸�è¦�ã�ªãƒ¢ãƒ‡ãƒ«ã‚’è§£æ”¾ä¸­...")


# # å­¦ç¿’æ¸ˆã�¿Ridgeãƒ¢ãƒ‡ãƒ«ã‚‚è§£æ”¾ (äºˆæ¸¬å€¤ã�¯ã�™ã�§ã�« oof_preds/test_preds ã�«æ ¼ç´�æ¸ˆã�¿)
# #del ridge_all, ridge_name, ridge_desc
# # Name/Desc ã�®åˆ‡ã‚Šå‡ºã�—ç”¨ä¸€æ™‚é…�åˆ—ã‚‚è§£æ”¾
# del X_train_name_only, X_train_desc_only

# gc.collect() 
# print("âœ… ãƒ¡ãƒ¢ãƒªè§£æ”¾å®Œäº†ã€‚LightGBMå­¦ç¿’ã�¸ç§»è¡Œã�—ã�¾ã�™ã€‚")


del X_train,combined_matrix_train,combined_matrix_test,X_brand_name,X_item_condition_id,X_shipping,X_category_3,X_category_1,X_category_2
gc.collect() 


print("\n" + "="*60)
print("ğŸ”§ LightGBMç”¨ç‰¹å¾´é‡�ã�®äº‹å‰�æŠ½å‡ºï¼ˆãƒ¡ãƒ¢ãƒªå‰Šæ¸›å¯¾ç­–ï¼‰")
print("="*60)

# 1. å¿…è¦�ã�ªæƒ…å ±ã‚’å…ˆã�«å�–å¾—
nrow_train = X_train_full.shape[0]
start_col_dense = name_dim + desc_dim

# 2. å¯†ã�ªç‰¹å¾´é‡�ã‚’æŠ½å‡ºï¼ˆğŸš¨ ç–�è¡Œåˆ—ã�®ã�¾ã�¾ä¿�æŒ�ï¼‰
print("ğŸ”§ å¯†ã�ªç‰¹å¾´é‡�ã‚’æŠ½å‡ºä¸­...")
X_train_dense_features = X_train_full[:, start_col_dense:]  # â†� .toarray()ã‚’å‰Šé™¤ï¼�
X_test_dense_features = X_test_full[:, start_col_dense:]    # â†� .toarray()ã‚’å‰Šé™¤ï¼�
print(f"âœ… æŠ½å‡ºå®Œäº†: Train {X_train_dense_features.shape}, Test {X_test_dense_features.shape}")


del X_train_full, X_test_full,split_df_test,split_df 


# 3. SVDãƒ‡ãƒ¼ã‚¿ã‚’åˆ†å‰²
print("ğŸ”§ SVDãƒ‡ãƒ¼ã‚¿ã‚’åˆ†å‰²ä¸­...")
X_train_name_svd = X_name_svd[:nrow_train]
X_test_name_svd = X_name_svd[nrow_train:]
print(f"âœ… SVDåˆ†å‰²å®Œäº†: Train {X_train_name_svd.shape}, Test {X_test_name_svd.shape}")

X_descp_svd

print("ğŸ”§ SVDãƒ‡ãƒ¼ã‚¿ã‚’åˆ†å‰²ä¸­...")
X_train_descp_svd = X_descp_svd[:nrow_train]
X_test_descp_svd = X_descp_svd[nrow_train:]
print(f"âœ… SVDåˆ†å‰²å®Œäº†: Train {X_train_descp_svd.shape}, Test {X_test_descp_svd.shape}")

# ğŸš¨ ã�“ã�“ã�§å·¨å¤§ã�ªè¡Œåˆ—ã‚’å‰Šé™¤
print("\nğŸ”§ å·¨å¤§è¡Œåˆ—å‰Šé™¤ä¸­...")
del X_name_svd,X_descp_svd
gc.collect()
print("âœ… X_name_svd X_descp_svd å‰Šé™¤å®Œäº†")


import sys
import gc
import numpy as np
import pandas as pd
from scipy.sparse import issparse

def print_memory_usage(top_n=20):
    """
    ç�¾åœ¨ã�®ãƒ¡ãƒ¢ãƒªä½¿ç”¨é‡�ã‚’ãƒ©ãƒ³ã‚­ãƒ³ã‚°å½¢å¼�ã�§è¡¨ç¤º
    
    Parameters:
    -----------
    top_n : int
        è¡¨ç¤ºã�™ã‚‹å¤‰æ•°ã�®æ•°ï¼ˆãƒ‡ãƒ•ã‚©ãƒ«ãƒˆ: 20ï¼‰
    """
    
    def get_size(obj):
        """ã‚ªãƒ–ã‚¸ã‚§ã‚¯ãƒˆã�®ã‚µã‚¤ã‚ºã‚’ãƒ�ã‚¤ãƒˆå�˜ä½�ã�§å�–å¾—"""
        if issparse(obj):
            # ç–�è¡Œåˆ—
            return obj.data.nbytes + obj.indices.nbytes + obj.indptr.nbytes
        elif isinstance(obj, np.ndarray):
            # numpyé…�åˆ—
            return obj.nbytes
        elif isinstance(obj, pd.DataFrame):
            # pandas DataFrame
            return obj.memory_usage(deep=True).sum()
        elif isinstance(obj, pd.Series):
            # pandas Series
            return obj.memory_usage(deep=True)
        else:
            # ã��ã�®ä»–
            return sys.getsizeof(obj)
    
    # ã�™ã�¹ã�¦ã�®å¤‰æ•°ã‚’å�–å¾—
    variables = []
    for name, obj in globals().items():
        # ãƒ—ãƒ©ã‚¤ãƒ™ãƒ¼ãƒˆå¤‰æ•°ã‚„ãƒ¢ã‚¸ãƒ¥ãƒ¼ãƒ«ã�¯é™¤å¤–
        if not name.startswith('_') and not callable(obj) and name not in ['In', 'Out', 'get_ipython']:
            try:
                size_bytes = get_size(obj)
                size_mb = size_bytes / (1024**2)
                size_gb = size_bytes / (1024**3)
                
                # å�‹æƒ…å ±ã‚’å�–å¾—
                if issparse(obj):
                    obj_type = f"sparse {obj.format}"
                    shape = obj.shape
                elif isinstance(obj, np.ndarray):
                    obj_type = f"ndarray {obj.dtype}"
                    shape = obj.shape
                elif isinstance(obj, pd.DataFrame):
                    obj_type = "DataFrame"
                    shape = obj.shape
                elif isinstance(obj, pd.Series):
                    obj_type = "Series"
                    shape = (len(obj),)
                else:
                    obj_type = type(obj).__name__
                    shape = "-"
                
                variables.append({
                    'Variable': name,
                    'Type': obj_type,
                    'Shape': str(shape),
                    'Size_MB': size_mb,  # ã‚­ãƒ¼å��ã‚’å¤‰æ›´ï¼ˆæ‹¬å¼§ã�ªã�—ï¼‰
                    'Size_GB': size_gb   # ã‚­ãƒ¼å��ã‚’å¤‰æ›´ï¼ˆæ‹¬å¼§ã�ªã�—ï¼‰
                })
            except:
                pass
    
    # DataFrameã�«å¤‰æ�›ã�—ã�¦ã‚½ãƒ¼ãƒˆ
    df = pd.DataFrame(variables)
    if len(df) == 0:
        print("å¤‰æ•°ã�Œè¦‹ã�¤ã�‹ã‚Šã�¾ã�›ã‚“ã�§ã�—ã�Ÿ")
        return df
    
    df = df.sort_values('Size_MB', ascending=False).head(top_n)
    
    # è¡¨ç¤º
    print("\n" + "="*80)
    print(f"ğŸ“Š ãƒ¡ãƒ¢ãƒªä½¿ç”¨é‡�ãƒ©ãƒ³ã‚­ãƒ³ã‚° (Top {top_n})")
    print("="*80)
    
    for idx, row in df.iterrows():
        # ä¿®æ­£ï¼šå¤‰æ•°ã‚’å…ˆã�«å�–ã‚Šå‡ºã�™
        var_name = row['Variable']
        var_type = row['Type']
        var_shape = row['Shape']
        size_gb = row['Size_GB']
        size_mb = row['Size_MB']
        
        # ã‚µã‚¤ã‚ºè¡¨ç¤ºã�®æ±ºå®š
        if size_gb >= 1:
            size_str = f"{size_gb:.3f} GB"
        else:
            size_str = f"{size_mb:.2f} MB"
        
        print(f"{var_name:30s} | {var_type:20s} | {var_shape:20s} | {size_str}")
    
    print("="*80)
    total_gb = df['Size_GB'].sum()
    print(f"å�ˆè¨ˆãƒ¡ãƒ¢ãƒªä½¿ç”¨é‡�: {total_gb:.3f} GB")
    print("="*80 + "\n")
    
    return df

# ä½¿ç”¨ä¾‹
memory_df = print_memory_usage(top_n=30)



# ãƒ¡ãƒ¢ãƒªä½¿ç”¨é‡�ç¢ºèª�
import lightgbm as lgb
import psutil
import gc
import numpy as np
from scipy.sparse import csr_matrix, hstack

print("="*60)
print("ğŸ�¯ Step 5-1: LightGBMãƒ¡ã‚¿ãƒ¢ãƒ‡ãƒ«ã�®å­¦ç¿’é–‹å§‹")
print("="*60)

# ============================================================
# 1. ãƒ‡ãƒ¼ã‚¿å�‹æœ€é�©åŒ–
# ============================================================
print("ğŸ”§ ãƒ‡ãƒ¼ã‚¿å�‹ã‚’æœ€é�©åŒ–ä¸­...")
X_train_name_svd = X_train_name_svd.astype(np.float32)
X_train_descp_svd = X_train_descp_svd.astype(np.float32)
X_test_name_svd = X_test_name_svd.astype(np.float32)
X_test_descp_svd = X_test_descp_svd.astype(np.float32)

# test_idã‚’ä¿�å­˜ã�—ã�¦DataFrameå‰Šé™¤
test_ids = test_df['test_id'].values
del train_df, test_df
gc.collect()

print(f"ğŸ“Š ãƒ¡ãƒ¢ãƒªå‰Šæ¸›å¾Œ: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")


# ============================================================
import gc
from scipy.sparse import vstack, csr_matrix, hstack
import numpy as np

print("ğŸ”§ Creating meta features in chunks...")

# ãƒ�ãƒ£ãƒ³ã‚¯ã‚µã‚¤ã‚ºã�®è¨­å®šï¼ˆãƒ¡ãƒ¢ãƒªã�«å¿œã�˜ã�¦èª¿æ•´å�¯èƒ½ï¼‰
chunk_size = 500000  # 40ä¸‡è¡Œã�šã�¤å‡¦ç�†
n_train = len(y_train_full)
n_chunks = (n_train + chunk_size - 1) // chunk_size

print(f"ğŸ“Š Total rows: {n_train:,}")
print(f"ğŸ“Š Chunk size: {chunk_size:,}")
print(f"ğŸ“Š Number of chunks: {n_chunks}")

# ãƒ�ãƒ£ãƒ³ã‚¯ã‚’æ ¼ç´�ã�™ã‚‹ãƒªã‚¹ãƒˆ
X_train_meta_chunks = []

for i in range(n_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, n_train)
    
    print(f"\n--- Chunk {i+1}/{n_chunks}: rows {start_idx:,} to {end_idx:,} ---")
    
    # ã�“ã�®ãƒ�ãƒ£ãƒ³ã‚¯ã�®ç‰¹å¾´é‡�ã‚’çµ�å�ˆ
    chunk = hstack([
        X_train_dense_features[start_idx:end_idx],
        # Ridgeäºˆæ¸¬ï¼ˆ3ç¨®ï¼‰
        csr_matrix(np.array(oof_preds['all'][start_idx:end_idx], dtype=np.float32).reshape(-1, 1)),
        csr_matrix(np.array(oof_preds['name'][start_idx:end_idx], dtype=np.float32).reshape(-1, 1)),
        csr_matrix(np.array(oof_preds['desc'][start_idx:end_idx], dtype=np.float32).reshape(-1, 1)),
        # SGDäºˆæ¸¬ï¼ˆ3ç¨®ï¼‰
        csr_matrix(np.array(oof_preds['sgd_all'][start_idx:end_idx], dtype=np.float32).reshape(-1, 1)),
        csr_matrix(np.array(oof_preds['sgd_name'][start_idx:end_idx], dtype=np.float32).reshape(-1, 1)),
        csr_matrix(np.array(oof_preds['sgd_desc'][start_idx:end_idx], dtype=np.float32).reshape(-1, 1)),
         # FMäºˆæ¸¬ï¼ˆ1ç¨®ï¼‰
        csr_matrix(np.array(oof_preds['fm_all'][start_idx:end_idx], dtype=np.float32).reshape(-1, 1)),
        csr_matrix(np.array(oof_preds['mlp_all'][start_idx:end_idx], dtype=np.float32).reshape(-1, 1)),
        # SVDç‰¹å¾´é‡�
        csr_matrix(X_train_name_svd[start_idx:end_idx]),
        csr_matrix(X_train_descp_svd[start_idx:end_idx])
    ], format='csr', dtype=np.float32)
    
    X_train_meta_chunks.append(chunk)
    
    print(f"  âœ… Chunk {i+1} shape: {chunk.shape}, Memory: {chunk.data.nbytes / 1024**3:.3f} GB")
    
    # ãƒ�ãƒ£ãƒ³ã‚¯ã�”ã�¨ã�«ã‚¬ãƒ™ãƒ¼ã‚¸ã‚³ãƒ¬ã‚¯ã‚·ãƒ§ãƒ³
    gc.collect()

# å…¨ãƒ�ãƒ£ãƒ³ã‚¯ã‚’ç¸¦æ–¹å�‘ã�«çµ�å�ˆ
print("\nğŸ”— Combining all chunks...")
X_train_meta = vstack(X_train_meta_chunks, format='csr')

print(f"âœ… X_train_meta: {X_train_meta.shape}, {X_train_meta.data.nbytes / 1024**3:.2f} GB")

# ãƒ¡ãƒ¢ãƒªè§£æ”¾
del X_train_meta_chunks, X_train_dense_features, X_train_name_svd, X_train_descp_svd, oof_preds
gc.collect()

print("âœ… Memory cleaned up")

# ============================================================
# 3. ãƒ¢ãƒ‡ãƒ«å­¦ç¿’
# ============================================================
print("\nğŸ�“ ãƒ¢ãƒ‡ãƒ«å­¦ç¿’ä¸­...")
X_meta_tr, X_meta_val, y_meta_tr, y_meta_val = train_test_split(
    X_train_meta, y_train_full, test_size=0.2, random_state=0
)

del X_train_meta
gc.collect()

lgbm_meta = lgb.LGBMRegressor(
    objective='regression',
    metric='rmse',
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=7,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    verbose=-1
)

lgbm_meta.fit(
    X_meta_tr, y_meta_tr,
    eval_set=[(X_meta_val, y_meta_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
)

# æ¤œè¨¼
pred_val = lgbm_meta.predict(X_meta_val)
rmse = np.sqrt(mean_squared_error(y_meta_val, pred_val))
print(f"\nğŸ�† Validation RMSE: {rmse:.5f}")

del X_meta_tr, X_meta_val, y_meta_tr, y_meta_val, pred_val
gc.collect()

# ============================================================
# 4. ãƒ�ãƒ£ãƒ³ã‚¯å‡¦ç�†ã�§äºˆæ¸¬
# ============================================================
print("\nğŸ”® ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿äºˆæ¸¬ä¸­ï¼ˆãƒ�ãƒ£ãƒ³ã‚¯å‡¦ç�†ï¼‰...")
chunk_size = 500000
n_chunks = (len(test_ids) + chunk_size - 1) // chunk_size
predictions = []

for i in range(n_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(test_ids))
    
    print(f"  Chunk {i+1}/{n_chunks}: {start_idx:,} - {end_idx:,}")
    
    X_test_chunk = hstack([
        X_test_dense_features[start_idx:end_idx],
        # Ridgeäºˆæ¸¬ï¼ˆ3ç¨®ï¼‰
        csr_matrix(np.array(test_preds['all'][start_idx:end_idx], dtype=np.float32).reshape(-1, 1)),
        csr_matrix(np.array(test_preds['name'][start_idx:end_idx], dtype=np.float32).reshape(-1, 1)),
        csr_matrix(np.array(test_preds['desc'][start_idx:end_idx], dtype=np.float32).reshape(-1, 1)),
        # SGDäºˆæ¸¬ï¼ˆ3ç¨®ï¼‰â†� è¿½åŠ 
        csr_matrix(np.array(test_preds['sgd_all'][start_idx:end_idx], dtype=np.float32).reshape(-1, 1)),
        csr_matrix(np.array(test_preds['sgd_name'][start_idx:end_idx], dtype=np.float32).reshape(-1, 1)),
        csr_matrix(np.array(test_preds['sgd_desc'][start_idx:end_idx], dtype=np.float32).reshape(-1, 1)),
        # FMäºˆæ¸¬ï¼ˆ1ç¨®ï¼‰
        csr_matrix(np.array(test_preds['fm_all'][start_idx:end_idx], dtype=np.float32).reshape(-1, 1)),
        csr_matrix(np.array(test_preds['mlp_all'][start_idx:end_idx], dtype=np.float32).reshape(-1, 1)),
        # SVDç‰¹å¾´é‡�
        csr_matrix(X_test_name_svd[start_idx:end_idx]),
        csr_matrix(X_test_descp_svd[start_idx:end_idx])
    ], format='csr', dtype=np.float32)
    
    chunk_preds = lgbm_meta.predict(X_test_chunk)
    predictions.append(chunk_preds)
    
    del X_test_chunk
    gc.collect()

# çµ�å�ˆã�¨å¾Œå‡¦ç�†
preds_log = np.concatenate(predictions)
del predictions, X_test_dense_features, X_test_name_svd, X_test_descp_svd, test_preds
gc.collect()

preds_original = np.expm1(preds_log)
preds_original = np.clip(preds_original, 0, None)

# ============================================================
# 5. æ��å‡ºãƒ•ã‚¡ã‚¤ãƒ«ä½œæˆ�
# ============================================================
submission = pd.DataFrame({
    'test_id': test_ids,
    'price': preds_original
})

submission.to_csv('submission.csv', index=False)
print(f"\nâœ… æ��å‡ºãƒ•ã‚¡ã‚¤ãƒ«ä½œæˆ�å®Œäº†: submission.csv")
print(f"ğŸ“Š æœ€çµ‚ãƒ¡ãƒ¢ãƒª: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")

