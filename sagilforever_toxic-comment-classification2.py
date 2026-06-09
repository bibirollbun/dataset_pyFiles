import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
!pip install -U scikit-learn  # å�‡çº§åˆ°æœ€æ–°ç‰ˆæœ¬
import sklearn



import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression,LinearRegression,SGDRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score,log_loss,confusion_matrix,classification_report,roc_curve,auc
from sklearn import svm
from scipy import sparse
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from collections import defaultdict
import plotly.graph_objects as gobs
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split



from sklearn.model_selection import cross_val_score


import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer() 
nltk.download('wordnet')
%matplotlib inline
seed = 42
import os
from sklearn.utils import resample  # ç”¨äº�æ•°æ�®é™�é‡‡æ ·
import time  # ç”¨äº�è®¡æ—¶
from sklearn.calibration import CalibratedClassifierCV  # ç”¨äº�æ¦‚ç�‡æ ¡å‡†
os.environ['OMP_NUM_THREADS'] = '4'



# è®¾ç½®ä¸­æ–‡å­—ä½“
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # è§£å†³è´Ÿå�·æ˜¾ç¤ºé—®é¢˜

# å�¯è§†åŒ–å‡½æ•°
def plot_confusion_matrix(y_true, y_pred, labels, title="æ··æ·†çŸ©é˜µ"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("é¢„æµ‹æ ‡ç­¾")
    plt.ylabel("çœŸå®�æ ‡ç­¾")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
def plot_classification_report(y_true, y_pred, labels, title="åˆ†ç±»æŠ¥å‘Š"):
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # ç»˜åˆ¶çƒ­åŠ›å›¾
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    return report_df

def plot_roc_curve(models, X_val, y_val, target_names):
    plt.figure(figsize=(10, 8))
    
    for target in target_names:
        model = models[target]
        
        # è�·å�–é¢„æµ‹æ¦‚ç�‡
        y_score = model.predict_proba(X_val)[:, 1]
        
        # è®¡ç®—ROCæ›²çº¿
        fpr, tpr, _ = roc_curve(y_val[target], y_score)
        roc_auc = auc(fpr, tpr)
        
        # ç»˜åˆ¶ROCæ›²çº¿
        plt.plot(fpr, tpr, lw=2, 
                 label=f'{target} (AUC = {roc_auc:.3f})')
    
    # ç»˜åˆ¶å¯¹è§’çº¿
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('å�‡é˜³æ€§ç�‡ (FPR)')
    plt.ylabel('çœŸé˜³æ€§ç�‡ (TPR)')
    plt.title('é€»è¾‘å›�å½’æ¨¡å�‹çš„å¤šç±»åˆ«ROCæ›²çº¿')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# åŠ è½½è®­ç»ƒé›†ä¸�æµ‹è¯•é›†
train = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
test = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
print('Number of rows and columns in the train data set:',train.shape)
print('Number of rows and columns in the test data set:',test.shape)
train.head()
test.head()
# å¡«è¡¥ç¼ºå¤±å€¼
k = pd.DataFrame()
k['train'] = train.isnull().sum()
k['test'] = test.isnull().sum()
k
test[test['comment_text'].isnull()]
test.fillna(' ',inplace=True)
gc.collect()


# æ•°æ�®æ¸…æ´—
def clean_text(text):
    # ç”¨æ­£åˆ™è¡¨è¾¾å¼�æ›¿æ�¢HTMLæ ‡ç­¾ä¸ºç©ºæ ¼ï¼ˆä¾‹å¦‚"<p>Hello!</p>" â†’ "  Hello!  "ï¼‰
    text = re.sub('<.*?>', ' ', text)
    # åˆ é™¤æ‰€æœ‰æ ‡ç‚¹ç¬¦å�·
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))
    # å�ªä¿�ç•™å­—æ¯�å’Œæ•°å­—ï¼Œç¤ºä¾‹ï¼šè¾“å…¥ "Hello123!" â†’ è¾“å‡º "Hello123"
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    # å°†æ�¢è¡Œç¬¦æ›¿æ�¢ä¸ºç©ºæ ¼
    text = re.sub("\n", " ", text)
    # è½¬æ�¢ä¸ºå…¨å°�å†™
    text = text.lower()
    # å�»é™¤å¤šä½™ç©ºæ ¼ï¼šåˆ†å‰²å�•è¯�å��é‡�æ–°ç”¨å�•ç©ºæ ¼è¿�æ�¥æˆ�å­—ç¬¦ä¸²è¿”å›�
    text = ' '.join(text.split())
    return text

# ä»�æ–‡æœ¬ä¸­ç§»é™¤å�œç”¨è¯�ï¼ˆå¦‚ "the", "is" ç­‰ï¼‰
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def remove_stopwords(input_text, stop_words):
    # å°†å�œç”¨è¯�åˆ—è¡¨è½¬ä¸ºé›†å�ˆæ��é«˜æ•ˆç�‡
    stop_words = set(stop_words)
    # ä½¿ç”¨NLTKçš„åˆ†è¯�å·¥å…·å°†æ–‡æœ¬æ‹†åˆ†ä¸ºå�•è¯�åˆ—è¡¨
    word_tokens = word_tokenize(input_text)
    # è¿‡æ»¤å�œç”¨è¯�ï¼ˆå�•æ¬¡å¾ªç�¯ï¼‰
    filtered_words = [word for word in word_tokens if word not in stop_words]
    # åˆ†å‰²å�•è¯�å��é‡�æ–°ç”¨å�•ç©ºæ ¼è¿�æ�¥æˆ�å­—ç¬¦ä¸²è¿”å›�
    text = ' '.join(filtered_words)
    return text

unrelevant_words = ['wiki', 'wikipedia', 'page']

def clean(data, word):
    # æ•°æ�®æ¸…æ´—
    data[word] = data[word].apply(clean_text)
    # å�»é™¤æ— å…³è¯�
    data[word] = data[word].apply(lambda x: ' '.join([w for w in x.split() if w not in unrelevant_words]))
    # ç§»é™¤å�œç”¨è¯�
    data[word] = data[word].apply(lambda x: remove_stopwords(x, stop_words))
    # è¯�å½¢è¿˜å�Ÿ
    data[word] = data[word].apply(lambda x: ' '.join([lemmatizer.lemmatize(w) for w in x.split()]))


# è®­ç»ƒé›†é¢„å¤„ç�†
clean(train,"comment_text")
train.head()

# æµ‹è¯•é›†é¢„å¤„ç�†
clean(test,"comment_text")
test.head()


# åˆ†åˆ«å¯¹å�•è¯�å’Œå­—ç¬¦è¿›è¡ŒTF-IDFå�‘é‡�åŒ–ï¼Œæ�•æ�‰ä¸�å�Œç²’åº¦çš„æ–‡æœ¬æ¨¡å¼�ã€‚
# åˆ�å§‹åŒ–TF-IDFå�‘é‡�å™¨
vect_word = TfidfVectorizer(max_features=5000, lowercase=True, analyzer='word',
                        stop_words= 'english',ngram_range=(1,3),dtype=np.float32)
vect_char = TfidfVectorizer(max_features=10000, lowercase=True, analyzer='char',
                        ngram_range=(3,6),dtype=np.float32)
# å�•è¯�çº§n-gram
tr_vect = vect_word.fit_transform(train['comment_text'])
ts_vect = vect_word.transform(test['comment_text'])

# å­—ç¬¦çº§ n-gram
tr_vect_char = vect_char.fit_transform(train['comment_text'])
ts_vect_char = vect_char.transform(test['comment_text'])
gc.collect()  # å‡�å°‘å†…å­˜å� ç”¨

tr_vect = csr_matrix(tr_vect)  # å¼ºåˆ¶è½¬æ�¢ä¸º CSRï¼ˆå�³ä½¿å·²æ˜¯ CSRï¼Œä¹Ÿå®‰å…¨ï¼‰
tr_vect_char = csr_matrix(tr_vect_char)  # å�Œä¸Š
ts_vect = csr_matrix(ts_vect)  # ç¡®ä¿�æµ‹è¯•é›†ä¹Ÿæ˜¯CSRæ ¼å¼�
ts_vect_char = csr_matrix(ts_vect_char)  # å�Œä¸Š

# ä½¿ç”¨sparse.hstackå�ˆå¹¶ç‰¹å¾�ä»¥å‡�å°‘å†…å­˜å� ç”¨ï¼ˆæ–‡æœ¬ç‰¹å¾�é€šå¸¸æ˜¯é«˜ç»´ç¨€ç–�çš„ï¼‰
X = sparse.hstack([tr_vect, tr_vect_char])
x_test = sparse.hstack([ts_vect, ts_vect_char])

target_col = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
y = train[target_col]
del tr_vect, ts_vect, tr_vect_char, ts_vect_char
gc.collect()



# æ•°æ�®æ ‡å‡†åŒ–
scaler = MaxAbsScaler()
X_csr = X.tocsr()
X_scaled = scaler.fit_transform(X_csr)
x_test_scaled = scaler.transform(x_test.tocsr())

# åˆ’åˆ†è®­ç»ƒé›†å’ŒéªŒè¯�é›†
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ä¼˜åŒ–çš„é€»è¾‘å›�å½’æ¨¡å�‹ï¼ˆåŠ é€Ÿè®­ç»ƒï¼‰
logreg_model = LogisticRegression(
    C=2,
    random_state=42,
    class_weight='balanced',
    solver='saga',
    penalty='l2',
    max_iter=1000,           # è¿­ä»£æ¬¡æ•°
    tol=5e-3,               # æ”¶æ•›å®¹å·®
    verbose=1,
    n_jobs=-1
)

# ç»“æ�œå­˜å‚¨
results = {target: {} for target in target_col}
training_times = {target: {} for target in target_col}
fitted_models = {}

# é’ˆå¯¹æ¯�ä¸ªç›®æ ‡åˆ—è®­ç»ƒé€»è¾‘å›�å½’æ¨¡å�‹
for col in target_col:
    print(f"\n=== è®­ç»ƒ {col} åˆ—çš„é€»è¾‘å›�å½’æ¨¡å�‹ ===")
    y_col = y_train[col]
    
    # å¤�åˆ¶æ¨¡å�‹é…�ç½®
    model_instance = logreg_model.__class__(**logreg_model.get_params())
    
    # è®°å½•è®­ç»ƒå¼€å§‹æ—¶é—´
    start_time = time.time()
    
    # è®­ç»ƒæ¨¡å�‹
    model_instance.fit(X_train, y_col)
    fitted_models[col] = model_instance
    
    # è®°å½•è®­ç»ƒç»“æ�Ÿæ—¶é—´å¹¶è®¡ç®—è€—æ—¶
    end_time = time.time()
    elapsed_time = end_time - start_time
    training_times[col] = elapsed_time
    print(f"{col} æ¨¡å�‹è®­ç»ƒè€—æ—¶: {elapsed_time:.2f} ç§’")
    
    # è�·å�–é¢„æµ‹æ¦‚ç�‡
    pred_pro = model_instance.predict_proba(X_val)[:, 1]
    
    # è®¾å®šé˜ˆå€¼å¹¶ç”Ÿæˆ�é¢„æµ‹
    threshold = 0.7
    pred = (pred_pro >= threshold).astype(int)
    
    # è®¡ç®—è¯„ä¼°æŒ‡æ ‡
    cm = confusion_matrix(y_val[col], pred)
    report = classification_report(y_val[col], pred)
    fpr, tpr, _ = roc_curve(y_val[col], pred_pro)
    auc_val = auc(fpr, tpr)
    
    # å­˜å‚¨ç»“æ�œ
    results[col] = {
        'confusion_matrix': cm,
        'report': report,
        'fpr': fpr,
        'tpr': tpr,
        'auc': auc_val,
        'pred_pro': pred_pro,
        'pred': pred
    }
    
    # æ‰“å�°è¿­ä»£ä¿¡æ�¯
    iter_count = model_instance.n_iter_
    print(f"{col} æ¨¡å�‹ - AUC: {auc_val:.4f}, è¿­ä»£æ¬¡æ•°: {iter_count}")




# å�¯è§†åŒ–è¯„ä¼°ç»“æ�œ
print("\n=== æ¨¡å�‹è¯„ä¼°å�¯è§†åŒ– ===")

# 1. ç»˜åˆ¶ROCæ›²çº¿ï¼ˆæ‰€æœ‰ç±»åˆ«ï¼‰
plot_roc_curve(fitted_models, X_val, y_val, target_col)

# 2. ä¸ºæ¯�ä¸ªç±»åˆ«ç»˜åˆ¶æ··æ·†çŸ©é˜µå’Œåˆ†ç±»æŠ¥å‘Š
for col in target_col:
    print(f"\n=== {col} åˆ—çš„æ¨¡å�‹è¯„ä¼° ===")
    
    # ç»˜åˆ¶æ··æ·†çŸ©é˜µ
    plot_confusion_matrix(
        y_val[col], 
        results[col]['pred'], 
        labels=['é��' + col, col],
        title=f"{col} æ··æ·†çŸ©é˜µ"
    )
    
    # ç»˜åˆ¶åˆ†ç±»æŠ¥å‘Š
    report_df = plot_classification_report(
        y_val[col], 
        results[col]['pred'], 
        labels=['é��' + col, col],
        title=f"{col} åˆ†ç±»æŠ¥å‘Š"
    )
    
    # æ‰“å�°AUC
    print(f"{col} AUC: {results[col]['auc']:.4f}")

# 3. æ˜¾ç¤ºå�„æ¨¡å�‹æ€»è®­ç»ƒæ—¶é—´
print("\n=== æ¨¡å�‹è®­ç»ƒæ—¶é—´ ===")
total_time = sum(training_times.values())
print(f"é€»è¾‘å›�å½’æ€»è®­ç»ƒæ—¶é—´: {total_time:.2f} ç§’")

# 4. ç”Ÿæˆ�Kaggleæ��äº¤ç»“æ�œ
print("\n=== ç”Ÿæˆ�Kaggleæ��äº¤æ–‡ä»¶ ===")
submission = pd.DataFrame({'id': test['id']})

for col in target_col:
    # ä½¿ç”¨è®­ç»ƒå¥½çš„é€»è¾‘å›�å½’æ¨¡å�‹è¿›è¡Œé¢„æµ‹
    model = fitted_models[col]
    submission[col] = model.predict_proba(x_test_scaled)[:, 1]

# ä¿�å­˜æ��äº¤æ–‡ä»¶
submission.to_csv('submission.csv', index=False)
print("Kaggleæ��äº¤æ–‡ä»¶å·²ç”Ÿæˆ�: submission.csv")




