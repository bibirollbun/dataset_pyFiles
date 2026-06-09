!unzip -q /kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip -d /kaggle/working/
!unzip -q /kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip -d /kaggle/working/
!unzip -q /kaggle/input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip -d /kaggle/working/



import pandas as pd

# âœ… è¯»å�–æ•°æ�®
df_train_1 = pd.read_csv("/kaggle/input/moviereview/labeledTrainData.tsv", sep='\t')
df_train_2 = pd.read_csv("/kaggle/input/moviereview/unlabeledTrainData.tsv",sep='\t',quoting=3,engine='python')
df_train_3 = pd.read_csv("/kaggle/input/moviereview/testData.tsv", sep='\t')

df_train_full1 = df_train_1
df_train_full2= pd.concat([df_train_1, df_train_2])
df_train_full3 = pd.concat([df_train_full2, df_train_3])

df_test = df_train_3


# âœ… åŸºç¡€å¯¼å…¥
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from tqdm.notebook import tqdm
import os

# âœ… å�¯ç”¨ tqdm æ”¯æŒ� pandas
tqdm.pandas()

# âœ… åˆ†è¯�å‡½æ•°
def tokenize(text):
#    return str(text).lower().split()

# âœ… è®­ç»ƒ Word2Vec æ¨¡å�‹ï¼ˆç”¨æ›´å¤§æ•°æ�®é›†ï¼‰
#print("ğŸš€ æ­£åœ¨ç”¨å¸¦æ— æ ‡ç­¾æ•°æ�®è®­ç»ƒ Word2Vec æ¨¡å�‹...")
sentences = df_train_full2['review'].progress_apply(tokenize).tolist()
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4, sg=0)
#print("âœ… Word2Vec è®­ç»ƒå®Œæˆ�")

# âœ… å�¥å­�å�‘é‡�ç”Ÿæˆ�å‡½æ•°ï¼ˆå�¥å­�å�‘é‡� = è¯�å�‘é‡�å�‡å€¼ï¼‰
def sentence_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# âœ… æ�„å»ºè®­ç»ƒé›†å�¥å­�å�‘é‡�ï¼ˆä»…ç”¨æœ‰æ ‡ç­¾æ•°æ�®ï¼‰
#print("ğŸ§± æ­£åœ¨æ�„å»ºè®­ç»ƒé›†å�¥å­�å�‘é‡�...")
X_train = np.vstack(df_train_full1['review'].progress_apply(lambda x: sentence_vector(tokenize(x), w2v_model)))
y_train = df_train_full1['sentiment'].values
#print("âœ… è®­ç»ƒæ•°æ�®æ�„å»ºå®Œæˆ�")

# âœ… æ�„å»ºæµ‹è¯•é›†å�¥å­�å�‘é‡�
#print("ğŸ§± æ­£åœ¨æ�„å»ºæµ‹è¯•é›†å�¥å­�å�‘é‡�...")
X_test = np.vstack(df_test['review'].progress_apply(lambda x: sentence_vector(tokenize(x), w2v_model)))
#print("âœ… æµ‹è¯•æ•°æ�®æ�„å»ºå®Œæˆ�")

# âœ… è®­ç»ƒé€»è¾‘å›�å½’æ¨¡å�‹
#print("ğŸ¤– æ­£åœ¨è®­ç»ƒé€»è¾‘å›�å½’æ¨¡å�‹...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
#print("âœ… æ¨¡å�‹è®­ç»ƒå®Œæˆ�")

# âœ… é¢„æµ‹æµ‹è¯•é›†
y_pred = clf.predict(X_test)
#print("âœ… é¢„æµ‹å®Œæˆ�")

# âœ… ç”Ÿæˆ�æ��äº¤æ–‡ä»¶
submission = pd.DataFrame({'id': df_test['id'], 'sentiment': y_pred})
submission = submission.sort_values(by='id')
submission.to_csv('submission.csv', index=False, quoting=1)

#print("ğŸ“� æ��äº¤æ–‡ä»¶å·²ç”Ÿæˆ�: submission.csv")
#print("ğŸ“‚ å½“å‰�ç›®å½•æ–‡ä»¶åˆ—è¡¨:", os.listdir("/kaggle/working"))



#import pandas as pd
#import gensim
#from gensim.models import Word2Vec
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#import numpy as np
#!unzip -q /kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip -d /kaggle/working/
#!unzip -q /kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip -d /kaggle/working/
#!unzip -q /kaggle/input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip -d /kaggle/working/

#df_labeled_train = pd.read_csv("/kaggle/working/labeledTrainData.tsv",
#                               sep='\t',
#                               engine='python',          # ä½¿ç”¨æ›´å®½å®¹çš„è§£æ��å™¨
#    on_bad_lines='skip'   # æ–°ç‰ˆæœ¬æ›¿ä»£ error_bad_lines
#)
#df_test = pd.read_csv("/kaggle/working/testData.tsv", 
#                      sep='\t',
#                      engine='python',          # ä½¿ç”¨æ›´å®½å®¹çš„è§£æ��å™¨
#    on_bad_lines='skip'   # æ–°ç‰ˆæœ¬æ›¿ä»£ error_bad_lines
#)
#df_unlabeled_train = pd.read_csv("/kaggle/working/unlabeledTrainData.tsv",
#                                 sep='\t',
#                                 engine='python',          # ä½¿ç”¨æ›´å®½å®¹çš„è§£æ��å™¨
#    on_bad_lines='skip'   # æ–°ç‰ˆæœ¬æ›¿ä»£ error_bad_lines
#)




# å�‡è®¾å·²ç»�è¯»å�–äº†ä¸¤ä¸ªDataFrameï¼šdf_labeled_train, df_unlabeled_train

# å…ˆç»™æ— æ ‡ç­¾æ•°æ�®æ·»åŠ ä¸€ä¸ªç¼ºå¤±æ ‡ç­¾åˆ—ï¼Œæ¯”å¦‚-1
#df_unlabeled_train['sentiment'] = -1

# ä¿�ç•™åˆ—é¡ºåº�ä¸€è‡´
##df_labeled_subset = df_labeled_train[['id', 'review', 'sentiment']]
#df_unlabeled_subset = df_unlabeled_train[['id', 'review', 'sentiment']]

# çºµå�‘å�ˆå¹¶
#df_train_full1 = df_labeled_subset
#df_train_full2 = pd.concat([df_labeled_subset, df_unlabeled_subset], ignore_index=True)

#print(df_train_full.shape)
#print(df_train_full.head())

# 1. è¯»å�–æ•°æ�®ï¼ˆå�‡è®¾å·²ç»�åŠ è½½åˆ°df_train_full1å’Œdf_testï¼‰
# è¿™é‡Œç¤ºèŒƒä»�æ–‡ä»¶è¯»å�–ï¼ŒæŒ‰ä½ å®�é™…æƒ…å†µæ›¿æ�¢
#df_train_full1 = pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv', sep='\t')
#df_test = pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/testData.tsv', sep='\t')

# 2. ç®€å�•åˆ†è¯�å‡½æ•°ï¼ˆç©ºæ ¼åˆ†è¯�ç¤ºä¾‹ï¼Œå®�é™…ä½ å�¯ä»¥ç”¨æ›´å¤�æ�‚åˆ†è¯�å™¨ï¼‰
#def tokenize(text):
#    return str(text).lower().split()

# 3. è®­ç»ƒWord2Vecè¯�å�‘é‡�æ¨¡å�‹ï¼Œä½¿ç”¨è®­ç»ƒé›†æ–‡æœ¬
#sentences = df_train_full1['review'].apply(tokenize).tolist()
#w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4, sg=0)  # é»˜è®¤å�‚æ•°

# 4. ç”Ÿæˆ�å�¥å­�å�‘é‡�ï¼šå¯¹æ¯�æ�¡è¯„è®ºçš„è¯�å�‘é‡�æ±‚å�‡å€¼ï¼ˆå¿½ç•¥æ²¡åœ¨è¯�å�‘é‡�ä¸­çš„è¯�ï¼‰
#def sentence_vector(tokens, model):
#    vectors = [model.wv[word] for word in tokens if word in model.wv]
#    if len(vectors) == 0:
#        return np.zeros(model.vector_size)
#    return np.mean(vectors, axis=0)

#X_train = np.vstack(df_train_full1['review'].apply(lambda x: sentence_vector(tokenize(x), w2v_model)))
#y_train = df_train_full1['sentiment'].values

# 5. è®­ç»ƒé€»è¾‘å›�å½’åˆ†ç±»å™¨
#clf = LogisticRegression(max_iter=1000)
#clf.fit(X_train, y_train)

# 6. ç”Ÿæˆ�æµ‹è¯•é›†å�¥å­�å�‘é‡�
#X_test = np.vstack(df_test['review'].apply(lambda x: sentence_vector(tokenize(x), w2v_model)))

# 7. é¢„æµ‹æµ‹è¯•é›†æƒ…æ„Ÿ
#y_pred = clf.predict(X_test)

# 8. ç”Ÿæˆ�æ��äº¤æ–‡ä»¶ï¼Œæ ¼å¼�ä¸º id å’Œ sentiment ä¸¤åˆ—
#submission = pd.DataFrame({'id': df_test['id'], 'sentiment': y_pred})
#submission.to_csv('submission.csv', index=False)

#print("Submission file generated: submission.csv")



# âœ… åŸºç¡€å¯¼å…¥
#import pandas as pd
#import numpy as np
#from gensim.models import Word2Vec
#from sklearn.linear_model import LogisticRegression
#from tqdm.notebook import tqdm
#import os

# âœ… å�¯ç”¨ tqdm æ”¯æŒ� pandas
#tqdm.pandas()

# âœ… è§£å�‹æ•°æ�®

# âœ… è¯»å�–æ•°æ�®
#df_train_full1 = pd.read_csv("/kaggle/input/moviereview/labeledTrainData.tsv",
#                             sep='\t')
#df_test = pd.read_csv("/kaggle/input/moviereview/testData.tsv",
#                      sep='\t'
#                      )

#print(f"âœ… è®­ç»ƒé›†å¤§å°�: {df_train_full1.shape}")
#print(f"âœ… æµ‹è¯•é›†å¤§å°�: {df_test.shape}")

# âœ… åˆ†è¯�å‡½æ•°
#def tokenize(text):
#    return str(text).lower().split()

# âœ… è®­ç»ƒ Word2Vec æ¨¡å�‹
#print("ğŸš€ æ­£åœ¨è®­ç»ƒ Word2Vec æ¨¡å�‹...")
#sentences = df_train_full1['review'].progress_apply(tokenize).tolist()
#w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4, sg=0)
#print("âœ… Word2Vec è®­ç»ƒå®Œæˆ�")

# âœ… å�¥å­�å�‘é‡�ç”Ÿæˆ�å‡½æ•°ï¼ˆå�¥å­� = å�«è¯�çš„å�‡å€¼ï¼‰
#def sentence_vector(tokens, model):
#    vectors = [model.wv[word] for word in tokens if word in model.wv]
#    if len(vectors) == 0:
#        return np.zeros(model.vector_size)
#    return np.mean(vectors, axis=0)

# âœ… æ�„å»ºè®­ç»ƒé›†å�¥å­�å�‘é‡�
#print("ğŸ§± æ­£åœ¨æ�„å»ºè®­ç»ƒé›†å�¥å­�å�‘é‡�...")
#X_train = np.vstack(df_train_full1['review'].progress_apply(lambda x: sentence_vector(tokenize(x), w2v_model)))
#y_train = df_train_full1['sentiment'].values
#print("âœ… è®­ç»ƒæ•°æ�®æ�„å»ºå®Œæˆ�")

# âœ… æ�„å»ºæµ‹è¯•é›†å�¥å­�å�‘é‡�
#print("ğŸ§± æ­£åœ¨æ�„å»ºæµ‹è¯•é›†å�¥å­�å�‘é‡�...")
#X_test = np.vstack(df_test['review'].progress_apply(lambda x: sentence_vector(tokenize(x), w2v_model)))
#print("âœ… æµ‹è¯•æ•°æ�®æ�„å»ºå®Œæˆ�")

# âœ… è®­ç»ƒé€»è¾‘å›�å½’æ¨¡å�‹
#print("ğŸ¤– æ­£åœ¨è®­ç»ƒé€»è¾‘å›�å½’æ¨¡å�‹...")
#clf = LogisticRegression(max_iter=1000)
#clf.fit(X_train, y_train)
#print("âœ… æ¨¡å�‹è®­ç»ƒå®Œæˆ�")

# âœ… é¢„æµ‹æµ‹è¯•é›†
#y_pred = clf.predict(X_test)
#print("âœ… é¢„æµ‹å®Œæˆ�")

# âœ… ç”Ÿæˆ�æ��äº¤æ–‡ä»¶
#submission = pd.DataFrame({'id': df_test['id'], 'sentiment': y_pred})
#submission = submission.sort_values(by='id')
#submission.to_csv('submission.csv', index=False, quoting=1)

#print("ğŸ“� æ��äº¤æ–‡ä»¶å·²ç”Ÿæˆ�: submission.csv")
#print("ğŸ“‚ å½“å‰�ç›®å½•æ–‡ä»¶åˆ—è¡¨:", os.listdir("/kaggle/working"))



# âœ… åŸºç¡€å¯¼å…¥
#import pandas as pd
#import numpy as np
#from gensim.models import Word2Vec
#from tqdm.notebook import tqdm
#import os
#import lightgbm as lgb  # âœ… æ›¿æ�¢ä¸º LightGBM

# âœ… å�¯ç”¨ tqdm æ”¯æŒ� pandas
#tqdm.pandas()

# âœ… è¯»å�–æ•°æ�®
#df_train_full1 = pd.read_csv("/kaggle/input/moviereview/labeledTrainData.tsv",
#                             sep='\t'
#                             )
#df_test = pd.read_csv("/kaggle/input/moviereview/testData.tsv",
#                      sep='\t')

#print(f"âœ… è®­ç»ƒé›†å¤§å°�: {df_train_full1.shape}")
#print(f"âœ… æµ‹è¯•é›†å¤§å°�: {df_test.shape}")

# âœ… åˆ†è¯�å‡½æ•°
#def tokenize(text):
#    return str(text).lower().split()

# âœ… è®­ç»ƒ Word2Vec æ¨¡å�‹
#print("ğŸš€ æ­£åœ¨è®­ç»ƒ Word2Vec æ¨¡å�‹...")
#sentences = df_train_full1['review'].progress_apply(tokenize).tolist()
#w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4, sg=0)
#print("âœ… Word2Vec è®­ç»ƒå®Œæˆ�")

# âœ… å�¥å­�å�‘é‡�ç”Ÿæˆ�å‡½æ•°
#def sentence_vector(tokens, model):
#    vectors = [model.wv[word] for word in tokens if word in model.wv]
#    if len(vectors) == 0:
#        return np.zeros(model.vector_size)
#    return np.mean(vectors, axis=0)

# âœ… æ�„å»ºè®­ç»ƒé›†å�¥å­�å�‘é‡�
#print("ğŸ§± æ­£åœ¨æ�„å»ºè®­ç»ƒé›†å�¥å­�å�‘é‡�...")
#X_train = np.vstack(df_train_full1['review'].progress_apply(lambda x: sentence_vector(tokenize(x), w2v_model)))
#y_train = df_train_full1['sentiment'].values
#print("âœ… è®­ç»ƒæ•°æ�®æ�„å»ºå®Œæˆ�")

# âœ… æ�„å»ºæµ‹è¯•é›†å�¥å­�å�‘é‡�
#print("ğŸ§± æ­£åœ¨æ�„å»ºæµ‹è¯•é›†å�¥å­�å�‘é‡�...")
#X_test = np.vstack(df_test['review'].progress_apply(lambda x: sentence_vector(tokenize(x), w2v_model)))
#print("âœ… æµ‹è¯•æ•°æ�®æ�„å»ºå®Œæˆ�")


# âœ… ä½¿ç”¨ LightGBM æ¨¡å�‹è®­ç»ƒ
#print("ğŸŒ² æ­£åœ¨è®­ç»ƒ LightGBM æ¨¡å�‹...")
#clf = lgb.LGBMClassifier(
#    n_estimators=1000,
#    learning_rate=0.05,
#    num_leaves=64,
#    subsample=0.8,
#    colsample_bytree=0.8,
#    random_state=42,
#    verbosity=-1  # âœ… å…³é—­æ‰€æœ‰è¾“å‡º
#)
#clf.fit(X_train, y_train)
######################################################################
#print("âœ… æ¨¡å�‹è®­ç»ƒå®Œæˆ�")
#import xgboost as xgb

#clf = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', verbosity=0)
#clf.fit(X_train, y_train)
########################################################################
#from catboost import CatBoostClassifier

#clf = CatBoostClassifier(verbose=0, random_state=42)
#clf.fit(X_train, y_train)

#print("âœ… æ¨¡å�‹è®­ç»ƒå®Œæˆ�")

# âœ… é¢„æµ‹æµ‹è¯•é›†
#y_pred = clf.predict(X_test)
#print("âœ… é¢„æµ‹å®Œæˆ�")

# âœ… ç”Ÿæˆ�æ��äº¤æ–‡ä»¶
#submission = pd.DataFrame({'id': df_test['id'], 'sentiment': y_pred})
#submission = submission.sort_values(by='id')
#submission.to_csv('submission.csv', index=False, quoting=1)

#print("ğŸ“� æ��äº¤æ–‡ä»¶å·²ç”Ÿæˆ�: submission.csv")
#print("ğŸ“‚ å½“å‰�ç›®å½•æ–‡ä»¶åˆ—è¡¨:", os.listdir("/kaggle/working"))



# âœ… åŸºç¡€å¯¼å…¥
#import pandas as pd
#import numpy as np
#from gensim.models import Word2Vec
#from sklearn.linear_model import LogisticRegression
#from tqdm.notebook import tqdm
#import os
#import lightgbm as lgb  # âœ… æ›¿æ�¢ä¸º LightGBM

# âœ… å�¯ç”¨ tqdm æ”¯æŒ� pandas
#tqdm.pandas()

# âœ… è¯»å�–æ•°æ�®
#df_train_full1 = pd.read_csv("/kaggle/input/moviereview/labeledTrainData.tsv", sep='\t')
#df_train_full2_1 = pd.read_csv("/kaggle/input/moviereview/labeledTrainData.tsv", sep='\t')
#df_train_full2_2 = pd.read_csv(
#    "/kaggle/input/moviereview/unlabeledTrainData.tsv",
#    sep='\t',
#3    quoting=3,              # ä¸�å¤„ç�†å¼•å�·ï¼ŒæŒ‰çº¯æ–‡æœ¬å¤„ç�†
#    engine='python',
#)
#df_train_full2_3 = pd.concat([df_train_full2_1, df_train_full2_2])

#df_test = pd.read_csv("/kaggle/input/moviereview/testData.tsv", sep='\t')
#df_train_full2 = pd.concat([df_train_full2_3, df_test])

#print(f"è®­ç»ƒé›†æ ‡ç­¾æ•°æ�®å¤§å°�: {df_train_full1.shape}")
#print(f"è®­ç»ƒé›†å¸¦æ— æ ‡ç­¾æ•°æ�®å¤§å°�: {df_train_full2.shape}")
#print(f"æµ‹è¯•é›†å¤§å°�: {df_test.shape}")

# âœ… åˆ†è¯�å‡½æ•°
#def tokenize(text):
#    return str(text).lower().split()

# âœ… è®­ç»ƒ Word2Vec æ¨¡å�‹ï¼ˆç”¨æ›´å¤§æ•°æ�®é›†ï¼‰
#print("ğŸš€ æ­£åœ¨ç”¨å¸¦æ— æ ‡ç­¾æ•°æ�®è®­ç»ƒ Word2Vec æ¨¡å�‹...")
#sentences = df_train_full2['review'].progress_apply(tokenize).tolist()
#w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4, sg=0)
#print("âœ… Word2Vec è®­ç»ƒå®Œæˆ�")

# âœ… å�¥å­�å�‘é‡�ç”Ÿæˆ�å‡½æ•°ï¼ˆå�¥å­�å�‘é‡� = è¯�å�‘é‡�å�‡å€¼ï¼‰
#def sentence_vector(tokens, model):
#    vectors = [model.wv[word] for word in tokens if word in model.wv]
#    if len(vectors) == 0:
#        return np.zeros(model.vector_size)
#    return np.mean(vectors, axis=0)

# âœ… æ�„å»ºè®­ç»ƒé›†å�¥å­�å�‘é‡�ï¼ˆä»…ç”¨æœ‰æ ‡ç­¾æ•°æ�®ï¼‰
#print("ğŸ§± æ­£åœ¨æ�„å»ºè®­ç»ƒé›†å�¥å­�å�‘é‡�...")
#X_train = np.vstack(df_train_full1['review'].progress_apply(lambda x: sentence_vector(tokenize(x), w2v_model)))
#y_train = df_train_full1['sentiment'].values
#print("âœ… è®­ç»ƒæ•°æ�®æ�„å»ºå®Œæˆ�")

# âœ… æ�„å»ºæµ‹è¯•é›†å�¥å­�å�‘é‡�
#print("ğŸ§± æ­£åœ¨æ�„å»ºæµ‹è¯•é›†å�¥å­�å�‘é‡�...")
#X_test = np.vstack(df_test['review'].progress_apply(lambda x: sentence_vector(tokenize(x), w2v_model)))
#print("âœ… æµ‹è¯•æ•°æ�®æ�„å»ºå®Œæˆ�")

# âœ… è®­ç»ƒé€»è¾‘å›�å½’æ¨¡å�‹
#print("ğŸ¤– æ­£åœ¨è®­ç»ƒé€»è¾‘å›�å½’æ¨¡å�‹...")
#clf = LogisticRegression(max_iter=1000)
#clf.fit(X_train, y_train)
#print("âœ… æ¨¡å�‹è®­ç»ƒå®Œæˆ�")
# âœ… ä½¿ç”¨ LightGBM æ¨¡å�‹è®­ç»ƒ
#print("ğŸŒ² æ­£åœ¨è®­ç»ƒ LightGBM æ¨¡å�‹...")
#clf = lgb.LGBMClassifier(
#    n_estimators=1000,
#    learning_rate=0.05,
#    num_leaves=64,
#    subsample=0.8,
#    colsample_bytree=0.8,
#    random_state=42,
#    verbosity=-1  # âœ… å…³é—­æ‰€æœ‰è¾“å‡º
#)
#clf.fit(X_train, y_train)
######################################################################
#print("âœ… æ¨¡å�‹è®­ç»ƒå®Œæˆ�")
#import xgboost as xgb

#clf = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', verbosity=0)
#clf.fit(X_train, y_train)
########################################################################
#from catboost import CatBoostClassifier

#clf = CatBoostClassifier(verbose=0, random_state=42)
#clf.fit(X_train, y_train)
# âœ… é¢„æµ‹æµ‹è¯•é›†
#y_pred = clf.predict(X_test)
#print("âœ… é¢„æµ‹å®Œæˆ�")

# âœ… ç”Ÿæˆ�æ��äº¤æ–‡ä»¶
#submission = pd.DataFrame({'id': df_test['id'], 'sentiment': y_pred})
#submission = submission.sort_values(by='id')
#submission.to_csv('submission.csv', index=False, quoting=1)

#print("ğŸ“� æ��äº¤æ–‡ä»¶å·²ç”Ÿæˆ�: submission.csv")
#print("ğŸ“‚ å½“å‰�ç›®å½•æ–‡ä»¶åˆ—è¡¨:", os.listdir("/kaggle/working"))



# âœ… å®‰è£…ä¾�èµ–ï¼ˆå�ªéœ€è¿�è¡Œä¸€æ¬¡ï¼‰
#!pip install -q transformers datasets accelerate

# âœ… å¯¼å…¥åº“
#import pandas as pd
#import numpy as np
#import torch
#from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
#from datasets import Dataset
#from sklearn.model_selection import train_test_split
#import os

# âœ… è®¾ç½® GPU
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("è®¾å¤‡:", device)


#df_train_full1 = pd.read_csv("/kaggle/input/moviereview/labeledTrainData.tsv", sep='\t')
#df_train_full2_1 = pd.read_csv("/kaggle/input/moviereview/labeledTrainData.tsv", sep='\t')
#df_train_full2_2 = pd.read_csv(
#    "/kaggle/input/moviereview/unlabeledTrainData.tsv",
#    sep='\t',
#    quoting=3,              # ä¸�å¤„ç�†å¼•å�·ï¼ŒæŒ‰çº¯æ–‡æœ¬å¤„ç�†
#    engine='python',
#)
#df_train_full2_3 = pd.concat([df_train_full2_1, df_train_full2_2])

#df_test = pd.read_csv("/kaggle/input/moviereview/testData.tsv", sep='\t')
#df_train_full2 = pd.concat([df_train_full2_3, df_test])

#df_train = df_train_full2

# âœ… è½¬æ�¢ä¸º Huggingface Dataset æ ¼å¼�
#train_dataset = Dataset.from_pandas(df_train[['review', 'sentiment']])
#test_dataset = Dataset.from_pandas(df_test[['review']])

# âœ… åŠ è½½ tokenizer
#model_name = "bert-base-uncased"
#tokenizer = AutoTokenizer.from_pretrained(model_name)

# âœ… Tokenize å‡½æ•°
#def preprocess(example):
#    return tokenizer(
#        example["review"],
#        truncation=True,
#        padding="max_length",
#        max_length=256
#    )

#train_dataset = train_dataset.map(preprocess, batched=True)
#train_dataset = train_dataset.rename_column("sentiment", "labels")
#train_dataset.set_format(
#    type="torch",
#    columns=["input_ids", "attention_mask", "labels"]
#)

# âœ… æµ‹è¯•é›† tokenize
#test_dataset = test_dataset.map(preprocess, batched=True)
#test_dataset.set_format(
#    type="torch",
#    columns=["input_ids", "attention_mask"]
#)

# âœ… åŠ è½½æ¨¡å�‹
#model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# âœ… TrainingArguments
#training_args = TrainingArguments(
#    output_dir="./results",
#    num_train_epochs=2,
#    per_device_train_batch_size=8,
#    learning_rate=2e-5,
#    warmup_steps=0,
#    weight_decay=0.01,
#    logging_steps=100,
#    evaluation_strategy="no",
#    save_strategy="no"
#)

# âœ… Trainer
#trainer = Trainer(
 #   model=model,
 #   args=training_args,
 #   train_dataset=train_dataset,
#)

# âœ… è®­ç»ƒæ¨¡å�‹
#trainer.train()

# âœ… æµ‹è¯•é›†é¢„æµ‹
#preds = trainer.predict(test_dataset).predictions.argmax(axis=-1)

# âœ… ç”Ÿæˆ�æ��äº¤æ–‡ä»¶
#submission = pd.DataFrame({
#    "id": df_test["id"],
#    "sentiment": preds
#})
#submission.to_csv("submission.csv", index=False, quoting=1)

#print("âœ… å·²ç”Ÿæˆ� submission.csv")





