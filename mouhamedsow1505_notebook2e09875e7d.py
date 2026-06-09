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


## Import libraries


from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model, metrics
import regex as re
import joblib
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MaxAbsScaler



class BPETokenizer:
    SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()])
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    
    def train(self, texts):
        trainer = trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.SPECIAL_TOKENS)
        self.tokenizer.train_from_iterator(texts, trainer=trainer)
        return self
    
    def tokenize(self, texts):
        return [" ".join(self.tokenizer.encode(text).tokens) for text in texts]



# Load dataset
text_data = pd.read_csv("/kaggle/input/train-v2-drcat-02/train_v2_drcat_02.csv")

# Extract the text column 
texts = text_data['text'].dropna().tolist()

# Train BPE Tokenizer
bpe_tokenizer = BPETokenizer(vocab_size=5000).train(texts)

# Apply Tokenization
text_data["bpe_text"] = bpe_tokenizer.tokenize(text_data["text"])


# TF-IDF Vectorization on BPE Tokenized Text
vectorizer = TfidfVectorizer(
    ngram_range=(3, 7), 
    lowercase=False, 
    sublinear_tf=True, 
    analyzer='word',
    min_df=2  
)

# Transform the text
X = vectorizer.fit_transform(text_data["bpe_text"])
y = text_data["label"].values  


def fit_and_do_prediction(X_train, y_train,X_test):
    # Normalizing the data
    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    #naive_bayes = MultinomialNB(alpha=0.02)
    #logistic = LogisticRegression(C=10, solver="liblinear")
    sgd = SGDClassifier(alpha=1e-4, max_iter=10000, tol=1e-4, loss="modified_huber")

    #naive_bayes_bagging = BaggingClassifier(base_estimator=naive_bayes, n_estimators=20, max_samples=0.8, bootstrap=True)
    # Store predictions in a list
    predictions = []

    print("start Naive Bayes")
    # Train Naive Bayes
    sgd.fit(X_train, y_train)
    predictions.append(sgd.predict_proba(X_test)[:, 1]) 
    del sgd
    print("ends of Naive Bayes")
       
    # Combine predictions 
    final_preds = predictions
    

    return final_preds[0]


csv_eval = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/test_essays.csv") #ici on fait avec ce dataset pour essayer, on changera le path vers test_essay dans le notebook kaggle


# Extract the text column
texts = csv_eval['text'].dropna().tolist()

# Apply Tokenization
csv_eval["bpe_text"] = bpe_tokenizer.tokenize(csv_eval["text"])
X_eval = vectorizer.transform(csv_eval["bpe_text"],
                            )



y_pred_eval = fit_and_do_prediction(X,y,X_eval)


d = {'id' : csv_eval['id'], 'generated': y_pred_eval}
df_submission = pd.DataFrame(data=d)


df_submission.head()


df_submission.to_csv("submission.csv", index=False)

