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


import pandas as pd
import seaborn as sns
import numpy as np
import nltk
nltk.download('punkt_tab')
from nltk import tokenize
from matplotlib import pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model, metrics
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectFromModel
import regex as re
from transformers import BertTokenizer, BertModel
from transformers import LongformerTokenizer, LongformerModel
import torch
from tqdm import tqdm
import pandas as pd


text_data = pd.read_csv("/kaggle/input/embedding/embedding.csv")
#test_data = pd.read_csv("/kaggle/input/dataset/train_v2_drcat_02.csv")
eval_data = pd.read_csv("/kaggle/input/test-essay2/test_essays.csv")


liste = text_data['vect']
df_vect = text_data["vect"].str.replace(r"[\[\]]", "", regex=True).str.split(expand=True)

df_vect = df_vect.astype(float)

# Renommer les colonnes
df_vect.columns = [f"vect_{i}" for i in range(df_vect.shape[1])]

# Ajouter au DataFrame original si nécessaire
text_data = text_data.join(df_vect)

# Supprimer l'ancienne colonne
text_data.drop(columns=["vect"], inplace=True)

# Afficher le résultat
text_data.head()


text_data = text_data.drop(['text', 'prompt_name', 'RDizzl3_seven', 'source', 'Unnamed: 0'], axis=1)


X = text_data.drop(['label'], axis=1)
y = text_data['label']


X_train = X
y_train = y
print(X_train)


from sklearn.decomposition import PCA

pca = PCA(n_components=100)  # On réduit à 100 dimensions
X_train_pca = pca.fit_transform(X_train)



from sklearn import tree

#ensemble = XGBClassifier(scale_pos_weight=2)

ensemble = tree.DecisionTreeClassifier()


# Étape 2 : Appliquer la calibration



ensemble.fit(X_train_pca, y_train)


eval_data.head()


from tqdm import tqdm
tqdm.pandas()

MODEL_PATH = "/kaggle/input/bert_model/transformers/default/1/bert-base-uncased"

def vectoriser_model(text, model):
    if model == "BERT":
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model= BertModel.from_pretrained(MODEL_PATH)
    else :
        raise Exception("entrer BERT ou longform")
    
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    output = model(**encoded_input)
    vector_cls = output.pooler_output.squeeze().detach().numpy()
    
    return vector_cls


eval_data['vect'] = eval_data['text'].progress_apply(lambda x: vectoriser_model(x, "BERT"))



eval_data['vect'].head()


eval_data.head()


eval_data["vect"] = eval_data["vect"].astype(str)
print(eval_data["vect"].dtype)


eval_data.head()


df_vect2 = eval_data["vect"].str.replace(r"[\[\]]", "", regex=True).str.split(expand=True)


df_vect2 = df_vect2.astype(float)


df_vect2.head()


#df_vect = eval_data["vect"].str.replace(r"[\[\]]", "", regex=True).str.split(expand=True)

#df_vect = df_vect.astype(float)

# Renommer les colonnes
df_vect2.columns = [f"vect_{i}" for i in range(df_vect2.shape[1])]

# Ajouter au DataFrame original si nécessaire


# Supprimer l'ancienne colonne
#df_vect2.drop(columns=["vect"], inplace=True)


df_vect2.head()
X_test_pca = pca.transform(df_vect2)


df_vect2.head()

#print(X_test_pca)


y_eval_pred_proba = ensemble.predict_proba(X_test_pca)


print(type(y_eval_pred_proba[:, 1][0]))
print(type(y_eval_pred_proba[:, 1][0]))


print(eval_data['id'])
print(y_eval_pred_proba[:,1])


d = {'id' : eval_data['id'], 'generated': y_eval_pred_proba[:,1]}
df_submission = pd.DataFrame(data=d)
df_submission.head()


print(df_submission)


print(df_submission.columns) 


print(df_submission.dtypes)


print(df_submission.isnull().sum())
print(len(df_submission))
print(df_submission["generated"].dtype)


df_submission.to_csv("submission.csv", index=False)



df_submission.to_csv("submission.csv", index=False, encoding="utf-8")

