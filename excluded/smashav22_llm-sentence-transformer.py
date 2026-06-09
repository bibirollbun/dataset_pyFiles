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


train_sentences=pd.read_csv("/kaggle/input/kazakhstan-respa-final-day-2-late-competition/train_sentences.csv")
train_timeseries=pd.read_csv("/kaggle/input/kazakhstan-respa-final-day-2-late-competition/train_timeseries.csv")
test_sentences=pd.read_csv("/kaggle/input/kazakhstan-respa-final-day-2-late-competition/test_sentences.csv")



train_sentences


train_timeseries


train_sentences["submitted_date"]=pd.to_datetime(train_sentences["submitted_date"])
min_date=train_sentences["submitted_date"].min()
train_sentences["seconds"]=(train_sentences["submitted_date"]-min_date).dt.total_seconds()
train_sentences


from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
model = SentenceTransformer('/kaggle/input/paraphrase-minilm-l3-v2/transformers/default/1/paraphrase-MiniLM-L3-v2')
tqdm.pandas()
texts = train_sentences['sentences'].tolist()
embeddings_np = model.encode(texts, show_progress_bar=True, batch_size=32)
train_sentences['sentences'] = list(embeddings_np)


embeddings_np.shape[0]


len(list(embeddings_np))





from lightgbm import LGBMRegressor
X=np.vstack(train_sentences['sentences'].values)
X_df=pd.DataFrame(X,columns=[f"embedding{i}" for i in range(X.shape[1])])

y=train_sentences["seconds"]

modelReg=LGBMRegressor()
modelReg.fit(X_df,y)


test_sentences


len(test_sentences)


pred1=0
pred2=0
pred=[]
for i in range(len(test_sentences)):
    text1=test_sentences.loc[i,"first_sentence"]
    text2=test_sentences.loc[i,"second_sentence"]
    embedding_one=model.encode(text1,batch_size=256, convert_to_numpy=True)
    embedding_two=model.encode(text2,batch_size=256, convert_to_numpy=True)
    pred1=modelReg.predict(embedding_one.reshape(1,-1))
    pred2=modelReg.predict(embedding_two.reshape(1,-1))
    if pred1[0]>pred2[0]:
        pred.append(0)
    elif pred1[0]<pred2[0]:
        pred.append(1)


    
        
    




