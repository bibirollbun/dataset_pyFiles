import pandas as pd 
train_sentences = pd.read_csv('/kaggle/input/kazakhstan-respa-final-day-2-late-competition/train_sentences.csv')
train_timeseries = pd.read_csv('/kaggle/input/kazakhstan-respa-final-day-2-late-competition/train_timeseries.csv')
test_sentences = pd.read_csv('/kaggle/input/kazakhstan-respa-final-day-2-late-competition/test_sentences.csv')


train_sentences


train_timeseries


test_sentences


from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
tqdm.pandas()
texts = train_sentences['sentences'].tolist()
embeddings_np = model.encode(texts, show_progress_bar=True, batch_size=32)
train_sentences['sentences'] = list(embeddings_np)


train_sentences


train_sentences['submitted_date'] = pd.to_datetime(train_sentences['submitted_date'])


train_sentences


train_sentences['time'] = (train_sentences['submitted_date'].dt.year-2000) + train_sentences['submitted_date'].dt.month*12


from lightgbm import LGBMRegressor
X = np.vstack(train_sentences['sentences'].values)
X_df = pd.DataFrame(X, columns=[f'embedding_{i}' for i in range(X.shape[1])])

y = train_sentences['time']

modelgbm = LGBMRegressor()
modelgbm.fit(X_df, y)



sample_submission = pd.read_csv('/kaggle/input/kazakhstan-respa-final-day-2-late-competition/sample_submission.csv')


pred = []
for i in range(len(test_sentences)):
    pred1 = modelgbm.predict([model.encode(test_sentences.loc[i,'first_sentence'])])
    pred2 = modelgbm.predict([model.encode(test_sentences.loc[i,'second_sentence'])])
    if pred1[0]>pred2[0]:
        pred.append(0)
    else:
        pred.append(1)


submission = pd.DataFrame({
    'ID': sample_submission['ID'],
    'label': pred
})


submission.to_csv('submission.csv',index = False)

