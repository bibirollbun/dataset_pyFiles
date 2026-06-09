!pip install -U transformers
!pip install "numpy<2" --force-reinstall
!pip install protobuf==3.20.3
!pip install detoxify


from detoxify import Detoxify
super_model = Detoxify('original')
super_model.model.cuda()


import pandas as pd
import numpy as np

test_df = pd.read_csv("/kaggle/input/iaio-2026-sf-r-comments-classification/new_test.csv")
test_texts = test_df['comment_text'].tolist()
test_df.head()


from tqdm.auto import tqdm

predicts = []

batch_size = 16
for i in tqdm(range(0, len(test_texts), batch_size)):
    batch = test_texts[i:i + batch_size]
    predict = super_model.predict(batch)
    predicts.append(predict)


dfs = []
for predict in tqdm(predicts):
    dfs.append(pd.DataFrame(predict))
df = pd.concat(dfs, ignore_index=True)


mapped = {
    'toxicity': 'toxic',
    'severe_toxicity': 'severe_toxic',
    'obscene': 'obscene',
    'threat': 'threat',
    'insult': 'insult',
    'identity_attack': 'identity_hate'
}
thresholds = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
result = (df.rename(columns=mapped) >= np.array(thresholds)).astype(int)
result.head()


submission_df = pd.DataFrame({'id': test_df['id'].values}).join(result)
submission_df.to_csv("predobuchka.csv", index=False)
submission_df

