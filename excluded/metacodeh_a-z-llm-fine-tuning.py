!pip install bitsandbytes


import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, DataCollatorWithPadding

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


!nvidia-smi


#수정불가
train_df = pd.read_csv('/kaggle/input/60k-data-with-context-v2/all_12_with_context2.csv')
valid_df = pd.read_csv('/kaggle/input/kaggle-llm-science-exam/train.csv')
test_df = valid_df.copy()
train_df


train_df.isnull().sum()


NUM_TRAIN_SAMPLES = 10000 # <= 10000
train_df = train_df.fillna('').sample(NUM_TRAIN_SAMPLES, random_state=42)





from datasets import load_dataset
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader

test_dataset = Dataset.from_pandas(test_df)
test_dataloader = None #TODO: dataloader 정의


model.eval()
test_predictions = []
for batch in test_dataloader:
    for k in batch.keys():
        batch[k] = batch[k].cuda()
    with torch.no_grad():
        outputs = model(**batch)
    test_predictions.append(outputs['logits'].cpu().detach())

test_predictions = torch.cat(test_predictions)
test_predictions = test_predictions.float().numpy()
predictions_as_ids = np.argsort(-test_predictions, 1)
predictions_as_answer_letters = np.array(list('ABCDE'))[predictions_as_ids]
predictions_as_string = test_df['prediction'] = [
    ' '.join(row) for row in predictions_as_answer_letters[:, :3]
]


#수정불가
#https://www.kaggle.com/code/philippsinger/h2ogpt-perplexity-ranking
import numpy as np
def precision_at_k(r, k):
    """Precision at k"""
    assert k <= len(r)
    assert k != 0
    return sum(int(x) for x in r[:k]) / k

def MAP_at_3(predictions, true_items):
    """Score is mean average precision at 3"""
    U = len(predictions)
    map_at_3 = 0.0
    for u in range(U):
        user_preds = predictions[u].split()
        user_true = true_items[u]
        user_results = [1 if item == user_true else 0 for item in user_preds]
        for k in range(min(len(user_preds), 3)):
            map_at_3 += precision_at_k(user_results, k+1) * user_results[k]
    return map_at_3 / U


#수정불가
m = MAP_at_3(test_df.prediction.values, test_df.answer.values)
print( 'CV MAP@3 =',m )




