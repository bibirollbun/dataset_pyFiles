import os
import gc
import sys
import torch
import datasets
import transformers
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import ( decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer,)


from collections import Counter
from itertools import chain

from tqdm.auto import tqdm

from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MaxAbsScaler

from sklearn.svm import LinearSVR
from scipy.sparse import vstack as spvstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt


# sigmoid function for probability correction 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  


#Section 1 : All Transfoermers (if any) predictions happen here


from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

def evaluate_model(true_labels, predicted_probs):
    predicted_classes = (predicted_probs >= 0.5).astype(int) 
    auc = roc_auc_score(true_labels, predicted_probs)
    accuracy = accuracy_score(true_labels, predicted_classes)
    f1 = f1_score(true_labels, predicted_classes)
    return {"AUC": auc, "Accuracy": accuracy, "F1-Score": f1}


# DistilRoberta predictions 

#current model : checkpoint-13542 , score 0.913 

model_checkpoint = "/kaggle/input/detect-llm-models/distilroberta-finetuned_v5/checkpoint-13542"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    return tokenizer(examples['text'], max_length = 512 , padding=True, truncation=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))  

num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Move your model and data to the GPU
model.to(device);
trainer = Trainer(
    model,
    tokenizer=tokenizer,
)
# test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
test =  pd.read_csv('/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv').tail(9000)

test_ds = Dataset.from_pandas(test)
test_ds_enc = test_ds.map(preprocess_function, batched=True)
test_preds = trainer.predict(test_ds_enc)
logits = test_preds.predictions
final_preds_trans_DistilRoberta = sigmoid(logits)[:,0]
final_preds_trans_DistilRoberta




metrics_distilroberta = evaluate_model(test['label'], final_preds_trans_DistilRoberta)
print("DistilRoBERTa 평가 결과:", metrics_distilroberta)


# define data reading methods 

def read_sub():
    return pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv')

def read_test():
    return pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')

# def read_dummy_test():
#     t =  pd.read_csv('/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv').tail(9000)
#     t['id'] = range(0, len(t))
#     t = t[['id', 'text']]
#     return t


def read_train(only_7_prompts = True):
    train = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv")
    train_old =  pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/train_essays.csv")
    train_old.rename(columns={'generated': 'label'}, inplace=True)
    train_old = train_old[['text', 'label']]

    lm_7b =  pd.read_csv("/kaggle/input/llm-mistral-7b-instruct-texts/Mistral7B_CME_v7.csv")
    lm_ali_1 =  pd.read_csv("/kaggle/input/llm-dataset/gen_llm_fac_v1.csv")
    #lm_ali_2 =  pd.read_csv("/kaggle/input/llm-dataset/gen_llm_elec_v1.csv")
    #lm_ali_3 =  pd.read_csv("/kaggle/input/llm-dataset/gen_llm_car_free_v1.csv")
    lm_ali_4 =  pd.read_csv("/kaggle/input/llm-dataset/gen_llm_exploring_venus_v1.csv")
    lm_ali_5 =  pd.read_csv("/kaggle/input/llm-dataset/gen_llm_face_on_mars_v1.csv")
    lm_ali_6 =  pd.read_csv("/kaggle/input/llm-dataset/gen_llm_driveless_cars_v1.csv")
    lm_ali_7 =  pd.read_csv("/kaggle/input/llm-dataset/gen_llm_cowboy_v1.csv")
    lm_ali_8 =  pd.read_csv("/kaggle/input/llm-dataset/gen_llm_cowboy_v2.csv")
    lm_ali_9 =  pd.read_csv("/kaggle/input/llm-dataset/gen_llm_face_on_mars_v2.csv")
    gemini = pd.read_csv("/kaggle/input/llm-dataset/gemini_pro_llm_text.csv")
    gemini = gemini[gemini['typos']=="no"]
    
    #lm_data = pd.concat([lm_7b, lm_ali_1, lm_ali_2, lm_ali_3,lm_ali_4,lm_ali_5,lm_ali_6,lm_ali_7,lm_ali_8,lm_ali_8,gemini], ignore_index=True)
    lm_data = pd.concat([lm_7b, lm_ali_1 ,lm_ali_4,lm_ali_5,lm_ali_6,lm_ali_7,lm_ali_8,lm_ali_8,gemini], ignore_index=True)

    lm_data.rename(columns={'generated': 'label'}, inplace=True)
    lm_data = lm_data[['text', 'label']]

    
    del gemini
    gc.collect()
    
    train =  pd.concat([train, lm_data, train_old])
    
    #print("train len1: ",len(train))
    
    if only_7_prompts:
        train = train[train['prompt_name'] != 'Distance learning']
        train = train[train['prompt_name'] != 'Seeking multiple opinions']
        train = train[train['prompt_name'] != 'Mandatory extracurricular activities']
        train = train[train['prompt_name'] != 'Summer projects']
        train = train[train['prompt_name'] != 'Cell phones at school']
        train = train[train['prompt_name'] != 'Grades for extracurricular activities']
        train = train[train['prompt_name'] != 'Community service']
        train = train[train['prompt_name'] != 'Phones and driving']
        
        train = train[train['prompt_name'] != 'Does the electoral college work?']
        train = train[train['prompt_name'] != 'Car-free cities']
    #print("train len2: ",len(train))
    
    train = train[['text', 'label']]
    train.reset_index(drop=True, inplace=True)


    return train, lm_data, train_old
    

def append_train_from_sub_phase(org_train_data, train_from_sub):
    
    train_from_sub.drop('generated', axis=1, inplace=True)
    train_from_sub.reset_index(drop=True, inplace=True)

    train_from_sub = train_from_sub[['text', 'label']]
    
    train =  pd.concat([org_train_data, train_from_sub])
    
    return train



# define PBE Tokenizer class

class BPETokenizer:
    ST = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    
    def __init__(
        self,
        vocab_size,
    ):
        self.vocab_size = vocab_size
        self.tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tok.normalizer = normalizers.Sequence([normalizers.NFC()])
        self.tok.pre_tokenizer = pre_tokenizers.ByteLevel()
        
    @classmethod
    def chunk_dataset(cls, dataset, chunk_size=1_000):
        for i in range(0, len(dataset), chunk_size):
            yield dataset[i : i + chunk_size]["text"]
        
    def train(self, data):
        trainer = trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.ST)
        dataset = Dataset.from_pandas(data[["text"]])
        self.tok.train_from_iterator(self.chunk_dataset(dataset), trainer=trainer)
        return self
    
    def tokenize(self, data):
        tokenized_texts = []
        for text in tqdm(data['text'].tolist()):
            tokenized_texts.append(self.tok.encode(text))
        return tokenized_texts


def train_tokenizer(train, lm_data, train_old, test):

    tokenizer_train_data = pd.concat([lm_data,train_old])
    tok_data = pd.concat([ tokenizer_train_data[["text"]],  test[["text"]] ]).reset_index(drop=True)
    vc_counters = {}
    for vs in [5_000]: # for if needed to train multi vocab_size 
        bpe_tok = BPETokenizer(vs).train(tok_data)
        ctr = Counter(chain(*[x.ids for x in bpe_tok.tokenize(tok_data)]))
        vc_counters[vs] = (bpe_tok, ctr)
        tqdm.write(f"completed tokenization with {vs:,} vocab size")
    return vc_counters


def tokenize_datasets (vc_counters, train, lm_data, test):
    
    bpe_tok = vc_counters[5_000][0]
    test_extend = pd.concat([lm_data,test])
    tokenized_texts_train = [x.tokens for x in bpe_tok.tokenize(train)]
    tokenized_texts_test = [x.tokens for x in bpe_tok.tokenize(test)]
    tokenized_texts_lm_data = [x.tokens for x in bpe_tok.tokenize(lm_data)]
    tokenized_texts_test2 = tokenized_texts_lm_data + tokenized_texts_test
    
    del tokenized_texts_lm_data
    gc.collect()
    
    return tokenized_texts_train, tokenized_texts_test, tokenized_texts_test2



def dummy(text):
    return text

def vectorizer_of_data(tokenized_texts_train,tokenized_texts_test,tokenized_texts_test2, min_diff ):
    len_test = len(test)
    #print(len_test)
    if len_test < 10:
        min_diff = 0
    
    print("vectorizer - prepare vocab ..")
    vectorizer = TfidfVectorizer(ngram_range=(3, 7), lowercase=False, sublinear_tf=True, analyzer = 'word',min_df=min_diff, 
                                 tokenizer = dummy, preprocessor = dummy, token_pattern = None, strip_accents='unicode')
    
    vectorizer.fit(tokenized_texts_test2)
    vocab = vectorizer.vocabulary_
    
    del vectorizer
    gc.collect()
    
    vectorizer = TfidfVectorizer(ngram_range=(3, 7), lowercase=False, sublinear_tf=True, vocabulary=vocab, analyzer = 'word',
                                 tokenizer = dummy, preprocessor = dummy, min_df=min_diff, token_pattern = None,
                                 strip_accents='unicode')

    print("vectorizer - fit transform on train ..")

    tf_train = vectorizer.fit_transform(tokenized_texts_train)
    
    print("vectorizer - transform on test ..")

    tf_test = vectorizer.transform(tokenized_texts_test)
    
    del tokenized_texts_test2
    del tokenized_texts_test
    del tokenized_texts_train
    
    del vectorizer
    gc.collect()

    return tf_train, tf_test


def get_predictions_linear(tf_train,tf_test, y_train):
    
    scaler = MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(tf_train)
    X_test_scaled = scaler.transform(tf_test)
    
    model = LinearSVR(C = 0.5, epsilon=0.001)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled.copy())
    preds = sigmoid(preds)
    
    return preds
    


def build_new_train_from_sub_all(final_preds_linear_tmp, test, X, Y):
    
    test.loc[:, 'generated'] = final_preds_linear_tmp
    sorted_df = test.sort_values(by='generated', ascending=False)
    top_rows = sorted_df.head(X).copy()
    top_rows['label'] = 1
    
    # Select the bottom Y rows and set 'generated' to 0
    bottom_rows = sorted_df.tail(Y).copy()
    bottom_rows['label'] = 0
    
    # Concatenate the two subsets
    train_from_sub = pd.concat([top_rows, bottom_rows])
    
    return train_from_sub

def build_new_train_from_sub_by_classs(final_preds_linear_tmp, test, X, Y):
    test.loc[:, 'generated'] = final_preds_linear_tmp
    class_dfs = {}
    
    # Iterate over each unique class value and create a separate DataFrame
    for class_value in test['prompt_id'].unique():
        class_dfs[class_value] = test[test['prompt_id'] == class_value]
    #print(class_dfs[class_value])
    
    sorted_class_dfs = {class_value: df.sort_values(by='generated', ascending=False) for class_value, df in class_dfs.items()}
    
    new_class_dfs_with_generated = {}
    
    new_class_dfs_filtered = {}
    for class_value, df in sorted_class_dfs.items():
        if len(df) >= (X + Y):
            top_rows = df.head(X).copy()
            top_rows['label'] = 1
            
            bottom_rows = df.tail(Y).copy()
            bottom_rows['label'] = 0
            
            combined = pd.concat([top_rows, bottom_rows], axis=0)
            new_class_dfs_filtered[class_value] = combined
            
    if len(test) > 10:
        train_from_sub = pd.concat(new_class_dfs_filtered.values(), ignore_index=True)
    else:
        train_from_sub = test
        test['label'] = 1

    return train_from_sub


def build_new_train_from_sub_add_all_data(final_preds_linear_tmp, test):
    
    test.loc[:, 'generated'] = final_preds_linear_tmp
    median_label = test['generated'].median()
    test['label'] = (test['generated'] >= median_label).astype(int)
    train_from_sub = test.copy()
    return train_from_sub






# demo for one run of predictions 

first_time = True
feedback_times = 4
final_predictions = []
X = 1000
Y = 1000
train_from_sub = read_test()
for i in range(1, feedback_times+1):
    
    print("reading datasets ... iter: ", i)
    sub = read_sub()
    test = read_test() #read_dummy_test() #read_test()
    train, lm_data, train_old = read_train(only_7_prompts = True)
    
    if first_time == False:
        train = append_train_from_sub_phase(train, train_from_sub)
        
    print("train tokenizer ... iter: ", i)
    vs_counters = train_tokenizer(train, lm_data, train_old, test)
    
    print("tokenize datasets ... iter: ", i)
    tokenized_texts_train, tokenized_texts_test, tokenized_texts_test2 = tokenize_datasets (vs_counters, train, lm_data, test)
    
    print("verctorize datasets ...iter: ", i)
    tf_train, tf_test = vectorizer_of_data(tokenized_texts_train,tokenized_texts_test,tokenized_texts_test2, 2 )
    
    del tokenized_texts_test2, tokenized_texts_test, tokenized_texts_train, vs_counters
    gc.collect()
    
    print("predictions ... iter: ", i)
    final_preds_linear_tmp = get_predictions_linear(tf_train,tf_test, train['label'].values)
    
    del tf_train, tf_test
    gc.collect()
    
    print("predeictions from iter : ",i, " is : ", final_preds_linear_tmp)
    
    final_preds_phase_tmp = 0.5*final_preds_linear_tmp + 0.5*final_preds_trans_DistilRoberta
    
    print("final predeictions from iter : ",i, " is : ", final_preds_phase_tmp)

    final_predictions = final_preds_phase_tmp #final_preds_phase_tmp
    
    train_from_sub = build_new_train_from_sub_all(final_preds_phase_tmp, test, X, Y)
    X = X + int(250/i)
    Y = Y + int(250/i)
    print("new x,y: ",X,Y)
    
    first_time = False



# 평가 데이터 준비
test = pd.read_csv('/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv').tail(9000)
test['text'] = test['text'].fillna("")  # 결측값 처리

# 테스트 데이터 토큰화
test_ds = Dataset.from_pandas(test)
test_ds_enc = test_ds.map(preprocess_function, batched=True)

# 모델 예측
test_preds = trainer.predict(test_ds_enc)
logits = test_preds.predictions  # 모델 출력값
final_preds_trans_DistilRoberta = sigmoid(logits)[:, 1]  # 확률값 변환

# 평가 수행
metrics_distilroberta = evaluate_model(test['label'], final_preds_trans_DistilRoberta)
print("DistilRoBERTa 평가 결과:", metrics_distilroberta)

# 확률값 분포 시각화
plt.hist(final_preds_trans_DistilRoberta, bins=50, alpha=0.7, label='DistilRoBERTa Probabilities')
plt.title("Distribution of Predicted Probabilities")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.legend()
plt.show()


print(test.head())  # 테스트 데이터 확인
print(test['label'].value_counts())  # 레이블 분포 확인



print(logits[:10])  # 로그 확률 출력
print(final_preds_trans_DistilRoberta[:10])  # 시그모이드 변환 후 확률


train, lm_data, train_old = read_train(only_7_prompts = True)
train


train = append_train_from_sub_phase(train, train_from_sub)

train


test = read_test() 
test = test[['id', 'text']]


from sklearn.model_selection import StratifiedKFold
sk = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
train0 = train
for i, (tr,val) in enumerate(sk.split(train0,train0.label)):
    train = train0.iloc[tr]
    valid = train0.iloc[val]
    break


train.label.value_counts()


valid.label.value_counts()


train.text = train.text.fillna("")
valid.text = valid.text.apply(lambda x: x.strip('\n'))
train.text = train.text.apply(lambda x: x.strip('\n'))
train.head()


from datasets import Dataset
ds_train = Dataset.from_pandas(train)
ds_valid = Dataset.from_pandas(valid)


from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
model_checkpoint = "/kaggle/input/distilroberta-base/distilroberta-base" #base model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    return tokenizer(examples['text'], max_length=128, padding=True, truncation=True)


ds_train_enc = ds_train.map(preprocess_function, batched=True)
ds_valid_enc = ds_valid.map(preprocess_function, batched=True)


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)


import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Move your model and data to the GPU
model.to(device)


from transformers import EarlyStoppingCallback
early_stopping = EarlyStoppingCallback(early_stopping_patience=2)


num_train_epochs=4.0
metric_name = "roc_auc"
model_name = "distilroberta"#"deberta-large"
batch_size = 2

args = TrainingArguments(
    f"{model_name}-finetuned_v5",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    lr_scheduler_type = "cosine",
    save_safetensors = False,

    optim="adamw_torch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=8,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
#     report_to='none', # change to wandb after enabling internet access
    save_total_limit=2,

)


from sklearn.metrics import roc_auc_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    auc = roc_auc_score(labels, probs[:,1], multi_class='ovr')
    return {"roc_auc": auc}


trainer = Trainer(
    model,
    args,
    train_dataset=ds_train_enc,
    eval_dataset=ds_valid_enc,
    tokenizer=tokenizer,
    callbacks = [early_stopping],
    compute_metrics=compute_metrics
)


import wandb
wandb.init(mode="disabled")


trainer.train()



trained_model = trainer.model



import transformers
import datasets
import pandas as pd
import numpy as np
from datasets import Dataset
import os
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from transformers import AutoTokenizer



tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
#model_checkpoint = "/kaggle/input/distilroberta-ali-fin-100"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
def preprocess_function(examples):
    return tokenizer(examples['text'], max_length = 512 , padding=True, truncation=True)
num_labels = 2
#model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Move your model and data to the GPU
model.to(device);
trainer = Trainer(
    model,
    tokenizer=tokenizer,
)



test_ds = Dataset.from_pandas(test)
test_ds_enc = test_ds.map(preprocess_function, batched=True)
test_preds = trainer.predict(test_ds_enc)
logits = test_preds.predictions
probs = sigmoid(logits)[:,1]



len(probs)


final_predictions_deep = 0.5*final_predictions+0.5*probs


test_ds_enc = Dataset.from_pandas(test).map(preprocess_function, batched=True)
test_preds = trainer.predict(test_ds_enc)
logits = test_preds.predictions
final_preds_roberta = sigmoid(logits)[:, 1]

# 평가
metrics_roberta = evaluate_model(test['label'], final_preds_roberta)
print("RobertaBase 평가 결과:", metrics_roberta)


print("DistilRoBERTa 성능:")
for metric, value in metrics_distilroberta.items():
    print(f"{metric}: {value:.4f}")

print("\DistilRoBERTa2 성능:")
for metric, value in metrics_roberta.items():
    print(f"{metric}: {value:.4f}")



labels = ['AUC', 'Accuracy', 'F1-Score']
distilroberta_scores = [metrics_distilroberta[label] for label in labels]
roberta_scores = [metrics_roberta[label] for label in labels]

x = range(len(labels))
plt.bar(x, distilroberta_scores, width=0.4, label='DistilRoBERTa', align='center')
plt.bar([p + 0.4 for p in x], roberta_scores, width=0.4, label='RobertaBase', align='center')
plt.xticks([p + 0.2 for p in x], labels)
plt.ylabel("Score")
plt.title("Model Comparison")
plt.legend()
plt.show()




