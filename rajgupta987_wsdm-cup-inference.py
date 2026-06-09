!pip install -U bitsandbytes


import numpy as np
import pandas as pd    
from transformers import Gemma2ForSequenceClassification,GemmaTokenizerFast    
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
import warnings
from tqdm import tqdm
from peft import PeftModel 
from timeit import default_timer as timer
from concurrent.futures import ThreadPoolExecutor 
from nltk.tokenize import sent_tokenize 
import joblib
import gc
import os
import torch 
import seaborn as sns 
import optuna 
import json  
import matplotlib.pyplot as plt 
from scipy.special import logit  
from lightgbm import LGBMClassifier 
from xgboost import XGBClassifier  
from catboost import CatBoostClassifier     
from sklearn.feature_extraction.text import TfidfVectorizer    
from sklearn.model_selection import StratifiedKFold 
from sklearn.linear_model import LogisticRegression,Ridge    
from sklearn.metrics import accuracy_score 
from sklearn.base import clone 

warnings.filterwarnings('ignore')


class CFG:
    train_path='/kaggle/input/wsdm-cup-multilingual-chatbot-arena/train.parquet'
    
    test_path="/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet" 

    sample_sub_path = '/kaggle/input/wsdm-cup-multilingual-chatbot-arena/sample_submission.csv'

    data_path = '/kaggle/input/wsdm-cup-gemma-2-9b-4-bit-qlora'   

    gemma_dir = "/kaggle/input/gemma-2-9b-4bit-it-unsloth/transformers/default/1/gemma-2-9b-it-4bit-unsloth_old"   

    lora_dir = ("/kaggle/input/wsdm-cup-gemma-2-9b-4-bit-qlora/gemma2-9b-4bit/fold-0/gemma-2-9b-it-bnb-4bit-3072-8-f0/checkpoint-2900")  

    max_length = 3072
    batch_size = 4    

    target = 'winner'
    n_folds = 5
    seed = 42

    
    char_vectorizer_params = {
        'analyzer': "char",
        "lowercase": False,
        "max_df": 0.605,
        "max_features": 331,
        "min_df": 0.075,
        "ngram_range": (1, 3),
        "strip_accents": "unicode"
    }

    word_vectorizer_params = {
        "analyzer": "word",
        "lowercase": True,
        "max_df": 0.985,
        "max_features": 769,
        "min_df": 0.01,
        "ngram_range": (1, 2),
        "strip_accents": "unicode"
    }


test=pd.read_parquet(CFG.test_path).fillna('')


if len(test) > 10_000:
    time_limit = int(3600 * 12) 
else:
    time_limit = int(3600 * 4.75)


def tokenize(tokenizer,prompt,response_a,response_b,max_length=CFG.max_length):
    prompt=["<prompt>: " + t for t in prompt] 
    response_a=["\n\n<response_a>: " + t for t in response_a] 
    response_b=["\n\n<response_b>: " + t for t in response_b] 

    text=[p_r + r_a + r_b for p_r,r_a,r_b in zip(prompt,response_a,response_b)] 

    tokenized=tokenizer(text,max_length=max_length,truncation=True)  

    return tokenized['input_ids'],tokenized['attention_mask']


tokenizer=GemmaTokenizerFast.from_pretrained(CFG.gemma_dir) 
tokenizer.add_eos_token=True 
tokenizer.padding_side='right'


for col in ['prompt','response_a','response_b']: 
    test[col]=test[col].fillna('')    
    text_list=[] 
    if col=='prompt':
        max_no=512 
        s_no=255 
        e_no=-256 
    else:
        max_no=3072 
        s_no=1535 
        e_no=-1536   

    for text in tqdm(test[col]):
        encoded=tokenizer(text,return_offsets_mapping=True) 
        if len(encoded['input_ids'])>max_no:
            start_idx,end_idx=encoded['offset_mapping'][s_no]   
            new_text=text[:end_idx]    
            start_idx,end_idx=encoded['offset_mapping'][e_no]       
            new_text=new_text + "\n(snip)\n" + text[start_idx:]  
            text=new_text  
        text_list.append(text)  
    test[col]=text_list 
    


data=pd.DataFrame() 
data["id"]=test['id'] 
data['input_ids'],data['attention_mask']=tokenize(tokenizer,test['prompt'],test['response_a'],test['response_b'])   
data['length']=data['input_ids'].apply(len)  

aug_data = pd.DataFrame()
aug_data["id"] = test["id"]
aug_data['input_ids'], aug_data['attention_mask'] = tokenize(tokenizer, test["prompt"], test["response_b"], test["response_a"])
aug_data["length"] = aug_data["input_ids"].apply(len)


model_0=Gemma2ForSequenceClassification.from_pretrained(CFG.gemma_dir,
                                        device_map=torch.device("cuda:0"),
                                        use_cache=False) 

model_1=Gemma2ForSequenceClassification.from_pretrained(CFG.gemma_dir,
                                        device_map=torch.device("cuda:1"),
                                        use_cache=False)


model_0=PeftModel.from_pretrained(model_0,CFG.lora_dir) 
model_1=PeftModel.from_pretrained(model_1,CFG.lora_dir)


model_0.eval() 
model_1.eval() 


@torch.no_grad() 
@torch.cuda.amp.autocast() 
def inference(df,model,device,batch_size,max_length=CFG.max_length):   
    winners=[] 

    for start_idx in range(0,len(df),batch_size):
        end_idx=min(start_idx+batch_size,len(df))
        tmp=df.iloc[start_idx:end_idx]   
        input_ids=tmp['input_ids'].to_list() 
        attention_mask=tmp['attention_mask'].to_list() 
        inputs=pad_without_fast_tokenizer_warning(
            tokenizer,
            {"input_ids":input_ids,"attention_mask":attention_mask}, 
            padding="longest", 
            pad_to_multiple_of=None, 
            return_tensors="pt"
        )   
        outputs=model(**inputs.to(device))   
        proba=outputs.logits.softmax(-1).cpu()    

        winners.extend(proba[:,1].tolist())      

    df['winner']=winners

    return df


global_timer=timer() 


data['index']=np.arange(len(data),dtype=np.int32)   
data=data.sort_values('length',ascending=False)


data


data_dict={} 
data_dict[0]=data[data['length']>1024].reset_index(drop=True)
data_dict[1]=data[data['length']<=1024].reset_index(drop=True)


data_dict[0]


result_df=[]   
for i,batch_size in enumerate([CFG.batch_size,CFG.batch_size]):   
    if len(data_dict[i])==0:   
        continue 
    sub_1=data_dict[i].iloc[0::2].copy() 
    sub_2=data_dict[i].iloc[1::2].copy()  

    with ThreadPoolExecutor(max_workers=2) as executor:   
        results=executor.map(
            inference,
            (sub_1,sub_2), 
            (model_0,model_1), 
            (torch.device("cuda:0"),torch.device("cuda:1")), 
            (batch_size,batch_size)
        )   

    result_df.append(pd.concat(list(results),axis=0))


result_df=pd.concat(result_df).sort_values('index').reset_index(drop=True)   


aug_data['index']=np.arange(len(aug_data),dtype=np.int32)  
aug_data=aug_data.sort_values('length',ascending=False) 


CONFIDENCE_THRESHOLD=0.2 
not_confident_mask=abs(result_df['winner']-0.5)<CONFIDENCE_THRESHOLD    

aug_data=aug_data[aug_data['index'].isin(result_df[not_confident_mask]['index'])]


aug_data_dict = {}
aug_data_dict[0] = aug_data[aug_data["length"] > 1024].reset_index(drop=True)
aug_data_dict[1] = aug_data[aug_data["length"] <= 1024].reset_index(drop=True)


aug_result_df = []
for i, batch_size in enumerate([CFG.batch_size, CFG.batch_size]):
    if len(aug_data_dict[i]) == 0:
        continue

    if timer() - global_timer > (time_limit - 900):
        break
        
    sub_1 = aug_data_dict[i].iloc[0::2].copy()
    sub_2 = aug_data_dict[i].iloc[1::2].copy()
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = executor.map(
            inference, 
            (sub_1, sub_2), 
            (model_0, model_1), 
            (torch.device("cuda:0"), torch.device("cuda:1")), 
            (batch_size, batch_size)
        )
        
    aug_result_df.append(pd.concat(list(results), axis=0))


if len(aug_result_df)>0:
    aug_result_df=pd.concat(aug_result_df).sort_values('index').reset_index(drop=True)         

    aug_result_df['winner']=1-aug_result_df['winner']  

    result_df=result_df.merge(
        aug_result_df[['index','winner']], 
        on='index',
        how='left', 
        suffixes=('','_aug')
    )   

    mask=result_df['winner_aug'].notna()   
    result_df.loc[mask,'winner']=(result_df.loc[mask,'winner']+result_df.loc[mask,'winner_aug'])/2    

    result_df=result_df.drop('winner_aug',axis=1)


gemma_test_pred_probs = result_df['winner'].values


sub = pd.read_csv(CFG.sample_sub_path)
sub[CFG.target] = (gemma_test_pred_probs > 0.5).astype(int)
sub[CFG.target] = sub[CFG.target].map({0: "model_a", 1: "model_b"})
sub.to_csv('submission.csv', index=False)
sub.head()

