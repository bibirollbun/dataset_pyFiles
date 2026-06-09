import argparse
import torch
import os
import json
import time
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import random, pickle, math, warnings
import itertools, nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import transformers, torch
import gc, os, logging
from math import exp
stop_words = set(stopwords.words('english'))

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
MODEL_PATH = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"
DEVICE = torch.device('cuda')
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, device_map="auto",
            torch_dtype=torch.float16,)
loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
model.eval() 

# Fix float16
model = model.half()


s2 = """reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament
reindeer sleep walk the night and drive mistletoe scrooge laugh chimney jump elf bake gingerbread family give advent fireplace ornament
sleigh yuletide beard carol cheer chimney decorations gifts grinch holiday holly jingle magi naughty nice nutcracker ornament polar workshop stocking
sleigh of the magi yuletide cheer is unwrap gifts and eat cheer holiday decorations holly jingle relax sing carol visit workshop grinch naughty nice chimney stocking ornament nutcracker polar beard
from and of to the as in that it we with not you have milk chocolate candy peppermint eggnog cookie fruitcake toy doll game puzzle greeting card wrapping paper bow wreath poinsettia snowglobe candle fireplace wish dream hope believe wonder night star angel peace joy season merry hohoho kaggle workshop
from and and as we and have the in is it of not that the to with you advent card angel bake beard believe bow candy candle carol cheer cheer chocolate chimney cookie decorations doll dream drive eat eggnog family fireplace fireplace chimney fruitcake game gifts give gingerbread greeting grinch holiday holly hohoho hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night elf nutcracker ornament ornament of the wrapping paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wreath yuletide"""


# num to text


def num_to_text(num , word_list ): 
    #num = torch.tensor([1,4])
    return " ".join([word_list[i ] for i in num.int()])


# model


def loss_model(text,  shows= 0 ):

            text_with_special = f"{tokenizer.bos_token}{text}{tokenizer.eos_token}"
            model_inputs = tokenizer(text_with_special, return_tensors='pt', add_special_tokens=False,)
            model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}

            logits = model(**model_inputs, use_cache=False)['logits']

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = model_inputs['input_ids'][..., 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1))
            sequence_loss = loss.sum() / len(loss)
            loss_list = sequence_loss.cpu().item()

            torch.cuda.empty_cache()
            return loss_list


# generator


def get_neighbors(  states_list, generator_list  ):
        """Return neighboring states for each state in the batch."""
        #states_list=torch.tensor( [ [1,2],[0,1],[2,3] ])
         
        por=0
        for states_i  in states_list:
            b=torch.tensor( generator_list  )
            gener = list(set(b.tolist()) - set(states_i.tolist()))    
            for gener_i in gener:
                element = torch.cat( (states_i, torch.tensor( [gener_i])) ,0)
                if por==0:
                    neighbors=    element.view(1,-1)
                else:    
                    neighbors= torch.cat( (neighbors ,  element.view(1,-1)), 0)  
                por+=1             
        return neighbors


# beam


from operator import itemgetter 

def bim_search(   generator_list , word_list ='nn nnn nnn' , B=70):
 
    batch_states=torch.tensor( [ [] ])
    for i in generator_list:
        states = get_neighbors(batch_states, generator_list)
        print(f'{i} ', end='' )#, states)
        loss_states=[]
        for state in states:
            text =  num_to_text(state , word_list  )  
            loss_states=loss_states+ [loss_model(text  )]
        idx2 = torch.argsort(  torch.tensor(loss_states) )[:B]
        batch_states = list(itemgetter(*idx2)(states))
        l=list(itemgetter(*idx2)(loss_states))
        l.sort()
        print(l[:5], len(l) )
    nk=0
    #for state in batch_states:
    #    print('Answer:',num_to_text(state, word_list)) 
    #    print( list(itemgetter(*idx2)(loss_states))[nk]  )
    #    nk+=1     
    answer=num_to_text(batch_states[0] , word_list)
    print(f'Answer:{answer}')
    return [answer,l[0] ]



# 6 sentense Santa 2025


#  2**1,2**2,2**3,2**4,2**5,2**6,2**7,2**8,2**9, 2**10,2**11,2**12,2**13,2**14,
for B in [ 2**15,2**16,2**17,2**18 ] :
    #B=2
    print(f'******************************************************************************************')
    print(f'***********************B = {B}************************************************************')
    for j in [0, 1,2,3,4,5]:
        text = " ".join(s2.split('\n')[j].split(" ")  ) # generator_list, word_list , B=70
        word_list = text.split(" ")
        generator_list = list(range(0,len(word_list)))
        print(f'_________________________ {j} _____ len: {len(word_list)}_______________________________________________________')
        print(generator_list) 
        print(word_list)
        print(f'_____________________________________________________________________________________________________')
        ans = bim_search(   generator_list=generator_list, word_list=word_list , B=B)
        if j==0: 
             df=pd.DataFrame([ans],columns= [ 'text','score'] )
        else: 
             df= pd.concat( [df,    pd.DataFrame([ans] ,columns= [ 'text','score'] ) ]  ) 
        df.index.name = "id"  
        df.index=list(range(0,len(df) ))
        df.to_csv(f'submission_{B}.csv')  



# chek by hand


           text = " ".join(s2.split('\n')[0].split(" ")  )   #  " ".join(num_to_word( batch_states[0]  ).split(" ")[0:10]) #num_to_word(batch_states[1] )
           #text = 'reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament'
           #text = 'scrooge mistletoe ornament family advent fireplace chimney elf reindeer gingerbread'
           #text = 'reindeer mistletoe scrooge elf gingerbread chimney fireplace ornament family advent'
           #text = 'reindeer mistletoe elf scrooge gingerbread chimney fireplace ornament family advent' 
           text = answer
           print(text) 
           with torch.no_grad():
                text_with_special = f"{tokenizer.bos_token}{text}{tokenizer.eos_token}"
                model_inputs = tokenizer(text_with_special, return_tensors='pt', add_special_tokens=False,)
                model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}
                print(model_inputs)
                logits = model(**model_inputs, use_cache=False)['logits']
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = model_inputs['input_ids'][..., 1:].contiguous()
                y0, x = shift_logits.view(-1, shift_logits.size(-1)),  shift_labels.view(-1)
                loss =   [  ]
                n_batch, n_class = y0.shape
                for y1, x1 in zip(y0, x) :
                    class_index = int(x1.item())
                    pred_i=class_index 
                    loss =  loss +  [-torch.log(torch.exp(y1[class_index])/(torch.exp(y1).sum()))  ]  
                loss= torch.tensor(loss).to(DEVICE)    
                sequence_loss = loss.sum() / len(loss)
                loss_list = sequence_loss.cpu().item()
                print (loss_list)
                print( math.exp(loss_list))
           print(text)    
                 

