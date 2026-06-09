# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
print('walking dir')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


print('unin proto')
!pip uninstall -y tensorflow protobuf


print('import')
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from torch.optim import Adam
import torch
import pandas as pd
import json



print('load test')
test=[]
with open('/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl') as f:
    for line in f.readlines():
        test.append(json.loads(line))


print('load model')
from transformers import AutoTokenizer, DistilBertModel

MODEL_DIR = "/kaggle/input/huggingface-bert-variants/"
tokenizer = AutoTokenizer.from_pretrained('/kaggle/input/huggingface-bert-variants/distilbert-base-uncased/distilbert-base-uncased')
model = DistilBertModel.from_pretrained('/kaggle/input/huggingface-bert-variants/distilbert-base-uncased/distilbert-base-uncased')


def get_context_start(tokenizer, input_ids):
    for i,v in enumerate(input_ids):
        if tokenizer.decode(v) == '[SEP]':
            return i+1


def get_char_pos_for_candidate(lc, orig_tok):    
    start_idx= 0    
    for i in range(0,lc['start_token']):
        start_idx += len(orig_tok[i]) + 1
    end_idx = start_idx
    for i in range(lc['start_token'],lc['end_token']+1):
        end_idx += len(orig_tok[i]) + 1        
    return (start_idx, end_idx)


def get_start_end_logit_indices(offset, start, end, context_start):
    idx_start = -1
    idx_end = -1    
    for i in range(context_start, len(offset)):
        if idx_start == -1:
            res = offset[i]            
            if res[0]<=start and start <= res[1]:                
                idx_start = i
        if idx_end == -1:
            res = offset[i]
            if res[0]<=end and end <=res[1]:                
                idx_end = i
    return idx_start, idx_end


def calc_best_end(end_logits, s, e):
    best_end = {}
    for i in range(s,e+1):
        best_end[i] = i
    best_end[e] = e
    best_end_idx = e
    for i in range (e,s, -1):
        if(end_logits[best_end[i]] < end_logits[best_end_idx]):
            best_end[i] = best_end_idx
        best_end_idx = best_end[i]
    return best_end


def get_best_short_answer_idx(s, e, st_logits, end_logits, best_end):    
    best_score = st_logits[s] + end_logits[e]
    bs = s
    be = e    
    for i in range(s,e+1):                
        score = st_logits[i] + end_logits[best_end[i]]
        if score > best_score:
            bs = i
            be = best_end[i]
            best_score = score
    return bs, be, best_score
            


def get_best_short_answer(st, e, off, origtok, start):    
    sc = off[st][0] + start
    ec = off[e][1] + start    
    idxs = -1
    idxe = -1
    cur = 0
    for i in range (0, len(origtok)): 
        if idxs == -1 and cur>=sc:
            idxs = i
        if idxe == -1 and cur>=ec:    
            idxe = i
        cur+= len(origtok[i]) + 1
    return (idxs, idxe)


import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class MyModule(torch.nn.Module):
    def __init__(self, model, tokenizer, classification_layer, start_span_layer, end_span_layer):
        super(MyModule, self).__init__()
        self.model = model
        self.classification_layer = classification_layer
        self.start_span_layer = start_span_layer
        self.end_span_layer = end_span_layer
        self.tokenizer = tokenizer

    def forward(self, ex):
        sl, el, off, cls = self._compute(ex)        
        loss = self._calc_loss(ex, sl, el, off, cls) 
        return loss
    
    def _compute(self, ex):
        chunk_size = 250
        batch_size = 3
        off = []
        sl = []
        el = [] 
        cls = []
        context_start = None
        for i in range(0, len(ex['document_text']), 250):
            tokz = self.tokenizer(ex['question_text'], ex['document_text'][i:i+250], return_offsets_mapping=True)
            inp = torch.tensor(tokz['input_ids']).unsqueeze(0).to(device)
            if context_start is None:
                context_start = get_context_start(tokenizer, tokz['input_ids'])    
            atn = torch.tensor(tokz['attention_mask']).unsqueeze(0).to(device)
            res = self.model(input_ids=inp, attention_mask=atn)
            c = res.last_hidden_state[:, 0, :]
                        
            classification = self.classification_layer(c)
            cls.append(torch.squeeze(classification).to(device))
            
            res.last_hidden_state.shape
            
            start_logits =  self.start_span_layer(res.last_hidden_state)
            sl.append(torch.squeeze(start_logits[0,context_start:,0]).view(-1))
            end_logits = self.end_span_layer(res.last_hidden_state)
            el.append(torch.squeeze(end_logits[0,context_start:,0]).view(-1))
            
            temp = []    
            for o in tokz['offset_mapping'][context_start:]:                
                temp.append((o[0] + i, o[1] + i))

            temp_tensor = torch.tensor(temp)
            if temp_tensor.dim() == 1:
                temp_tensor = temp_tensor.unsqueeze(0)
            off.append(temp_tensor)            
                
        sl = torch.cat(sl, dim=-1)
        el = torch.cat(el, dim=-1)
        off = torch.cat(off, dim=0)
        cls = torch.vstack(cls)
        cls = torch.mean(cls, dim=0)        
        return sl, el, off, cls

    def _calc_loss(self, ex, sl, el, off, cls):    
        #sl = sl.unsqueeze(0)
        #el = el.unsqueeze(0)
        anno = ex['annotations'][0]
        cx = get_class(ex).to(device)
        classification_loss = torch.nn.CrossEntropyLoss()
        total = classification_loss(cls.to(device), cx)
        
        if cx > 0:
            ls, le =  map_to_model_space(off, anno['long_answer']['start_token'], 
                                         anno['long_answer']['end_token'])      
            long_loss = torch.nn.CrossEntropyLoss()
            total += long_loss(sl, torch.tensor(ls).to(device))
            total += long_loss(sl, torch.tensor(le).to(device))
        
        if cx == 1:
            ss, se = map_to_model_space(off, ex['annotations'][0]['short_answers'][0]['start_token'], 
                               ex['annotations'][0]['short_answers'][0]['end_token'])    
            short_loss = torch.nn.CrossEntropyLoss()
            total += short_loss(sl, torch.tensor(ss).to(device))
            total += short_loss(el, torch.tensor(se).to(device))
        return total


classification_layer  = torch.nn.Linear(768, 5)
start_span_layer = torch.nn.Linear(768,1)
end_span_layer = torch.nn.Linear(768,1)
my_module = MyModule(model, tokenizer, classification_layer, start_span_layer, end_span_layer)
my_module.load_state_dict(torch.load('/kaggle/input/tf-qa-100-1/trained_model.pth', map_location=torch.device('cpu')), strict=False)
my_module = my_module.to(device)


ids = []
preds = []
torch.cuda.empty_cache()
torch.set_grad_enabled(False)
import numpy as np

for idx, t in enumerate(test):
    ds=[]
    de=[]
    try:
        ogtok = t['document_text'].split()
        for index, lc in enumerate(t['long_answer_candidates']):
            s,e =  get_char_pos_for_candidate(lc, ogtok)
            ds.append(s)
            de.append(e)
        start = int(np.percentile(ds,15))
        end = int(np.percentile(de, 75))        
        if end-start > 15000:
            median = np.median(ds + de)
            #print(f"media is {median}")
            start = max(median-7500, 0)
            end = min(median +7500, len(t['document_text'])-1)
        start = int(start)
        end = int(end)    
        #print(f"processing example {idx} with index {start} and {end}")
        ex = {}
        ex['question_text'] = t['question_text']
        ex['document_text'] = t['document_text'][start:end+1]    
        sl, el, off,cls = my_module._compute(ex)        
        cls_index = torch.argmax(cls)
        # no answer
        if cls_index == 0:
            print('exiting because class index is 0')
            #print(cls)
            ids.append(f"{t['example_id']}_long")
            ids.append(f"{t['example_id']}_short")
            preds.append('')
            preds.append('')
            continue
            
        max_score = 0
        best_index = 0
        best_long_st = -1
        best_long_e = -1
        
        for index, lc in enumerate(t['long_answer_candidates']):
            sc,ec = get_char_pos_for_candidate(lc, ogtok)
            if sc < start or ec > end:
                continue # outisde truncate doc range
            delta = ec - sc
            sc = sc-start
            ec = sc + delta
            st_l, end_l = get_start_end_logit_indices(off,sc,ec,0)
            score = sl[st_l] + el[end_l]
            if index == 0 or score > max_score:
                max_score = score
                best_index = index
                best_long_st = st_l
                best_long_e = end_l
            
        best_long_answer = t['long_answer_candidates'][best_index] 
        ids.append(f"{t['example_id']}_long")
        preds.append(f"{best_long_answer['start_token']}:{best_long_answer['end_token']}")
    
        if cls_index == 4:
            ids.append(f"{t['example_id']}_short")
            preds.append('')
            continue
        elif cls_index == 2:
            ids.append(f"{t['example_id']}_short")
            preds.append('YES')
        elif cls_index == 3:
            ids.append(f"{t['example_id']}_short")
            preds.append('NO')
        else:
            best_end = calc_best_end(el, best_long_st, best_long_e)
            best_short_st, best_short_e, best_short_score = get_best_short_answer_idx(best_long_st, best_long_e, sl, el, best_end)        
            best_short_answer = get_best_short_answer(best_short_st, best_short_e, off, ogtok, start)
            ids.append(f"{t['example_id']}_short")
            preds.append(f"{best_short_answer[0]}:{best_short_answer[1]}")
        torch.cuda.empty_cache()
    except Exception as e:
        ids.append(f"{t['example_id']}_long")
        ids.append(f"{t['example_id']}_short")
        preds.append('')
        preds.append('')


print('convert')
sub = {'example_id': ids, 'PredictionString': preds}
frame = pd.DataFrame(sub)
frame.to_csv('/kaggle/working/submission.csv', index=False)

