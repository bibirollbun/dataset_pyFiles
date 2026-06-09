import os

os.chdir('/kaggle/working/')
!git clone https://github.com/k1242/pilgrim.git
os.chdir('/kaggle/working/pilgrim')
!pip install schedulefree


#!python test_grid.py --model_ids {model_ids} --epochs 8192 --B 16777216 --tests_num 10 --device_id 0 --shift 10 --tests=datasets/santa_word5.scv  ## 20 30


import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import random, pickle, math, warnings
import itertools, nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
#warnings.simplefilter('ignore')

#p = '/kaggle/input/santa-2024/sample_submission.csv'
#df = pd.read_csv(p) # 	id 	text
#print(df['text'].map(lambda x: len(str(x).split(' '))).values)

import transformers, torch
import gc, os, logging
from math import exp

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

DEVICE = torch.device('cuda')


tests = torch.load(tests_path, weights_only=False, map_location=device)


t.split('\n')[0]





l=[]
for i in [0,1,2,3,4,5]:
    print(len(t.split('\n')[i].split(' ')))
    l=l+[t.split('\n')[i]]


l


df = pd.DataFrame(l)


df.to_csv('datasets/santa_word5.scv')


df= pd.read_csv('datasets/santa_word5.scv')
df[['0']].values.tolist()


df


df[['0']].values.tolist()


torch.tensor(l)


t = """reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament
reindeer sleep walk the night and drive mistletoe scrooge laugh chimney jump elf bake gingerbread family give advent fireplace ornament
sleigh yuletide beard carol cheer chimney decorations gifts grinch holiday holly jingle magi naughty nice nutcracker ornament polar workshop stocking
sleigh of the magi yuletide cheer is unwrap gifts and eat cheer holiday decorations holly jingle relax sing carol visit workshop grinch naughty nice chimney stocking ornament nutcracker polar beard
from and of to the as in that it we with not you have merry game night season greeting peace angel believe candle bow card candy chocolate cookie doll dream eggnog fireplace fruitcake hohoho hope joy kaggle milk peppermint poinsettia puzzle snowglobe star toy wish wonder workshop wrapping paper wreath
from and and as we and have the in is it of not that the to with you advent card angel bake beard believe bow candy candle carol cheer cheer chocolate chimney cookie decorations doll dream drive eat eggnog family fireplace fireplace chimney fruitcake game gifts give gingerbread greeting grinch holiday holly hohoho hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night elf nutcracker ornament ornament of the wrapping paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wreath yuletide"""

for i in [0,1,2,3,4,5]:
    print(len(t.split('\n')[i].split(' ')))

df = pd.read_csv('/kaggle/input/santa-2024/sample_submission.csv')
df['text'] = t.split('\n')
df.to_csv("submission.csv", index=False)


s2='from and and as we and have the in is it of not that the to with you advent card angel bake beard believe bow candy candle carol cheer cheer chocolate chimney cookie decorations doll dream drive eat eggnog family fireplace fireplace chimney fruitcake game gifts give gingerbread greeting grinch holiday holly hohoho hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night elf nutcracker ornament ornament of the wrapping paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wreath yuletide'


i=0
s2=t.split('\n')[i]


for i in [0]:#,1,2,3,4,5]:
    s2 =  t.split('\n')[i]

    la=0
    kk=0
     
    for iii in range(0, len(s2.split(" ")) ) : #len(s2.split(" "))):   2
           text = " ".join(s2.split(" ")[0:iii+1]  )  #[0:10]  [0:iii] ## [0:iii+1]
           with torch.no_grad():
                text_with_special = f"{tokenizer.bos_token}{text}{tokenizer.eos_token}"
                model_inputs = tokenizer(text_with_special, return_tensors='pt', add_special_tokens=False,)
                model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}
                logits = model(**model_inputs, use_cache=False)['logits']
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = model_inputs['input_ids'][..., 1:].contiguous()
                y0, x = shift_logits.view(-1, shift_logits.size(-1)),  shift_labels.view(-1)
                loss =   [  ]
                n_batch, n_class = y0.shape
                for y1, x1 in zip(y0, x) :
                    class_index = int(x1.item())
                    pred_i=class_index
                    #print(class_index,torch.exp(y1[class_index]).item() , f'sum of {len(y1)}:',torch.exp(y1).sum().item(),
                    #      torch.log(torch.exp(y1[class_index])/(torch.exp(y1).sum())).item(), sep='\t')
                    loss =  loss +  [-torch.log(torch.exp(y1[class_index])/(torch.exp(y1).sum()))  ]  
                loss= torch.tensor(loss).to(DEVICE)    
                sequence_loss = loss.sum() / len(loss)
                loss_list = sequence_loss.cpu().item()
                #print (loss_list)
                #print( math.exp(loss_list))
                print(kk,iii,1, shift_logits.shape[1]-la,shift_logits.shape[1],pred_i, text.split(" ") [-1] , sep='\t'   )
                kk+=1
                for ish in range(2,shift_logits.shape[1]-la+1 ): 
                    print(kk,iii,ish, shift_logits.shape[1]-la,shift_logits.shape[1],pred_i, text.split(" ") [-1]  , sep='\t'  )
                    kk+=1 
                 
                la=shift_logits.shape[1]


model_inputs


model_inputs['input_ids'].type()


s2 = """reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament
reindeer sleep walk the night and drive mistletoe scrooge laugh chimney jump elf bake gingerbread family give advent fireplace ornament
sleigh yuletide beard carol cheer chimney decorations gifts grinch holiday holly jingle magi naughty nice nutcracker ornament polar workshop stocking
sleigh of the magi yuletide cheer is unwrap gifts and eat cheer holiday decorations holly jingle relax sing carol visit workshop grinch naughty nice chimney stocking ornament nutcracker polar beard
from and of to the as in that it we with not you have merry game night season greeting peace angel believe candle bow card candy chocolate cookie doll dream eggnog fireplace fruitcake hohoho hope joy kaggle milk peppermint poinsettia puzzle snowglobe star toy wish wonder workshop wrapping paper wreath
from and and as we and have the in is it of not that the to with you advent card angel bake beard believe bow candy candle carol cheer cheer chocolate chimney cookie decorations doll dream drive eat eggnog family fireplace fireplace chimney fruitcake game gifts give gingerbread greeting grinch holiday holly hohoho hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night elf nutcracker ornament ornament of the wrapping paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wreath yuletide"""



 s2.split('\n')[0]


s2 = """reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament
reindeer sleep walk the night and drive mistletoe scrooge laugh chimney jump elf bake gingerbread family give advent fireplace ornament
sleigh yuletide beard carol cheer chimney decorations gifts grinch holiday holly jingle magi naughty nice nutcracker ornament polar workshop stocking
sleigh of the magi yuletide cheer is unwrap gifts and eat cheer holiday decorations holly jingle relax sing carol visit workshop grinch naughty nice chimney stocking ornament nutcracker polar beard
from and of to the as in that it we with not you have merry game night season greeting peace angel believe candle bow card candy chocolate cookie doll dream eggnog fireplace fruitcake hohoho hope joy kaggle milk peppermint poinsettia puzzle snowglobe star toy wish wonder workshop wrapping paper wreath
from and and as we and have the in is it of not that the to with you advent card angel bake beard believe bow candy candle carol cheer cheer chocolate chimney cookie decorations doll dream drive eat eggnog family fireplace fireplace chimney fruitcake game gifts give gingerbread greeting grinch holiday holly hohoho hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night elf nutcracker ornament ornament of the wrapping paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wreath yuletide"""

text = " ".join(s2.split('\n')[0].split(" ")  )
text


           with torch.no_grad():
                text_with_special = f"{tokenizer.bos_token}{text}{tokenizer.eos_token}"


text_with_special


                model_inputs = tokenizer(text_with_special, return_tensors='pt', add_special_tokens=False,)


model_inputs_i['input_ids']


torch.cat(( model_inputs_i['input_ids'],model_inputs_i['input_ids']) )


tens = torch.stack((tens_1, tens_2, tens_3), -1) 


model_inputs_i['input_ids']


model_inputs_i['input_ids'].shape[1]


m=0

for word1 in text.split():
    model_inputs_i = tokenizer(word1, return_tensors='pt', add_special_tokens=False,)
    if m==0:
       res = model_inputs_i['input_ids']
       res_len=[model_inputs_i['input_ids'].shape[1]]
    else:
       res = torch .cat(  (res, model_inputs_i['input_ids']), 1)
       res_len=res_len + [model_inputs_i['input_ids'].shape[1]]        
    m+=1
res, res_len


res_len


res


model_inputs_i['input_ids']


 torch.cat( (res,model_inputs_i['input_ids']) ) 


len


model_inputs_i['input_ids']


res = torch .stack( (res, model_inputs_i['input_ids'])  , 1)


torch .stack( (model_inputs_i['input_ids'], model_inputs_i['input_ids'])  , 1)


torch.cat(( res, model_inputs_i['input_ids']) )


model_inputs['input_ids']


         
           with torch.no_grad():
                text_with_special = f"{tokenizer.bos_token}{text}{tokenizer.eos_token}"
                model_inputs = tokenizer(text_with_special, return_tensors='pt', add_special_tokens=False,)
                model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}


model_inputs


list(  x )


(469.77489246939956+
424.3898557845707+
298.930340487336+
201.87383775438812+
85.39145819673074+
34.58065528074536)/6


(0) 469.5 here    469,7748925
(1) 424.5 here    424,3898558
(2) 298.5 here    298,9303405
(3) 197.5 here    201,8738378*
(4)  67.5 here     85,3914582*
(5)  32.5 here     34,58065528*


       import pandas as pd
       text = " ".join(s2.split(" ") ) #[0:10]
       with torch.no_grad():
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
            print (loss_list)
            print( math.exp(loss_list))
            print(shift_logits.shape)

 
            shift_logits = torch.exp(shift_logits )
            x=shift_logits.view(-1, shift_logits.size(-1))
            px =   pd.DataFrame(torch.transpose(x, 0, 1)  .cpu().numpy())
            px.to_csv('8_.csv')


#show loss: 
px


px[px.index.isin(words_ind)][4].sort_values()


words_ind_df[words_ind_df[1]==43485    ]


jo=px.join(words_ind_df, lsuffix='_caller', rsuffix='_other')
resu=jo[jo['1_other'] >0]
resu[2]








jo=px.join(words_ind_df, lsuffix='_caller', rsuffix='_other')
resu=jo[jo['1_other'] >0]
resu.to_csv('7_.csv')


px[px[0].any(words_ind)]


words_ind_df.reset_index(inplace=True)
words_ind_df.set_index(1)


words_ind_df.set_index(1)


words_ind


words_ind = [2273  ,
578   ,
578   ,
685   ,
783   ,
578   ,
791   ,
573   ,
575   ,
603   ,
665   ,
576   ,
780   ,
674   ,
573   ,
577   ,
675   ,
692   ,
12002 ,
4076  ,
22448 ,
44528 ,
38175 ,
4564  ,
7181  ,
25720 ,
28162 ,
138763,
22867 ,
22867 ,
13171 ,
67905 ,
17467 ,
42768 ,
7474  ,
6523  ,
6109  ,
7812  ,
7815  ,
1312  ,
869   ,
2730  ,
43485 ,
43485 ,
67905 ,
9471  ,
23144 ,
2398  ,
17196 ,
2734  ,
136507,
32338 ,
2660  ,
14111 ,
12083 ,
108548,
1965  ,
215898,
4077  ,
204063,
9902  ,
10300 ,
124555,
2315  ,
10084 ,
198447,
46301 ,
9512  ,
7727  ,
165493,
97840 ,
4866  ,
3354  ,
3354  ,
52931 ,
16621 ,
99946 ,
29138 ,
29138 ,
576   ,
573   ,
56178 ,
4368  ,
7124  ,
149218,
16573 ,
83096 ,
881   ,
9437  ,
24754 ,
103360,
10228 ,
1513  ,
80108 ,
541   ,
3891  ,
2800  ,
155702,
6284  ,
8529  ,
112671,
2343  ,
77515 ,
12849 ,
748   ,
14660 ,
3532  ,
5376  ,
6199  ,
5144  ,
20257 ,
20257 ,
58409 ,
597   ,
1731  ,
23675 ,
1     ]
words_ind=sorted(list(set(words_ind)) )
words_ind_df=pd.DataFrame(list(set(words_ind)))
words_ind_df


word_num = [['from',2273],                                     
['and',578],
['and',578],
['as',685],
['we',783],
['and',578],
['have',791],
['the',573              ],
['in',575               ],
['is',603               ],
['it',665               ],
['of',576               ],
['not',780              ],
['that',674             ],
['the',573              ],
['to',577               ],
['with',675             ],
['you',692              ],
['advent',12002         ],
['card',4076            ],
['angel',22448          ],
['bake',44528           ],
['beard',38175          ],
['believe',4564         ],
['bow',7181             ],
['candy',25720          ],
['candle',28162         ],
['carol',138763         ],
['cheer',22867          ],
['cheer',22867          ],
['chocolate',13171      ],
['chimney',67905        ],
['cookie',17467         ],
['decorations',42768    ],
['doll',7474            ],
['dream',6523           ],
['drive',6109           ],
['eat',7812             ],
['eggnog',7815          ],
['eggnog',1312          ],
['eggnog',869           ],
['family',2730          ],
['fireplace',43485      ],
['fireplace',43485      ],
['chimney',67905        ],
['fruitcake',9471       ],
['fruitcake',23144      ],
['game',2398            ],
['gifts',17196          ],
['give',2734            ],
['gingerbread',136507   ],
['greeting',32338       ],
['grinch',2660          ],
['grinch',14111         ],
['holiday',12083        ],
['holly',108548         ],
['hohoho',1965          ],
['hohoho',215898        ],
['hope',4077            ],
['jingle',204063        ],
['jump',9902            ],
['joy',10300            ],
['kaggle',124555        ],
['kaggle',2315          ],
['laugh',10084          ],
['magi',198447          ],
['merry',46301          ],
['milk',9512            ],
['mistletoe',7727       ],
['mistletoe',165493     ],
['naughty',97840        ],
['nice',4866            ],
['night',3354           ],
['night',3354           ],
['elf',52931            ],
['nutcracker',16621     ],
['nutcracker',99946     ],
['ornament',29138       ],
['ornament',29138       ],
['of',576               ],
['the',573              ],
['wrapping',56178       ],
['paper',4368           ],
['peace',7124           ],
['peppermint',149218    ],
['polar',16573          ],
['poinsettia',83096     ],
['poinsettia',881       ],
['poinsettia',9437      ],
['puzzle',24754         ],
['reindeer',103360      ],
['relax',10228          ],
['scrooge',1513         ],
['scrooge',80108        ],
['scrooge',541          ],
['season',3891          ],
['sing',2800            ],
['sleigh',155702        ],
['sleep',6284           ],
['snowglobe',8529       ],
['snowglobe',112671     ],
['star',2343            ],
['stocking',77515       ],
['toy',12849            ],
['unwrap',748           ],
['unwrap',14660         ],
['visit',3532           ],
['walk',5376            ],
['wish',6199            ],
['wonder',5144          ],
['workshop',20257       ],
['workshop',20257       ],
['yuletide',58409       ],
['yuletide',597         ],
['yuletide',1731        ],
['wreath',23675         ],
['wreath',1             ]]

 
words_ind_df=pd.DataFrame(word_num)
words_ind_df


words_ind_df


desc_w=['jingle',
'magi',
'sleigh',
'peppermint',
'carol',
'gingerbread',
'kaggle',
'holly',
'reindeer',
'naughty',
'poinsettia',
'stocking',
'chimney',
'chimney',
'yuletide',
'wrapping',
'elf',
'merry',
'bake',
'fireplace',
'fireplace',
'decorations',
'beard',
'greeting',
'ornament',
'ornament',
'candle',
'candy',
'puzzle',
'wreath',
'cheer',
'cheer',
'angel',
'workshop',
'workshop',
'cookie',
'gifts',
'nutcracker',
'polar',
'chocolate',
'toy',
'holiday',
'advent',
'joy',
'relax',
'laugh',
'jump',
'milk',
'fruitcake',
'snowglobe',
'eggnog',
'eat',
'mistletoe',
'doll',
'bow',
'peace',
'dream',
'sleep',
'wish',
'drive',
'walk',
'wonder',
'nice',
'believe',
'paper',
'hope',
'card',
'season',
'visit',
'night',
'night',
'sing',
'give',
'family',
'grinch',
'game',
'star',
'from',
'hohoho',
'scrooge',
'have',
'we',
'not',
'unwrap',
'you',
'as',
'with',
'that',
'it',
'is',
'and',
'and',
'and',
'to',
'of',
'of',
'in',
'the',
'the',
'the']


asc_word=['the',
'the',
'the',
'in',
'of',
'of',
'to',
'and',
'and',
'and',
'is',
'it',
'that',
'with',
'as',
'you',
'unwrap',
'not',
'we',
'have',
'scrooge',
'hohoho',
'from',
'star',
'game',
'grinch',
'family',
'give',
'sing',
'night',
'night',
'visit',
'season',
'card',
'hope',
'paper',
'believe',
'nice',
'wonder',
'walk',
'drive',
'wish',
'sleep',
'dream',
'peace',
'bow',
'doll',
'mistletoe',
'eat',
'eggnog',
'snowglobe',
'fruitcake',
'milk',
'jump',
'laugh',
'relax',
'joy',
'advent',
'holiday',
'toy',
'chocolate',
'polar',
'nutcracker',
'gifts',
'cookie',
'workshop',
'workshop',
'angel',
'cheer',
'cheer',
'wreath',
'puzzle',
'candy',
'candle',
'ornament',
'ornament',
'greeting',
'beard',
'decorations',
'fireplace',
'fireplace',
'bake',
'merry',
'elf',
'wrapping',
'yuletide',
'chimney',
'chimney',
'stocking',
'poinsettia',
'naughty',
'reindeer',
'holly',
'kaggle',
'gingerbread',
'carol',
'peppermint',
'sleigh',
'magi',
'jingle']


sum(torch.nn.CrossEntropyLoss(reduction='none')(shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1))) 


shift_logits  # [1, 15, 256 000]


shift_logits.view(-1, shift_logits.size(-1)).shape  #torch.Size([15, 256000])


shift_labels.view(-1)


       text = " ".join('reindeer'.split(" ")) 
       with torch.no_grad():
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
            print (loss_list)
            print( math.exp(loss_list))


model_inputs


class compute_crossentropyloss_manual:
    """
    y0 is the vector with shape (batch_size,C)
    x shape is the same (batch_size), whose entries are integers from 0 to C-1
    """
    def __init__(self, ignore_index=-100) -> None:
        self.ignore_index=ignore_index
    
    def __call__(self, y0, x):
        loss = 0.
        n_batch, n_class = y0.shape
        # print(n_class)
        for y1, x1 in zip(y0, x):
            class_index = int(x1.item())
            if class_index == self.ignore_index:  # <------ I added this if-statement
                continue
            loss = loss + torch.log(torch.exp(y1[class_index])/(torch.exp(y1).sum()))
        loss = - loss/n_batch
        return loss






class CrossEntropyLossManual:
    """
    y0 is the vector with shape (batch_size,C)
    x shape is the same (batch_size), whose entries are integers from 0 to C-1
    """
    def __init__(self, ignore_index=-100) -> None:
        self.ignore_index=ignore_index
    
    def __call__(self, y0, x):
        loss = 0.
        n_batch, n_class = y0.shape
        # print(n_class)
        for y1, x1 in zip(y0, x):
            class_index = int(x1.item())
            if class_index == self.ignore_index:
                n_batch -= 1
                continue
            loss = loss + torch.log(torch.exp(y1[class_index])/(torch.exp(y1).sum()))
        loss = - loss/n_batch
        return loss


 compute_crossentropyloss_manual() (
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)) 


        y0, x = shift_logits.view(-1, shift_logits.size(-1)),  shift_labels.view(-1)
        loss = 0.
        n_batch, n_class = y0.shape
        for y1, x1 in zip(y0, x) :
            class_index = int(x1.item())
            loss = loss + torch.log(torch.exp(y1[class_index])/(torch.exp(y1).sum()))
        loss = - loss/n_batch
        loss


n_batch


y1, x1 = list(zip(y0, x))[0]


x1.item()


torch.exp(y1).sum()


class_index


y1[class_index]


torch.log(torch.exp(y1[class_index])/(torch.exp(y1).sum()))


torch.log( torch.exp( torch.tensor( y1[478]))/torch.exp( torch.tensor( 0.0011)) )


torch.exp(y1).sum()


torch.tensor(-21.3438/0.0011)


torch.log(torch.tensor(-21.3438/0.0011))


n_batch


loss


shift_logits.shape #[1, 3, 256 000]


list(shift_logits[0][0])


 model_inputs['input_ids'] [0] 


 math.exp(loss_list)





sum(list(loss))


math.exp(92.2664/15)


len(loss)


math.exp(2.617802858352661)


list(loss) 





l=list(itertools.permutations('reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament'.split(" ") ))


scorer.get_perplexity(" ".join(i) )


s=[] 
nn =0
for i in   itertools.batched(
    itertools.permutations('reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament'.split(" "))
                             
                             ,n= 10):
    print(i)
    #s=  scorer.get_perplexity(" ".join(i) ) -469.77489246939956  
    #nn+=1
 
    #if nn%50==0:
    #    print(nn, s )
    #if s<0:
    #        df = pd.DataFrame([" ".join(i)])
    #        df.to_csv(f'f_{nn}.csv')
    #        print(nn, s,   " ".join(i) ,'********')


#tests = pd.read_csv('datasets/santa_word5.scv')
#tests = tests[['0']].values.tolist()   


s2 = """reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament
reindeer sleep walk the night and drive mistletoe scrooge laugh chimney jump elf bake gingerbread family give advent fireplace ornament
sleigh yuletide beard carol cheer chimney decorations gifts grinch holiday holly jingle magi naughty nice nutcracker ornament polar workshop stocking
sleigh of the magi yuletide cheer is unwrap gifts and eat cheer holiday decorations holly jingle relax sing carol visit workshop grinch naughty nice chimney stocking ornament nutcracker polar beard
from and of to the as in that it we with not you have merry game night season greeting peace angel believe candle bow card candy chocolate cookie doll dream eggnog fireplace fruitcake hohoho hope joy kaggle milk peppermint poinsettia puzzle snowglobe star toy wish wonder workshop wrapping paper wreath
from and and as we and have the in is it of not that the to with you advent card angel bake beard believe bow candy candle carol cheer cheer chocolate chimney cookie decorations doll dream drive eat eggnog family fireplace fireplace chimney fruitcake game gifts give gingerbread greeting grinch holiday holly hohoho hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night elf nutcracker ornament ornament of the wrapping paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wreath yuletide"""

text = " ".join(s2.split('\n')[0].split(" ")  )
text


import torch
import torch.nn as nn
import torch.nn.functional as F

 
def batch_process(model, data, device, batch_size):
    """
    Process data through a model in batches.

    :param data: Tensor of input data
    :param model: A PyTorch model with a forward method that accepts data
    :param device: Device to perform computations (e.g., 'cuda', 'cpu')
    :param batch_size: Number of samples per batch
    :return: Concatenated tensor of model outputs
    """
    model.eval()
    model.to(device)

    outputs = torch.empty(data.size(0), dtype=torch.float32, device=device)

    # Process each batch
    for i in range(0, data.size(0), batch_size):
        batch = data[i:i+batch_size].to(device)
        with torch.no_grad():
            batch_output = model(batch).flatten()
        outputs[i:i+batch_size] = batch_output

    return outputs


import torch
import time
from collections import deque
from tqdm import tqdm
 

  
def get_neighbors(  states):
        """Return neighboring states for each state in the batch."""
        neighbors = torch.empty(states.size(0), n_gens, state_size, device=device, dtype=states.dtype)
        for i in range(0, states.size(0), batch_size):
            batch_states = states[i:i + batch_size]
            neighbors[i:i + batch_size] = torch.gather(
                batch_states.unsqueeze(1).expand(batch_states.size(0), n_gens, state_size), 
                2, 
                all_moves.unsqueeze(0).expand(batch_states.size(0), n_gens, state_size)
            )
        return neighbors
    
def apply_move(  states, moves):
        moved_states = torch.empty(states.size(0), state_size, device=device, dtype=states.dtype)
        for i in range(0, states.size(0), batch_size):
            moved_states[i:i+batch_size] = torch.gather(states[i:i+batch_size], 1, all_moves[moves[i:i+batch_size]])
        return moved_states
    

   
def do_greedy_step(  states,   B=1000):
        """Perform a greedy step to find the best neighbors."""
        idx0 = torch.arange(states.size(0), device=device).repeat_interleave(n_gens)
        moves = torch.arange(n_gens, device=device).repeat(states.size(0))

        neighbors_hashed = torch.empty(moves.size(0), dtype=torch.int64, device=device)
        for i in range(0, states.size(0), batch_size):
            batch_states = states[i:i+ batch_size]
            neighbors = get_neighbors(batch_states).flatten(end_dim=1)
            neighbors_hashed[i*n_gens:(i+batch_size)*n_gens] = state2hash(neighbors, hash_vec, batch_size)
        idx1 = get_unique_hashed_states_idx(neighbors_hashed )
        
        value = torch.empty(idx1.size(0), dtype=torch.float32, device=device)
        for i in range(0, idx1.size(0), batch_size):
            batch_states = apply_move(states[idx0[idx1[i:i+batch_size]]], moves[idx1[i:i+batch_size]])
            value[i:i+batch_size] = pred_d(batch_states)[0]
        idx2 = torch.argsort(value)[:B]
        
        next_states = torch.empty(idx2.size(0), state_size, dtype=states.dtype, device=device)
        for i in range(0, idx2.size(0), batch_size):
            next_states[i:i+batch_size] = apply_move(
                states[idx0[idx1[idx2[i:i+batch_size]]]], 
                moves[idx1[idx2[i:i+batch_size]]])

        return next_states, value[idx2], moves[idx1[idx2]], idx0[idx1[idx2]]
    
def check_stagnation(  states_log):
        """Check if the process is in a stagnation state."""
        return torch.isin(torch.concat(list(states_log)[2:]), torch.concat(list(states_log)[:2])).all().item()

    
def get_solution(  state, B=2**12, num_steps=200, num_attempts=10, return_tree=False):
        """Main solution-finding loop that attempts to solve the cube."""
        
        for J in range(num_attempts):
            
            states = state#.unsqueeze(0).clone()
            
            tree_move = -torch.ones((num_steps, B), dtype=torch.int64)
            tree_idx = -torch.ones((num_steps, B), dtype=torch.int64)
#             tree_value = -torch.ones((num_steps, B), dtype=torch.int64)
             
            
          
            for j in range(num_steps):
                states, y_pred, moves, idx = do_greedy_step(states,   B)
                if verbose:
                    pbar.set_description(
                        f"  y_min = {y_pred.min().item():.1f}, y_mean = {y_pred.mean().item():.1f}, y_max = {y_pred.max().item():.1f}"
                    )
                states_hash_log.append(state2hash(states, hash_vec))
                leaves_num = states.size(0)
                tree_move[j, :leaves_num] = moves
                tree_idx[j, :leaves_num] = idx

                if (states == V0).all(dim=1).any():
                    break
                elif (j > 3 and check_stagnation(states_hash_log)):
                    states_bad_hashed = torch.concat((states_bad_hashed, torch.concat(list(states_hash_log))))
                    states_bad_hashed = torch.unique(states_bad_hashed)
                    break

            if (states == V0).all(dim=1).any():
                break
        
        if not (states == V0).all(dim=1).any():
            return None, J
        
        # Reverse the tree to reconstruct the path
        tree_idx, tree_move = tree_idx[:j+1].flip((0,)), tree_move[:j+1].flip((0,))
        
        V0_pos = torch.nonzero((states == V0).all(dim=1), as_tuple=True)[0].item()
        
        # Construct the path
        path = [tree_idx[0, V0_pos].item()]
        for k in range(1, j+1):
            path.append(tree_idx[k, path[-1]].item())
        
        moves_seq = torch.tensor([tree_move[k, path[k-1]] if k > 0 else tree_move[k, V0_pos] for k in range(j+1)], dtype=torch.int64)
        if return_tree:
            return moves_seq.flip((0,)), J, torch.concat((tree_idx.unsqueeze(0), tree_move.unsqueeze(0))).cpu()
        else:
            return moves_seq.flip((0,)), J
    
def pred_d(  states):
        """Predict values for states using the model."""
        pred = batch_process(model, states, device, 2**14)
#         pred[(states == V0).all(dim=-1)] = 0
        return pred.unsqueeze(0)






res


import argparse
import torch
import os
import json
import time
#from pilgrim import Pilgrim, Searcher
#from pilgrim import count_parameters, generate_inverse_moves, load_cube_data
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



cube_size=10
cube_type=""
tests=""
weights=""
B=4096
num_attempts=1
num_steps=100
tests_num=1
device_id=0
verbose=0
shift=0
skip_list=[]
return_tree=0


log_dir = "logs"
forest_dir = "forest"



# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu",  device_id)
#  device = torch.device("cpu")
timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
print(f"[{timestamp}] Start testing with {device}.")

# Load cube data (moves and names)
#all_moves, move_names = load_cube_data(args.cube_size, args.cube_type, device)

# Derive important cube parameters from the loaded data
#n_gens = 99 # all_moves.size(0)  # Number of moves
#state_size = 1 #all_moves.size(1)  # Size of the state representation
#face_size = state_size // 6  # Size of one face of the cube

# Generate inverse moves
#inverse_moves = torch.tensor(generate_inverse_moves(move_names), dtype=torch.int64, device=device)
#V0 = #torch.arange(6, dtype=torch.int8, device=device).repeat_interleave(face_size)

# Load model and weights
#model = Pilgrim(state_size=state_size, 
#                hd1=info['hd1'], hd2=info['hd2'], nrd=info['nrd'], 
#                activation_function=info.get('activation', 'relu'), 
#                use_batch_norm=info.get('use_batch_norm', True))
#model.load_state_dict(torch.load(args.weights, weights_only=False, map_location=device))
#model.eval()

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
#model.dtype = torch.float16

tests=6
# Load test dataset




model.device



#tests = torch.load(tests_path, weights_only=False, map_location=device)
#tests = tests[args.shift:args.shift+args.tests_num]
#tests = tests.to(device)


# Initialize Searcher object
#searcher = Searcher(model=model,   device=device, verbose= verbose)


# Extract epoch information from weights file name

# Prepare log file
os.makedirs(log_dir, exist_ok=True)
log_file_add = ""

log_file = f"{log_dir}/test__B{B}.json"

results = []
total_length = 0
t1 = time.time()


 
for i, state in enumerate(tests, start=0):
    print(i, state) 
    solution_time_start = time.time()
    result = get_solution(
        state, B= B, 
        num_steps= num_steps, num_attempts= num_attempts, 
        return_tree= return_tree
    )
    moves, attempts = result[:2]
    if  return_tree and moves is not None:
        tree = result[2]
        os.makedirs(forest_dir, exist_ok=True)
        torch.save(tree.cpu(), f"{forest_dir}/tree_B{B:08d}.pt")  
        torch.save(state.cpu(), f"{forest_dir}/state_B{B:08d}.pt") 

    solution_time_end = time.time()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    
    if moves is not None:
        solution_length = len(moves)
        total_length += solution_length
        
        result = {
            "test_num": i+shift,
            "solution_length": solution_length,
            "attempts": attempts + 1,
            "time": round(solution_time_end - solution_time_start, 2),
            "moves": moves.tolist()
        }
        
        # Print solution length for each solved cube
        print(f"[{timestamp}] Solution {i+shift}: Length = {solution_length}")
    else:
        # If no solution is found
        result = {
            "test_num": i+shift,
            "solution_length": None,
            "attempts": None,
            "time": round(solution_time_end - solution_time_start, 2),
            "moves": None
        }
        print(f"[{timestamp}] Solution {i+shift} not found")
    
    results.append(result)

    # Append new result to the log file
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=4)

t2 = time.time()

# Calculate average solution length
solved_results = [r for r in results if r["solution_length"] is not None]
avg_length = total_length / len(solved_results) if solved_results else 0

# Print completion message with average solution length
print(f"Test completed in {(t2 - t1):.2f}s.")
print(f"Average solution length: {avg_length:.2f}.")
print(f"Solved {len(solved_results)}  cubes.")
print(f"Results saved to {log_file}.")



1





states=torch.tensor([])


states


    #for i, state in enumerate(tests, start=0):
    #print(i, state) 
    states= torch.tensor([])
    states, y_pred, moves, idx = do_greedy_step(  states,   B=1000)


        idx0 = torch.arange(states.size(0), device=device)#.repeat_interleave(n_gens)
        moves = torch.arange(n_gens, device=device).repeat(states.size(0))





neighbors = get_neighbors(batch_states).flatten(end_dim=1)


        idx0 = torch.arange(states.size(0), device=device)#.repeat_interleave(n_gens)
        moves = torch.arange(n_gens, device=device).repeat(states.size(0))

        neighbors_hashed = torch.empty(moves.size(0), dtype=torch.int64, device=device)
        for i in range(0, states.size(0), batch_size):
            batch_states = states[i:i+ batch_size]
            neighbors = get_neighbors(batch_states).flatten(end_dim=1)
            neighbors_hashed[i*n_gens:(i+batch_size)*n_gens] = state2hash(neighbors, hash_vec, batch_size)
        idx1 = get_unique_hashed_states_idx(neighbors_hashed )
        
        value = torch.empty(idx1.size(0), dtype=torch.float32, device=device)
        for i in range(0, idx1.size(0), batch_size):
            batch_states = apply_move(states[idx0[idx1[i:i+batch_size]]], moves[idx1[i:i+batch_size]])
            value[i:i+batch_size] = pred_d(batch_states)[0]
        idx2 = torch.argsort(value)[:B]
        
        next_states = torch.empty(idx2.size(0), state_size, dtype=states.dtype, device=device)
        for i in range(0, idx2.size(0), batch_size):
            next_states[i:i+batch_size] = apply_move(
                states[idx0[idx1[idx2[i:i+batch_size]]]], 
                moves[idx1[idx2[i:i+batch_size]]])

        #return next_states, value[idx2], moves[idx1[idx2]], idx0[idx1[idx2]]





states


# 
# 


states, y_pred, moves, idx = do_greedy_step(  states,   B=1000)


text


           with torch.no_grad():
                text_with_special = f"{tokenizer.bos_token}{text}{tokenizer.eos_token}"
                model_inputs = tokenizer(text_with_special, return_tensors='pt', add_special_tokens=False,)
                model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}
                logits = model(**model_inputs, use_cache=False)['logits']
                shift_logits = logits[..., :-1, :].contiguous()


model_inputs


        batch_states=torch.tensor([])
        neighbors = get_neighbors(batch_states) 
        neighbors


len(neighbors[0][0])


torch.tensor( [1] * (len(neighbors[0][0])+2 ) )


r = torch.cat(  (torch.tensor([2]),neighbors[0][0],torch.tensor([1]) )  ) .int() 
r.view( 1, r.shape[0] )



r = torch.cat(  (torch.tensor([2]),neighbors[0][0],torch.tensor([1]) )  ) .int() 
r2=r.view( 1, r.shape[0] )
{'input_ids': r2.to(DEVICE),
 'attention_mask' : torch.tensor( [1] * (len(neighbors[0][0])+2 ) ).view( 1, r.shape[0] ).to(DEVICE) }


model_inputs = {'input_ids': torch.tensor([[     2,    478,  74070,   7727, 165493,  52931, 136507,   2730,  12002,
            1513,  80108,    541,  67905,  43485,  29138,      1]],
        device='cuda:0'), 'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0')}


model_inputs





def loss_model(state):
            print(state)
            #state = neighbors[0][0]
            r = torch.cat(  (torch.tensor([2]),state,torch.tensor([1]) ) ) . long() 
            r2=r.view( 1, r.shape[0] )
            model_inputs = {'input_ids': r2.to(DEVICE),
             'attention_mask' : torch.tensor( [1] * (len(state)+2 ) ).view( 1, r.shape[0] ).to(DEVICE) }
            logits = model(**model_inputs, use_cache=False)['logits']
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = model_inputs['input_ids'][..., 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1))
            sequence_loss = loss.sum() / len(loss)
            loss_list = sequence_loss.cpu().item()
            #print (loss_list)
            #del temp
            torch.cuda.empty_cache()
            return loss_list


neighbors


loss_model(neighbors[3][0])


    del temp
    torch.cuda.empty_cache()


        batch_states=torch.tensor([])
        states = get_neighbors(batch_states) 


state[0]


            torch.cuda.empty_cache()


for name in dir():
    if not name.startswith('_'):
        if name not in ['loss_model','state','get_neighbors', 'torch']:
             del globals()[name]


loss_model(state[0])


from operator import itemgetter 
batch_states=torch.tensor([])

for i in [0 ]:
    
    states = get_neighbors(batch_states)
    print(i, states)

    loss_states=[]
    for state in states:
        loss_states=loss_states+ [loss_model(state[0])]
    # states  loss_states
    B=2
    idx2 = torch.argsort(  torch.tensor(loss_states) )[:B]
    batch_states = list(itemgetter(*idx2)(states))


batch_states


num_to_word(batch_states )


 idx2.tolist()


from operator import itemgetter 
 
print(itemgetter(*idx2)(states))


states


states[idx2.tolist()]


states[idx0[idx1[idx2[i:i+batch_size]]]]


loss_model(state)


        """Perform a greedy step to find the best neighbors."""
        batch_states=torch.tensor([])
        neighbors = get_neighbors(batch_states) 
         
        value = torch.empty(idx1.size(0), dtype=torch.float32, device=device)
        for i in range(0, idx1.size(0), batch_size):
       
            value[i:i+batch_size] = pred_d(batch_states)[0]
        idx2 = torch.argsort(value)[:B]
        
        next_states = torch.empty(idx2.size(0), state_size, dtype=states.dtype, device=device)
        for i in range(0, idx2.size(0), batch_size):
            next_states[i:i+batch_size] = apply_move(
                states[idx0[idx1[idx2[i:i+batch_size]]]], 
                moves[idx1[idx2[i:i+batch_size]]])

        print(  next_states, value[idx2], moves[idx1[idx2]], idx0[idx1[idx2]])


s2 = """reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament
reindeer sleep walk the night and drive mistletoe scrooge laugh chimney jump elf bake gingerbread family give advent fireplace ornament
sleigh yuletide beard carol cheer chimney decorations gifts grinch holiday holly jingle magi naughty nice nutcracker ornament polar workshop stocking
sleigh of the magi yuletide cheer is unwrap gifts and eat cheer holiday decorations holly jingle relax sing carol visit workshop grinch naughty nice chimney stocking ornament nutcracker polar beard
from and of to the as in that it we with not you have merry game night season greeting peace angel believe candle bow card candy chocolate cookie doll dream eggnog fireplace fruitcake hohoho hope joy kaggle milk peppermint poinsettia puzzle snowglobe star toy wish wonder workshop wrapping paper wreath
from and and as we and have the in is it of not that the to with you advent card angel bake beard believe bow candy candle carol cheer cheer chocolate chimney cookie decorations doll dream drive eat eggnog family fireplace fireplace chimney fruitcake game gifts give gingerbread greeting grinch holiday holly hohoho hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night elf nutcracker ornament ornament of the wrapping paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wreath yuletide"""

text = " ".join(s2.split('\n')[0].split(" ")  )

m=0

for word1 in text.split():
    model_inputs_i = tokenizer(word1, return_tensors='pt', add_special_tokens=False,)
    if m==0:
       res = model_inputs_i['input_ids']
       res_len=[model_inputs_i['input_ids'].shape[1]]
    else:
       res = torch .cat(  (res, model_inputs_i['input_ids']), 1)
       res_len=res_len + [model_inputs_i['input_ids'].shape[1]]        
    m+=1
res, res_len


len(torch.tensor([]))


get_neighbors(  torch.tensor([]))


def get_neighbors(  statesi):
        """Return neighboring states for each state in the batch."""
        neighbors=[]
        if len(statesi)>0: 
             
            for sentense in statesi:
                #print(sentense)
                s=0 
                for next_word_len in res_len:
                    next_word = res[0][s:s+next_word_len] .view(1,next_word_len )
                    if sum(sum(sentense   == next_word[0][0].item() )).item()==0:
             
                        s+= next_word_len   
                        #print(    torch.cat((sentense,next_word ),1))
                        neighbors = neighbors + [torch.cat((sentense,next_word ),1)]
        else:   
                sentense = statesi
                #print(sentense)
                s=0 
                for next_word_len in res_len:
                    next_word = res[0][s:s+next_word_len] .view(1,next_word_len )
                    s+= next_word_len  
                    #print(   next_word)
                    #print(    torch.cat((sentense,next_word ),1))
                    neighbors= neighbors + [torch.cat((sentense,next_word ),1)]              
        return neighbors


state1[0]


text.split(" ")


for i in state1[0][0]:
    print(i)


sentense


res


res[0]


def num_to_word(sentense):
    #sentense=state1[1] 
    s=0
    dec_n=0
    s_res=[]
    for next_word_len in res_len:
                    next_word = res[0][s:s+next_word_len] .view(1,next_word_len )
                    index= sum(sum(sentense   == next_word[0][0].item() )).item()
                    #print( next_word[0][0], index)
                    s+= next_word_len  
                    if index:
                        s_res = s_res+[text.split(" ")[dec_n] ]
                    dec_n+=1
    return " ".join(s_res  )                 


num_to_word(state1[1] )


state0=get_neighbors(  torch.tensor([]))
state1=get_neighbors(  state0)


#1 step
state1=get_neighbors(  state0)


state1


states


get_neighbors(  states)


torch.tensor([[]])


get_neighbors(  torch.tensor([ ])  )


res, res_len


s=0 
for next_word_len in res_len:
    next_word = res[0][s:s+next_word_len] .view(1,next_word_len )
    s+=next_word_len
    print(next_word)


 torch.gather()


state1[0][0][0]


res == state1[0][0][0]


(res == state1[0][0][0] ).nonzero(as_tuple=True)[0][0]


state1


sentense[0][0]


torch.cat((sentense,sentense),1)


resu=[]
for sentense in state1:
    #print(sentense)
    s=0 
    for next_word_len in res_len:
        next_word = res[0][s:s+next_word_len] .view(1,next_word_len )
        if sum(sum(sentense   == next_word[0][0].item() )).item()==0:
             
            s+= next_word_len   
            print(    torch.cat((sentense,next_word ),1))
            resu=resu + [torch.cat((sentense,next_word ),1)]


state2=resu


resu=[]
for sentense in state2:
    #print(sentense)
    s=0 
    for next_word_len in res_len:
        next_word = res[0][s:s+next_word_len] .view(1,next_word_len )
        if sum(sum(sentense   == next_word[0][0].item() )).item()==0:
             
            s+= next_word_len   
            print(    torch.cat((sentense,next_word ),1))
            resu=resu + [torch.cat((sentense,next_word ),1)]


sum(sum(next_word   == 483)).item()

