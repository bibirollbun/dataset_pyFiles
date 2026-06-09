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

p = '/kaggle/input/santa-2024/sample_submission.csv'
df = pd.read_csv(p) # 	id 	text
print(df['text'].map(lambda x: len(str(x).split(' '))).values)


print(0)


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


import transformers, torch
import gc, os, logging
from math import exp

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
MODEL_PATH = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"
DEVICE = torch.device('cuda')

class PerplexityCalculator:
    def __init__(self,):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, device_map="auto",
            torch_dtype=torch.float16,)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        self.model.eval()

    def get_perplexity(self, text: str) -> float:
        with torch.no_grad():
            text_with_special = f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}"
            model_inputs = self.tokenizer(text_with_special, return_tensors='pt', add_special_tokens=False,)
            model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}
            logits = self.model(**model_inputs, use_cache=False)['logits']
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = model_inputs['input_ids'][..., 1:].contiguous()
            loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1))
            sequence_loss = loss.sum() / len(loss)
            loss_list = sequence_loss.cpu().item()
            print (loss_list)
        return math.exp(loss_list)
print(0.1)        
scorer = PerplexityCalculator()


import pandas as pd





print(1)


scorer.get_perplexity(l )


DEVICE = torch.device('cuda')



DEVICE


logits


text


t='reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament'


s2


s2='from and and as we and have the in is it of not that the to with you advent card angel bake beard believe bow candy candle carol cheer cheer chocolate chimney cookie decorations doll dream drive eat eggnog family fireplace fireplace chimney fruitcake game gifts give gingerbread greeting grinch holiday holly hohoho hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night elf nutcracker ornament ornament of the wrapping paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wreath yuletide'


s2='from and as we and have the in is it of not that the to with you advent card angel bake beard believe bow candy candle carol cheer cheer chocolate chimney cookie decorations doll dream drive eat eggnog family fireplace fireplace chimney fruitcake game gifts give gingerbread greeting grinch holiday holly hohoho hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night elf nutcracker ornament ornament of the and wrapping paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wreath yuletide'


s2='from and you as we and have the in is it of not that the to with advent card angel bake beard believe bow candy candle carol cheer cheer chocolate chimney cookie decorations doll dream drive eat eggnog family fireplace fireplace chimney fruitcake game gifts give gingerbread greeting grinch holiday holly hohoho hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night elf nutcracker ornament ornament of the and wrapping paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wreath yuletide'


s2='from and bake you as we and have the in is it of not that the to with advent card angel beard believe bow candy candle carol cheer cheer chocolate chimney cookie decorations doll dream drive eat eggnog family fireplace fireplace chimney fruitcake game gifts give gingerbread greeting grinch holiday holly hohoho hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night elf nutcracker ornament ornament of the and wrapping paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wreath yuletide'


#eggnog
s2='from eggnog and and as we and have the in is it of not that the to with you advent card angel bake beard believe bow candy candle carol cheer cheer chocolate chimney cookie decorations doll dream drive eat family fireplace fireplace chimney fruitcake game gifts give gingerbread greeting grinch holiday holly hohoho hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night elf nutcracker ornament ornament of the wrapping paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wreath yuletide'


#laugh
s2='from eggnog laugh and and as we and have the in is it of not that the to with you advent card angel bake beard believe bow candy candle carol cheer cheer chocolate chimney cookie decorations doll dream drive eat family fireplace fireplace chimney fruitcake game gifts give gingerbread greeting grinch holiday holly hohoho hope jingle jump joy kaggle magi merry milk mistletoe naughty nice night night elf nutcracker ornament ornament of the wrapping paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wreath yuletide'


#sleep
s2='from eggnog laugh sleep and and as we and have the in is it of not that the to with you advent card angel bake beard believe bow candy candle carol cheer cheer chocolate chimney cookie decorations doll dream drive eat family fireplace fireplace chimney fruitcake game gifts give gingerbread greeting grinch holiday holly hohoho hope jingle jump joy kaggle magi merry milk mistletoe naughty nice night night elf nutcracker ornament ornament of the wrapping paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wreath yuletide'



#fireplace
s2='from eggnog laugh sleep and and as we and have the in is it of not that the to with you advent card angel bake beard believe bow candy candle carol cheer cheer chocolate chimney cookie decorations doll dream drive eat family fireplace fireplace chimney fruitcake game gifts give gingerbread greeting grinch holiday holly hohoho hope jingle jump joy kaggle magi merry milk mistletoe naughty nice night night elf nutcracker ornament ornament of the wrapping paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wreath yuletide'


la=0
kk=0
for iii in range(0,1): #len(s2.split(" ")) ) : #len(s2.split(" "))):   2
       text = " ".join(s2.split(" ")  )  #[0:10]  [0:iii] ## [0:iii+1]
       with torch.no_grad():
            text_with_special = f"{tokenizer.bos_token}{text}{tokenizer.eos_token}"
            model_inputs = tokenizer(text_with_special, return_tensors='pt', add_special_tokens=False,)
            model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}
            logits = model(**model_inputs, use_cache=False)['logits']
            shift_logits = logits[..., :-1, :].contiguous()
           
            shift_labels = model_inputs['input_ids'][..., 1:].contiguous()
            #loss = loss_fct(
            #    shift_logits.view(-1, shift_logits.size(-1)),
            #    shift_labels.view(-1))


            y0, x = shift_logits.view(-1, shift_logits.size(-1)),  shift_labels.view(-1)
            loss =   [  ]
            n_batch, n_class = y0.shape
            for y1, x1 in zip(y0, x) :
                class_index = int(x1.item())
                print(class_index,torch.exp(y1[class_index]).item() , len(y1),torch.exp(y1).sum().item(),
                      torch.log(torch.exp(y1[class_index])/(torch.exp(y1).sum())).item(), sep='\t')
                loss =  loss +  [-torch.log(torch.exp(y1[class_index])/(torch.exp(y1).sum()))  ]
                 
            loss= torch.tensor(loss).to(DEVICE)    
            
    
            sequence_loss = loss.sum() / len(loss)
            loss_list = sequence_loss.cpu().item()
            print (loss_list)
            print( math.exp(loss_list))
            #print(kk,iii,1, shift_logits.shape[1]-la,shift_logits.shape[1], text.split(" ") [-1] , sep='\t'   )
            kk+=1
            for ish in range(2,shift_logits.shape[1]-la+1 ): 
                #print(kk,iii,ish, shift_logits.shape[1]-la,shift_logits.shape[1], text.split(" ") [-1]  , sep='\t'  )
                kk+=1 
            la=shift_logits.shape[1]


3.5423548221588135
34.548178293732335


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

