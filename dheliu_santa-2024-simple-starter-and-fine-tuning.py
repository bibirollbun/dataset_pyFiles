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
#from Levenshtein import distance

p = '/kaggle/input/santa-2024/sample_submission.csv'
df = pd.read_csv(p) # 	id 	text
print(df['text'].map(lambda x: len(str(x).split(' '))).values)
# äº”ä¸ªæ•°


import gc
import os
from math import exp
from collections import Counter
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import transformers
import torch

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8

class ParticipantVisibleError(Exception):
    pass


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    model_path: str = '/kaggle/input/gemma-2/transformers/gemma-2-9b/2',
    load_in_8bit: bool = True,
    clear_mem: bool = False,
) -> float:
    
    # Check that each submitted string is a permutation of the solution string
    sol_counts = solution.loc[:, 'text'].str.split().apply(Counter)
    sub_counts = submission.loc[:, 'text'].str.split().apply(Counter)
    invalid_mask = sol_counts != sub_counts
    if invalid_mask.any():
        raise ParticipantVisibleError(
        )

    # Calculate perplexity for the submitted strings
    sub_strings = [
        ' '.join(s.split()) for s in submission['text'].tolist()
    ]  # Split and rejoin to normalize whitespace
    scorer = PerplexityCalculator(
        model_path=model_path,
        load_in_8bit=load_in_8bit,
    )  # Initialize the perplexity calculator with a pre-trained model
    perplexities = scorer.getb_perplexity(
        sub_strings
    )  # Calculate perplexity for each submitted string

    if clear_mem:
        # Just move on if it fails. Not essential if we have the score.
        try:
            scorer.clear_gpu_memory()
        except:
            print('GPU memory clearing failed.')

    return float(np.mean(perplexities))


class PerplexityCalculator:
    
    def __init__(
        self,
        model_path: str,
        load_in_8bit: bool = False,
        device_map: str = 'auto',
    ):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path,padding_side="right")
        # Configure model loading based on quantization setting and device availability
        if load_in_8bit:
            if DEVICE.type != 'cuda':
                raise ValueError('8-bit quantization requires CUDA device')

            #quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
            #quantization_config = transformers.BitsAndBytesConfig(load_in_4bit=True)

            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_quant_type = "fp4", #fp4 nf4
                bnb_4bit_use_double_quant = False,
                bnb_4bit_compute_dtype=torch.float16,
            )

            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map=device_map,
            )
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
                device_map=device_map,
            )

        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

        self.model.eval()
        #if not load_in_8bit:
        #    self.model.to(DEVICE)  # Explicitly move the model to the device

    def get_perplexity(
        self, input_texts: Union[str, List[str]], batch_size: 32
    ) -> Union[float, List[float]]:
       
        single_input = isinstance(input_texts, str)
        input_texts = [input_texts] if single_input else input_texts

        loss_list = []

        batches = len(input_texts)//batch_size + (len(input_texts)%batch_size != 0)
        for j in range(batches):

            a = j*batch_size
            b = (j+1)*batch_size
            input_batch = input_texts[a:b]

            with torch.no_grad():

                # Explicitly add sequence boundary tokens to the text
                text_with_special = [f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}" for text in input_batch]

                # Tokenize
                model_inputs = self.tokenizer(
                    text_with_special,
                    return_tensors='pt',
                    add_special_tokens=False,
                    padding=True
                )

                if 'token_type_ids' in model_inputs:
                    model_inputs.pop('token_type_ids')

                model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}

                # Get model output
                output = self.model(**model_inputs, use_cache=False)
                logits = output['logits']

                label = model_inputs['input_ids']
                label[label == self.tokenizer.pad_token_id] = PAD_TOKEN_LABEL_ID

                # Shift logits and labels for calculating loss
                shift_logits = logits[..., :-1, :].contiguous()  # Drop last prediction
                shift_labels = label[..., 1:].contiguous()  # Drop first input

                # Calculate token-wise loss
                loss = self.loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                loss = loss.view(len(logits), -1)
                valid_length = (shift_labels != PAD_TOKEN_LABEL_ID).sum(dim=-1)
                loss = torch.sum(loss, -1) / valid_length

                loss_list += loss.cpu().tolist()
                
        ppl = [exp(i) for i in loss_list]
        
        return ppl[0] if single_input else ppl

    def clear_gpu_memory(self) -> None:
        """Clears GPU memory by deleting references and emptying caches."""
        if not torch.cuda.is_available():
            return

        # Delete model and tokenizer if they exist
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer

        # Run garbage collection
        gc.collect()

        # Clear CUDA cache and reset memory stats
        with DEVICE:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()


# test
# class PerplexityCalculator:
#    def __init__(self,):
#        self.test = 0

#    def get_perplexity(self, text: str) -> float:
#        return 9999999999999999.0

# scorer = PerplexityCalculator()


santa_2024_path="/kaggle/input/santa-2024"

google_gemma_2_transformers_gemma_2_9b_2_path='/kaggle/input/gemma-2/transformers/gemma-2-9b/2'

scorer = PerplexityCalculator(google_gemma_2_transformers_gemma_2_9b_2_path)


samples=pd.read_csv(santa_2024_path+"/sample_submission.csv")
samples.loc[0,"text"]='reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament'

samples.loc[1,"text"]='reindeer sleep walk the night and drive mistletoe scrooge laugh chimney jump elf bake gingerbread family give advent fireplace ornament'

samples.loc[2,"text"]='sleigh yuletide beard carol cheer chimney decorations gifts grinch holiday holly jingle magi naughty nice nutcracker ornament polar workshop stocking'

#samples.loc[3,"text"] = 'sleigh of the magi yuletide cheer is unwrap gifts and eat cheer holiday decorations holly jingle relax sing carol visit workshop grinch naughty nice chimney stocking nutcracker polar beard ornament'
samples.loc[3,"text"]='sleigh of the magi yuletide cheer is unwrap gifts and eat cheer holiday decorations holly jingle relax sing carol visit workshop grinch naughty nice chimney stocking ornament nutcracker polar beard'

#samples.loc[4,"text"] = 'from and of to the as in that it we with not you have merry game night season greeting peace angel believe candle bow card candy chocolate cookie doll dream eggnog fireplace fruitcake hohoho hope joy kaggle milk peppermint poinsettia puzzle snowglobe star toy wish wonder workshop wrapping paper wreath'
samples.loc[4,"text"]='from and of to the as in that it we with not you have milk chocolate candy peppermint eggnog cookie fruitcake toy doll game puzzle greeting card wrapping paper bow candle fireplace wreath poinsettia snowglobe angel star wish dream night season wonder believe hope joy peace merry hohoho kaggle workshop'
samples.loc[5,"text"]='from and and as we and have the in is it of not that the to with you advent card angel bake beard believe bow candy candle carol cheer cheer chocolate chimney cookie decorations doll dream drive eat eggnog family fireplace fireplace chimney fruitcake game gifts give gingerbread greeting grinch holiday holly hohoho hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night elf nutcracker ornament ornament of the wrapping paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wreath yuletide'


perplexities2 = []
PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
for index, row in samples.iterrows(): # Step 1: Reorder the words based on POS tagging reordered_text = reorder_text(row["text"])
    score=scorer.get_perplexity(row['text'],batch_size=BATCH_SIZE)
    perplexities2.append(score)
    print(f"i={index}:{score}")
np.sum(perplexities2)/6


# samples
samples.to_csv("submission.csv", index=False)


past = {}
# You can comment out the following lines to run on a faster GPU with different scoring
with open('/kaggle/input/santa-2024-perplexity-permutation-puzzle-scores/past.pickle', 'rb') as handle:
    past = pickle.load(handle)


perms = {0:3, 1:3, 2:2, 3:2, 4:1, 5:1}
dBaFI = {0:False, 1:False, 2:False, 3:True, 4:False, 5:False}
dChosen = {0:[], 1:['reindeer'], 2:[], 3:[], 4:[], 5:[]}
def custom_sort(words):
    stop_words_in_text = sorted([word for word in words if word.lower() in stop_words])
    other_words = sorted([word for word in words if word.lower() not in stop_words])
    return stop_words_in_text + other_words

def getStart(words, r=1, BaFI=True, chosen=[]): #permutation repeats
    words = words.split(' ')
    #words = sorted(words)
    for w_ in chosen:
        words.remove(w_)
    bestt = []
    resp = ''
    while len(words)>0:
        if len(words)<r: r = len(words)
        p = list(set(itertools.permutations(words, r=r)))
        best = 99999999999
        print('ROUND OF: ', len(p))
        for w in tqdm(p):
            temp = chosen[:] + list(w)
            temp_words = words[:]
            for w_ in list(w): #Used to not remove all duplicate words
                temp_words.remove(w_)
            t = ' '.join(temp + temp_words)
            if t in past:
                s = past[t]
            else:
                s = scorer.get_perplexity(t,8)
                past[t] = s
            if s < best:
                best = s
                bestt = temp[:]
                resp = t
                print(s, t)
                additions = list(w)
        for w_ in additions:
            words.remove(w_)
        chosen = bestt[:]
        #print(best, resp)
        if BaFI: break #Break at First Iteration
    print(best, resp)
    return resp

#samples.at[4, 'text'] = getStart(samples['text'][4], 2, dBaFI[3], dChosen[3])


perms = {0:3, 1:3, 2:2, 3:2, 4:1, 5:1}
dBaFI = {0:False, 1:False, 2:False, 3:True, 4:False, 5:False}
dChosen = {0:[], 1:['reindeer'], 2:[], 3:[], 4:[], 5:[]}

# samples['score'] = df['text'].map(lambda x: scorer.get_perplexity(x))
# samples.to_csv("submission.csv", index=False)
# print(np.mean(samples['score']))
# samples['score']


def permutations_generator(lst):
    if len(lst) == 0:
        yield []
    else:
        for i in range(len(lst)):
            rest = lst[:i] + lst[i + 1:]
            for perm in permutations_generator(rest):
                yield [lst[i]] + perm

best = 999999999
# for permutation in tqdm(permutations_generator(words)):
#     t = ' '.join(permutation)
#     if t in past:
#         s = past[t]
#     else:
#         s = scorer.get_perplexity(t,batch_size= 8)
#         past[t] = s
#     if s < best:
#         best = s
#         #bestt = temp[:]
#         resp = t
#         print(s, resp)


perms = {0:6, 1:3, 2:3, 3:4, 4:2, 5:1}
dBaFI = {0:False, 1:False, 2:False, 3:False, 4:False, 5:False}
dChosen = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}

#https://www.kaggle.com/code/asalhi/sorting-sample-6-stopwords-first
def custom_sort(words):
    stop_words_in_text = sorted([word for word in words if word.lower() in stop_words])
    other_words = sorted([word for word in words if word.lower() not in stop_words])
    return stop_words_in_text + other_words

def getStart(words, a,r, BaFI=True, chosen=[]): #permutation repeats
    print(words)
    
    words = words.split(' ')
    
    # ä¿�å­˜text
    bestt = []
    # ä¿�å­˜æ–‡æœ¬
    resp = ''
      
    print(len(words),words)
    p = list(set(itertools.permutations(words[a:r])))

    best = 99999999999
    print('ROUND OF: ', len(p))
    # p:[('carol', 'decorations'), ('sing', 'nutcracker'), ('workshop', 'unwrap'), ('nutcracker', 'polar'), ('stocking', 'jingle'), ('gifts', 'naughty'), ('grinch', 'visit'), ('gifts', 'unwrap'), ('nutcracker', 'nice'), ('sleigh', 'chimney'), ('unwrap', 'sleigh') ]
    
    for w in tqdm(p):
        temp = list(w)
        #temp_words = words[:]
        
        t = ' '.join(words[:int(a)] + temp + words[r:])
        if t in past:
            s = past[t]
        else:
            s = scorer.get_perplexity(t,batch_size= 8)
            past[t] = s
        if s < best:
            best = s
            bestt = temp[:]
            resp = t
            print(s, t)
    #print(best, resp)
    print(best, resp)
    return best,resp

# epoch = 50
# for i in range(0,epoch,5):
#     if epoch- i < 5:
#         #df.at[4, 'text'] = getStart(samples.loc[4,"text"],50-i, 50, dBaFI[3], dChosen[3])
#         break
#     s,df.at[4, 'text'] = getStart(samples.loc[4,"text"], i, i+5, dBaFI[3], dChosen[3])
#     if s<70:
#         print(s)
#         break
    


# SA
temp_start = 10    #how high a temperature we start with (prior 10)
temp_end = 1e-8       #final temperature (prior 0.2)
cooling_rate = 0.98  #how quick we cool each time we drop temp (prior 0.95)
steps_per_temp = 10  #steps at each temperature (prior 20)    <---- Increase this for a longer run (20 steps is about 3 hours)
 
def simulated_annealing_optimize(text: str, temp_start=temp_start, temp_end=temp_end, cooling_rate=cooling_rate, steps_per_temp=steps_per_temp, verbose=False):
    """Optimize word sequence using simulated annealing, handling NaN scores by randomizing.

    Args:
       text: Input string of space-separated words to optimize
       temp_start: Starting temperature - higher means more random exploration
       temp_end: Ending temperature - lower means more selective at end
       cooling_rate: How fast temperature decreases each step
       steps_per_temp: How many swaps to try at each temperature
       verbose: Whether to print detailed progress
    """
    
    words = text.split()
    #words = sorted(words)
    # first_five = words[:7]
    # random.shuffle(first_five)
    # words = first_five + words[7:]
    current = words.copy()
    
    current_score = scorer.get_perplexity(' '.join(current),batch_size= 8)

    # Handling any NaNs...
    if math.isnan(current_score):
        # Keep shuffling until we find a valid sequence
        while True:
            current = words.copy()
            random.shuffle(current)
            current_score = scorer.get_perplexity(' '.join(current),batch_size= 8)
            if not math.isnan(current_score):
                break
            
    best = current.copy()
    best_score = current_score
    temp = temp_start
    print(f"Start Temperature: {temp:.2f}, Initial score: {current_score:.2f}")
    
    # Main annealing loop - keep trying until we've cooled down enough
    while temp > temp_end:
        for _ in range(steps_per_temp):  # Do multiple attempts at each temperature
            # Try improving sequence by swapping random pairs of words
            i, j = random.sample(range(len(words)), 2)
            neighbor = current.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            
            # Get score for this arrangement, skip if invalid
            t = ' '.join(neighbor)
            if t in past:
                neighbor_score = past[t]
            else:
                neighbor_score = scorer.get_perplexity(t,batch_size= 8)
                past[t] = neighbor_score
            #neighbor_score = scorer.get_perplexity(' '.join(neighbor))
            if math.isnan(neighbor_score):
                continue
            
            # Accept better scores, sometimes accept worse ones based on temperature
            delta = neighbor_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current = neighbor
                current_score = neighbor_score
                
                if current_score < best_score :
                    best = current.copy()
                    best_score = current_score
                    print(">", end="")
                    print(best,best_score)
                else: print("<", end="")
            else:print("-", end="")

        
        # Reduce temperature according to cooling schedule (AFTER all steps at this temperature)
        temp *= cooling_rate
        if verbose: print(f"\nTemperature: {temp:.2f}, Current score: {current_score:.2f}, Current text: {' '.join(current)}")
    
    print(f"\nFinal score: {best_score:.2f}")
    
    return ' '.join(best), best_score

# text = samples.iloc[4].text
# #text = 'sleigh of the magi yuletide cheer is unwrap gifts and eat cheer holiday decorations holly jingle relax sing carol visit workshop stocking naughty nice grinch ornament chimney nutcracker polar beard'
# #print(text)
# for _ in range(50):
#     text,s = simulated_annealing_optimize(text, verbose=True)
#     if s < 71:
#         break
    



used = []

def part_perm_brutem(st, start=0, end=3, skips=1, best=100):
    global past, used
    bestt = st
    #best = scorer.get_perplexity(st)
    st = st.split(' ')
    part = st[start:end]
    if start>0:
        st1 =  ' '.join(st[:start]) + ' '
    else:
        st1 = ''
    if end<len(st): 
        st2 =  ' ' + ' '.join(st[end:])
    else:
        st2 = ''
    p = list(itertools.permutations(part))
    for i in range(0, len(p), skips): #removed tqdm
        t =  st1 + ' '.join(list(p[i])) + st2
        #add_new = False
        if t in past:  #check only once
            s = past[t]
        else:
            s =  scorer.get_perplexity(t,8)
            past[t] = s
            #add_new = True
        if s <= best and t not in used: # and add_new:
            used.append(t)
            print("New Score: ", s, t)
            best = s
            bestt = t
    return bestt, best


# perms = {0:9, 1:6, 2:5, 3:8, 4:6, 5:4}
# dbest = {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0}

# for i in range(6):
#     bestt = samples['text'][4]
#     best = scorer.get_perplexity(bestt,8) + dbest[i]
#     l = len(bestt.split(' '))
#     for p in range(2, perms[3]):
#         for start in tqdm(range(0,l-p+1)):
#             bestt, best = part_perm_brutem(bestt, start, start+p, 1, best)
#     samples.at[3, 'text'] = bestt

# df['score'] = df['text'].map(lambda x: scorer.get_perplexity(x))
# df.to_csv("submission.csv", index=False)
# print(np.mean(df['score']))
# df['text']


def find_best_reorder(input_str, index):
    words = input_str.split()
    if index < 0 or index >= len(words):
        raise ValueError("Index is out of range")

    best_sentence = input_str
    best_perplexity = scorer.get_perplexity(input_str, 8)
    #print(f"Start perplexity: {best_perplexity}, for sentence: {input_str}")

    word_to_move = words.pop(index)

    for i in range(len(words) + 1):
        #print(i)
        reordered_words = words[:]
        reordered_words.insert(i, word_to_move)
        reordered_sentence = ' '.join(reordered_words)

        #print(reordered_sentence)
        
        perplexity = scorer.get_perplexity(reordered_sentence, 8)

        if perplexity < best_perplexity:
            print(f"Checking permutation: {reordered_sentence} :with perplexity: {perplexity}")

def optimize_sentence(input_str):
    words = input_str.split()
    best_sentence = input_str
    best_perplexity = scorer.get_perplexity(input_str, 8)
    print(f"Initial perplexity: {best_perplexity}, for sentence: {input_str}")

    for index in range(len(words)):
        print(f"Index: {index}")
        find_best_reorder(best_sentence, index)


# optimize_sentence(samples.loc[4,"text"])


#import pickle

#with open('past.pickle', 'wb') as f:
#    pickle.dump(past, f, protocol=pickle.HIGHEST_PROTOCOL)


# samples
#samples.to_csv("submission.csv", index=False)

