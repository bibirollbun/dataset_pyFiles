import warnings
warnings.filterwarnings('ignore')


import gc
import os
import torch
import yaml
import time
import datetime
import itertools
import transformers

from math import exp

import numpy as np
import pandas as pd 
from tqdm import tqdm

from collections import Counter
from typing import List, Optional, Union
import random, pickle, math 

os.environ['OMP_NUM_THREADS'] = '1' # omp_num_threads = 1 means program executed serially
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ParticipantVisibleError(Exception):
    pass

#solution : DataFrame
#me containing the permuted text in a column named 'text'.
def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, model_path: str='/kaggle/input/gemma-2/transformers/gemma-2-9b/2', load_in_8bit: bool=True, clear_mem: bool=False)-> float:

    # Check that each submitted string sentence is solution's text permutation 
    sol_counts = solution.loc[:, 'text'].str.split().apply(Counter)
    sub_coutns = submission.loc[:, 'text'].str.split().apply(Counter)

    invalid_mask = sol_counts != sub_counts
    if invalid_mask.any():
        raise ParticipantVisibleError('Atleast one submitted string is not a valid permutation of solution string')

    # Split and rejoin to normalize the white space
    sub_strings = [
        ' '.join(s.split()) for s in submitted['text'].tolist()
    ]

    # Initialize the perplexity calculator with a pretrained model
    scorer = PerplexityCalculator(
        model_path = model_path,
        load_in_8bit = load_in_8bit
    )

    # calculate the perplexity for each submitted string
    perplexities = scorer.get_perplexity(sub_strings)

    # Just move on if it fails. Not essential if we have the score
    try:
        scorer.clear_gpu_memory()
    except:
        print('GPU memory clearing failed')

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


from nltk.corpus import stopwords

# Load stop words (you might need to download the NLTK stopwords if you haven't already)
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Sort stop words and other words separately
def custom_sort(text):
    words = text.split(" ")
    stop_words_in_text = sorted([word for word in words if word.lower() in stop_words])
    other_words = sorted([word for word in words if word.lower() not in stop_words])
    return " ".join(stop_words_in_text + other_words)


path = '/kaggle/input/santa-2024/sample_submission.csv'
data = pd.read_csv(path)
print(data['text'].map(lambda x: len(str(x).split(' '))).values)


def perm_from_start(sen, m=5, skips=1000):
    final_sen = sen
    prev_scr = scorer.get_perplexity(sen, batch_size=1)
    first_m = sen.split(' ')[:m]
    sen = ' ' + ' '.join(sen.split(' ')[m:])
    all_perm = list(itertools.permutations(first_m))
    for i in range(0, len(all_perm), skips):
        new_sen = ' '.join(list(all_perm[i]))+sen
        new_scr = scorer.get_perplexity(new_sen, batch_size=1)
        if new_scr<prev_scr:
            print(f'New Score: {round(new_scr)} {""*3}  ::: Updated Sentence: {new_sen}')
            prev_scr=new_scr
            final_sen=new_sen

    return final_sen

def perm_from_last(sen, m=5, skips=1000):
    final_sen = sen
    prev_scr = scorer.get_perplexity(sen, batch_size=1)

    last_m = sen.split(' ')[-m:]
    sen = ' '.join(sen.split(' ')[:-m])+ ' '
    all_perm = list(itertools.permutations(last_m))
    for i in range(0, len(all_perm), skips):
        new_sen = sen+' '.join(list(all_perm[i]))
        new_scr = scorer.get_perplexity(new_sen, batch_size=1)
        if new_scr<prev_scr:
            print(f'New Score: {round(new_scr)} {""*3}  ::: Updated Sentence: {new_sen}')
            prev_scr=new_scr
            final_sen=new_sen
    return final_sen


model_path = '/kaggle/input/gemma-2/transformers/gemma-2-9b/2'
# scorer = PerplexityCalculator(model_path)


# st=0
# total=6
# print('ROW :', st, 'PERMUTATION SIZE : ', total)
# text = data['text'][st]
# text = perm_from_start(text, total, 1)
# text = perm_from_last(text, total, 1)
# print(text)
# print('Lets Reverse The Sentence.........')
# text = list(text.split(' '))
# text.reverse()
# text = ' '.join(text)
# text = perm_from_start(text, total, 1)
# text = perm_from_last(text, total, 1)

# data.at[0, 'text'] = text
# print(text)


def format_time(elapsed):
    """Take a time in seconds and return a string hh:mm:ss."""
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))
    
class SimulatedAnnealing:
    def __init__(self, Tmax, Tmin, nsteps, nsteps_per_T, log_freq, random_state, cooling, k):
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.nsteps = nsteps
        self.nsteps_per_T = nsteps_per_T
        self.log_freq = log_freq
        self.cooling = cooling
        self.k = k
        random.seed(random_state)

    def _generate_neighbor(self, solution):
        r = random.choice(range(2))
        if r == 0:
            neighbor = solution.copy()
            i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            return neighbor
        elif r == 1:
            shift = solution.copy()
            extract, insert = random.sample(range(len(shift) - 1), 2)
            shift_words = shift[extract : extract + 1]
            shift = shift[:extract] + shift[extract + 1 :]
            shift = shift[:insert] + shift_words + shift[insert:]
            return shift

    def _acceptance_probability(self, current_energy, new_energy, temperature):
        """
        Calculate the probability of accepting a new solution.
        """
        if new_energy < current_energy:
            return 1.0
        return math.exp(self.k * (current_energy - new_energy) / temperature)

    def solve(self, text):

        t0 = time.time()  # Measure staring time
        current_solution = text.split()
        
        A = " ".join(current_solution)
        if A in USED:
            current_energy = USED[A]
        else:
            current_energy = scorer.get_perplexity(A, batch_size=config["batch_size"])
            USED[A] = current_energy

        best_solution = current_solution.copy()
        best_energy = current_energy

        temperature = self.Tmax
        Tfactor=None
        
        if self.cooling=='exponential':
            Tfactor=-math.log(self.Tmax/self.Tmin)
        elif self.cooling=='linear':
            Tfactor=(self.Tmax-self.Tmin)/self.nsteps
        
        Tfactor = -math.log(self.Tmax / self.Tmin)  # for exponentil cooling

        temperatures = [temperature]
        log_energies = [current_energy]

        for step in range(self.nsteps):
            cntused = 0
            accept = 0

            for step1 in range(self.nsteps_per_T):
                # generate neighbor
                new_solution = self._generate_neighbor(current_solution)
                A = " ".join(new_solution)
                if A in USED:
                    new_energy = USED[A]
                    cntused += 1
                else:
                    new_energy = scorer.get_perplexity(A, batch_size=config["batch_size"])
                    USED[A] = new_energy

                # calculation of acceptance probability
                acceptance = self._acceptance_probability(current_energy, new_energy, temperature)

                # update current solution
                if acceptance > random.random():
                    current_solution = new_solution
                    current_energy = new_energy
                    accept += 1
                # update best solution
                if new_energy < best_energy:
                    best_solution = new_solution.copy()
                    best_energy = new_energy
                    if new_energy < 10000:
                        print(f"\nNew best score: {best_energy:8.3f}")
                        print("New text: ", " ".join(best_solution), "\n", flush=True)

                # log
                log_energies.append(current_energy)
                temperatures.append(temperature)

                t1 = format_time(time.time() - t0)

                if step1 % self.log_freq == 0 or step1 == (self.nsteps_per_T - 1):
                    print(
                        f"T: {temperature:8.3f}  Step: {step1:6} CntUsed: {cntused}  Acceptance Rate: {accept/(step1+1):7.4f}  Score: {current_energy:8.3f}  Best Score: {best_energy:8.3f}  Elapsed Time: {t1}",
                        flush=True,
                    )

            # lower the temperature
            if self.cooling == "linear":
                temperature -= (self.Tmax - self.Tmin) / self.nsteps
            elif self.cooling == "exponential":
                temperature = self.Tmax * math.exp(Tfactor * (step + 1) / self.nsteps)
            elif self.cooling == "logarithmic":
                temperature = self.Tmax / math.log10(step + 10)

            if best_energy < 30.:
                print("Stop! Target value is achieved.")
                break

        return " ".join(best_solution), best_energy, log_energies, temperatures


USED = {}


%%writefile config.yaml

subfile: "/kaggle/input/santa-2024/sample_submission.csv"
sample: 3
batch_size: 1024
model_path: '/kaggle/input/gemma-2/transformers/gemma-2-9b/2'

params:
    Tmax: 2
    Tmin: 1
    nsteps: 30
    nsteps_per_T: 3000
    log_freq: 1000
    random_state: 77
    cooling: 'exponential'
    k: 1.


with open("config.yaml", "r") as file_obj:
    config = yaml.safe_load(file_obj)


scorer = PerplexityCalculator(config["model_path"])


optimizer = SimulatedAnnealing(**config["params"])


# text = data['text'][1]
# print(text)
# solution, score, log_scores, log_ts = optimizer.solve(text)


sub_data=data.copy()


solution0='reindeer mistletoe elf gingerbread ornament family advent scrooge chimney fireplace'
sub_data.at[0, 'text']=solution0


solution1='reindeer mistletoe elf gingerbread ornament advent scrooge chimney fireplace family laugh sleep walk jump bake drive the night and give'
score1 = 514.1392310443808
sub_data.at[1, 'text']=solution1


# text2 = data['text'][2]
# text2 = custom_sort(text)
# solution, score, log_scores, log_ts = optimizer.solve(text2)


solution2 = 'stocking beard grinch carol cheer chimney decorations gifts holiday holly jingle magi naughty nice nutcracker ornament polar sleigh workshop yuletide'
score2 = 345.4619435584505
sub_data.at[2, 'text']=solution2


# text3 = data['text'][3]
# solution, score, log_scores, log_ts = optimizer.solve(text3)


score3 = 302.493
solution3 = 'magi yuletide cheer cheer grinch visit sleigh polar beard workshop chimney stocking sing carol holly jingle naughty nice nutcracker ornament decorations of the holiday is unwrap gifts eat and relax'
sub_data.at[3, 'text']=solution3


# text4 = data['text'][4]
# text4 = custom_sort(text4)
# solution, score, log_scores, log_ts = optimizer.solve(text4)


score4 = 80.681
solution4 = 'from and of to the as in that it we with not you have merry season joy peace hope dream believe angel bow candy candle chocolate cookie doll eggnog fireplace fruitcake game night greeting card hohoho kaggle milk peppermint poinsettia puzzle snowglobe star toy workshop wish wonder wrapping paper wreath'
sub_data.at[4, 'text']=solution4


text5 = data['text'][5]
text5 = custom_sort(text5)


scorer.get_perplexity(text5, batch_size=1)


solution5 = 'and and and as from have in is it not of of that the the the to we with you advent angel bake beard believe bow candle candy card carol cheer cheer chimney chimney chocolate cookie decorations doll dream drive eat eggnog elf family fireplace fireplace fruitcake game gifts gingerbread give greeting grinch hohoho holiday holly hope jingle joy jump kaggle laugh magi merry milk mistletoe naughty nice night night nutcracker ornament ornament paper peace peppermint poinsettia polar puzzle reindeer relax scrooge season sing sleep sleigh snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wrapping wreath yuletide'
sub_data.at[5, 'text']=solution5



df = sub_data.copy()
df.to_csv('submission.csv', index=False)
df['score']=df['text'].map(lambda x: scorer.get_perplexity(x, batch_size=1))
print(np.mean(df['score']))
df['score']




