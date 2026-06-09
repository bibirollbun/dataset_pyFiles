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
    """
    Calculates the mean perplexity of submitted text permutations compared to an original text.

    Parameters
    ----------
    solution : DataFrame
        DataFrame containing the original text in a column named 'text'.
        Includes a row ID column specified by `row_id_column_name`.

    submission : DataFrame
        DataFrame containing the permuted text in a column named 'text'.
        Must have the same row IDs as the solution.
        Includes a row ID column specified by `row_id_column_name`.

    row_id_column_name : str
        Name of the column containing row IDs.
        Ensures aligned comparison between solution and submission.

    model_path : str
        Path to the serialized LLM.

    clear_mem : bool
        Clear GPU memory after scoring by clearing the CUDA cache.
        Useful for testing.

    Returns
    -------
    float
        The mean perplexity score. Lower is better.

    Raises
    ------
    ParticipantVisibleError
        If the submission format is invalid or submitted strings are not valid permutations.

    Examples
    --------
    >>> import pandas as pd
    >>> model_path = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"
    >>> solution = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'text': ["this is a normal english sentence", "the quick brown fox jumps over the lazy dog"]
    ... })
    >>> submission = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'text': ["sentence english normal a is this", "lazy the over jumps fox brown quick the dog"]
    ... })
    >>> score(solution, submission, 'id', model_path=model_path, clear_mem=True) > 0
    True
    """
    # Check that each submitted string is a permutation of the solution string
    sol_counts = solution.loc[:, 'text'].str.split().apply(Counter)
    sub_counts = submission.loc[:, 'text'].str.split().apply(Counter)
    invalid_mask = sol_counts != sub_counts
    if invalid_mask.any():
        raise ParticipantVisibleError(
            'At least one submitted string is not a valid permutation of the solution string.'
        )

    # Calculate perplexity for the submitted strings
    sub_strings = [
        ' '.join(s.split()) for s in submission['text'].tolist()
    ]  # Split and rejoin to normalize whitespace
    scorer = PerplexityCalculator(
        model_path=model_path,
        load_in_8bit=load_in_8bit,
    )  # Initialize the perplexity calculator with a pre-trained model
    perplexities = scorer.get_perplexity(
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
    """
    Calculates perplexity of text using a pre-trained language model.

    Adapted from https://github.com/asahi417/lmppl/blob/main/lmppl/ppl_recurrent_lm.py

    Parameters
    ----------
    model_path : str
        Path to the pre-trained language model

    load_in_8bit : bool, default=False
        Use 8-bit quantization for the model. Requires CUDA.

    device_map : str, default="auto"
        Device mapping for the model.
    """

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
        """
        Calculates the perplexity of given texts.

        Parameters
        ----------
        input_texts : str or list of str
            A single string or a list of strings.

        batch_size : int, default=None
            Batch size for processing. Defaults to the number of input texts.

        verbose : bool, default=False
            Display progress bar.

        Returns
        -------
        float or list of float
            A single perplexity value if input is a single string,
            or a list of perplexity values if input is a list of strings.

        Examples
        --------
        >>> import pandas as pd
        >>> model_path = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"
        >>> scorer = PerplexityCalculator(model_path=model_path)

        >>> submission = pd.DataFrame({
        ...     'id': [0, 1, 2],
        ...     'text': ["this is a normal english sentence", "thsi is a slihgtly misspelled zr4g sentense", "the quick brown fox jumps over the lazy dog"]
        ... })
        >>> perplexities = scorer.get_perplexity(submission["text"].tolist())
        >>> perplexities[0] < perplexities[1]
        True
        >>> perplexities[2] < perplexities[0]
        True

        >>> perplexities = scorer.get_perplexity(["this is a sentence", "another sentence"])
        >>> all(p > 0 for p in perplexities)
        True

        >>> scorer.clear_gpu_memory()
        """
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

                # Debug output
                #print(f"\nProcessing: '{text}'")
                #print(f"With special tokens: '{text_with_special}'")
                #print(f"Input tokens: {model_inputs['input_ids'][0].tolist()}")
                #print(f"Target tokens: {shift_labels[0].tolist()}")
                #print(f"Input decoded: {self.tokenizer.decode(model_inputs['input_ids'][0])}")
                #print(f"Target decoded: {self.tokenizer.decode(shift_labels[0])}")
                #print(f"Individual losses: {loss.tolist()}")
                #print(f"Average loss: {sequence_loss.item():.4f}")

        ppl = [exp(i) for i in loss_list]

        # print("\nFinal perplexities:")
        # for text, perp in zip(input_texts, ppl):
        #     print(f"Text: '{text}'")
        #     print(f"Perplexity: {perp:.2f}")

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


santa_2024_path="/kaggle/input/santa-2024"

google_gemma_2_transformers_gemma_2_9b_2_path='/kaggle/input/gemma-2/transformers/gemma-2-9b/2'

scorer = PerplexityCalculator(google_gemma_2_transformers_gemma_2_9b_2_path)


samples=pd.read_csv(santa_2024_path+"/sample_submission.csv")

import numpy as np
BATCH_SIZE = 64

best_score = 1e6
import gc
import os
from math import exp
from collections import Counter
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import transformers
import torch
import math


from functools import lru_cache

import random
import math
import random, pickle, warnings

samples.loc[0,"text"]='reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament'

samples.loc[1,"text"]='reindeer sleep walk the night and drive mistletoe scrooge laugh chimney jump elf bake gingerbread family give advent fireplace ornament'

samples.loc[2,"text"]='sleigh yuletide beard carol cheer chimney decorations gifts grinch holiday holly jingle magi naughty nice nutcracker ornament polar workshop stocking'

samples.loc[3,"text"]='sleigh of the magi yuletide cheer is unwrap gifts and eat cheer holiday decorations holly jingle relax sing carol visit workshop grinch naughty nice chimney stocking ornament nutcracker polar beard'

samples.loc[4,"text"]='from and of to the as in that it we with not you have milk chocolate candy peppermint eggnog cookie fruitcake toy doll game puzzle greeting card wrapping paper bow wreath poinsettia snowglobe candle fireplace wish dream hope believe wonder night star angel peace joy season merry hohoho kaggle workshop'
samples.loc[5,"text"]='from and and as we and have the in is it of not that the to with you advent card angel bake beard believe bow candy candle carol cheer cheer chocolate chimney cookie decorations doll dream drive eat eggnog family fireplace fireplace chimney fruitcake game gifts give gingerbread greeting grinch holiday holly hohoho hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night elf nutcracker ornament ornament of the wrapping paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wreath yuletide'

perplexities2 = []


for index, row in samples.iterrows(): # Step 1: Reorder the words based on POS tagging reordered_text = reorder_text(row["text"])
    score=scorer.get_perplexity(row['text'],batch_size=BATCH_SIZE)
    perplexities2.append(score)
    print(f"i={index}:{score}")
np.sum(perplexities2)/6


import itertools
def get_best_plex(df,ix, shift=0, rev=False, tf=False):
    print('SHIFT: ', shift, ' Score: ', df.loc[ix,'score'])
    for r in [ix]:
        if rev: shift+=1
        if len(df['text'][r].split(' ')) > shift:
            bGood = True
            while bGood == True:
                bGood = False
                t = df['text'][r].split(' ')
                if rev: shift *= -1
                last = [t[shift]]
                del t[shift]
                t = t + last
                best = df['score'][r]
                for x in range(len(t)-1):
                    new = t[:x] + last + t[x:-1]
                    new = ' '.join(new)
                    s = scorer.get_perplexity(new, batch_size=BATCH_SIZE)
                    if s < best:
                        bGood = True
                        best = s
                        df.at[r, 'score'] = s
                        df.at[r, 'text'] = new
                        print(r, x, "New Score: ", s)
                        print(new)
                        if tf: break #take first shift
    return df




def simulated_annealing_optimize_nb3(text: str, temp_start=6.0, temp_end=1.0, cooling_rate=0.2,
                                     n_neighbor=2, steps_per_temp=4, verbose=False, seq_to_choose=None,
                                     factor_acceptance=3,
                                     shuffle_start: bool = False, shuffle_pos=[], check_komb=True):


    words = text.split()
    current = words[:]
    current_score = scorer.get_perplexity(' '.join(current), batch_size=BATCH_SIZE)
    if check_komb:
        res=komb_words.loc[komb_words["text"]==text]
        if len(res)>0:
            print(f"Kombination has been looked before")
            return text,current_score
    while math.isnan(current_score):
        random.shuffle(current)
        current_score = scorer.get_perplexity(' '.join(current), batch_size=BATCH_SIZE)
    print(f"current={' '.join(current)}: {current_score}")
    if shuffle_start:
        shuffle_pos = [i for i in shuffle_pos if i < len(current)]
        print(f"Shuffle positions: {shuffle_pos}")
        shuffled_values = random.sample([current[i] for i in shuffle_pos], len(shuffle_pos))
        for i, pos in enumerate(shuffle_pos):
            current[pos] = shuffled_values[i]
        current_score = scorer.get_perplexity(' '.join(current), batch_size=BATCH_SIZE)
        print(f"After shuffle: current={' '.join(current)}: {current_score}")
    best = current[:]
    best_score = current_score
    temp = temp_start

    seq_no_k = seq_to_choose if seq_to_choose is not None else list(range(len(words)))
    print(f"Seq to choose from: {seq_no_k}")
    while temp > temp_end:
        for _ in range(steps_per_temp):

            indices = random.sample(seq_no_k, min(n_neighbor, len(seq_no_k)))
            if n_neighbor == 2:
                neighbor = current[:]
                neighbor[indices[0]], neighbor[indices[1]] = neighbor[indices[1]], neighbor[indices[0]]
                neighbor_score = scorer.get_perplexity(' '.join(neighbor), batch_size=BATCH_SIZE)
            elif n_neighbor==3:
                neighbor = current[:]
                neighbor_variants = []
                for perm in [(1, 0, 2), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 2, 0)]:
                    variant = current[:]
                    variant[indices[0]], variant[indices[1]], variant[indices[2]] = \
                        neighbor[indices[perm[0]]], neighbor[indices[perm[1]]], neighbor[indices[perm[2]]]
                    neighbor_variants.append(' '.join(variant))

                #perplexities = [scorer.get_perplexity(text, batch_size=BATCH_SIZE) for text in neighbor_variants]
                perplexities = scorer.get_perplexity(neighbor_variants, batch_size=BATCH_SIZE)
                best_variant_idx = perplexities.index(min(perplexities))
                neighbor = neighbor_variants[best_variant_idx].split()
                neighbor_score = scorer.get_perplexity(' '.join(neighbor), batch_size=BATCH_SIZE)
            else:
                print("n_neigbor must be either 2 or 3")
                return 0
            if math.isnan(neighbor_score):
                continue

            delta = neighbor_score - current_score
            acceptance_prob = math.exp(-delta / temp) * factor_acceptance
            if delta < 0 or random.random() < acceptance_prob:
                #print(f"acceptance_prob: {acceptance_prob}")
                current, current_score = neighbor[:], neighbor_score
                #if verbose: print(f"Temperature: {temp:.2f}, Current score:{current} {current_score:.2f}")
                #if verbose: print(f"Temperature: {temp:.2f}, Neigbor score: {neighbor_score:.2f}")
                if current_score < best_score:
                    best, best_score = current[:], current_score
                    if verbose>0:
                        print(f"New best: {' '.join(best)}")
                        print(f"New best score: {best_score:.2f}")

        temp -= cooling_rate
        if verbose>0:
            print(f"Temperature: {temp:.2f}, Current: {' '.join(current)}")
            print(f"Temperature: {temp:.2f}, Current score: {current_score:.2f}")


    if verbose:
        print(f"Final score: {best_score:.2f}")
    return ' '.join(best), best_score


def part_perm_brute_lastf(start_word, psize=3):
    """
    Efficiently finds the best permutation of the last part of the input string
    to minimize perplexity.
    """
    # Initial best word and score
    best_word = start_word
    best_score = scorer.get_perplexity(start_word, batch_size=BATCH_SIZE)
    print(f"Start score: {best_score}")

    # Split input into parts
    words = start_word.split(' ')
    rest = ' '.join(words[:psize]) + ' '
    part = words[psize:]

    import pandas as pd



    # Early exit if there are too many permutations
    if len(part) > 10:
        print("Too many permutations to process efficiently.")
        return start_word



    print(f"There are {math.factorial(len(part))} permutations.")
    # Iterate over permutations
    for i, perm in enumerate(itertools.permutations(part)):
        # Create the new word order
        new_word = rest + ' '.join(perm)
        #kombis.loc[i,"text"]=new_word

        # Calculate perplexity for the new word order
        new_score = scorer.get_perplexity(new_word, batch_size=BATCH_SIZE)

        #kombis.loc[i,"score"]=new_score
        # Log progress every 1000 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Current best score = {best_score}")
            #print(f"Tested sequence: {new_word} score: {new_score}")

        # Update best score and word if an improvement is found
        if new_score < best_score:
            print(f"New best score found: {new_score}")
            print(f"New best sequence: {new_word}")
            best_score = new_score
            best_word = new_word


    return best_word, best_score


##collection poor starting points

texts = [
    'wreath wish bow doll hope joy peace dream believe wonder star candle night with in and the have of it from as to not you we that poinsettia snowglobe angel fruitcake cookie peppermint candy greeting card fireplace kaggle puzzle game season merry hohoho chocolate milk eggnog toy workshop wrapping paper',
    'merry hohoho the season of eggnog and fruitcake greeting card kaggle workshop from with to in as that we you it have not toy doll puzzle game night chocolate milk cookie peppermint candy candle wreath snowglobe fireplace poinsettia wrapping paper bow angel star wish dream wonder believe hope joy peace'
]
komb_words=pd.DataFrame({'text':['']*(len(texts)+10), 'score':[1000.0]*(len(texts)+10)})
max_text=len(texts)
for idx, t in enumerate(texts):
    komb_words.loc[idx, "text"] = t


for ix in range(len(texts)):
    komb_words.loc[ix,"score"]=scorer.get_perplexity(komb_words.loc[ix,"text"], batch_size=BATCH_SIZE)
#text1=text
#komb_words.loc[komb_words["text"]==text1]
#komb_words.loc[0,"score"]=np.round(score,5)
komb_words.loc[:len(texts),:]


##initial solution
txt1=samples.loc[4,"text"]
scorer.get_perplexity(txt1,batch_size=BATCH_SIZE)


## sample 4
ii=4

## setting seq_to_choose to list(range(21,50)) 
#txt1=samples.loc[ii,"text"]
temp_text, temp_score=simulated_annealing_optimize_nb3(text=txt1, temp_start=6.0,
                                                    temp_end=1.0, cooling_rate=0.2,
                                n_neighbor=3,
                                  steps_per_temp=5, verbose=2, seq_to_choose=list(range(22,50)),
                                                           shuffle_start=False,
                                shuffle_pos=range(21,50), factor_acceptance=0.15,check_komb=True)


## I get a new good starting point
txt1='from and of to the as in that it we with not you have milk chocolate candy peppermint eggnog cookie fruitcake toy doll puzzle game night season greeting card wrapping paper bow wreath poinsettia candle fireplace snowglobe angel star wish wonder dream believe hope joy peace merry hohoho kaggle workshop'
scorer.get_perplexity(txt1,batch_size=BATCH_SIZE)


ii=4
samples_new=samples.copy()
samples_new.loc[ii,"text"]=txt1
samples_new['score'] = samples_new['text'].map(lambda x: scorer.get_perplexity(x, batch_size=BATCH_SIZE))
for sh in range(6,10):
    df = get_best_plex(df=samples_new, ix=ii, shift=sh, rev=True, tf=True)



def find_best_reorder(input_str, index):
    words = input_str.split()
    if index < 0 or index >= len(words):
        raise ValueError("Index is out of range")

    best_sentence = input_str
    best_perplexity = scorer.get_perplexity(input_str, 4)
    #print(f"Start perplexity: {best_perplexity}, for sentence: {input_str}")

    word_to_move = words.pop(index)

    for i in range(len(words) + 1):
        #print(i)
        reordered_words = words[:]
        reordered_words.insert(i, word_to_move)
        reordered_sentence = ' '.join(reordered_words)

        #print(reordered_sentence)
        
        perplexity = scorer.get_perplexity(reordered_sentence, 4)

        if perplexity < best_perplexity:
            print(f"Checking permutation: {reordered_sentence} :with perplexity: {perplexity}")

def optimize_sentence(input_str):
    words = input_str.split()
    best_sentence = input_str
    best_perplexity = scorer.get_perplexity(input_str, 4)
    print(f"Initial perplexity: {best_perplexity}, for sentence: {input_str}")

    for index in range(len(words)):
        print(f"Index: {index}")
        find_best_reorder(best_sentence, index)


## I get a new good starting point
text_new = 'from and of to the as in that it we with not you have milk chocolate candy peppermint eggnog cookie fruitcake toy doll game puzzle greeting card wrapping paper bow candle wreath poinsettia snowglobe fireplace angel star wish dream night season wonder believe hope joy peace merry hohoho kaggle workshop'
optimize_sentence(text_new)


new_best_text = 'from and of to the as in that it we with not you have milk chocolate candy peppermint eggnog cookie fruitcake toy doll game puzzle greeting card wrapping paper bow candle fireplace wreath poinsettia snowglobe angel star wish dream night season wonder believe hope joy peace merry hohoho kaggle workshop'


new_best_text_5 = "from and and as we and have the in is it of not that the to with you advent card angel bake beard believe bow candy candle carol cheer cheer chocolate chimney cookie decorations doll dream drive eat eggnog elf family fireplace fireplace chimney fruitcake game give gifts gingerbread greeting grinch holiday holly hohoho hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night of the nutcracker ornament ornament wrapping paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wreath yuletide"


samples.loc[4,"text"]=new_best_text
samples.loc[5,"text"]=new_best_text_5
perplexities2 = []


for index, row in samples.iterrows(): # Step 1: Reorder the words based on POS tagging reordered_text = reorder_text(row["text"])
    score=scorer.get_perplexity(row['text'],batch_size=BATCH_SIZE)
    perplexities2.append(score)
    print(f"i={index}:{score}")
np.sum(perplexities2)/6


samples.to_csv("submission.csv", index=False)

