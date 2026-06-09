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


scorer = PerplexityCalculator('/kaggle/input/gemma-2/transformers/gemma-2-9b/2')


import random
import math
import numpy as np
from collections import deque
from rich.live import Live
from rich.table import Table
from rich.console import Console

def random_swap(words, rng):
    if len(words) < 2:
        return words
    w = words[:] 
    i, j = rng.sample(range(len(w)), 2)
    w[i], w[j] = w[j], w[i]
    return w

def rotate(words, rng):
    if len(words) < 3:
        return words
    w = words[:] 
    indices = rng.sample(range(len(w)), 3)
    i, j, k = indices[0], indices[1], indices[2]
    new_w = w[:] 
    new_w[i] = w[k]
    new_w[j] = w[i]
    new_w[k] = w[j]
    return new_w

def local_rotate(words, rng):
    if len(words) < 5:
        return words
    w = words[:] 
    start = rng.randint(0, len(w) - 3)
    max_block_len = min(7, len(w) - start)
    L = rng.randint(3, max_block_len)
    block_indices = list(range(start, start + L))
    i, j, k = rng.sample(block_indices, 3)
    temp = w[i]
    w[i] = w[k]
    w[k] = w[j]
    w[j] = temp
    return w

MOVES = [random_swap, rotate, local_rotate]
MOVE_WEIGHTS = [3, 1, 1] 

class SingleAnnealer:
    def __init__(self, state, name, batch_size, temp, Tmin, max_iter, seed=None):
        self.name = name
        self.state = state[:]      
        self.cur_e = float('inf')  
        self.bst_e = float('inf')   
        self.bst_state = state[:]  
        self.count = 0             
        self.iter = 0
        self.batch_size = batch_size
        self.start_temp = temp
        self.end_temp = Tmin
        self.max_iter = max_iter
        self.temp = temp
        self.rng = random.Random(seed if seed is not None else random.randint(0, 1000000))
        self.tabu_list = set()
        self.tabu_deque = deque(maxlen=1000)
        self.local_tabu = set()

        w = np.array(MOVE_WEIGHTS, dtype=float)
        self.move_probs = w / w.sum()

    def propose_moves(self):
        self.local_tabu.clear()
        candidates = []
        base = self.state[:]
        while len(candidates) < self.batch_size:
            move_fn = self.rng.choices(MOVES, weights=self.move_probs, k=1)[0]
            new_words = move_fn(base, self.rng)
            key = tuple(new_words)
            if key not in self.local_tabu and key not in self.tabu_list:
                candidates.append(new_words)
                self.local_tabu.add(key)
        return candidates

    def accept_move(self, best_candidate, best_e):
        dE = best_e - self.cur_e
        if dE < 0 or self.rng.random() < math.exp(-dE / self.temp):
            self.state = best_candidate
            self.cur_e = best_e
            self.tabu_list.add(tuple(best_candidate))
            self.tabu_deque.append(tuple(best_candidate))
            if best_e < self.bst_e:
                self.bst_e = best_e
                self.bst_state = best_candidate[:]
                self.count += 1

        self.temp = self._lower_temperature(self.temp, self.iter)

    def increment_iter(self):
        self.iter += 1

    def _lower_temperature(self, temperature, iteration):
        t1 = self.end_temp + self.start_temp / (1 + math.log(iteration + 1))
        t2 = self.start_temp + (self.end_temp - self.start_temp) * (iteration / self.max_iter)
        return max(t1, t2)

    def reinsert(self, scorer, update_fn=None, batch_size=32):
        improved = True
        while improved:
            improved = False
            for i in range(len(self.state)):
                candidate_state = self.state[:] 
                word = candidate_state.pop(i)    

                candidate_states = []
                for j in range(len(candidate_state) + 1):
                    new_state = candidate_state[:]  
                    new_state.insert(j, word)
                    candidate_states.append(new_state)

                candidate_texts = [" ".join(state) for state in candidate_states]
                energies = []

                for start in range(0, len(candidate_texts), batch_size):
                    batch_texts = candidate_texts[start:start + batch_size]
                    perplexities = scorer.get_perplexity(batch_texts, batch_size=len(batch_texts))
                    energies.extend([math.log(p) for p in perplexities])

                best_index = min(range(len(energies)), key=lambda idx: energies[idx])
                best_energy = energies[best_index]

                if best_energy < self.cur_e:
                    self.state = candidate_states[best_index][:]
                    self.cur_e = best_energy
                    if best_energy < self.bst_e:
                        self.bst_e = best_energy
                        self.bst_state = self.state[:]
                        self.count += 1
                    improved = True
                    if update_fn:
                        update_fn()  
                    break

class ParallelManager:
    def __init__(self, population, scorer, max_iter, swap_interval=100):
        self.population = population
        self.scorer = scorer
        self.max_iter = max_iter
        self.swap_interval = swap_interval  
        self.swaps = 0
        self.console = Console()
        self.live = Live(console=self.console, auto_refresh=False)
        
        self.global_best_e = float('inf')
        self.global_best_state = None

        self.rng = random.Random(random.randint(0, 1000000))

    def run(self):
        self.live.start()
        try:
            for iteration in range(1, self.max_iter + 1):
                all_candidates = []
                sizes = []
                for ann in self.population:
                    cands = ann.propose_moves()
                    all_candidates.extend(cands)
                    sizes.append(len(cands))
                
                text_batch = [" ".join(candidate) for candidate in all_candidates]
                energies = self.scorer.get_perplexity(text_batch, batch_size=len(text_batch))
                
                idx = 0
                for ann, sz in zip(self.population, sizes):
                    subset = energies[idx: idx + sz]
                    cands = all_candidates[idx: idx + sz]
                    idx += sz

                    best_i = np.argmin(subset)
                    best_state = cands[best_i]
                    best_e = math.log(subset[best_i])

                    ann.accept_move(best_state, best_e)
                    ann.increment_iter()
                
                for ann in self.population:
                    if ann.cur_e < self.global_best_e:
                        if iteration > 50:
                            ann.reinsert(
                                self.scorer,
                                update_fn=lambda: self._update_display(iteration)
                            )
                        self.global_best_e = ann.cur_e
                        self.global_best_state = ann.state[:]
                
                if iteration % self.swap_interval == 0:
                    self.parallel_tempering_swap()

                self._update_display(iteration)

                # JUST TO END
                if math.exp(self.global_best_e) < 470:
                    break
        
        except KeyboardInterrupt:
            self.live.stop()
        self.live.stop()

    def parallel_tempering_swap(self):
        for i in range(0, len(self.population) - 1, 2):
            a1, a2 = self.population[i], self.population[i+1]
            dE = a2.cur_e - a1.cur_e
            dT = (1 / a1.temp - 1 / a2.temp)
            swap_prob = min(1, math.exp(dE * dT))
            if self.rng.random() < swap_prob:
                a1.state, a2.state = a2.state, a1.state
                a1.cur_e, a2.cur_e = a2.cur_e, a1.cur_e
                a1.temp, a2.temp = a2.temp, a1.temp
                self.swaps += 1

    def _update_display(self, iteration):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Annealer")
        table.add_column("Iteration")
        table.add_column("Current Perplexity")
        table.add_column("Best Perplexity")
        table.add_column("Temperature")
        # table.add_column("Improvements")

        for ann in self.population:
            table.add_row(
                ann.name,
                str(ann.iter),
                f"{math.exp(ann.cur_e):.4f}",
                f"{math.exp(ann.bst_e):.4f}",
                f"{ann.temp:.6f}",
                # str(ann.count)
            )

        table.add_row(
            "GLOBAL",
            "-",
            "-",
            f"{math.exp(self.global_best_e):.8f}",
            "-"
        )
            
        self.live.update(table)
        self.live.refresh()


%%time

text ='advent chimney elf family fireplace gingerbread mistletoe ornament reindeer scrooge'
tokens = text.split()

Tmax = 0.70
Tmin = 0.01
N = 4
r = (Tmin / Tmax) ** (1 / (N - 1))
iterations = 1000

population = []

for i in range(N):
    np.random.shuffle(tokens)
    population.append(SingleAnnealer(
        tokens, name=f"ann-{i}", batch_size=1, temp=Tmax * r**i, Tmin=Tmin, max_iter=iterations
    ))
    
manager = ParallelManager(population, scorer, max_iter=iterations, swap_interval=10)
manager.run()

# --- done
print(' '.join(manager.global_best_state))
print(math.exp(manager.global_best_e))

