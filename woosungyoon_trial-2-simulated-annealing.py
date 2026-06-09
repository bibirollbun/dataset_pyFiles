import gc
import os
import time
import math
from math import exp
from collections import Counter
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import transformers
import torch
import heapq
from tqdm import tqdm

import random
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pprint import pprint


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HuggingFaceModelLoader:
    def __init__(self, model_path: str, load_in_4bit: bool, device_map: str):
        self.model_path = model_path
        self.load_in_8bit = load_in_4bit
        self.device_map = device_map

    def load_model(self) -> transformers.PreTrainedModel:
        if self.load_in_8bit:
            if DEVICE.type != 'cuda':
                raise ValueError('8-bit quantization requires a CUDA device')

            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_use_double_quant=False,
                bnb_4bit_compute_dtype=torch.float16,
            )

            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map=self.device_map
            )
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map=self.device_map
            )

        model.eval()
        return model


class HuggingFaceTokenizer:
    def __init__(self, model_path: str):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, padding_side="right")
        self.bos_token = self.tokenizer.bos_token or self.tokenizer.cls_token
        self.eos_token = self.tokenizer.eos_token or self.tokenizer.sep_token
        if self.bos_token is None:
            self.bos_token = ""
        if self.eos_token is None:
            self.eos_token = ""

    def tokenize(self, texts: List[str]) -> dict:
        processed_texts = []

        for text in texts:
            combined_text = f"{self.bos_token}{text}{self.eos_token}"
            processed_texts.append(combined_text)

        model_inputs = self.tokenizer(
            processed_texts,
            return_tensors='pt',
            add_special_tokens=False,
            padding=True
        )

        if 'token_type_ids' in model_inputs:
            model_inputs.pop('token_type_ids')

        return model_inputs


class PerplexityCalculator:
    def __init__(self, model_loader, tokenizer, exp_mode=False):
        self.model = model_loader.load_model()
        self.tokenizer = tokenizer
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        self.exp_mode = exp_mode
    
    def get_perplexity(
        self,
        input_texts: Union[str, List[str]],
        batch_size: int = 32
    ) -> Union[float, List[float]]:

        single_input = isinstance(input_texts, str)
        input_texts = [input_texts] if single_input else input_texts

        loss_list = []
        num_texts = len(input_texts)
        batches = num_texts // batch_size + (num_texts % batch_size != 0)
        with torch.no_grad():
            for j in range(batches):
                start_idx = j * batch_size
                end_idx = (j + 1) * batch_size
                input_batch = input_texts[start_idx:end_idx]

                sequence_loss = self._compute_sequence_loss(input_batch)
                loss_list.extend(sequence_loss)

        if self.exp_mode:
            ppl = [exp(i) for i in loss_list]
        else:
            ppl = loss_list

        return ppl[0] if single_input else ppl

    def _compute_sequence_loss(self, input_batch):
        with torch.no_grad():
            model_inputs = self.tokenizer.tokenize(input_batch)
            model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}

            output = self.model(**model_inputs, use_cache=False)
            logits = output['logits']

            label = model_inputs['input_ids']
            if hasattr(self.model.config, 'pad_token_id') and self.model.config.pad_token_id is not None:
                label[label == self.model.config.pad_token_id] = PAD_TOKEN_LABEL_ID

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = label[..., 1:].contiguous()

            token_loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(len(logits), -1)

            valid_length = (shift_labels != PAD_TOKEN_LABEL_ID).sum(dim=-1)
            sequence_loss = torch.sum(token_loss, -1) / valid_length
        return sequence_loss.cpu().tolist()


model_path = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"
model_loader = HuggingFaceModelLoader(model_path=model_path, load_in_4bit=False, device_map='auto')
tokenizer = HuggingFaceTokenizer(model_path)
scorer = PerplexityCalculator(model_loader, tokenizer)


import random
from typing import List

class TSPListOperations:
    @staticmethod
    def two_rotation(item: List[str]) -> List[str]:
        i, j = random.sample(range(len(item)), 2)
        # Rotate two elements: (i, j) -> (j, i)
        item[i], item[j] = item[j], item[i]
        return item

    @staticmethod
    def point_insertion(item: List[str]) -> List[str]:
        i = random.randrange(len(item))
        elem = item.pop(i)
        j = random.randrange(len(item) + 1)
        item.insert(j, elem)
        return item

    @staticmethod
    def block_insertion(item: List[str]) -> List[str]:
        block_size = random.randrange(2, min(5, len(item)) + 1)
        start = random.randrange(len(item) - block_size + 1)
        block = item[start:start + block_size]
        del item[start:start + block_size]
        insert_pos = random.randrange(len(item) + 1)
        item = item[:insert_pos] + block + item[insert_pos:]
        return item

    @staticmethod
    def block_reverse(item: List[str]) -> List[str]:
        start = random.randrange(len(item) - 2)
        end = random.randrange(start + 1, min(start + 5, len(item)))
        item[start:end + 1] = reversed(item[start:end + 1])
        return item

    @staticmethod
    def random_rotation(item: List[str]) -> List[str]:
        if len(item) < 2:
            return item

        block_size = random.randrange(2, min(8, len(item)) + 1)
        start = random.randrange(len(item) - block_size + 1)
        sublist = item[start:start + block_size]

        shift = random.randint(1, len(item) - 1)
        shift = shift % len(sublist)
        sublist = sublist[shift:] + sublist[:shift]

        item[start:start + block_size] = sublist

        return item



class SimulatedAnnealing:
    def __init__(self, start_temp, end_temp, max_iterations, cost_fn, prefix=0, suffix=0):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.max_iterations = max_iterations
        self.cost_fn = cost_fn
        self.prefix = prefix
        self.suffix = suffix

    def _generate_neighbor(self, solution):
        """
        Generate a neighboring solution by randomly modifying the word sequence.
        """
        item = solution.split()
        operations = [TSPListOperations.two_rotation,
                      TSPListOperations.point_insertion,
                      TSPListOperations.block_insertion,
                      TSPListOperations.block_reverse,
                      TSPListOperations.random_rotation
                      ]

        op = random.choice(operations)
        item_modified = item[:self.prefix] + op(item[self.prefix:len(item)-self.suffix]) + item[len(item)-self.suffix:]
        return ' '.join(item_modified)


    def _acceptance_probability(self, diff, temperature):
        if diff <= 0:
            return 1.0
        return math.exp(- diff/ temperature)

    def _lower_temperature(self, temperature, iteration):
        t1 = self.end_temp + self.start_temp/(1 + math.log(iteration+1))
        t2 = self.start_temp + (self.end_temp - self.start_temp)*(iteration/self.max_iterations)
        return max(t1, t2)

    def _print_progress(
        self,
        iteration: int,
        best_solutions: List[List[str]],
        best_energies: List[float],
        current_solutions: List[List[str]],
        current_energies: List[float],
        temperature: float,
        start_time: float,
        spend_minute: int
    ) -> int:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        elapsed_time = time.time() - start_time

        # Check if 60 seconds have passed since the last update
        if elapsed_time - 60 * spend_minute > 60:
            spend_minute += 1
            progress = iteration / self.max_iterations * 100  # Progress as percentage

            # Print progress in a structured format
            print("===== Simulated Annealing Progress =====")
            print(f"Time: {current_time}")
            print(f"Iteration: {iteration}/{self.max_iterations} ({progress:.2f}%)")
            print(f"Temperature: {temperature:.4f}")
            print(f"Elapsed Time: {elapsed_time:.2f} seconds")

            # Print best solutions and energies
            print("\nBest Solutions:")
            for i, solution in enumerate(best_solutions):
                print(f"  Solution {i+1}: {solution}")
            print("\nBest Energies:")
            print("  " + ", ".join(f"{exp(energy):.4f}" for energy in best_energies))

            # Print current solutions and energies
            print("\nCurrent Solutions:")
            for i, solution in enumerate(current_solutions):
                print(f"  Solution {i+1}: {solution}")
            print("\nCurrent Energies:")
            print("  " + ", ".join(f"{exp(energy):.4f}" for energy in current_energies))

            print("========================================\n")

        return spend_minute

    def solve_batch(self, text_list):
        """
        Perform Simulated Annealing for multiple texts at once.
        """
        solutions = text_list[:]
        current_energies = self.cost_fn(solutions)

        best_solutions = solutions[:]
        best_energies = current_energies[:]

        log_energies = [[] for _ in range(len(text_list))]
        for i in range(len(text_list)):
            log_energies[i].append(current_energies[i])

        temperature = self.start_temp
        start_time = time.time()
        spend_minute = 0

        for iteration in range(self.max_iterations):
            # 1) Generate neighbors
            new_solutions = [self._generate_neighbor(sol) for sol in solutions]

            # 2) Calculate new energies in batch
            new_energies = self.cost_fn(new_solutions)

            # 3) Acceptance and update
            for i in range(len(text_list)):

                diff = new_energies[i] - current_energies[i]
                ap = self._acceptance_probability(diff, temperature)

                if random.random() < ap:
                    solutions[i] = new_solutions[i]
                    current_energies[i] = new_energies[i]

                if current_energies[i] < best_energies[i]:
                    best_solutions[i] = solutions[i][:]
                    best_energies[i] = current_energies[i]

            # 4) Lower temperature
            temperature = self._lower_temperature(temperature, iteration)

            # 5) Log current energies
            for i in range(len(text_list)):
                log_energies[i].append(current_energies[i])

            # 6) Print progress (extracted into separate method)
            spend_minute = self._print_progress(
                iteration,
                best_solutions,
                best_energies,
                solutions,
                current_energies,
                temperature,
                start_time,
                spend_minute
            )

            # 7) Early stop if temperature is below threshold
            if temperature <= self.end_temp:
                print("Reached the minimum temperature. Exiting.")
                break

        print(f"Execution time: {time.time() - start_time:.4f}s")

        # Convert best solutions back to strings
        return best_solutions, best_energies, log_energies


from collections import OrderedDict

class CostFunction:
    """
    Wraps a 'scorer' object (which can calculate perplexity or similar)
    and provides a caching mechanism for repeated queries.
    """

    def __init__(self, scorer, batch_size=32, capacity: int = None):
        """
        scorer: An object with get_perplexity(input_texts, batch_size=...) method.
                Typically an instance of PerplexityCalculator.
        capacity: Maximum size of the LRU cache.
        """
        self.scorer = scorer
        self.capacity = capacity
        self.batch_size = batch_size
        self.cache = OrderedDict() 


    def _cache_get(self, key: str) -> Union[float, None]:
        if key in self.cache:
            self.cache.move_to_end(key)  
            return self.cache[key]
        return None

    def _cache_set(self, key: str, value: float):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if self.capacity and len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Evict LRU

    def __call__(self, input_texts: Union[str, List[str]], batch_size: int = 200) -> Union[float, List[float]]:
        """
        Returns a cost value (e.g., perplexity) for the given text(s).
        """
        single_input = isinstance(input_texts, str)
        texts = [input_texts] if single_input else input_texts

  
        new_texts = []
        for text in texts:
            new_texts.append(' '.join(text.split('#')).strip())
        texts = new_texts

        results = [None] * len(texts)
        to_compute = []

        for i, txt in enumerate(texts):
            txt_key = txt.strip()  
            cached_val = self._cache_get(txt_key)
            if cached_val is not None:
                results[i] = cached_val
            else:
                to_compute.append((i, txt_key))

        if not to_compute:
            return results[0] if single_input else results

        to_compute.sort(key=lambda x: x[0])
        pending_texts = [x[1] for x in to_compute]


        cost_values = scorer.get_perplexity(pending_texts, batch_size=batch_size)

        for (idx, txt_key), cost_val in zip(to_compute, cost_values):
            results[idx] = cost_val
            self._cache_set(txt_key, cost_val)

        return results[0] if single_input else results


cost_fn = CostFunction(scorer)


sa_params = {
    'start_temp': 0.01,             # Initial temperature
    'end_temp': 0.001,          # Final temperature, decreasing linearly
    'max_iterations': 40000,    # Number of iterations (approximately 4 hours for 100,000 iterations)
    "cost_fn": cost_fn,
    "prefix": 0,
    "suffix": 0
}

sa_optimizer = SimulatedAnnealing(**sa_params)


solutions = ['initial_solutions']


best_solutions, best_energies, log_scores = sa_optimizer.solve_batch(solutions)

