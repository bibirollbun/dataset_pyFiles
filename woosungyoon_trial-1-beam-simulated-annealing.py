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


from dataclasses import dataclass

@dataclass
class AlgorithmParameters:
    p0: float = 0.8
    Lmax: int = 50
    beam_width: int = 150
    iterations: int = 50
    no_improve_limit: int = 5
    branch_interval: int = 10
    rho_target: float = 0.5
    lambda_m: float = 5
    M_min: int = 5
    M_max: int = 30
    verbose: bool = True
    cooling_rate: float = 0.95
    num_trials: int = 1

    num_binary_search_iterations: int = 100
    max_initial_trials: int = 5
    initial_temperature_scale_factor: float = 100.0



class TemperatureManager:
    def __init__(self,
                 cost_fn: Callable[[Union[Any, List[Any]]], Union[float, List[float]]],
                 neighbor_fn: Callable[[Any], Any],
                 params: AlgorithmParameters):

        self.init_accept_prob = params.p0
        self.max_uphill_diffs = params.Lmax
        self.cost_fn = cost_fn
        self.neighbor_fn = neighbor_fn
        self.num_binary_search_iterations = params.num_binary_search_iterations
        self.max_initial_trials = params.max_initial_trials
        self.initial_temperature_scale_factor = params.initial_temperature_scale_factor

    def _collect_uphill_cost_differences(self, current_solutions: List[Any], current_costs: List[float]) -> Union[List[float], List[Any], List[float]]:
        improved_solutions = current_solutions[:]
        improved_costs = current_costs[:]
        collected_uphill_diffs = []

        for _ in range(self.max_initial_trials):
            neighbor_solutions = [self.neighbor_fn(sol) for sol in improved_solutions]
            neighbor_solutions_costs = self.cost_fn(neighbor_solutions)

            for idx, (neighbor_sol, neighbor_cost) in enumerate(zip(neighbor_solutions, neighbor_solutions_costs)):
                cost_diff = neighbor_cost - improved_costs[idx]

                # Collect uphill differences
                if cost_diff > 0:
                    collected_uphill_diffs.append(cost_diff)

                # If improvement is found, update solution and cost
                if neighbor_cost < improved_costs[idx]:
                    improved_solutions[idx] = neighbor_sol
                    improved_costs[idx] = neighbor_cost

            if len(collected_uphill_diffs) >= self.max_uphill_diffs:
                break

        return collected_uphill_diffs, improved_solutions, improved_costs

    def _calc_temperature_value(self, temp: float, uphill_diffs: List[float], num_diffs: int) -> float:
        epsilon = 1e-12
        avg_exp = sum(math.exp(-diff / (temp + epsilon)) for diff in uphill_diffs) / num_diffs
        return avg_exp - self.init_accept_prob

    def _estimate_initial_temperature(self, uphill_diffs: List[float]) -> float:
        num_uphill_diffs = len(uphill_diffs)
        if num_uphill_diffs == 0:
            return -1.0 / math.log(self.init_accept_prob + 1e-12)

        epsilon = 1e-12
        max_uphill_diff = max(uphill_diffs)
        low = epsilon
        high = max_uphill_diff * self.initial_temperature_scale_factor

        for _ in range(self.num_binary_search_iterations):
            mid = (low + high) / 2.0
            val = self._calc_temperature_value(mid, uphill_diffs, num_uphill_diffs)
            if val > 0:
                high = mid
            else:
                low = mid

        return (low + high) / 2.0

    def get_initial_temperature(self, current_solutions: List[Any], current_costs: List[float]) -> Union[float, List[Any], List[float]]:
        if isinstance(current_solutions, str):
            current_solutions = [current_solutions]
        if isinstance(current_costs, float):
            current_costs = [current_costs]

        uphill_diffs, improved_solutions, improved_costs = self._collect_uphill_cost_differences(current_solutions, current_costs)

        if len(uphill_diffs) == 0:
            uphill_diffs = [1.0] * self.max_uphill_diffs

        initial_temp = self._estimate_initial_temperature(uphill_diffs)

        return initial_temp, improved_solutions, improved_costs


class BeamManager:
    def __init__(self, params: AlgorithmParameters):
        self.params = params
        self.beam_width = self.params.beam_width
        self.beam: List[tuple[Any, float]] = []
        self.branch_archives: List[List[tuple[Any,float]]] = []
        self.no_improve_counts: List[int] = []
        self.prev_best_costs: List[float] = []

    def initialize_beam(self, initial_solutions: List[Any], cost_fn: Callable[[Any], float]):
        candidate_solutions = random.sample(initial_solutions, min(self.beam_width*2, len(initial_solutions)))
        candidate_costs = cost_fn(candidate_solutions)
        initial_beam = sorted(zip(candidate_solutions, candidate_costs), key=lambda x: x[1])[:self.beam_width]

        self.beam = [(sol, cost) for sol, cost in initial_beam]
        self.branch_archives = [[] for _ in range(len(self.beam))]
        self.no_improve_counts = [0]*len(self.beam)
        self.prev_best_costs = [c for (_, c) in self.beam]

    def record_branch_points(self, iteration: int):
        if iteration % self.params.branch_interval == 0:
            for i, (sol, cost) in enumerate(self.beam):
                self.branch_archives[i].append((sol, cost))

    def update_beam(self, candidates: List[tuple[Any,float]]):
        old_info = {
            sol: (nic, pbc, ba)
            for (sol, _), nic, pbc, ba in zip(self.beam,
                                              self.no_improve_counts,
                                              self.prev_best_costs,
                                              self.branch_archives)
        }
        unique_map = {}
        for sol, cost in self.beam + candidates:
            if sol not in unique_map or unique_map[sol] > cost:
                unique_map[sol] = cost

        sorted_candidates = sorted(unique_map.items(), key=lambda x: x[1])

        new_beam = [(k, v) for k, v in sorted_candidates[:self.beam_width]]

        new_info = [old_info.get(sol, (0, cost, [])) for sol, cost in new_beam]
        new_no_improve_counts, new_prev_best_costs, new_branch_archives = map(list, zip(*new_info))

        self.beam = new_beam
        self.no_improve_counts = new_no_improve_counts
        self.prev_best_costs = new_prev_best_costs
        self.branch_archives = new_branch_archives


    def check_improvement(self, i: int, best_local_cost: float) -> bool:
        improved = False
        if best_local_cost < self.prev_best_costs[i]:
            self.no_improve_counts[i] = 0
            improved = True
        else:
            self.no_improve_counts[i] += 1
        self.prev_best_costs[i] = best_local_cost
        return improved

    def return_to_branch_point(self, i: int):
        if self.no_improve_counts[i] > self.params.no_improve_limit and self.branch_archives[i]:
            chosen_point = min(self.branch_archives[i], key=lambda x: x[1])
            sol, cost = chosen_point
            self.beam[i] = (sol, cost)
            self.no_improve_counts[i] = 0
            if self.params.verbose:
                print(f"[Info] Beam member {i}: No improvement. Returning to best past branch point.")


class SAController:
    """
    Simulated Annealing Controller integrated with a beam manager and temperature manager.
    """
    def __init__(self,
                 R: List[Any],
                 prefix: int,
                 suffix: int,
                 cost_function: Callable[[List[Any]], List[float]],
                 neighbor_function: Callable[[Any, int, int], Any],
                 params: AlgorithmParameters):

        self.params = params
        self.beam_manager = BeamManager(self.params)
        self.beam_manager.initialize_beam(R, cost_fn=cost_function)

        self.temp_manager = TemperatureManager(
            cost_fn=cost_function,
            neighbor_fn=lambda x: neighbor_function(x, prefix, suffix),
            params=self.params
        )

        self.cost_function = cost_function
        self.prefix = prefix
        self.suffix = suffix
        self.neighbor_function = lambda x: neighbor_function(x, self.prefix, self.suffix)
        self.recent_improvements = [0]*5
        self.M = 10
        self.cooling_rate = self.params.cooling_rate
        self.num_trials = self.params.num_trials

        self.global_best_cost = float('inf')
        self.global_best_sol = None
        self.R = R

        # Example solution initialization
        example_sol = self.beam_manager.beam[0][0]
        if isinstance(example_sol, str):
            elements = example_sol.split()
        else:
            elements = example_sol
        # Removed optimizer initialization

    def adjust_M(self):
        improvement_rate = sum(self.recent_improvements)/len(self.recent_improvements)
        M_new = self.M + self.params.lambda_m*(self.params.rho_target - improvement_rate)
        self.M = max(self.params.M_min, min(self.params.M_max, int(M_new)))

    def get_acceptance_prob(self, diff, T):
        p = exp(-diff/T+1e-12)
        return p

    def run(self):
        for trial in range(self.num_trials):
            current_solutions = [sol for (sol, cost) in self.beam_manager.beam]
            current_costs = [cost for (sol, cost) in self.beam_manager.beam]
            T, current_solutions, current_costs = self.temp_manager.get_initial_temperature(
                current_solutions=current_solutions,
                current_costs=current_costs
            )
            step_k = 0

            self.M = self.params.M_min

            for it in range(self.params.iterations):
                iteration_improved = False
                self.beam_manager.record_branch_points(it)
                self.adjust_M()

                current_solutions = [sol for (sol, cost) in self.beam_manager.beam]
                current_costs = [cost for (sol, cost) in self.beam_manager.beam]

                for _ in range(self.M):

                    neighbors = [self.neighbor_function(sol) for sol in current_solutions]
                    neighbors_cost = self.cost_function(neighbors)

                    adjusted_neighbors_cost = neighbors_cost

                    for i in range(len(current_solutions)):
                        diff = adjusted_neighbors_cost[i] - current_costs[i]
                        if diff < 0:
                            current_solutions[i] = neighbors[i]
                            current_costs[i] = adjusted_neighbors_cost[i]
                        else:
                            p = self.get_acceptance_prob(diff, T)
                            if random.random() < p:
                                current_solutions[i] = neighbors[i]
                                current_costs[i] = adjusted_neighbors_cost[i]
                    T *= self.cooling_rate

                # Update beam
                candidates = []
                for i, (sol, old_cost) in enumerate(self.beam_manager.beam):
                    best_local_sol = current_solutions[i]
                    best_local_cost = current_costs[i]
                    improved = self.beam_manager.check_improvement(i, best_local_cost)
                    if not improved:
                        self.beam_manager.return_to_branch_point(i)
                    candidates.append((best_local_sol, best_local_cost))

                self.beam_manager.update_beam(candidates)

                # Update global best
                current_best_sol, current_best_cost = min(self.beam_manager.beam, key=lambda x:x[1])
                if current_best_cost < self.global_best_cost:
                    self.global_best_cost = current_best_cost
                    self.global_best_sol = current_best_sol
                    iteration_improved = True

                self.recent_improvements.pop(0)
                self.recent_improvements.append(1 if iteration_improved else 0)
                ir = sum(self.recent_improvements)/len(self.recent_improvements)


                if self.params.verbose:
                    print(f"Trial {trial}, Iteration {it}: T={T:.4f}, Best cost={exp(self.global_best_cost):.4f}, "
                          f"M={self.M}, improvement_rate={ir:.2f}")
                    pprint([(x[0], exp(x[1])) for x in self.beam_manager.beam[:3]])
                    print()

        current_solutions = [sol for (sol, cost) in self.beam_manager.beam]
        current_costs = [cost for (sol, cost) in self.beam_manager.beam]
        return current_solutions, current_costs



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


def neighbor_fn(arr: str, prefix: int = 0, suffix: int = 0) -> str:
    item = arr.split()
    operations = [
        TSPListOperations.two_rotation,
        TSPListOperations.point_insertion,
        TSPListOperations.block_insertion,
        TSPListOperations.block_reverse,
        TSPListOperations.random_rotation,
    ]

    op = random.choice(operations)
    item_modified = item[:prefix] + op(item[prefix:len(item)-suffix]) + item[len(item)-suffix:]
    return ' '.join(item_modified)


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




# Given parameter values from the user example:
p0 = 0.306
beam_width = 20
iterations = 30
cooling_rate = 0.985
num_trials = 40

num_binary_search_iterations = 100
max_initial_trials = 5
initial_temperature_scale_factor = 100.0

params = AlgorithmParameters(
    p0=p0,
    Lmax=50,
    beam_width=beam_width,
    iterations=iterations,
    no_improve_limit=20,
    branch_interval=20,
    rho_target=0.5,
    lambda_m=5,
    M_min=5,
    M_max=30,
    verbose=True,
    cooling_rate=cooling_rate,
    num_trials=num_trials,
    num_binary_search_iterations=num_binary_search_iterations,
    max_initial_trials=max_initial_trials,
    initial_temperature_scale_factor=initial_temperature_scale_factor
)



RR = ['initial_solutions']

cost_fn = CostFunction(scorer=scorer, batch_size=20, capacity=2000000)
prefix_length = 0
suffix_length = 0


sa = SAController(RR, prefix_length, suffix_length, cost_fn, neighbor_fn, params)
improved_beam = sa.run()
improved_beam

