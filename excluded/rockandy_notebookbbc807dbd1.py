# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


"""Evaluation metric for Santa 2024."""

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
    load_in_8bit: bool = False,
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

    model_path : str, default='/kaggle/input/gemma-2/transformers/gemma-2-9b/2'
        Path to the serialized LLM.

    load_in_8bit : bool, default=False
        Use 8-bit quantization for the model. Requires CUDA.

    clear_mem : bool, default=False
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
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        # Configure model loading based on quantization setting and device availability
        if load_in_8bit:
            if DEVICE.type != 'cuda':
                raise ValueError('8-bit quantization requires CUDA device')
            quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
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
        self, input_texts: Union[str, List[str]], debug=False
    ) -> Union[float, List[float]]:
        """
        Calculates the perplexity of given texts.

        Parameters
        ----------
        input_texts : str or list of str
            A single string or a list of strings.

        batch_size : int, default=None
            Batch size for processing. Defaults to the number of input texts.

        debug : bool, default=False
            Print debugging information.

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
        with torch.no_grad():
            # Process each sequence independently
            for text in input_texts:
                # Explicitly add sequence boundary tokens to the text
                text_with_special = f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}"

                # Tokenize
                model_inputs = self.tokenizer(
                    text_with_special,
                    return_tensors='pt',
                    add_special_tokens=False,
                )

                if 'token_type_ids' in model_inputs:
                    model_inputs.pop('token_type_ids')

                model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}

                # Get model output
                output = self.model(**model_inputs, use_cache=False)
                logits = output['logits']

                # Shift logits and labels for calculating loss
                shift_logits = logits[..., :-1, :].contiguous()  # Drop last prediction
                shift_labels = model_inputs['input_ids'][..., 1:].contiguous()  # Drop first input

                # Calculate token-wise loss
                loss = self.loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                # Calculate average loss
                sequence_loss = loss.sum() / len(loss)
                loss_list.append(sequence_loss.cpu().item())

                # Debug output
                if debug:
                    print(f"\nProcessing: '{text}'")
                    print(f"With special tokens: '{text_with_special}'")
                    print(f"Input tokens: {model_inputs['input_ids'][0].tolist()}")
                    print(f"Target tokens: {shift_labels[0].tolist()}")
                    print(f"Input decoded: {self.tokenizer.decode(model_inputs['input_ids'][0])}")
                    print(f"Target decoded: {self.tokenizer.decode(shift_labels[0])}")
                    print(f"Individual losses: {loss.tolist()}")
                    print(f"Average loss: {sequence_loss.item():.4f}")

        ppl = [exp(i) for i in loss_list]

        if debug:
            print("\nFinal perplexities:")
            for text, perp in zip(input_texts, ppl):
                print(f"Text: '{text}'")
                print(f"Perplexity: {perp:.2f}")

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


# initialize score
model_name='/kaggle/input/gemma-2/transformers/gemma-2-9b/2'
scorer = PerplexityCalculator(model_name)


sample_submission = pd.read_csv('/kaggle/input/santa-2024/sample_submission.csv')
texts = sample_submission.loc[:, 'text']
texts


scores = scorer.get_perplexity(texts)
scores


import random
import matplotlib.pyplot as plt
class AntColonyOptimization:
    def __init__(self, num_ants, max_iterations, evaporation_rate, pheromone_deposit_factor, initial_heuristic_importance, initial_exploration_importance):
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit_factor = pheromone_deposit_factor
        self.initial_heuristic_importance = initial_heuristic_importance
        self.initial_exploration_importance = initial_exploration_importance

    def _initialize_pheromones(self, num_words):
        return [[1.0 for _ in range(num_words)] for _ in range(num_words)]

    def _calculate_probabilities(self, current_word, unvisited, pheromones, heuristics, heuristic_importance, exploration_importance):
        probabilities = []
        denominator = 0.0
        for word in unvisited:
            numerator = (pheromones[current_word][word] ** heuristic_importance) * (heuristics[current_word][word] ** exploration_importance)
            probabilities.append(numerator)
            denominator += numerator
        probabilities = [p / denominator for p in probabilities]
        return probabilities

    def _construct_tour(self, words, pheromones, heuristics, heuristic_importance, exploration_importance):
        tour = []
        visited = set()
        current_word = random.randint(0, len(words) - 1)
        tour.append(current_word)
        visited.add(current_word)
        
        while len(tour) < len(words):
            unvisited = [i for i in range(len(words)) if i not in visited]
            probabilities = self._calculate_probabilities(current_word, unvisited, pheromones, heuristics, heuristic_importance, exploration_importance)
            next_word = random.choices(unvisited, weights=probabilities)[0]
            tour.append(next_word)
            visited.add(next_word)
            current_word = next_word
        
        return tour

    def _update_pheromones(self, pheromones, tours, energies):
        pheromones = [[pheromone * (1 - self.evaporation_rate) for pheromone in row] for row in pheromones]
        for tour, energy in zip(tours, energies):
            for i in range(len(tour)):
                j = (i + 1) % len(tour)
                pheromones[tour[i]][tour[j]] += self.pheromone_deposit_factor / energy
        
        return pheromones

    def _fitness(self, solution, scorer):
        return scorer.get_perplexity(' '.join(solution))

    def _local_search(self, solution, scorer):
        n = len(solution)
        for _ in range(10):  # Perform a limited number of local search steps
            i, j = random.sample(range(n), 2)
            new_solution = solution[:]
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            new_energy = self._fitness(new_solution, scorer)
            if new_energy < self._fitness(solution, scorer):
                solution = new_solution
        return solution

    def solve(self, text, scorer):
        words = text.split()
        num_words = len(words)
        pheromones = self._initialize_pheromones(num_words)
        heuristics = [[1.0 / abs(i - j) if i != j else 0.0 for j in range(num_words)] for i in range(num_words)]
        
        best_solution = None
        best_energy = float('inf')
        scores = []  # List to store scores for plotting
        
        for iteration in range(self.max_iterations):
            # Dynamically adjust heuristic and exploration importance
            heuristic_importance = self.initial_heuristic_importance + (iteration / self.max_iterations) * (1 - self.initial_heuristic_importance)
            exploration_importance = self.initial_exploration_importance + (1 - iteration / self.max_iterations) * (1 - self.initial_exploration_importance)
            
            tours = []
            energies = []
            for _ in range(self.num_ants):
                tour = self._construct_tour(words, pheromones, heuristics, heuristic_importance, exploration_importance)
                solution = [words[word] for word in tour]
                energy = self._fitness(solution, scorer)
                tours.append(tour)
                energies.append(energy)
                
                if energy < best_energy:
                    best_solution = solution[:]
                    best_energy = energy
            
            # Local search on the best solution found so far
            best_solution = self._local_search(best_solution, scorer)
            best_energy = self._fitness(best_solution, scorer)
            
            pheromones = self._update_pheromones(pheromones, tours, energies)
            scores.append(best_energy)  # Append the best energy of this iteration
            print(f"Iteration {iteration + 1}/{self.max_iterations}: Best Energy = {best_energy}")
        
        return ' '.join(best_solution), best_energy


# text = "advent chimney elf family fireplace gingerbread mistletoe ornament reindeer scrooge"
# aco = AntColonyOptimization(num_ants=30, max_iterations=100, evaporation_rate=0.6, pheromone_deposit_factor=1.0, initial_heuristic_importance=0.1, initial_exploration_importance=0.9)
# best_solution, best_energy = aco.solve(text, scorer)
# plt.plot(scores, label=f'Text ID: 1')
# plt.title(f'Score Evolution for Text ID: 1')
# plt.xlabel('Iteration')
# plt.ylabel('Score')
# plt.legend()
# plt.grid(True)
# plt.show()
# print("Best Solution:", best_solution)
# print("Best Energy:", best_energy)


texts[0]
output_file='submission.csv'
aco = AntColonyOptimization(num_ants=50, max_iterations=200, evaporation_rate=0.5, pheromone_deposit_factor=1.0, initial_heuristic_importance=0.1, initial_exploration_importance=0.9)
results = []

for index, row in sample_submission.iterrows():
    id_value = row['id']
    text = row['text']
    best_solution, best_energy = aco.solve(text, scorer)
    results.append({'id': id_value, 'text': best_solution})
    
    # plt.plot(scores, label=f'Text ID: {id_value}')
    # plt.title(f'Score Evolution for Text ID: {id_value}')
    # plt.xlabel('Iteration')
    # plt.ylabel('Score')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

results_df = pd.DataFrame(results)

results_df.to_csv(output_file, index=False)


sub = pd.read_csv('/kaggle/working/submission.csv')
text1 = sub.loc[:, 'text']
text1


scores = scorer.get_perplexity(text1)
scores


#此部分已废弃
# import random

# class AntColonyOptimization:
#     def __init__(self, num_ants, max_iterations, evaporation_rate, pheromone_deposit_factor, heuristic_importance, exploration_importance):
#         self.num_ants = num_ants
#         self.max_iterations = max_iterations
#         self.evaporation_rate = evaporation_rate
#         self.pheromone_deposit_factor = pheromone_deposit_factor
#         self.heuristic_importance = heuristic_importance
#         self.exploration_importance = exploration_importance

#     def _initialize_pheromones(self, num_words):
#         return [[1.0 for _ in range(num_words)] for _ in range(num_words)]

#     def _calculate_probabilities(self, current_word, unvisited, pheromones, heuristics):
#         probabilities = []
#         denominator = 0.0
#         for word in unvisited:
#             numerator = (pheromones[current_word][word] ** self.heuristic_importance) * (heuristics[current_word][word] ** self.exploration_importance)
#             probabilities.append(numerator)
#             denominator += numerator
#         probabilities = [p / denominator for p in probabilities]
#         return probabilities

#     def _construct_tour(self, words, pheromones, heuristics):
#         tour = []
#         visited = set()
#         current_word = random.randint(0, len(words) - 1)
#         tour.append(current_word)
#         visited.add(current_word)
        
#         while len(tour) < len(words):
#             unvisited = [i for i in range(len(words)) if i not in visited]
#             probabilities = self._calculate_probabilities(current_word, unvisited, pheromones, heuristics)
#             next_word = random.choices(unvisited, weights=probabilities)[0]
#             tour.append(next_word)
#             visited.add(next_word)
#             current_word = next_word
        
#         return tour

#     def _update_pheromones(self, pheromones, tours, energies):
#         pheromones = [[pheromone * (1 - self.evaporation_rate) for pheromone in row] for row in pheromones]
#         for tour, energy in zip(tours, energies):
#             for i in range(len(tour)):
#                 j = (i + 1) % len(tour)
#                 pheromones[tour[i]][tour[j]] += self.pheromone_deposit_factor / energy
        
#         return pheromones

#     def _fitness(self, solution, scorer):
#         return scorer.get_perplexity(' '.join(solution))

#     def solve(self, text, scorer):
#         words = text.split()
#         num_words = len(words)
#         pheromones = self._initialize_pheromones(num_words)
#         heuristics = [[1.0 / abs(i - j) if i != j else 0.0 for j in range(num_words)] for i in range(num_words)]
        
#         best_solution = None
#         best_energy = float('inf')
        
#         for iteration in range(self.max_iterations):
#             heuristic_importance = self.initial_heuristic_importance + (iteration / self.max_iterations) * (1 - self.initial_heuristic_importance)
#             exploration_importance = self.initial_exploration_importance + (1 - iteration / self.max_iterations) * (1 - self.initial_exploration_importance)
#             tours = []
#             energies = []
#             for _ in range(self.num_ants):
#                 tour = self._construct_tour(words, pheromones, heuristics)
#                 solution = [words[word] for word in tour]
#                 energy = self._fitness(solution, scorer)
#                 tours.append(tour)
#                 energies.append(energy)
                
#                 if energy < best_energy:
#                     best_solution = solution[:]
#                     best_energy = energy
            
#             pheromones = self._update_pheromones(pheromones, tours, energies)
        
#         return ' '.join(best_solution), best_energy

