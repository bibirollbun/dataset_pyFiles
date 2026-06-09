pip install transformers==4.45.1


import gc
import os
from math import exp
from collections import Counter
from typing import List, Optional, Union
import pickle
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

# LOAD GEMMA SCORER
scorer = PerplexityCalculator('/kaggle/input/gemma-2/transformers/gemma-2-9b/2')

past = {}
# You can comment out the following lines to run on a faster GPU with different scoring
with open('/kaggle/input/santa-2024-perplexity-permutation-puzzle-scores/past.pickle', 'rb') as handle:
    past = pickle.load(handle)
print(len(past))


import re, sys

class Reprinter:
    def __init__(self):
        self.text = ''

    def moveup(self, lines):
        for _ in range(lines):
            sys.stdout.write("\x1b[A")

    def reprint(self, text):
        # Clear previous text by overwritig non-spaces with spaces
        self.moveup(self.text.count("\n"))
        sys.stdout.write(re.sub(r"[^\s]", " ", self.text))

        # Print new text
        lines = min(self.text.count("\n"), text.count("\n"))
        self.moveup(lines)
        sys.stdout.write(text)
        self.text = text

reprinter = Reprinter()


words_0 = [
    'reindeer', 'mistletoe', 'elf', 'gingerbread', 'family', 'advent', 'scrooge', 'chimney', 
    'fireplace', 'ornament'
]
words_1 = [
    'reindeer', 'sleep', 'walk', 'the', 'night', 'and', 'drive', 'mistletoe', 'scrooge', 'laugh', 
    'chimney', 'jump', 'elf', 'bake', 'gingerbread', 'family', 'give', 'advent', 'fireplace', 
    'ornament'
]
words_2 = [
    'sleigh', 'yuletide', 'beard', 'carol', 'cheer', 'chimney', 'decorations', 'gifts', 'grinch', 
    'holiday', 'holly', 'jingle', 'magi', 'naughty', 'nice', 'nutcracker', 'ornament', 'polar', 
    'workshop', 'stocking'
]
words_3 = [
    'sleigh', 'of', 'the', 'magi', 'yuletide', 'cheer', 'is', 'unwrap', 'gifts', 'and', 'eat', 
    'cheer', 'holiday', 'decorations', 'holly', 'jingle', 'relax', 'sing', 'carol', 'visit', 
    'workshop', 'grinch', 'naughty', 'nice', 'chimney', 'stocking', 'ornament', 'nutcracker', 
    'polar', 'beard'
]
words_4 = [
    'from', 'and', 'of', 'to', 'the', 'as', 'in', 'that', 'it', 'we', 'with', 'not', 'you', 
    'have', 'milk', 'chocolate', 'candy', 'peppermint', 'eggnog', 'cookie', 'fruitcake', 'toy', 
    'doll', 'game', 'puzzle', 'greeting', 'card', 'wrapping', 'paper', 'bow', 'wreath', 'poinsettia', 
    'snowglobe', 'candle', 'fireplace', 'wish', 'dream', 'hope', 'believe', 'wonder', 'night', 
    'star', 'angel', 'peace', 'joy', 'season', 'merry', 'hohoho', 'kaggle', 'workshop'
]
words_5 = [
    'from', 'and', 'and', 'as', 'we', 'and', 'have', 'the', 'in', 'is', 'it', 'of', 'not', 
    'that', 'the', 'to', 'with', 'you', 'advent', 'card', 'angel', 'bake', 'beard', 'believe', 
    'bow', 'candy', 'candle', 'carol', 'cheer', 'cheer', 'chocolate', 'chimney', 'cookie', 
    'decorations', 'doll', 'dream', 'drive', 'eat', 'eggnog', 'family', 'fireplace', 'fireplace', 
    'chimney', 'fruitcake', 'game', 'gifts', 'give', 'gingerbread', 'greeting', 'grinch', 'holiday', 
    'holly', 'hohoho', 'hope', 'jingle', 'jump', 'joy', 'kaggle', 'laugh', 'magi', 'merry', 'milk', 
    'mistletoe', 'naughty', 'nice', 'night', 'night', 'elf', 'nutcracker', 'ornament', 'ornament', 
    'of', 'the', 'wrapping', 'paper', 'peace', 'peppermint', 'polar', 'poinsettia', 'puzzle', 
    'reindeer', 'relax', 'scrooge', 'season', 'sing', 'sleigh', 'sleep', 'snowglobe', 'star', 'stocking', 
    'toy', 'unwrap', 'visit', 'walk', 'wish', 'wonder', 'workshop', 'workshop', 'wreath', 'yuletide'
]






import json
import random
import pandas as pd
import time

# Assuming `score` function and PerplexityCalculator (scorer) are already available
iteration = 0

start_time = time.time()
import math

def simulated_annealing_v4(text: str, temp_start=6.0, temp_end=1.0, cooling_rate=0.2,
                           n_neighbor=2, steps_per_temp=4, verbose=False, seq_to_choose=None,
                           factor_acceptance=1.0):
    import math
    import random
    
    words = text.split()
    current = words[:]
    current_score = scorer.get_perplexity(' '.join(current), batch_size=BATCH_SIZE)
    max_retries = 10
    retries = 0

    while math.isnan(current_score) and retries < max_retries:
        random.shuffle(current)
        current_score = scorer.get_perplexity(' '.join(current), batch_size=BATCH_SIZE)
        retries += 1
    if math.isnan(current_score):
        raise ValueError("Unable to compute a valid perplexity score after retries.")

    best = current[:]
    best_score = current_score
    temp = temp_start

    seq_no_k = seq_to_choose if seq_to_choose is not None else list(range(len(words)))
    iteration = 0

    while temp > temp_end:
        for _ in range(steps_per_temp):
            if len(seq_no_k) < n_neighbor:
                n_neighbor = len(seq_no_k)

            indices = random.sample(seq_no_k, n_neighbor)
            neighbor = current[:]
            if n_neighbor == 2:
                neighbor[indices[0]], neighbor[indices[1]] = neighbor[indices[1]], neighbor[indices[0]]
            else:
                continue  # Simplify for now, extend as needed for larger n_neighbor
            
            neighbor_score = scorer.get_perplexity(' '.join(neighbor), batch_size=BATCH_SIZE)
            if math.isnan(neighbor_score):
                continue

            delta = neighbor_score - current_score
            acceptance_prob = math.exp(-delta / (temp * factor_acceptance))
            if delta < 0 or random.random() < acceptance_prob:
                current, current_score = neighbor[:], neighbor_score
                if current_score < best_score:
                    best, best_score = current[:], current_score
                    if verbose:
                        print(f"New best: {' '.join(best)} | Score: {best_score:.2f}")

        temp *= cooling_rate
        iteration += 1
        if verbose:
            print(f"Iteration {iteration}: Temp={temp:.2f}, Current Score={current_score:.2f}")

    return ' '.join(best), best_score


# Genetic Algorithm Functions
def create_population(size, words):
    """ Create an initial population of random permutations of the given words. """
    population = []
    for _ in range(size):
        random.shuffle(words)
        population.append(words[:])  # make a copy of the shuffled list
    return population

def calculate_fitness(population, solution_df, row_id_column_name):
    """ Calculate the fitness of each individual (lower perplexity is better). """
    fitness_scores = []
    for individual in population:
        # Create a string for the permuted text
        permuted_text = ' '.join(individual)
        if permuted_text in past:
            perplexity = past[permuted_text]
        else:
            perplexity = scorer.get_perplexity(permuted_text, 4)
            past[permuted_text] = perplexity

        # Calculate perplexity using the scorer (PerplexityCalculator)
        perplexity = scorer.get_perplexity(permuted_text, 4)
        fitness_scores.append(perplexity)  # lower perplexity is better
        
        # Optionally print iteration and perplexity (using reprinter)
        global iteration
        iteration += 1
        reprinter.reprint(f"Iteration: {iteration}, Perplexity: {perplexity:.2f}\r")

    return fitness_scores

def tournament_selection(population, fitness_scores, num_parents, tournament_size=3):
    selected_parents = []
    for _ in range(num_parents):
        # Tournament selection: pick a random sample of individuals and select the best one
        tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
        tournament.sort(key=lambda x: x[1])  # Sort by fitness (lower perplexity is better)
        selected_parents.append(tournament[0][0])  # Select the best
    return selected_parents
    
def roulette_wheel_selection(population, fitness_scores, num_parents=2):
    """ Perform roulette wheel selection to select parents. """
    total_fitness = sum(fitness_scores)
    selection_probs = [1 - (score / total_fitness) for score in fitness_scores]  # Lower perplexity -> higher probability
    selected_parents = random.choices(population, weights=selection_probs, k=num_parents)
    return selected_parents



def elitism(population, fitness_scores, num_elites=2):
    """ Select the best individuals (elites) to survive. """
    sorted_population = sorted(zip(population, fitness_scores), key=lambda x: x[1])
    elites = [indiv[0] for indiv in sorted_population[:num_elites]]
    return elites


def pmx_crossover(parent1, parent2, best_fitness, temperature, solution_df, row_id_column_name):
    """ Perform Partially Matched Crossover (PMX) with acceptance of worse solutions. """
    size = len(parent1)
    point1, point2 = sorted(random.sample(range(size), 2))  # Select two random crossover points

    # Create offspring with the same structure as parents
    offspring1 = [None] * size
    offspring2 = [None] * size

    # Copy the segments from parents
    for i in range(point1, point2):
        offspring1[i] = parent2[i]
        offspring2[i] = parent1[i]

    # Mapping to resolve duplicates
    mapping1 = {parent2[i]: parent1[i] for i in range(point1, point2)}
    mapping2 = {parent1[i]: parent2[i] for i in range(point1, point2)}

    # Fill the remaining positions using the mapping
    for i in range(size):
        if offspring1[i] is None:
            word = parent1[i]
            while word in offspring1:  # Handle duplicates in offspring1
                word = mapping1[word]
            offspring1[i] = word

        if offspring2[i] is None:
            word = parent2[i]
            while word in offspring2:  # Handle duplicates in offspring2
                word = mapping2[word]
            offspring2[i] = word

    # Accept worse offspring with probability based on fitness difference and temperature
    current_perplexity1 = calculate_fitness([offspring1], solution_df, row_id_column_name)[0]
    current_perplexity2 = calculate_fitness([offspring2], solution_df, row_id_column_name)[0]

    if not accept_worse_solution(current_perplexity1, best_fitness, temperature):
        offspring1 = parent1  # Revert if worse solution isn't accepted

    if not accept_worse_solution(current_perplexity2, best_fitness, temperature):
        offspring2 = parent2  # Revert if worse solution isn't accepted

    return offspring1, offspring2




def simulated_annealing_schedule(initial_temp, cooling_rate, iteration, max_iterations):
    """Calculate the current temperature and mutation probability based on the annealing schedule."""
    temperature = initial_temp * (cooling_rate ** (iteration / max_iterations))
    return temperature

def accept_worse_solution(perplexity, best_perplexity, temperature):
    """ Accept a worse solution based on the simulated annealing probability. """
    delta = perplexity - best_perplexity
    if delta < 0:
        return True  # Always accept a better solution
    else:
        probability = math.exp(-delta / temperature)
        return random.random() < probability


def mutate(individual, mutation_rate, temperature, best_perplexity, current_perplexity):
    """ Apply mutation with simulated annealing acceptance criterion. """
    if random.random() < mutation_rate:
        # Perform mutation (swap two random elements)
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
        
        # Check if we should accept this mutation based on temperature
        if not accept_worse_solution(current_perplexity, best_perplexity, temperature):
            # Revert mutation if it's not accepted
            individual[i], individual[j] = individual[j], individual[i]

    return individual



def genetic_algorithm(words, solution_df, row_id_column_name, generations=100, population_size=20, mutation_rate=0.1, num_parents=2, log_filename='generation_log.json', fixed_parents=None, num_elites=3, sa_iterations=1000, initial_temp=1000, cooling_rate=0.99, sa_frequency=5):
    population = create_population(population_size, words)
    best_solution = None
    best_fitness = float('inf')
    logs = []  # To store logs for each iteration
    stagnation_count = 0  # Counter to track stagnation

    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")
        
        # Calculate fitness scores for the current population
        fitness_scores = calculate_fitness(population, solution_df, row_id_column_name)

        # Find the best solution in the current population
        best_generation_fitness = min(fitness_scores)
        best_generation_solution = population[fitness_scores.index(best_generation_fitness)]

        # Update the global best solution if necessary
        if best_generation_fitness < best_fitness:
            best_fitness = best_generation_fitness
            best_solution = best_generation_solution

        # Log the generation data (text and perplexity)
        for i, individual in enumerate(population):
            permuted_text = ' '.join(individual)
            logs.append({
                'generation': generation + 1,
                'individual_index': i,
                'permuted_text': permuted_text,
                'perplexity': fitness_scores[i]  # Ensure we are using the updated fitness_scores
            })

        # Select parents
        parents = []
        if fixed_parents:
            # Use the manually set first two parents, convert them to word lists if necessary
            parent1 = fixed_parents[0].split()
            parent2 = fixed_parents[1].split()
            parents.extend([parent1, parent2])

            # Fill the remaining parents using tournament selection
            while len(parents) < num_parents:
                selected_parents = roulette_wheel_selection(population, fitness_scores, num_parents=num_parents - len(parents))
                parents.extend(selected_parents)
        else:
            # If no fixed parents are set, use tournament selection for all parents
            parents = roulette_wheel_selection(population, fitness_scores, num_parents=num_parents - len(parents))

        # Create the next generation using crossover and mutation
        next_generation = []
        # New mutation handling:
        for i in range(0, len(parents), 2):  # Ensure we are pairing parents properly
            offspring1, offspring2 = pmx_crossover(parents[i], parents[i+1], best_fitness, initial_temp, solution_df, row_id_column_name)  # Pass row_id_column_name here
        
            # Set current_perplexity based on best fitness (perplexity)
            current_perplexity = best_fitness  # This is the best fitness we've found so far
            
            # Calculate the temperature using the current generation
            temperature = simulated_annealing_schedule(initial_temp, cooling_rate, generation, generations)
            
            # Apply mutation with additional parameters
            offspring1 = mutate(offspring1, mutation_rate, temperature, best_fitness, current_perplexity)  # Mutate the offspring
            offspring2 = mutate(offspring2, mutation_rate, temperature, best_fitness, current_perplexity)  # Mutate the offspring
        
            next_generation.append(offspring1)  # Add mutated offspring to the next generation
            next_generation.append(offspring2)  # Add mutated offspring to the next generation

        # Optionally apply elitism to keep the best individuals
        elites = elitism(population, fitness_scores, num_elites=num_elites)
        for i, elite in enumerate(elites):
            elites[i], _ = simulated_annealing_v4(
    text=elite,
    temp_start=6.0,
    temp_end=1.0,
    cooling_rate=0.85,  # Adjust as needed
    n_neighbor=2,
    steps_per_temp=10,
    verbose=True,  # Enable verbose logging
    seq_to_choose=None,  # Use all indices
    factor_acceptance=1.0
)
        next_generation.extend(elites)

        population = next_generation

        # Apply Simulated Annealing every `sa_frequency` generations or if a significant improvement is found
        if generation % sa_frequency == 0:  # Apply SA every `sa_frequency` generations
            current_solution = ' '.join(best_solution)
            current_perplexity = best_fitness
        if generation % 10 == 0:  # Every 10 generations
            random_individuals = create_population(int(0.2 * population_size), words)
            population[-len(random_individuals):] = random_individuals
        if stagnation_count >= 10:
            mutation_rate = 0.5  # Temporarily increase mutation rate
        if stagnation_count >= 10:
            random_individuals = create_population(int(0.5 * population_size), words)
            population[-len(random_individuals):] = random_individuals
        if stagnation_count >= 10:
            best_solution, best_fitness = simulated_annealing_v4(
    text=text,
    temp_start=6.0,
    temp_end=1.0,
    cooling_rate=0.85,  # Adjust as needed
    n_neighbor=2,
    steps_per_temp=10,
    verbose=True,  # Enable verbose logging
    seq_to_choose=None,  # Use all indices
    factor_acceptance=1.0
)

        
            # Optionally, you could check if there's been enough improvement to trigger SA
            if current_perplexity > best_fitness:
                best_solution, best_fitness = simulated_annealing_v4(
    text=text,
    temp_start=6.0,
    temp_end=1.0,
    cooling_rate=0.85,  # Adjust as needed
    n_neighbor=2,
    steps_per_temp=10,
    verbose=True,  # Enable verbose logging
    seq_to_choose=None,  # Use all indices
    factor_acceptance=1.0
)
    # Save the logs to a JSON file
        with open(log_filename, 'w') as f:
            json.dump(logs, f, indent=4)

    # Return the best solution found
    return ' '.join(best_solution), best_fitness



# Words list
words = words_4  # Assuming 'words_1' is your list of words
# Create a dummy solution DataFrame (replace with actual solution dataframe)
solution_df = pd.DataFrame({
    'id': [0],
    'text': "words"
})

# Apply the genetic algorithm, passing the fixed_parents
best_solution, best_fitness = genetic_algorithm(
    words, solution_df, row_id_column_name="id", 
    generations=100, 
    population_size=50, 
    mutation_rate=0.2,  # Higher mutation rate
    log_filename="/kaggle/working/genetic_algorithm_log.json", 
    fixed_parents=None, 
    num_elites=5,
    sa_iterations=500,  # More SA iterations
    initial_temp=500,  # Higher initial temperature
    cooling_rate=0.98,  # Slower cooling
    sa_frequency=10  # More frequent SA application
)


print()
print(f"Best permutation: {best_solution}")
print(f"Best perplexity: {best_fitness}")
print(f"Time taken: {time.time() - start_time:.2f} seconds")



#words_1
fixed_parents = ['reindeer sleep walk the night and drive mistletoe scrooge laugh chimney jump elf bake gingerbread family give advent fireplace ornament', 'ornament mistletoe fireplace chimney and advent elf the family gingerbread reindeer scrooge walk give jump drive bake night sleep laugh']  # Parents are the first two individuals


fixed_parents=None


import pickle

with open('past.pickle', 'wb') as f:
    pickle.dump(past, f, protocol=pickle.HIGHEST_PROTOCOL)

