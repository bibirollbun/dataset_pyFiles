i = 3
PRELOAD = True
PRELOAD_CACHE = False
POPULATION_SIZE = 100
BATCH_SIZE = 1024
GENERATIONS = 100

LOCAL = False


# Metric updated on 03.12.2024


"""Evaluation metric for Santa 2024."""

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

import random

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
                attn_implementation="sdpa"
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
        self, input_texts: Union[str, List[str]], batch_size: 32, debug=False
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


if LOCAL:

    model_path = '/mnt/ssd1/Kaggle/Models/gemma-2-transformers-gemma-2-9b-v2'
    cache_path = os.path.join(str(i),'evaluator_cache.pkl')
    cache_path_res = os.path.join(str(i),'evaluator_cache.pkl')
    sol_path = 'submission.csv'
else:
    
    model_path = '/kaggle/input/gemma-2/transformers/gemma-2-9b/2'
    cache_path = '/kaggle/input/evaluator-cache-4/evaluator_cache_newer.pkl'
    cache_path_res = 'evaluator_cache.pkl'
    sol_path = '/kaggle/input/santa-2024/sample_submission.csv'


df = pd.read_csv(sol_path)


sents = df['text'].tolist()


original_string = sents[i]
words = original_string.split()





DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PerplexityEvaluatorWithCache:
    
    def __init__(self, model_path: str, batch_size: int):
        self.calculator = PerplexityCalculator(model_path=model_path)
        self.cache: Dict[bytes, float] = {}
        self.batch_size = batch_size

    def evaluate(self, permutation: List[str]) -> float:

        key = " ".join(permutation)

        if key in self.cache:
            return self.cache[key]
        
        # Calculate perplexity if not cached
        text = " ".join(permutation)
        perplexity = self.calculator.get_perplexity([text], batch_size=self.batch_size)[0]
        self.cache[key] = perplexity  # Cache the result
        return perplexity

    def evaluate_batch(self, texts: list[str]) -> list[float]:
        """
        Evaluates the perplexity of a batch of texts, using caching.
        
        Parameters
        ----------
        texts : list[str]
            A list of texts to evaluate.

        Returns
        -------
        list[float]
            A list of perplexity scores corresponding to the input texts.
        """
        results = []
        uncached_texts = []
        uncached_indices = []

        # Check cache for existing perplexities
        for idx, text in enumerate(texts):
            key = " ".join(text)
            # print('evaluate_batch key check', idx, key)
            if key in self.cache:
                results.append(self.cache[key])
            else:
                uncached_texts.append(key) # should pass the joined string to evaluate it
                uncached_indices.append(idx)

        # Process uncached texts in batches
        if uncached_texts:
            uncached_perplexities = self.calculator.get_perplexity(
                uncached_texts, batch_size=self.batch_size
            )

            # Update results and cache
            for i, perplexity in enumerate(uncached_perplexities):
                text = uncached_texts[i] # words already joined into a single string 
                # key = " ".join(text)
                # print('evaluate_batch key update', text)
                results.insert(uncached_indices[i], perplexity)
                self.cache[text] = perplexity

        return results

    def load_cache(self, path):
        with open(path, "rb") as file:
            self.cache = pickle.load(file)       
        print("Evaluator cache loaded")    

def local_search(solution, evaluator, max_iterations=10, sample_fraction=0.1):
    print("        Local search started")
    """
    Refines a permutation solution using loacl (tabu) search.

    Args:
        solution (list): The current solution (permutation of words).
        evaluator (Evaluator): Object to evaluate fitness (perplexity).
        max_iterations (int): Maximum number of tabu search iterations.
        sample_fraction (float): Fraction of neighbors to randomly sample.

    Returns:
        list: The improved solution.
        float: The fitness of the improved solution.
    """
    best_solution = solution[:]
    best_score = evaluator.evaluate(best_solution)
    # tabu_list = evaluator.cache  # Use the evaluator cache as a tabu list - UPDATE: better use directly the cache as it grows during the iterations

    for i in range(max_iterations):
        print(f"            Iteration {i}")
        neighbors = []
        total_neighbors = (len(solution) * (len(solution) - 1)) // 2
        sample_size = max(1, int(sample_fraction * total_neighbors))  # Sample a fraction of neighbors

        # Generate sampled neighbors
        for _ in range(sample_size):
            # Randomly pick two indices to swap
            i, j = random.sample(range(len(solution)), 2)
            neighbor = best_solution[:]
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

            # Add to neighbors only if it's not tabu (not in cache)
            neighbor_key = ' '.join(neighbor)
            if neighbor_key not in evaluator.cache:
                neighbors.append(neighbor)

        # Evaluate all sampled neighbors
        if not neighbors:
            print("            No valid neighbors to explore; exiting early.")
            break  # No valid (non-tabu) neighbors, terminate early

        scored_neighbors = [(neighbor,score) for neighbor,score in zip(neighbors, evaluator.evaluate_batch(neighbors))]     
        scored_neighbors.sort(key=lambda x: x[1])  # Sort by fitness (perplexity)

        # Update if a better neighbor is found
        if scored_neighbors[0][1] < best_score:
            best_solution, best_score = scored_neighbors[0]
            print(f"            Found a better solution: Score = {best_score}")
        else:
            print("            No improvement found; terminating.")
            # break  # No improvement, stop searching

    return best_solution, best_score

def local_search_vns(solution, evaluator, max_iterations=10, max_neighborhood_size=5, sample_fraction=0.1):
    print("        Local search with VNS started")
    """
    Refines a permutation solution using Variable Neighborhood Search (VNS).

    Args:
        solution (list): The current solution (permutation of words).
        evaluator (Evaluator): Object to evaluate fitness (perplexity).
        max_iterations (int): Maximum number of tabu search iterations.
        max_neighborhood_size (int): Maximum size of neighborhoods to explore.
        sample_fraction (float): Fraction of neighbors to randomly sample.

    Returns:
        list: The improved solution.
        float: The fitness of the improved solution.
    """
    best_solution = solution[:]
    best_score = evaluator.evaluate(best_solution)
    neighborhood_size = 2  # Start with 2-word swaps
    
    for iteration in range(max_iterations):
        print(f"            Iteration {iteration}, Neighborhood size: {neighborhood_size}")
        neighbors = []
        total_neighbors = (len(solution) * (len(solution) - 1)) // 2  # Approximate for larger neighborhoods
        sample_size = max(1, int(sample_fraction * total_neighbors))  # Sample a fraction of neighbors

        # Generate sampled neighbors based on current neighborhood size
        for _ in range(sample_size):
            indices = random.sample(range(len(solution)), neighborhood_size)
            neighbor = best_solution[:]
            permuted_indices = random.sample(indices, len(indices))
            for idx, new_idx in zip(indices, permuted_indices):
                neighbor[idx] = best_solution[new_idx]

            # Add to neighbors only if it's not tabu (not in cache)
            neighbor_key = ' '.join(neighbor)
            if neighbor_key not in evaluator.cache:
                neighbors.append(neighbor)

        # Evaluate all sampled neighbors
        if not neighbors:
            neighborhood_size += 1  # Increase neighborhood size
            if neighborhood_size > max_neighborhood_size:
                print("            Reached maximum neighborhood size; terminating.")
                break
            else:
                continue

        scored_neighbors = [(neighbor, score) for neighbor, score in zip(neighbors, evaluator.evaluate_batch(neighbors))]
        scored_neighbors.sort(key=lambda x: x[1])  # Sort by fitness (perplexity)

        # Update if a better neighbor is found
        if scored_neighbors[0][1] < best_score:
            best_solution, best_score = scored_neighbors[0]
            neighborhood_size = 2  # Reset to smallest neighborhood size
            print(f"            Found a better solution: Score = {best_score}")
        else:
            # neighborhood_size += 1  # Increase neighborhood size
            print("            No improvement found.")

    return best_solution, best_score

def create_child(parent1, population, words, crossover_point):
    words2 = words.copy()
    # Get the required words after the crossover point from parent1
    required_words = Counter(parent1[crossover_point:])
    
    # Extract words already in the first part of parent1
    first_part_words = parent1[:crossover_point]
    words2 = parent1[crossover_point:]

    try:
        # try to find matching partner
        a = Counter(parent1[crossover_point:])
        parent2 = random.choice([x for x in population if (Counter(x[crossover_point:])==a) and (x[crossover_point:] != words2)])
    except:
        # Shuffle the remaining words
        random.shuffle(words2)
    
    # Construct a new parent with the same first part and shuffled remaining words
    child = parent1[:crossover_point] + words2
    
    return child

def perform_crossover(parent1, parent2, words, crossover_point):
    # Include first part from parent1
    child = parent1[:crossover_point]

    # Track counts to handle duplicates
    child_counts = Counter(child)
    word_counts = Counter(words)  # Original counts of all words

    # Add remaining words from parent2, respecting duplicates
    for w in parent2:
        if child_counts[w] < word_counts[w]:  # Ensure duplicates are added correctly
            child.append(w)
            child_counts[w] += 1  # Update the count in the child

    # Ensure child length matches words length
    assert len(child) == len(words), f"Child length mismatch: {len(child)} != {len(words)}"
    assert Counter(child) == word_counts, f"Child mismatch: {Counter(child)} != {word_counts}"

    return child

def reseed_from_cache(cache, top_share=0.1, drop_leading_group=True):
    
    '''
    Gets the cache, extracts the first word, selects the elite share of each group, removes the top group and shuffle the rows.
    Returns a list of strings with rel. high score to feed the reseeding.
    '''
    
    df = pd.DataFrame.from_dict(cache, orient='index').sort_values(0)
    df = df.reset_index()
    df.columns = ['perm', 'score']

    print('Current top 10')
    print(df.head(10))
    
    df['first_word'] = df['perm'].map(lambda x: x.split(' ')[0])
    df['rank_in_group'] = (df.groupby('first_word')['score'].rank('min')).astype(int)
    df['group_min'] = df.groupby('first_word')['score'].transform('min')#.transform('rank')
    df['count_in_group'] = df.groupby('first_word')['score'].transform('count')

    df['max_rank'] = (df['count_in_group']*top_share).astype(int)
    df_stratified = df.loc[df['rank_in_group']<= df['max_rank']]

    # drop the leading group
    if drop_leading_group:
        df_stratified = df_stratified.loc[df_stratified['group_min']>df_stratified['group_min'].min()]
    res = df_stratified.sample(frac=1)['perm'].tolist()
    res = [x.split(' ') for x in res]

    # print(len(set(len(x) for x in res)))

    assert set(len(x) for x in res) == set([len(words)]), "reseed_from_cache - Not all strings in the list have the same length"
    
    return res

def get_random_for_each_first_word(words, size=POPULATION_SIZE):
    res = []
    while len(res)<size:
        for w in words:
            aux = words.copy()
            aux.remove(w)
            random.shuffle(aux)
            res.append([w]+aux)
        
    assert set(len(x) for x in res) == set([len(words)]), "get_random_for_each_first_word - Not all strings in the list have the same length"
    
    return res

def evolutionary_algorithm(
    words: List[str],
    evaluator: PerplexityEvaluatorWithCache,
    preloads: List[str],
    population_size: int = 50,
    generations: int = 100,
    base_mutation_rate: float = 0.2,
    elite_fraction: float = 0.1,
    reseed_interval: int = 10,
    reseed_random_fraction: float = 0.2,
    reseed_from_cache_fraction: float = 0.2,
    max_attempts: int = 100,
    local_search__max_iterations: int = 10,
    local_search__sample_fraction: float = 0.1,
    local_search__max_neighborhood_size: int = 5,
):
    print("Evolution started")
    # Generate initial population
    population = [x.split(' ') for x in preloads] + [random.sample(words, len(words)) for _ in range(population_size - len(preloads))]

    if PRELOAD_CACHE:
        # 
        generate_from_cache_count = int(reseed_from_cache_fraction * (population_size-len(preloads))) # use the same fraction as the reseed for simplicity
        population[len(preloads):(len(preloads)+generate_from_cache_count)] = random.sample(reseed_from_cache(evaluator.cache,drop_leading_group=False), generate_from_cache_count)    

    assert set(len(x) for x in population) == set([len(words)]), "Generate initial population - Not all strings in the list have the same length"
    
    # print([len(x) for x in population])
    best_solution, best_score = None, float("inf")
    mutation_rate = base_mutation_rate

    for gen in range(generations):
        print(f"    Generation {gen}")
        # Evaluate fitness of each individual
        # fitness = [
        #     (ind, evaluator.evaluate(ind))
        #     for ind in population
        # ]

        fitness = [(ind,score) for ind,score in zip(population, evaluator.evaluate_batch(population))]  
        fitness.sort(key=lambda x: x[1])  # Sort by perplexity score

        # Apply local search depending on the score based probability
        probs = [fitness[0][1]/float(x[1]) for x in fitness]
        print(probs)

        for i in range(population_size):
            if random.random() < probs[i]:
                print(f"        Local search on fitness[{i}], score {fitness[i][1]}: {' '.join(fitness[i][0])}")
                refined_solution, refined_score = local_search_vns(fitness[i][0], evaluator, local_search__max_iterations, local_search__max_neighborhood_size, local_search__sample_fraction)
                if refined_score < fitness[i][1]:  # Replace if improved
                    fitness[i] = (refined_solution, refined_score)

        fitness.sort(key=lambda x: x[1])  # Sort by perplexity score

        # Save the best solution so far
        if fitness[0][1] < best_score:
            best_solution, best_score = fitness[0]
        
        print(f"    Generation {gen+1}, Best Score: {best_score}, Solution: {' '.join(best_solution)}")

        # Select elites
        elite_count = int(elite_fraction * population_size)
        elites = [ind for ind, _ in fitness[:elite_count]]

        # Apply local search to a subset of elites
        for i in range(len(elites)):
            print(f"        Elites[{i}], score {fitness[i][1]}: {' '.join(elites[i])}")
            refined_solution, refined_score = local_search_vns(elites[i], evaluator, local_search__max_iterations, local_search__max_neighborhood_size, local_search__sample_fraction)
            if refined_score < fitness[i][1]:  # Replace if improved
                fitness[i] = (refined_solution, refined_score)

        # update elites with the refined values
        elites = [ind for ind, _ in fitness[:elite_count]]   

        assert set(len(x) for x in elites) == set([len(words)]), "Apply local search to a subset of elites - Not all strings in the list have the same length"

        # Adjust mutation rate dynamically
        # mutation_rate = base_mutation_rate * (1 - gen / generations) + 0.05

        # Generate offspring
        new_population = elites[:]
        while len(new_population) < population_size:
            attempts = 0
            child = None
            while attempts < max_attempts:
                # Dynamic crossover: bias toward elites as generations progress
                elite_bias = min(1.0, gen / generations)
                crossover_point = random.randint(1, len(words) - 1) 
                
                if random.random() < elite_bias and len(elites) >= 2:
                    parent1 = random.choice(elites)
                    # parent2 = select_second_parent(parent1, elites, words, crossover_point)
                    child = create_child(parent1, elites, words, crossover_point)
                else:
                    parent1 = random.choice(population)
                    # parent2 = select_second_parent(parent1, population, words, crossover_point)
                    child = create_child(parent1, population, words, crossover_point)
                

                # Create child using crossover

                #child = parent1[:crossover_point] + [
                #    w for w in parent2 if w not in parent1[:crossover_point]
                #]
                #missing_words = [w for w in words if w not in child]
                #child += missing_words
                # child = perform_crossover(parent1, parent2, words, crossover_point)

                # Ensure child contains all words with the correct frequency
                assert Counter(child) == Counter(words), f"Child mismatch: {Counter(child)} != {Counter(words)}"
                # print('child', len(child))

                # Check if child exists in the cache
                child_key = ' '.join(child)
                if child_key not in evaluator.cache:
                    break  # Found unseen child
                attempts += 1

            # If attempts exceeded, add a random unseen solution instead
            if attempts >= max_attempts:
                child = random.sample(words, len(words))
                print(f"Max attempts reached. Adding random unseen solution.")

            new_population.append(child)

        # Add the best solution to the next generation
        # new_population.append(best_solution)
        new_population = new_population[:population_size]

        assert set(len(x) for x in new_population) == set([len(words)]), "new_population 1 - Not all strings in the list have the same length"


        # Mutation
        for i in range(elite_count, population_size):
            if random.random() < mutation_rate:
                swap_idx1, swap_idx2 = random.sample(range(len(words)), 2)
                # print(i, len(new_population[i]), swap_idx1, swap_idx2)
                new_population[i][swap_idx1], new_population[i][swap_idx2] = (
                    new_population[i][swap_idx2],
                    new_population[i][swap_idx1],
                )

        assert set(len(x) for x in new_population) == set([len(words)]), "new_population mutation - Not all strings in the list have the same length"


        # Reseed population periodically
        if gen % reseed_interval == 0:
            print("Reseed")
            # reseed_count = int(reseed_fraction * (population_size-elite_count))

            reseed_from_cache_count = int(reseed_from_cache_fraction * (population_size-elite_count))
            reseed_random_count = int(reseed_random_fraction * (population_size-elite_count-reseed_from_cache_count))
            
            # new_population[elite_count:(elite_count+reseed_from_cache_count)] = random.sample(reseed_from_cache(evaluator.cache, drop_leading_group=False), reseed_from_cache_count)
            # new_population[(elite_count+reseed_from_cache_count):(elite_count+reseed_from_cache_count+reseed_random_count)] = random.sample(
            #     get_random_for_each_first_word(words), reseed_random_count
            # )

            # the code above put the reseeded values just after the elites
            # the code below puts the reseeded values at the end
            new_population[-reseed_from_cache_count-reseed_random_count:] = random.sample(reseed_from_cache(evaluator.cache, drop_leading_group=False), reseed_from_cache_count)
            new_population[-reseed_random_count:] = random.sample(
                get_random_for_each_first_word(words), reseed_random_count
            )


            assert set(len(x) for x in new_population) == set([len(words)]), "new_population reseed - Not all strings in the list have the same length"

            
            # print('reseed new_population')
            # print(new_population)
            



        assert set(len(x) for x in new_population) == set([len(words)]), "Update population - Not all strings in the list have the same length"

        # Secure only unique elements in population

        new_population = np.unique(np.array(new_population), axis=0).tolist()

        while np.unique(np.array(new_population), axis=0).shape[0] < POPULATION_SIZE:
            new_unseen = words.copy()
            random.shuffle(new_unseen)
            if (new_unseen not in new_population) and (' '.join(new_unseen) not in evaluator.cache):
                new_population.append(new_unseen)

                
        # Update population       
        
        population = new_population
        # print('new_population')
        # print(new_population)

        # Save the cache to the pickle file
        with open(cache_path_res, "wb") as file:
            pickle.dump(evaluator.cache, file)
        
        print(f"Evaluator cache has been saved to {cache_path_res}")

    return best_solution, best_score, evaluator.cache


# population = [x.split(' ') for x in [original_string]] + [random.sample(words, len(words)) for _ in range(50 - len([original_string]))]
# assert set(len(x) for x in population) == set([len(words)]), "Generate initial population - Not all strings in the list have the same length"


# set(len(x) for x in population)


# Initialise evaluator
evaluator = PerplexityEvaluatorWithCache(model_path, batch_size=BATCH_SIZE)
if PRELOAD_CACHE:
    evaluator.load_cache(cache_path)



print(len(evaluator.cache))


evaluator.calculator.get_perplexity([original_string], BATCH_SIZE) 


tops = (pd.DataFrame.from_dict(evaluator.cache, orient='index')
         .reset_index()
         .sort_values(0)
         .head(20)
        )['index'].to_list()


# Run the EA

if PRELOAD:
    if PRELOAD_CACHE:
        preload = [*tops, 
                  ]
    else:
        preload = [
            original_string,
        ]
else:
    preload = []


best_permutation, best_perplexity, cache = evolutionary_algorithm(
    words,
    evaluator,
    population_size=POPULATION_SIZE,
    generations=GENERATIONS,
    base_mutation_rate=0.8,
    elite_fraction=0.2,
    preloads=preload,
    reseed_interval = 2,
    reseed_random_fraction = 0.2,
    reseed_from_cache_fraction = 0.2,
    max_attempts = 100,
    local_search__max_iterations = 20,
    local_search__sample_fraction = 0.04,
    local_search__max_neighborhood_size = 6
    # preloads=['reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament']
)

print("Best permutation:", " ".join(best_permutation))
print("Best perplexity:", best_perplexity)
print("Cache size:", len(cache))






len(evaluator.cache)


sub = pd.read_csv(sol_path)
print("Submission shape is",sub.shape)

if evaluator.evaluate(original_string.split(' ')) > best_perplexity:
    sub.loc[i,"text"] = " ".join(best_permutation)
    print("+-+-+-+-+-+ SOLLUTION IMPROVED +-+-+-+-+-+")
sub.to_csv("submission.csv",index=False)
sub.head()




