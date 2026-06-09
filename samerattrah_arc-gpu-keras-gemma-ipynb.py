
!pip install -U keras-hub
!pip install -U keras
!pip install rouge-score
# !pip install --upgrade numpy scipy

import re
import os

os.environ['KERAS_BACKEND'] = 'jax'

import keras
import jax
# from jax import numpy as jnp
# from jax.sharding import Mesh, PartitionSpec as P
import keras_hub
import numpy as np
import pandas as pd
from keras_hub.models import Gemma3CausalLM
# from keras.distribution import LayoutMap, DeviceMesh, ModelParallel, AutoLayoutMap
import json
from typing import Dict, Any, List, Optional
from rouge_score import rouge_scorer
import glob

golf_data_dir = "/kaggle/input/copy-neurips-2025-google-code-golf-championship"

training_challenges_file = "/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json"

evaluation_challenges_file = "/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json"

test_file_path = '/kaggle/input/arc-prize-prompts-v8/test_prompts_v8_with_st.csv'

my_dataframe = pd.read_csv(test_file_path)

ids = list(my_dataframe["input_id"])
prompts = list(my_dataframe["prompt"])
labels = list(my_dataframe["labels"])


def last_list(response, rag_solution):
    refined_pattern = r"\[\s*\[[^\]]*\](?:\s*,\s*\[[^\]]*\])*\s*\]"

    # Use regex to find the potential list-of-lists string first
    list_string_match = re.findall(refined_pattern, response)
    if list_string_match:
        if not re.search('[a-zA-Z]', list_string_match[-1]):
            return list_string_match[-1]
    else:
        if rag_solution != None:
          return rag_solution
        else:
            return response[0:1000]



def string_to_list(string):
    # print("this is string", list(string))
    main_list = []
    sub_list = []
    bracket_counter = 0
    for i in list(string):
        if i.isnumeric():
            sub_list.append(int(i))
            bracket_counter = 0
        elif i == "]" and bracket_counter == 1:
            return main_list
        elif i == "]":
            main_list.append(sub_list)
            sub_list = []
            bracket_counter+=1
            continue
        else:
            continue



# A type alias for a task dictionary
Task = Dict[str, Any]

class ARCPuzzleSearcher:
    """
    A class to search for similar ARC puzzles based on ROUGE score of their input grids.
    It indexes training examples from specified challenge files and directories.
    """
    def __init__(self, data_dirs: List[str], challenges_files: List[str], rouge_threshold: float = 0.9):
        """
        Initializes the ARCPuzzleSearcher.

        Args:
            data_dirs (List[str]): A list of paths to directories containing ARC task JSON files
                                    (e.g., 'golf_data/').
            challenges_files (List[str]): A list of paths to single ARC challenges JSON files
                                          (e.g., 'arc-agi_training_challenges.json').
            rouge_threshold (float): The minimum ROUGE-1 F-measure score required to consider a match.
        """
        self.data_dirs = data_dirs
        self.challenges_files = challenges_files
        self.rouge_threshold = rouge_threshold
        # Stores indexed training examples: {task_id, original_task_id, train_input_text, train_output_grid}
        self.indexed_puzzles: List[Dict[str, Any]] = []
        # Using RougeScorer with 'rouge1' (unigram) F-measure.
        # For numerical grids, stemming is not relevant, so use_stemmer is omitted (defaults to False).
        self.scorer = rouge_scorer.RougeScorer(['rouge1'])
        self._load_all_puzzles()

    def _grid_to_text(self, grid: List[List[int]]) -> str:
        """
        Converts a grid (list of lists of integers) to a compact JSON string.
        This string representation is used for ROUGE comparison.
        """
        return json.dumps(grid, separators=(',', ':'))

    def _load_puzzles_from_file(self, file_path: str):
        """
        Loads puzzles from a single JSON file (e.g., arc-agi_training_challenges.json).
        It extracts all 'train' input-output pairs for indexing.
        """
        if not os.path.exists(file_path):
            print(f"Warning: Challenges file not found at {file_path}")
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict):
                for task_id, task_details in data.items():
                    if 'train' in task_details and isinstance(task_details['train'], list):
                        for i, pair in enumerate(task_details['train']):
                            if 'input' in pair and 'output' in pair:
                                self.indexed_puzzles.append({
                                    'task_id': f"{task_id}_train_{i}", # Unique ID for each train example
                                    'original_task_id': task_id,
                                    'train_input_text': self._grid_to_text(pair['input']),
                                    'train_output_grid': pair['output']
                                })
            else:
                print(f"Warning: Unexpected JSON structure in {file_path}. Expected a dictionary of tasks.")

        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading puzzles from {file_path}: {e}")

    def _load_puzzles_from_directory(self, directory_path: str):
        """
        Loads puzzles from JSON files within a directory (e.g., 'golf_data/').
        It extracts all 'train' input-output pairs for indexing.
        """
        if not os.path.exists(directory_path):
            print(f"Warning: Data directory not found at {directory_path}")
            return

        # Look for files named 'task*.json'
        pattern = os.path.join(directory_path, 'task*.json')
        task_files = sorted(glob.glob(pattern))

        for file_path in task_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    task_details = json.load(f)

                task_id = os.path.splitext(os.path.basename(file_path))[0]
                if 'train' in task_details and isinstance(task_details['train'], list):
                    for i, pair in enumerate(task_details['train']):
                        if 'input' in pair and 'output' in pair:
                            self.indexed_puzzles.append({
                                'task_id': f"{task_id}_train_{i}",
                                'original_task_id': task_id,
                                'train_input_text': self._grid_to_text(pair['input']),
                                'train_output_grid': pair['output']
                            })
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading task from {file_path}: {e}")

    def _load_all_puzzles(self):
        """Loads all puzzles from specified directories and files into the index."""
        print("Loading puzzles for indexing...")
        for d_dir in self.data_dirs:
            self._load_puzzles_from_directory(d_dir)
        for c_file in self.challenges_files:
            self._load_puzzles_from_file(c_file)
        print(f"Indexed {len(self.indexed_puzzles)} training examples from all sources.")

    def find_similar_solution(self, query_input_grid: List[List[int]]) -> Optional[List[List[int]]]:
        """
        Searches for a similar input grid in the indexed training examples.
        If a match with a ROUGE-1 F-measure above the threshold is found,
        it returns the corresponding output grid of that training example.

        Args:
            query_input_grid (List[List[int]]): The input grid of the query task
                                                 for which a similar solution is sought.

        Returns:
            Optional[List[List[int]]]: The output grid (list of lists of integers)
                                       of the most similar training example if its ROUGE-1
                                       F-measure is above the configured threshold.
                                       Returns None if no sufficiently similar match is found.
        """
        query_input_text = self._grid_to_text(query_input_grid)
        best_score = -1.0
        best_match_output = None
        best_match_task_id = None

        for indexed_example in self.indexed_puzzles:
            scores = self.scorer.score(query_input_text, indexed_example['train_input_text'])
            rouge_1_fmeasure = scores['rouge1'].fmeasure

            if rouge_1_fmeasure > best_score:
                best_score = rouge_1_fmeasure
                best_match_output = indexed_example['train_output_grid']
                best_match_task_id = indexed_example['original_task_id']

        if best_score > self.rouge_threshold:
            print(f"Found a similar task (ID: {best_match_task_id}) with ROUGE-1 F-measure: {best_score:.4f}")
            return (best_match_output, best_score)
        else:
            print(f"No similar task found above threshold {self.rouge_threshold:.4f}. Best score: {best_score:.4f}")
            return (None, None)


print(f"Keras backend: {keras.backend.backend()}")
print(f"JAX devices: {len(jax.devices())} GPU Devices")
devices = keras.distribution.list_devices()
if len(devices) < 2:
    print(f"Warning: Only {len(devices)} devices found. Need 2 for optimal parallelism.")


# device_mesh = keras.distribution.DeviceMesh(
#     shape=(1, len(devices)), 
#     axis_names=["batch", "model"],
#     devices=devices
# )

# layout_map = AutoLayoutMap(device_mesh)

# distribution = keras.distribution.ModelParallel(
#     device_mesh=device_mesh,
#     layout_map = layout_map,
# )

# # Set the global distribution scope for the Keras model
# keras.distribution.set_distribution(distribution)


gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset("/kaggle/input/gemma3/keras/gemma3_instruct_1b/3",
                                                       dtype="bfloat16")
# gemma_lm = keras.models.load_model("/kaggle/input/gemma31bit-finetuned-arcprize2025-prompts-v3/keras/default/1/fine_tuned_gemma_datav3_batch8_epochs10_val.keras")
gemma_lm.summary()

searcher = ARCPuzzleSearcher(
    data_dirs=[golf_data_dir],
    challenges_files=[training_challenges_file, evaluation_challenges_file],
    rouge_threshold=0.85
)

counter = 1
submission_list = []
submission_dump = {}

for i in range(len(prompts)):
    last_input = last_list(prompts[i], _)
    rag_solution, score = searcher.find_similar_solution(last_input)
    response_1 = gemma_lm.generate(prompts[i]+f"take into consideration that the best potential answer is: {rag_solution} and return it in case uncertain about what you generated", max_length=5000, strip_prompt=True)
    response_2 = gemma_lm.generate(prompts[i]+f"take into consideration that the best potential answer is: {rag_solution} and return it in case uncertain about what you generated", max_length=3000, strip_prompt=True)
    output_1 = last_list(response_1, rag_solution)
    output_2 = last_list(response_2, rag_solution)
    if score == 1:
        list_attempt_1 = rag_solution
    elif isinstance(output_1, list):
        list_attempt_1 = output_1
    else:
        list_attempt_1 = string_to_list(output_1)
    if isinstance(output_2, list):
        list_attempt_2 = output_2
    else:    
        list_attempt_2 = string_to_list(output_2)

        
    if list_attempt_1 == None and list_attempt_2 == None:
        list_attempt_1 = [[0]]
        
        list_attempt_2 = [[0]]
        submission_list.append([ids[i], [{"attempt_1":list_attempt_1, "attempt_2":list_attempt_2}]])
    elif list_attempt_1 == None:
        list_attempt_1 = [[0]]
        submission_list.append([ids[i], [{"attempt_1":list_attempt_1, "attempt_2":list_attempt_2}]])
    elif list_attempt_2 == None:
        list_attempt_2 = [[0]]
        submission_list.append([ids[i], [{"attempt_1":list_attempt_1, "attempt_2":list_attempt_2}]])        
    else:
        submission_list.append([ids[i], [{"attempt_1":list_attempt_1, "attempt_2":list_attempt_2}]])

    if submission_list[i][0].split("_")[0] in submission_dump.keys():
        submission_dump[submission_list[i][0].split("_")[0]].append(submission_list[i][1])
    else:
        submission_dump[submission_list[i][0].split("_")[0]] = submission_list[i][1]
    print(counter)
    counter+=1


output_submission_file = "submission.json"
with open(output_submission_file, "w") as f:
    json.dump(submission_dump, f, indent = 4) # Using indent for readability\n",

