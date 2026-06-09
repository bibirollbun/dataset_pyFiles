import os
import logging
from datetime import datetime
# from dotenv import load_dotenv

# Load environment variables from .env file
# load_dotenv()

start_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

class Cfg:
    # app
    LOGGING_FILE_PATH = f"../logs/run_{start_timestamp}.log"
    LOGGING_LEVEL = logging.DEBUG
    DATASET_ROOT_PATH = "/kaggle/input/arc-prize-2025/"
    SUBMISSION_FILE_PATH = "submission.json" #f"../submissions/submission_{start_timestamp}.json"

    # puzzle
    N_RUNS = 1

    # llm
    OLLAMA_ENDPOINT = "http://localhost:11434/v1/chat/completions"
    # OLLAMA_ENDPOINT = "https://60fd-34-75-186-201.ngrok-free.app" + "/v1/chat/completions"
    LLM_TYPE = "ollama"
    MODEL = "meta-llama/llama-3-1-8b-instruct"
    IBM_ACCESS_TOKEN = os.getenv('IBM_ACCESS_TOKEN')
    REQUEST_TIMEOUT = 300
    N_REQUEST_RETRIES = 5
    TEMPERATURE = 0.5
    TOP_P = 0.9
    MAX_TOKENS = 22000
    
    sanitised_model_value = MODEL.replace(':', '_').replace('/', '_')
    TABULAR_RUN_INFO_PATH = f"../results/predictions_{sanitised_model_value}_{start_timestamp}.csv"


import json

def load_json_dataset(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

class ARCData:
    """
    Training data: Consists of n-1 examples from the training data
    Validation data: Consists of the last example from the training data
    Test data: Consists of the challenge input from the training data

    Challenge is an input which needs to be solved
    """

    def __init__(self):
        self.training_challenges: dict = load_json_dataset(Cfg.DATASET_ROOT_PATH + 'arc-agi_training_challenges.json')
        self.training_solutions: dict = load_json_dataset(Cfg.DATASET_ROOT_PATH + 'arc-agi_training_solutions.json')

        self.evaluation_challenges: dict = load_json_dataset(Cfg.DATASET_ROOT_PATH + 'arc-agi_evaluation_challenges.json')
        self.evaluation_solutions: dict = load_json_dataset(Cfg.DATASET_ROOT_PATH + 'arc-agi_evaluation_solutions.json')

        self.test_challenges: dict = load_json_dataset(Cfg.DATASET_ROOT_PATH + 'arc-agi_test_challenges.json')

        self.test_challenges: dict = load_json_dataset(Cfg.DATASET_ROOT_PATH + 'arc-agi_test_challenges.json')
        
    def get_train_example_keys(self, n: int = None):
        """Return a list of keys for training examples. If n is provided, return the first n keys."""

        if n is not None:
            return list(self.training_challenges.keys())[:n]
        else:
            return list(self.training_challenges.keys())
    
    def get_test_example_keys(self, n=None):
        """Return a list of keys for test examples. If n is provided, return the first n keys."""

        if n is not None:
            return list(self.test_challenges.keys())[:n]
        else:
            return list(self.test_challenges.keys())
        
    # updated
    def get_train_example_data(self, key, include_validation=False):
        """
        Return the training data consisting of input and output for a given key.
        If include_validation is True, return the entire training data including the validation set.
        """

        data = self.training_challenges[key]
        if include_validation:
            return data['train']
        else:
            return data['train'][:-1]
    
    def get_train_challenge_data(self, key):
        """Return the training challenge data consisting of only the input for a given key."""
        
        data = self.training_challenges[key]
        return data['test']
        
    def get_train_solution_data(self, key):
        """Return the training solution data consisting of only the output for a given key."""
        
        return self.training_solutions[key]
    
    # new
    def get_train_validation_data(self, key):
        """Return the training validation data consisting of input and output for a given key."""

        data = self.training_challenges[key]
        return data['train'][-1]
    
    # updated
    def get_test_example_data(self, key, include_validation=False):
        """Return the test example data consisting of input and output for a given key."""
        if include_validation:
            data = self.test_challenges[key]
            return data['train']
        else:
            data = self.test_challenges[key]
            return data['train'][:-1]
    
    def get_test_challenge_data(self, key):
        """Return the test challenge data consisting of only the output for a given key."""

        data = self.test_challenges[key]
        return data['test']
    
    # new
    def get_test_validation_data(self, key):
        """Return the test validation data consisting of input and output for a given key."""
        
        data = self.test_challenges[key]
        return data['train'][-1]



submission = {}
def init_base_submission(arcd: ARCData):
    keys = arcd.get_test_example_keys()
    for key in keys:
        n_puzzles = len(arcd.get_test_challenge_data(key))
        dummy_results = [{"attempt_1": [[0]], "attempt_2": [[0]]}] * n_puzzles
        submission[key] = dummy_results

def submit_prediction(key: str, solution: list, puzzle_index: int):
    key = str(key)
    results = submission.get(key)
    
    results[puzzle_index]["attempt_2"] = results[puzzle_index]["attempt_1"]
    results[puzzle_index]["attempt_1"] = solution

    answer = {key: results}
    print(f"Updating submission.json: {answer}")
    submission.update(answer)

def final_submission():
    with open(Cfg.SUBMISSION_FILE_PATH, 'w') as f:
        json.dump(submission, f)


arcd = ARCData()

init_base_submission(arcd)

for key in arcd.get_test_example_keys(5):
    solution = [[0, 0]]
    submit_prediction(key, solution, puzzle_index=0)

final_submission()


!ls /kaggle/working/submission.json

