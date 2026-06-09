!pip install -qU langchain 2>/dev/null
#!pip install -qU langchain-openai 2>/dev/null
# !pip install -qU langchain-anthropic 2>/dev/null
# !pip install -qU langchain-google-genai 2>/dev/null
# !%pip install -qU langchain-groq
#!pip install -qU langchain-groq
!pip install -qU langchain-ollama
!pip install -U ollama


from kaggle_secrets import UserSecretsClient # Used to manage secrets. Similar to .env

import langchain # Main LangChain import
#from langchain_openai import ChatOpenAI # To work with OpenAI
# from langchain_anthropic import ChatAnthropic # To work with Anthropic (optional)
# from langchain_google_genai import ChatGoogleGenerativeAI # To work with Gemini (optional)
from langchain_core.output_parsers import JsonOutputParser # To help with structured output
from langchain_core.prompts import PromptTemplate # To help create our prompt
from pydantic import BaseModel, Field # To help with defining what output structure we want

from typing import List, Tuple
import os
import json
#from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
import torch 
import subprocess


!export OLLAMA_CUDA=1
!export OLLAMA_MAX_LOADED=4


print ("Files included")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


task_sets = {
    'training' : {
        'challenges' : '/kaggle/input/arc-prize-2024/arc-agi_training_challenges.json',
        'solutions' : '/kaggle/input/arc-prize-2024/arc-agi_training_solutions.json',
    },
    'evaluation' : {
        'challenges' : '/kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json',
        'solutions' : '/kaggle/input/arc-prize-2024/arc-agi_evaluation_solutions.json',
    }
}

#kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json


def load_tasks_from_file(task_set):
    """
    Loads the tasks from the file and returns the challenges and solutions tasks
    """
    with open(task_set['challenges'], "r") as tasks:
        challenges = json.load(tasks)

    with open(task_set['solutions'], "r") as tasks:
        solutions = json.load(tasks)

    return challenges, solutions


challenges, solutions = load_tasks_from_file(task_set=task_sets['training'])
challenges['0520fde7']


#!/usr/bin/env python
# Complete GPT-OSS-20B Setup and Run in Kaggle with Ollama

# Step 1: Install required packages
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])

# Step 2: Import necessary libraries
import os
import time
from openai import OpenAI

# Step 3: Install Ollama using bash commands
print("Installing Ollama...")
os.system("curl -fsSL https://ollama.com/install.sh | sh")

# Step 4: Start Ollama server in the background
print("Starting Ollama server...")
os.system("nohup ollama serve > /tmp/ollama_serve_stdout.log 2>/tmp/ollama_serve_stderr.log &")

# Step 5: Wait for server to start
print("Waiting for server to initialize...")
time.sleep(5)

# Step 6: Check if ollama is running
print("Checking if Ollama is running...")
os.system("ps aux | grep -E 'ollama' | grep -v grep || true")

# Step 7: Download GPT-OSS:20B model (this will take significant time - ~13GB)
print("\n" + "="*50)
print("Downloading GPT-OSS:20B model...")
print("This will take several minutes (downloading ~13GB)")
print("="*50 + "\n")
os.system("ollama pull gpt-oss:20b")

# Step 8: Verify the model is downloaded
print("\nVerifying model installation...")
os.system("ollama list")

# Step 9: Initialize OpenAI client for Ollama
print("\nInitializing OpenAI client for Ollama...")


#llm = ChatOpenAI(model='gpt-3.5-turbo-0125', openai_api_key=UserSecretsClient().get_secret('OPENAI_API_KEY'), max_tokens=3000)
llm = ChatOllama(model='gpt-oss:20b', max_tokens=3000)

## And incase you want to try Anthropic
# llm = ChatAnthropic(model='claude-3-5-sonnet-20240620', api_key=UserSecretsClient().get_secret("ANTHROPIC_API_KEY"), max_tokens=3000)
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=UserSecretsClient().get_secret("GOOGLE_API_KEY"), max_tokens=3000)


def json_task_to_string(challenge_tasks: dict, task_id: str, test_input_index: int) -> str:
    """
    challenge_tasks: dict a list of tasks
    task_id: str the id of the task we want to convert to a string
    
    Convert your json task into a string so you can pass it to your LLM.
    This is a crucial step where you can use your creativity to edit how tasks are represented.
    """
    json_task = challenge_tasks[task_id]

    final_output = ""

    train_tasks = json_task['train']
    test_task = json_task['test']

    final_output = "Training Examples\n"

    for i, task in enumerate(train_tasks):
        final_output += f"Example {i + 1}: Input\n["
        for row in task['input']:
            final_output += f"\n{str(row)},"

        final_output += "]\n\n"
        final_output += f"Example {i + 1}: Output\n["

        for row in task['output']:
            final_output += f"\n{str(row)},"

        final_output += "]\n\n"

    final_output += "Test\n["
    for row in test_task[test_input_index]['input']:
        final_output += f"\n{str(row)}"

    final_output += "]\n\nYour Response:"

    return final_output


task_string = json_task_to_string(challenges, '0520fde7', 0)
print (task_string)


# Defining a prediction as a list of lists
class ARCPrediction(BaseModel):
    prediction: List[List] = Field(..., description="A prediction for a task")


def get_task_prediction(challenge_tasks, task_id, test_input_index) -> List[List]:
    """
    challenge_tasks: dict a list of tasks
    task_id: str the id of the task we want to get a prediction for
    test_input_index: the index of your test input. 96% of tests only have 1 input.

    Given a task, predict the test output
    """

    # Get the string representation of your task
    task_string = json_task_to_string(challenge_tasks, task_id, test_input_index)
    
    # Set up a parser to inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=ARCPrediction)

    # Create your prompt template. This is very rudimentary! You should edit this to do much better.
    # For example, we don't tell the model what it's first attempt was (so it can do a different one), that might help!
    prompt = PromptTemplate(
        template="You are a bot that is very good at solving puzzles. Below is a list of input and output pairs with a pattern."
                 "You are analyzing an ARC (Abstraction and Reasoning Corpus) puzzle. Each task contains input-output grid pairs. Your goal is to describe the transformations in abstract terms. Focus on spatial or pattern-based operations such as rotations, reflections, translations, scaling, repetition, adjacency, filling, counting, or symmetry. Do not merely restate the colors or shapes. Instead, explain the general rules that map the input to the output, in a way that could apply to unseen test cases."
                 "Output your reasoning as a set of abstract transformation rules, written clearly and systematically, so they could be implemented programmatically."
                 "Identify the pattern, then apply that pattern to the test input to give a final output"
                 "Just give valid json list of lists response back, nothing else. Do not explain your thoughts."
                    "{format_instructions}\n{task_string}\n",
        input_variables=["task_string"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Wrap up your chain with LCEL
    chain = prompt | llm | parser

    # Optional, print out the prompt if you want to see it. If you use LangSmith you could view this there as well.
    # print (f"Prompt:\n\n{prompt.format(task_string=task_string)}")
    
    # Finally, go get your prediction from your LLM. Ths will make the API call.
    output = chain.invoke({"task_string": task_string})

    # Because the output is structured, get the prediction key. If it isn't there, then just get the output
    if isinstance(output, dict):
        prediction = output.get('prediction', output)
    else:
        prediction = output

    # Safety measure to error out if you don't get a list of lists of ints back. This will spark a retry later.
    if not all(isinstance(sublist, list) and all(isinstance(item, int) for item in sublist) for sublist in prediction):
        print("Warning: Output must be a list of lists of integers.")
        print (f"Errored Output: {prediction}")
        raise ValueError("Output must be a list of lists of integers.")
    
    # Let's find the shape of our prediction
    num_rows = len(prediction)
    num_cols = len(prediction[0]) if num_rows > 0 else 0
    print(f"    Prediction Grid Size: {num_rows}x{num_cols}\n")
    
    return prediction


def run_model(challenges, NUM_ATTEMPTS=2, RETRY_ATTEMPTS=3, NUM_TASKS=None):
    """
    challenges: dict a list of challenges. This should come directly from your _challenges file
    NUM_ATTEMPTS: int the number of times to attempt a prediction. The official competition has 2 attempts.
    RETRY_ATTEMPTS: int the number of times to retry a prediction if it fails
    NUM_TASKS: int, If set, this represents the the number of tasks you'd like to test. If None then the all challeneges will be tested

    Loop through your challenges and produce a submission.json file you can submit for a score.
    """

    # A dict to hold your submissions that you'll return after all predictions are made
    submission = {}

    # Run through each task in your challenge set
    for i, task_id in enumerate(challenges):
        task_attempts = []  # List to store all attempts for the current task

        # Go through each test pair to get a prediction. 96% of challenges have 1 pair.
        for t, pair in enumerate(challenges[task_id]['test']):
            print(f"Starting task #{i + 1} ({task_id}), pair #{t+1}")

            # Dictionary to store attempts for the current test pair
            pair_attempts = {}  

            # Run through each prediction attempt
            for attempt in range(1, NUM_ATTEMPTS + 1):
                attempt_key = f"attempt_{attempt}"
                pair_attempts[attempt_key] = [] # Init your attempt

                # Try to get a prediction, with retries in case of failure
                for retry in range(RETRY_ATTEMPTS):
                    try:
                        print(f"    Predicting attempt #{attempt}, retry #{retry + 1}")
                        prediction = get_task_prediction(challenge_tasks=challenges,
                                                         task_id=task_id,
                                                         test_input_index=t)
                        
                        # If you get a valid prediction (list of lists of ints) with no error, then log the attempt
                        pair_attempts[attempt_key] = prediction
                        break  # Break the retry loop if prediction is successful
                    except Exception as e:
                        print(f"Retrying: {e}")
                        if retry == RETRY_ATTEMPTS - 1:
                            pair_attempts[attempt_key] = []  # Assign None if all retries fail

            # After you get your attempts, append them to the task attempts
            task_attempts.append(pair_attempts)

        # Append the task attempts to the submission with the task_id as the key
        submission[task_id] = task_attempts

        # If you want to stop after N tasks, uncomment the below
        if NUM_TASKS is not None and i + 1 == NUM_TASKS:
            break

    return submission


# Load up training tasks
challenges, solutions = load_tasks_from_file(task_set=task_sets['training'])

# Run the model on a single task
submission = run_model(challenges, NUM_TASKS=1)

# Print the submission
print (submission)


def create_submission_file(submission, file_name='submission.json'):
    """
    Save a submission file to the specified file name
    """
    with open(file_name, "w") as file:
        json.dump(submission, file)

    print (f"Submission saved to {file_name}")


def score_submission(submission_file_name, solutions) -> Tuple[float, int]:
    """
    submission_file_name: str, the file name of your submission file
    solutions: dict, the ground truth solutions you'd like to test against
    
    Read a submission from file, score it, then return the score
    """
    print (f"Scoring {submission_file_name}\n")

    # Open your submission file
    with open(submission_file_name, "r") as file:
        submission = json.load(file)

    total_score = 0
    total_tasks = 0

    # Loop through each task in your submission to grade it
    for task_id, task_submission in submission.items():
        total_tasks += 1
        task_score = 0
        num_pairs = len(task_submission)

        # Go through each task. Most will only have 1
        for pair_index, pair_attempts in enumerate(task_submission):
            print(f"Scoring Task {task_id} pair #{pair_index+1}")
            pair_correct = False

            # Look at both of your attempts
            for attempt_key, attempt in pair_attempts.items():
                
                # check to see if one is correct
                if attempt == solutions[task_id][pair_index]:
                    print(f"Task Id {task_id} pair {pair_index+1} {attempt_key} matches solution")
                    pair_correct = True
                    break # If it is correct, log it and break the loop

            if pair_correct:
                task_score += 1

        task_score /= num_pairs
        total_score += task_score

    return {
        'total_score': total_score,
        'total_tasks_scored': total_tasks
    }


def main(task_set='training', NUM_TASKS=None, submission_file_name='submission.json'):
    # Load datasets
    challenges, solutions = load_tasks_from_file(task_set=task_sets[task_set])

    # # Run the model
    submission = run_model(challenges, NUM_TASKS=NUM_TASKS)

    # Create (and overwrite) a submission file
    create_submission_file(submission, file_name=submission_file_name)

    # Score the submission
    score_result = score_submission(solutions = solutions, submission_file_name=submission_file_name)

    print(f"Final score: {score_result['total_score']} of {score_result['total_tasks_scored']} ({round(score_result['total_score']/score_result['total_tasks_scored'] * 100, 2)}%)")


main(task_set='evaluation', NUM_TASKS=None)




