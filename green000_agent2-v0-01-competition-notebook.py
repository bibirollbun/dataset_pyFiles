import io
import os
import shutil
import subprocess

import pandas as pd
import polars as pl

import kaggle_evaluation.konwinski_prize_inference_server


from collections import Counter
import os
import copy
import time
from typing import List
from agent2.agent.agent import Agent
from agent2.agent.tool import Tool
from agent2.file import File
from agent2.tools_common.element_tools.element_viewing import view_element, search_elements, view_file
from agent2.tools_common.element_tools.element_editing import replace_element, replace_element_with, open_element
from agent2.utils.utils import load_project_files, get_completion, get_rating_keys
from agent2.utils.agent_utils import load_agent_from_json
import ast


instance_count = None

def get_number_of_instances(num_instances: int) -> None:
    """ The very first message from the gateway will be the total number of instances to be served.
    You don't need to edit this function.
    """
    global instance_count
    instance_count = num_instances


from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from lmdeploy.cli.utils import get_chat_template

print("Begin loading...")
backend_config = TurbomindEngineConfig(tp=4, enable_prefix_caching=True, cache_max_entry_count = 0.6)
llm = pipeline('/kaggle/input/qwen2.5-coder/transformers/32b-instruct-awq/1',
                backend_config=backend_config)
print("Finished loading!")


gen_config_coder = GenerationConfig(do_sample=True,
                              min_p=0.1,
                              temperature=0.8,
                              max_new_tokens=3000)
coding_start_temp = 0.8
coding_temp_drop = 0.2
coding_end_temp = 0.2     # Temperature decreases over time, this means different runs diverge from one another but still make use of a low temperature


gen_config_rater = GenerationConfig(do_sample=True,
                              min_p=0.15,
                              temperature=0.8,
                              max_new_tokens=3000)


def get_responses(pipeline, gen_config, oai_inputs):
    responses = pipeline(oai_inputs, gen_config=gen_config)
    return [response.text for response in responses]


def get_n_most_common(lists, n):
    counter = Counter()
    for lst in lists:
        # Since each list has no duplicates, we can safely update counts
        for item in lst:
            counter[item] += 1
    # Get the n most common items, which are already unique
    return [item for item, _ in counter.most_common(n)]


i = 0
max_questions = 30
turn_limit = 10
agent_count = 4
rater_iterations = 4
def predict(problem_statement: str, repo_archive: io.BytesIO, pip_packages_archive: io.BytesIO, env_setup_cmds_templates: list[str]) -> str:
    """ Replace this function with your inference code.
    Args:
        problem_statement: The text of the git issue.
        repo_path: A BytesIO buffer path with a .tar containing the codebase that must be patched. The gateway will make this directory available immediately before this function runs.
        pip_packages_archive: A BytesIO buffer path with a .tar containing the wheel files necessary for running unit tests.
        env_setup_cmds_templates: Commands necessary for installing the pip_packages_archive.
    """
    
    # Unpack the codebase to be patched into a directory that won't be exported when
    # the notebook is saved.
    archive_path = '/tmp/repo_archive.tar'
    with open(archive_path, 'wb') as f:
        f.write(repo_archive.read())
    repo_path = 'repo'
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
    shutil.unpack_archive(archive_path, extract_dir=repo_path)
    os.remove(archive_path)

    """
    Unpack pip_packages if you want to run unit tests on your patch.
    Note that editing unit tests with your patch -- even to add valid tests -- can cause your submission to be flagged as a failure.
    Most of the relevant repos use pytest for running tests. You will almost certainly need to run only a subset of the unit tests to avoid running out of inference time.
    """
    pip_archive_dir = '/tmp/pip_packages_archive.tar'
    with open(pip_archive_dir, 'wb') as f:
        f.write(pip_packages_archive.read())
    pip_packages_path = '/path/to/pip_packages'
    if os.path.exists(pip_packages_path):
        shutil.rmtree(pip_packages_path)
    shutil.unpack_archive(pip_archive_dir, extract_dir=pip_packages_path)
    os.remove(pip_archive_dir)

    # Get env setup cmds by setting the pip_packages_path
    env_setup_cmds = [cmd.format(pip_packages_path=pip_packages_path) for cmd in env_setup_cmds_templates]

    # Run env setup for the repo
    subprocess.run(
        "\n".join(env_setup_cmds),
        shell=True,
        executable="/bin/bash",
        cwd=repo_path,
    )

    #### ACTUAL BEHAVIOR
    global i
    global max_questions
    i += 1
    if i > max_questions:
        return None
    
    # Initialize tools
    tools = [
        Tool(view_element),
        Tool(replace_element),
        Tool(replace_element_with),
        Tool(search_elements),
        Tool(view_file),
        Tool(open_element)
    ]

    start_time = time.time()
    print(f"==== STARTING ISSUE {i}/{max_questions} at time {start_time} ====")
    print(problem_statement)

    print("Loading files...")
    original_project_files = load_project_files("repo")

    global agent_count
    global rater_iterations
    total_agents = []
    agent_chats = []
    print("Loading agents...")
    # The tool tokens I used with mistral aren't working with qwen, likely because they are special tokens, so I change them out here
    rater_agent = load_agent_from_json("/kaggle/input/rater-agent/rater_agent.json", tools)
    for j in range(0, agent_count):
        new_agent = load_agent_from_json("/kaggle/input/codeact-agent/codeact_agent.json", tools)
        agent_chats += [new_agent.start(task=problem_statement, files=copy.deepcopy(original_project_files)).openai_completion]
        total_agents += [new_agent]
    for j in range(0, agent_count):
        new_agent = load_agent_from_json("/kaggle/input/md-agent/md_agent.json", tools)
        agent_chats += [new_agent.start(task=problem_statement, files=copy.deepcopy(original_project_files)).openai_completion]
        total_agents += [new_agent]
    for j in range(0, agent_count):
        new_agent = load_agent_from_json("/kaggle/input/xml-agent/xml_agent.json", tools)
        agent_chats += [new_agent.start(task=problem_statement, files=copy.deepcopy(original_project_files)).openai_completion]
        total_agents += [new_agent]
    
    print(f"==== DEPLOYING {len(total_agents)} AGENTS ====")
    global turn_limit
    turn_counter = 0
    current_temperature = coding_start_temp
    while len(agent_chats) > 0 and turn_counter < turn_limit:
        print(f"==== TURN {turn_counter} ====")
        gen_config_coder.temperature = current_temperature
        current_temperature -= coding_temp_drop
        if current_temperature < coding_end_temp:
            current_temperature = coding_end_temp
        responses = get_responses(llm, gen_config_coder, agent_chats)
        agent_counter = 0
        new_chats = []
        for x in total_agents:
            if x.frozen == False:                
                resp = x.step(responses[agent_counter])
                if resp.done == None:
                    print("Continue...")
                    new_chats += [resp.openai_completion]
                else:
                    print("Frozen agent...")
                    x.frozen = True
                agent_counter += 1
        if (time.time() - start_time)/60 > 20:
            print("!!!!!! EMERGENCY ERROR; RAN OUT OF TIME !!!!!!")
            return None
        agent_chats = new_chats
        turn_counter += 1
    print(f"Time taken to generate solutions: {(time.time() - start_time)/60} minutes")
    print(f"==== COLLECTING SOLUTIONS! ====")
    solutions = []
    aggregate_good_references = []
    for a in total_agents:
        diffs = []
        failed = False
        for f in a.cached_state.workspace:
            if f.original_content != f.updated_content:
                diffs += [f.diff(None)]
                try:
                    ast.parse(f.updated_content)
                except Exception:
                    failed = True
        if len(diffs) == 0 or failed:
            print("Discarded broken solution")
            continue
        else:
            print("Working solution got")
            solutions += ["\n".join(diffs)]
            aggregate_good_references += [a.cached_state.saved_elements]
    if len(solutions) == 0:
        print("All failures...")
        return None

    print(f"==== RATING SOLUTIONS! ====")
    print(aggregate_good_references)
    aggregated_files = get_n_most_common(aggregate_good_references, 10)
    print(aggregated_files)
    all_rating_chats = []
    scores = {}
    init_message_cache = rater_agent.init_message
    for sol in solutions:
        scores[sol] = 0
        rater_agent.init_message = (init_message_cache.replace("{{diffs}}", sol))
        all_rating_chats += [rater_agent.start(task=problem_statement, files=original_project_files, copy_saved_elements=aggregated_files).openai_completion] * rater_iterations
    
    responses = get_responses(llm, gen_config_rater, all_rating_chats)

    # Track the current position in the responses list
    current_index = 0
    for sol in solutions:
        # Get the chunk of responses for this solution
        solution_responses = responses[current_index : current_index + rater_iterations]
        # Sum the scores for this solution
        scores[sol] = sum(get_rating_keys(response) for response in solution_responses)
        # Move to the next chunk
        current_index += rater_iterations
        for solresp in solution_responses:
            print(solresp)
        print("SCORE:", scores[sol])
        print("SOLUTION:", sol)

    highest_rated_solution = max(scores.items(), key=lambda x: x[1])[0]
    print(f"Time taken to finish: {(time.time() - start_time)/60} minutes")
    if scores[highest_rated_solution] > 0:
        print(f"Returning highest rated solution with score of {scores[highest_rated_solution]}...")
        print(highest_rated_solution)
        return highest_rated_solution
    else:
        print(f"Highest solution only got score of {scores[highest_rated_solution]}, returning nothing...")
        return None


inference_server = kaggle_evaluation.konwinski_prize_inference_server.KPrizeInferenceServer(
    get_number_of_instances,   
    predict
)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/konwinski-prize/',  # Path to the entire competition dataset
            '/kaggle/tmp/konwinski-prize/',   # Path to a scratch directory for unpacking data.a_zip.
        ),
        use_concurrency=True,  # This can safely be disabled for purposes of local testing if necessary.
    )

