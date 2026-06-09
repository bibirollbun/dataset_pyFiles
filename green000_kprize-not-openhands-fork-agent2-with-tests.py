import os
import time
from datetime import datetime

os.environ['TZ'] = 'Asia/Kolkata'
time.tzset()
print('Starting')

TRUE_START_TIME = time.time()


import os 
is_interactive = False
is_non_interactive = not is_interactive
hide_output_and_error="2>&- >&-" if is_non_interactive else "" 
hide_output_and_error2="2>&- >&-"


!touch /kaggle/working/submission.csv
print('submission csv created')
from time import sleep
if is_non_interactive:sleep(5)


import io
import shutil
import subprocess
import traceback

import pandas as pd
import polars as pl

import kaggle_evaluation.konwinski_prize_inference_server
%env PIP_NO_INDEX=1


from IPython.core.magic import register_cell_magic

@register_cell_magic
def notify(line, cell):
    exec(cell, globals())


"""def start_server():
    !python3.10 -m pip install vllm --find-links /kaggle/input/vllm-nb-na -q  2>&- >&-
    !python3.10 -m pip install triton --find-links /kaggle/input/vllm-nb-na
    !python3.10 -m pip uninstall pynvml -y
    import subprocess
    log_file = open("/kaggle/working/vllm_output.log", "w")
    
    # background task
    command = [
        "python3.10", 
        "-m", 
        "vllm.scripts", 
        "serve",
        model_path,
        "--tensor_parallel_size", "4",
        "--gpu_memory_utilization", "0.9",
        "--enforce_eager",
        "--enable-chunked-prefill",
        "--enable_prefix_caching",
        "--max_model_len", "25000"
    ]
    
    process = subprocess.Popen(command, stdout=log_file, stderr=log_file, start_new_session=True)
    
    print(f"Background process started with PID: {process.pid}")
def start_server():
    !python3.10 -m pip install lmdeploy --find-links /kaggle/input/lmdeploy-thingy -v
    !python3.10 -m pip uninstall pynvml -y
    
    import subprocess
    log_file = open("/kaggle/working/vllm_output.log", "w")
    
    # background task
    command = [
        "python3.10", 
        "-m", 
        "lmdeploy", 
        "serve",
        "api_server",
        model_path,
        "--tp", "4",
        "--server-port", "8000",
        "--enable-prefix-caching",
        "--cache-max-entry-count", "0.6"
    ]
    
    process = subprocess.Popen(command, stdout=log_file, stderr=log_file, start_new_session=True)
    
    print(f"Background process started with PID: {process.pid}")"""


"""setup_done = False
SERVE = True
!cp -r /kaggle/input/openhands-fork-offline-version/Kevin /kaggle/working

def kaggle_setup():
    global setup_done
    if setup_done: return
    setup_done = True
    
    if SERVE: start_server()
    
    !dpkg -i  $(ls /kaggle/input/openhands-fork-offline-version/apt/*.deb) 2>&- >&-
    
    !python3.10 -m pip install poetry -q --no-index --find-links=/kaggle/input/openhands-fork-offline-version/pip 
    
    %cd /kaggle/working/Kevin
    
    !python3.10 -m poetry env use python3.12
    !python3.10 -m poetry run pip install -q setuptools --no-index --find-links /kaggle/input/openhands-fork-offline-version/poetry
    !python3.10 -m poetry run pip install -q --no-build-isolation grpclib --no-index --find-links /kaggle/input/openhands-fork-offline-version/poetry 
    #!python3.10 -m poetry run pip install -q -r /kaggle/input/openhands-fork-offline-version/Kevin/requirements.txt --no-index --find-links /kaggle/input/openhands-fork-offline-version/poetry

    if SERVE:
        import requests
        import time
        try:
            while True:
                try:
                    !tail -1 /kaggle/working/vllm_output.log
                    requests.get('http://localhost:8000/v1/models')
                    break
                except Exception as e:
                    print(end='.')
                    time.sleep(30)
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
    
    !git config --global init.defaultBranch main"""
setup_done = False
SERVE = True
!cp -r /kaggle/input/openhands-fork-offline-version/Kevin /kaggle/working

def kaggle_setup():
    global setup_done
    if setup_done: return
    setup_done = True
    
    !python3.10 -m pip install lmdeploy --find-links /kaggle/input/lmdeploy-thingy -v
    !python3.10 -m pip uninstall pynvml -y
    
    !dpkg -i  $(ls /kaggle/input/openhands-fork-offline-version/apt/*.deb) 2>&- >&-
    
    !python3.10 -m pip install poetry -q --no-index --find-links=/kaggle/input/openhands-fork-offline-version/pip 
    
    %cd /kaggle/working/Kevin
    
    !python3.10 -m poetry env use python3.12
    !python3.10 -m poetry run pip install -q setuptools --no-index --find-links /kaggle/input/openhands-fork-offline-version/poetry
    !python3.10 -m poetry run pip install -q --no-build-isolation grpclib --no-index --find-links /kaggle/input/openhands-fork-offline-version/poetry 
    
    !git config --global init.defaultBranch main


%env LOCAL_RUNTIME_MODE=1
%env DISABLE_BROWSER=1
%env USER=root
%env OPENHANDS_REPO_PATH=/kaggle/working/Kevin
%env POETRY_VIRTUALENVS_PATH=/root/.cache/pypoetry/virtualenvs
%env POETRY_CACHE_DIR=/kaggle/working/poetry
%env SKIP_DEPENDENCY_CHECK=1
%env USE_PEXPECT=1
%env DISABLE_METRICS=1
%env SINGLE_LOG_FOLDER=1
%env SWE_BENCH=1

os.environ['PYTHONPATH'] = '/kaggle/working/Kevin:' + os.environ['PYTHONPATH']


model_path1 = '/kaggle/input/qwen2.5-coder/transformers/32b-instruct-awq/1'
model_path2 = "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-32b/1"
model_path = model_path1
temperature = 0.5 if model_path == model_path2 else 0
model = f'hosted_vllm/{model_path}'
repo_path = '/testbed'
if is_non_interactive:
    base_url='http://localhost:8000/v1'
    max_iterations=100
else:
    base_url='https://91fb-34-86-166-77.ngrok-free.app/v1'
    max_iterations=25

config=f'''
[core]
workspace_base ='{repo_path}'
runtime='local'
max_iterations={max_iterations}

[llm]
model='{model}'
base_url='{base_url}'
max_input_tokens = 62_000
temperature={temperature}
# use_group='groq'

[llm.groq]
model='groq/deepseek-r1-distill-llama-70b'
'''
with open('/kaggle/working/Kevin/config.toml', 'w') as f:
    f.write(config)


instance_count = None

def get_number_of_instances(num_instances: int) -> None:
    """ The very first message from the gateway will be the total number of instances to be served.
    You don't need to edit this function.
    """
    global instance_count
    instance_count = num_instances


setup_done = 0
kaggle_setup()


os.environ['PATH'] = '/testbed/.venv/bin:' + os.environ['PATH']


import tomlkit
def get_project_name():
    try:
        pyproject_file = '/testbed/pyproject.toml'
    
        with open(pyproject_file, 'r') as f:
            pyproject = tomlkit.parse(f.read())
    
        project_name = pyproject['project']['name']
        return project_name
    except Exception as e:
        print(e)
        return ''
    if not pyproject.get('tool',{}).get('uv'):
        pyproject['tool']['uv'] = {
            'override-dependencies': [project_name],
            'sources': {
                project_name: { 'workspace': True }
            }
        }

    with open(pyproject_file, 'w') as f:
        tomlkit.dump(pyproject, f)


%env PIP_FIND_LINKS=/kaggle/input/wheels-for-kprize-instances


%env POETRY_ALIAS=python3.10 -m poetry


from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from lmdeploy.cli.utils import get_chat_template

print("Begin loading...")
backend_config = TurbomindEngineConfig(tp=4, enable_prefix_caching=True, cache_max_entry_count = 0.7)
llm = pipeline('/kaggle/input/qwen2.5-coder/transformers/32b-instruct-awq/1',
                backend_config=backend_config)
print("Finished loading!")

gen_config_replication = GenerationConfig(do_sample=True,
                              min_p=0.1,
                              temperature=0.6,
                              max_new_tokens=2500)

gen_config = GenerationConfig(do_sample=True,
                              min_p=0.1,
                              temperature=0.6,
                              max_new_tokens=2500)

gen_config_coder = GenerationConfig(do_sample=True,
                              min_p=0.1,
                              temperature=0.8,
                              max_new_tokens=2500)
coding_start_temp = 0.8
coding_temp_drop = 0.2
coding_end_temp = 0.2
def get_responses(pipeline, gen_config, oai_inputs):
    responses = pipeline(oai_inputs, gen_config=gen_config)
    return [response.text for response in responses]
    
print(get_responses(llm, gen_config, [[{"role":"user", "content":"What is 1+1*6"}],
                     [{"role":"user","content":"What is 7*12? Respond like a pirate?"}],
                     [{"role":"user","content":"Please get the weather using the get_weather tool, wrapped inside <tool> </tool> XML tool calling format."}]]))


from collections import Counter
import os
import copy
import time
import random
from typing import List
from pathlib import Path

from agent2.agent.agent import Agent
from agent2.agent.tool import Tool
from agent2.file import File
from agent2.element import Element
from agent2.utils.utils import load_project_files
from agent2.utils.agent_utils import load_agent_from_json
from agent2.agent.agent_state import AgentState
from agent2.agent.tool_settings import ToolSettings
import ast

def get_n_most_common(lists, n):
    counter = Counter()
    for lst in lists:
        # Since each list has no duplicates, we can safely update counts
        for item in lst:
            counter[item] += 1
    # Get the n most common items, which are already unique
    return [item for item, _ in counter.most_common(n)]

super_secret_test_str = ""
super_secret_project_dir = []
def run_test(state: AgentState, settings: ToolSettings, debug_message="Made edits, running code..."):
    """
    Run the issue replication script. Will error if the issue still persists, otherwise prints SUCCESS.
    
    Args:
        None
    
    Returns:
        The output of the issue replication script.
    
    Example:
        Run the issue replication script.
    Tool Call:
        {"name": "run_test", "arguments": {}}
    """
    print(debug_message)
    # repo_path <- cwd
    # super_secret_test_str <- the string to run
    # super_secret_project_dir
    for ffile in super_secret_project_dir:
        if ffile.original_content != ffile.updated_content:
            with open(ffile.path, 'w') as writefile:
                writefile.write(ffile.updated_content)
        else:
            with open(ffile.path, 'w') as writefile:
                writefile.write(ffile.original_content)
    with open("super_special_python_file.py", 'w') as writefile:
        writefile.write(super_secret_test_str)
    try:
        result = subprocess.run(
            "python super_special_python_file.py",
            shell=True,
            executable="/bin/bash",
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=15
        )
    except Exception as e:
        return (f"Error running code: {e}", None, None)
    return ("```output\n" + "\n".join(("Stdout:\n" + result.stdout + "\nStderr:\n" + result.stderr).splitlines()[-100:])[-7000:] + "\n```", None, None)


first_question = False
alloted_time = 8*60*60
comp_time_alloted = 22*60*60

stderr_count=0
pythonnotprint_count = 0
nopassfail_count = 0
allowed_fails=25

search_turn_limit = 9 # 8
search_batch = 3 # 3

fail_to_fail_count = 4

replication_top_select = 8
replication_gurantee_refs = 3
replication_random_refs = 3
replication_batch = 25 # 25
replication_min_req = 8 # 8

fixer_top_select = 6
fixer_random_refs = 3
fixer_turn_limit = 9 # 8 
fixer_batch = 4 # 4

pass_to_pass_percent = 0.65
def predict(problem_statement: str, repo_archive: io.BytesIO, pip_packages_archive: io.BytesIO, env_setup_cmds_templates: list[str]) -> str:
    """ Replace this function with your inference code.
    Args:
        problem_statement: The text of the git issue.
        repo_path: A BytesIO buffer path with a .tar containing the codebase that must be patched. The gateway will make this directory available immediately before this function runs.
        pip_packages_archive: A BytesIO buffer path with a .tar containing the wheel files necessary for running unit tests.
        env_setup_cmds_templates: Commands necessary for installing the pip_packages_archive.
    """
    global super_secret_test_str, super_secret_project_dir
    global comp_time_alloted, alloted_time, stderr_count, pythonnotprint_count, nopassfail_count, first_question
    global instance_count
    if instance_count is not None and instance_count > 100:
        print(instance_count)
        alloted_time = comp_time_alloted
    if first_question:
        first_question = False
        print("Skip first question.")
        return None
    if time.time() > TRUE_START_TIME + alloted_time:
        print("EARLY STOP!")
        return None
    else:
        print(time.time())
        print(TRUE_START_TIME)
        print(TRUE_START_TIME + alloted_time)

    start_time = time.time()
    print(f"==== STARTING ISSUE at time {start_time} ====")
    print(problem_statement)
    try:
        print("==== BEGINNING ENV SETUP AND QUICK TESTING ====")
        
        %cd /
        # Unpack the codebase to be patched
        with open('repo_archive.tar', 'wb') as f:
            f.write(repo_archive.read())
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
        shutil.unpack_archive('repo_archive.tar', extract_dir=repo_path)
        os.remove('repo_archive.tar')
    
        """
        Unpack pip_packages if you want to run unit tests on your patch.
        Note that editing unit tests with your patch -- even to add valid tests -- can cause your submission to be flagged as a failure.
        Most of the relevant repos use pytest for running tests. You will almost certainly need to run only a subset of the unit tests to avoid running out of inference time.
        """
        with open('pip_packages_archive.tar', 'wb') as f:
            f.write(pip_packages_archive.read())
        pip_packages_path = '/path/to/pip_packages'
        if os.path.exists(pip_packages_path):
            shutil.rmtree(pip_packages_path)
        shutil.unpack_archive('pip_packages_archive.tar', extract_dir=pip_packages_path)
        os.remove('pip_packages_archive.tar')
    
        # Get env setup cmds by setting the pip_packages_path
        env_setup_cmds = [cmd.format(pip_packages_path=pip_packages_path).replace('uv pip','pip').replace(' --link-mode=symlink','') for cmd in env_setup_cmds_templates[2:]]
        project_name=get_project_name()
        %cd $repo_path
        !uv venv --python python3.11
        !python -m ensurepip
        !git init
        !git add .
        if project_name=='astroid':
            !pip install pylint
        # Run env setup for the repo
        env_setup_cmds = "\n".join(env_setup_cmds)
        result = subprocess.run(
            env_setup_cmds,
            shell=True,
            executable="/bin/bash",
            cwd=repo_path,
            capture_output=True,
            text=True
            
        )
        print(f"Setup environment results:\n{result.stderr}\n{result.stdout}\n{result.returncode}\n{env_setup_cmds}")
        if result.returncode != 0 or "failed to build" in result.stderr.lower() or "importerror" in result.stderr.lower():
            print("NOTEBOOK FAILED TO SETUP")
            print(stderr_count)
            stderr_count += 1
            return None
        python_content = """
try:
    import astropy
except Exception as e:
    print(e)
try:
    import astroid
except Exception as e:
    print(e)
print(\"SUCCESS!\")"""
        with open("super_special_python_file.py", "w") as f:
            f.write(python_content)
        
        test_content = """
def test_assert_right_1():
    assert 1 == 1
def test_assert_wrong_1():
    assert 2+2==5
def test_assert_right_2():
    assert 1-1==0
def test_assert_wrong_2():
    assert 5+5==10"""
    
        with open("super_special_test_file.py", 'w') as f:
            f.write(test_content)

        print("==== CHECKING IF PYTHON FILES WORK ====")
        !ls
        result = subprocess.run(
            "set -e\npython super_special_python_file.py",
            shell=True,
            executable="/bin/bash",
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Failed to run python file!\nstderr:\n{result.stderr}\nSetup stdout:\n{result.stdout}\nSetup returncode\n{result.returncode}")
            pythonnotprint_count += 1
            print(pythonnotprint_count)
            return None
        else:
            print(f"Successfully ran python file!!\nstderr:\n{result.stderr}\nSetup stdout:\n{result.stdout}\nSetup returncode\n{result.returncode}")
    
        print("==== PYTEST INSTALLATION CHECK ====")
        result = subprocess.run(
            "set -e\npytest --version",
            shell=True,
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        print("Stdout:", result.stdout)
        print("Stderr:", result.stderr)
        print("Result code:", result.returncode)
        
        print("==== CHECKING IF PYTEST WORKS ====")
        result = subprocess.run(
            "set -e\npytest --color=no --ff -rA -q --tb=no super_special_test_file.py 2>&1",
            shell=True,
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        print("Stdout:", result.stdout)
        print("Stderr:", result.stderr) 
        print("Result code:", result.returncode)
        if ("passed" not in result.stdout.lower() and "failed" not in result.stdout.lower()) or (result.returncode != 0 and result.returncode != 1):
            print("NO TESTS WORKED!")
            nopassfail_count += 1
            print(nopassfail_count)
            return None
        else:
            print("SUCCESS!")

        print("*** LOADING FILES ***")
        original_project_files = load_project_files(".")
        print(original_project_files[0].path)
        
        print("==== FINISH SETUP! ====")
        search_agents = []
        test_agents = []
        search_chats = []
        search_start_time = time.time()
        print("Loading agents...")
        for i in range(0, search_batch):
            new_agent = load_agent_from_json("/kaggle/input/new-agent2-agents/search_codeact_agent.json")
            search_chats += [new_agent.start(task=problem_statement, files=original_project_files).openai_completion]
            search_agents += [new_agent]
        for i in range(0, search_batch):
            new_agent = load_agent_from_json("/kaggle/input/new-agent2-agents/search_md_agent.json")
            search_chats += [new_agent.start(task=problem_statement, files=original_project_files).openai_completion]
            search_agents += [new_agent]
        print("Loading test search agents...")
        
        new_agent = load_agent_from_json("/kaggle/input/new-agent2-agents/md_agent_tests.json")
        search_chats += [new_agent.start(task=problem_statement, files=original_project_files).openai_completion]
        search_agents += [new_agent]
        test_agents += [new_agent]
        new_agent = load_agent_from_json("/kaggle/input/new-agent2-agents/codeact_agent_tests.json")
        search_chats += [new_agent.start(task=problem_statement, files=original_project_files).openai_completion]
        search_agents += [new_agent]
        test_agents += [new_agent]
        
            
        print(f"==== DEPLOYING {len(search_agents)} SEARCH AGENTS ====")
        turn_counter = 0
        while len(search_chats) > 0 and turn_counter < search_turn_limit:
            print(f"==== TURN {turn_counter} ====")
            responses = get_responses(llm, gen_config, search_chats)
            agent_counter = 0
            new_chats = []
            for x in search_agents:
                if x.frozen == False:                
                    resp = x.step(responses[agent_counter])
                    if resp.done == None:
                        print("Continue...")
                        new_chats += [resp.openai_completion]
                    else:
                        print("Frozen agent...")
                        x.frozen = True
                    agent_counter += 1
            if (time.time() - start_time)/60 > 50:
                print("!!!!!! EMERGENCY ERROR; RAN OUT OF TIME !!!!!!")
                return None
            search_chats = new_chats
            turn_counter += 1
        print(f"Time taken to search for relevant elements: {(time.time() - search_start_time)/60} minutes")
        print(f"==== COLLECTING RESULTS! ====")
        reference_elements = []
        for a in search_agents:
            """
            if a in test_agents:
                continue
            """
            reference_elements += [a.cached_state.saved_elements]
        
        fail_to_fail_tests = []
        pytest_run_commands = []
        for t in test_agents:
            for testtorun in t.cached_state.saved_elements:
                file_path = testtorun[0]
                element_id = testtorun[1]
                file = next((f for f in original_project_files if f.path.lower() == file_path.lower()), None)
                if not file:
                    continue
                all_elements = []
                stack = list(file.elements)
                while stack:
                    element = stack.pop()
                    all_elements.append(element)
                    stack.extend(element.elements)
                
                element = next((e for e in all_elements if e.identifier.lower() == element_id.lower()), None)
                if not element:
                    continue
                if len(element.elements) > 2:
                    continue
                test_cmd_final = testtorun[0] + "::" + testtorun[1].replace(".", "::")
                if "test" in testtorun[0] and "test" in testtorun[1].lower() and test_cmd_final not in pytest_run_commands:
                    print("Got test:", test_cmd_final)
                    pytest_run_commands += [test_cmd_final]
                else:
                    print("Not test")
        if len(pytest_run_commands) < 3:
            print("NO TESTS FOUND")
        pytest_run_commands = pytest_run_commands[0:fail_to_fail_count]
        pytest_run_commands = " ".join(pytest_run_commands)
        print(pytest_run_commands)
        try:
            result = subprocess.run(
                f"pytest --color=no --ff -rA -q --tb=no {pytest_run_commands} 2>&1",
                shell=True,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
        except Exception as e:
            print("TIMEOUT")
            return None
        print("Stdout:", result.stdout)
        print("Stderr:", result.stderr) 
        print("Result code:", result.returncode)
        if "passed" not in result.stdout.lower() and "failed" not in result.stdout.lower():
            # Tests didnt work
            print("Test fail!")
            return None
        
        if len(reference_elements) < 4:
            print("Not enough references found...")
            return None
        print(f"*** All references: ***\n{reference_elements}")
        sorted_refs = get_n_most_common(reference_elements, replication_top_select + replication_gurantee_refs + 2)
        # Discard classes
        new_sorted_refs = []
        for ref in sorted_refs:
            file_path = ref[0]
            element_id = ref[1]
            file = next((f for f in original_project_files if f.path.lower() == file_path.lower()), None)
            if not file:
                continue
            all_elements = []
            stack = list(file.elements)
            while stack:
                element = stack.pop()
                all_elements.append(element)
                stack.extend(element.elements)
            
            element = next((e for e in all_elements if e.identifier.lower() == element_id.lower()), None)
            if not element:
                continue
            if len(element.elements) < 7:
                new_sorted_refs += [ref]
        sorted_refs = new_sorted_refs
        print(f"*** FILTERED: ***\n{sorted_refs}")
        replication_gurantee_refs_picked = sorted_refs[0:replication_gurantee_refs]
        replication_random_refs_picked = sorted_refs[replication_gurantee_refs:]
        fixer_random_refs_picked = sorted_refs[0:fixer_top_select]
        print(f"*** Replication guranteed references: ***\n{replication_gurantee_refs_picked}")
        print(f"*** Replication random references: ***\n{replication_random_refs_picked}")
        print(f"*** Replication mixed references: ***\n{fixer_random_refs_picked}")

        print("==== GENERATING REPLICATIONS ====")
        replication_start_time = time.time()
        all_replication_chats = []
        for x in range(0, replication_batch):
            new_agent = load_agent_from_json("/kaggle/input/new-agent2-agents/replication_maker.json")
            new_agent.get_import_block_saved = True
            rep_references = random.sample(replication_random_refs_picked, min(replication_random_refs, len(replication_random_refs_picked))) + replication_gurantee_refs_picked
            all_replication_chats += [new_agent.start(task=problem_statement, files=original_project_files, copy_saved_elements=rep_references).openai_completion]
        responses = get_responses(llm, gen_config_replication, all_replication_chats)
        if (time.time() - start_time)/60 > 50:
            print("!!!!!! EMERGENCY ERROR; RAN OUT OF TIME !!!!!!")
            return None
        for x in range(0, len(responses)):
            resp = responses[x]
            all_replication_chats[x].append({"role":"assistant", "content":resp})
            print(resp)
            if resp.count("```") < 2:
                all_replication_chats[x].append({"role":"user", "content":"No code block found... Please output your replication code wrapped in code blocks."})
                print("No replication found.")
                continue
            code_block_segments = resp.split("```")
            last_section = code_block_segments[-2]
            # Remove excess newline at the start
            last_section = last_section[last_section.find("\n"):].strip()
            code_block = last_section
            try:
                ast.parse(code_block)
                print("GOT CODE BLOCK")
                
                # Try to run replication
                with open("super_special_python_file.py", "w") as f:
                    f.write(code_block)
                try:
                    result = subprocess.run(
                        "python super_special_python_file.py",
                        shell=True,
                        executable="/bin/bash",
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                except Exception as e:
                    all_replication_chats[x].append({"role":"user", "content":"Replication script timed out... Please fix it."})
                    print("TIMEOUT")
                    continue
                print(result.stdout + result.stderr)                
                all_replication_chats[x].append({"role":"user", "content":"```output\n" + "\n".join((result.stdout + result.stderr).splitlines()[-100:])[-7000:] + "\n```\nBased on the output of your replication block, output a new and improved replication wrapped within code blocks. Fix any issues that occured, and add debug print statements if necessary. After you output your code block, do not output anything else, you should only have one code block in your output."})
                continue
            except Exception:
                print("CODE BLOCK FAILED TO PARSE")
                all_replication_chats[x].append({"role":"user", "content":"Replication script failed to parse... Please fix it."})
                continue
        responses = get_responses(llm, gen_config, all_replication_chats)
        true_replication_chats = []
        real_replications = []
        # Grab functional replications
        for x in range(0, len(responses)):
            resp = responses[x]
            print(resp)
            if resp.count("```") < 2:
                real_replications += [None]
                print("No test block found.")
                continue
            code_block_segments = resp.split("```")
            last_section = code_block_segments[-2]
            # Remove excess newline at the start
            last_section = last_section[last_section.find("\n"):].strip()
            code_block = last_section
            try:
                ast.parse(code_block)
                print("GOT CODE BLOCK")
                
                # Try to run replication
                with open("super_special_python_file.py", "w") as f:
                    f.write(code_block)
                try:
                    result = subprocess.run(
                        "python super_special_python_file.py",
                        shell=True,
                        executable="/bin/bash",
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                except Exception as e:
                    real_replications += [None]
                    print("TIMEOUT")
                    continue
                print(result.stdout + result.stderr)
                if "SUCCESS" in result.stdout or "SUCCESS" in result.stderr:
                    print("Found success...")
                    real_replications += [None]
                    continue
                else:
                    real_replications += [code_block]
                    all_replication_chats[x].append({"role":"assistant", "content":responses[x]})
                    all_replication_chats[x].append({"role":"user", "content":"```output\n" + "\n".join((result.stdout + result.stderr).splitlines()[-100:])[-7000:] + "\n```\nFirst review and output a quick analysis of the output of running your code. After that, determine if the replication was successful or not. If it was, output REPLICATION SUCCESS, otherwise output REPLICATION FAILURE. End your output after that, do not output anything else. Begin your analysis."})
                    true_replication_chats.append(all_replication_chats[x])
                    continue
            except Exception:
                print("CODE BLOCK FAILED TO PARSE")
                real_replications += [None]
                continue
        
        # Check which replications are actually successful
        print("==== ASSESSING REPLICATION SUCCESS ====")
        responses = get_responses(llm, gen_config, true_replication_chats)
        if (time.time() - start_time)/60 > 50:
            print("!!!!!! EMERGENCY ERROR; RAN OUT OF TIME !!!!!!")
            return None
        counter = 0
        verified_replications = []
        for x in range(0, len(real_replications)):
            if real_replications[x] == None:
                continue
            resp = responses[counter]
            print(resp)
            if "REPLICATION SUCCESS" in resp and "REPLICATION FAILURE" not in resp:
                print("GOOD!")
                verified_replications += [real_replications[x]]
            counter += 1
        print("SUCCESSFUL REPLICATIONS:", len(verified_replications))

        if len(verified_replications) < replication_min_req:
            print("NOT ENOUGH REPLICATIONS, QUITTING...")
            return None
        verified_replications = verified_replications[0:9]
        print(f"Time taken to create replications: {(time.time() - replication_start_time)/60} minutes")
        coding_start_time = time.time()
        code_agents = []
        code_agents_chats = []
        code_agents_replication_files = []
        replication_id = 0
        for j in range(0, fixer_batch):
            chosen_rep = verified_replications[replication_id]
            with open("super_special_python_file.py", 'w') as writefile:
                writefile.write(chosen_rep)
            try:
                result = subprocess.run(
                    "python super_special_python_file.py",
                    shell=True,
                    executable="/bin/bash",
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=20
                )
                resultout = "\n".join(("Stdout:\n" + result.stdout + "\nStderr:\n" + result.stderr).splitlines()[-100:])[-7000:]
            except Exception as e:
                print("TIMEOUT")
                resultout = "The replication script timed out when running for some reason..."
            fix_references = random.sample(fixer_random_refs_picked, min(replication_random_refs, len(fixer_random_refs_picked)))
            new_agent = load_agent_from_json("/kaggle/input/new-agent2-agents/codeact_agent_fixer.json")
            new_agent.tools_list += [Tool(run_test)]
            new_agent.init_message = new_agent.init_message.replace("{{replication_script}}", f"```python\n{chosen_rep}\n```")
            new_agent.init_message = new_agent.init_message.replace("{{replication_output}}", f"```output\n{resultout}\n```")
            code_agents_chats += [new_agent.start(task=problem_statement, files=copy.deepcopy(original_project_files), copy_saved_elements=fix_references).openai_completion]
            code_agents_replication_files += [chosen_rep]
            code_agents += [new_agent]
            
            replication_id += 1
            if replication_id == len(verified_replications):
                replication_id = 0
        for j in range(0, fixer_batch):
            chosen_rep = verified_replications[replication_id]
            with open("super_special_python_file.py", 'w') as writefile:
                writefile.write(chosen_rep)
            try:
                result = subprocess.run(
                    "python super_special_python_file.py",
                    shell=True,
                    executable="/bin/bash",
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=20
                )
                resultout = "\n".join(("Stdout:\n" + result.stdout + "\nStderr:\n" + result.stderr).splitlines()[-100:])[-7000:]
            except Exception as e:
                print("TIMEOUT")
                resultout = "The replication script timed out when running for some reason..."
            fix_references = random.sample(fixer_random_refs_picked, min(replication_random_refs, len(fixer_random_refs_picked)))
            new_agent = load_agent_from_json("/kaggle/input/new-agent2-agents/md_agent_fixer.json")
            new_agent.tools_list += [Tool(run_test)]
            new_agent.init_message = new_agent.init_message.replace("{{replication_script}}", f"```python\n{chosen_rep}\n```")
            new_agent.init_message = new_agent.init_message.replace("{{replication_output}}", f"```output\n{resultout}\n```")
            code_agents_chats += [new_agent.start(task=problem_statement, files=copy.deepcopy(original_project_files), copy_saved_elements=fix_references).openai_completion]
            code_agents_replication_files += [chosen_rep]
            code_agents += [new_agent]
            
            replication_id += 1
            if replication_id == len(verified_replications):
                replication_id = 0
        for j in range(0, fixer_batch):
            chosen_rep = verified_replications[replication_id]
            with open("super_special_python_file.py", 'w') as writefile:
                writefile.write(chosen_rep)
            try:
                result = subprocess.run(
                    "python super_special_python_file.py",
                    shell=True,
                    executable="/bin/bash",
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=20
                )
                resultout = "\n".join(("Stdout:\n" + result.stdout + "\nStderr:\n" + result.stderr).splitlines()[-100:])[-7000:]
            except Exception as e:
                print("TIMEOUT")
                resultout = "The replication script timed out when running for some reason..."
            fix_references = random.sample(fixer_random_refs_picked, min(replication_random_refs, len(fixer_random_refs_picked)))
            new_agent = load_agent_from_json("/kaggle/input/new-agent2-agents/xml_agent_fixer.json")
            new_agent.tools_list += [Tool(run_test)]
            new_agent.init_message = new_agent.init_message.replace("{{replication_script}}", f"```python\n{chosen_rep}\n```")
            new_agent.init_message = new_agent.init_message.replace("{{replication_output}}", f"```output\n{resultout}\n```")
            code_agents_chats += [new_agent.start(task=problem_statement, files=copy.deepcopy(original_project_files), copy_saved_elements=fix_references).openai_completion]
            code_agents_replication_files += [chosen_rep]
            code_agents += [new_agent]
            
            replication_id += 1
            if replication_id == len(verified_replications):
                replication_id = 0
        print(f"==== DEPLOYING {len(code_agents_chats)} CODER AGENTS ====")
        turn_counter = 0
        current_temperature = coding_start_temp
        while len(code_agents_chats) > 0 and turn_counter < fixer_turn_limit:
            print(f"==== TURN {turn_counter} ====")
            gen_config_coder.temperature = current_temperature
            current_temperature -= coding_temp_drop
            if current_temperature < coding_end_temp:
                current_temperature = coding_end_temp
            responses = get_responses(llm, gen_config_coder, code_agents_chats)
            agent_counter = 0
            new_chats = []
            for x in range (0, len(code_agents)):
                curagent = code_agents[x]
                super_secret_test_str = code_agents_replication_files[x]
                super_secret_project_dir = curagent.cached_state.workspace
                if curagent.frozen == False:                
                    resp = curagent.step(responses[agent_counter])
                    if resp.done == None:
                        print("Continue...")
                        new_chats += [resp.openai_completion]
                    else:
                        print("Frozen agent...")
                        curagent.frozen = True
                    agent_counter += 1
            if (time.time() - start_time)/60 > 50:
                print("!!!!!! EMERGENCY ERROR; RAN OUT OF TIME !!!!!!")
                return None
            code_agents_chats = new_chats
            turn_counter += 1
        print(f"Time taken to generate solutions: {(time.time() - coding_start_time)/60} minutes")
        print(f"==== COLLECTING SOLUTIONS! ====")
        solutions = []
        solutions_sources = []
        for a in code_agents:
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
                solutions_sources += [a.cached_state.workspace]
        print(f"==== CHECKING SOLUTIONS! ====")
        best_solution = None
        best_solution_score = pass_to_pass_percent
        for xx in range(0, len(solutions)):
            sol = solutions[xx]
            print(sol)
            sol_source = solutions_sources[xx]
            for ffile in sol_source:
                if ffile.original_content != ffile.updated_content:
                    with open(ffile.path, 'w') as writefile:
                        writefile.write(ffile.updated_content)
                else:
                    with open(ffile.path, 'w') as writefile:
                        writefile.write(ffile.original_content)
            correct = 0
            total = len(verified_replications)
            
            try:
                result = subprocess.run(
                    f"pytest --color=no --ff -rA -q --tb=no {pytest_run_commands} 2>&1",
                    shell=True,
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            except Exception as e:
                print("TIMEOUT")
                continue
            print(result.stdout)
            print(result.stderr)
            print(result.returncode)
            if "failed" in result.stdout.lower() or "passed" not in result.stdout.lower():
                print("FAILED TESTS")
                continue
            
            for yy in verified_replications:
                with open("super_special_python_file.py", 'w') as writefile:
                    writefile.write(yy)
                try:
                    result = subprocess.run(
                        "python super_special_python_file.py",
                        shell=True,
                        executable="/bin/bash",
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    print("STDOUT:\n" + result.stdout + "STDERR:\n" + result.stderr)
                except Exception as e:
                    print("TIMEOUT")
                    continue
                if "SUCCESS" in result.stdout or "SUCCESS" in result.stderr:
                    print("Get point")
                    correct += 1
            score = float(correct) / float(total)
            print("Score:", score)
            if score > best_solution_score:
                best_solution_score = score
                best_solution = sol
                print("IS NEW BEST")

        # Clear files:
        file_path = Path(f'super_special_python_file.py')
        if file_path.exists():
            file_path.unlink()
            print(f"File {file_path} has been deleted.")
        else:
            print(f"File {file_path} does not exist.")

        file_path = Path(f'super_special_test_file.py')
        if file_path.exists():
            file_path.unlink()
            print(f"File {file_path} has been deleted.")
        else:
            print(f"File {file_path} does not exist.")

        for ffile in original_project_files:
            with open(ffile.path, 'w') as writefile:
                writefile.write(ffile.original_content)
        print(f"TOTAL TIME TAKEN: {(time.time() - start_time)/60} minutes")
        if best_solution != None:
            print("Got best solution!")
            print("Score:", best_solution_score)
            print(best_solution)
            return best_solution
        return None
    except Exception as e:
        print(e)
        traceback.print_exc()
        return None


%%notify
try:
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
except KeyboardInterrupt:
    print('KeyboardInterrupt')
except Exception as e:
    print(e)
    traceback.print_exc()
finally:
    print('Inference done')

























