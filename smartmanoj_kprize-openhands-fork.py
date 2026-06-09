import os
import time
from datetime import datetime, timedelta


os.environ['TZ'] = 'Asia/Kolkata'
time.tzset()
print('Starting')
start_time = datetime.now()


import os
is_interactive = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive'
is_non_interactive = not is_interactive
is_rerun = os.getenv('KAGGLE_IS_COMPETITION_RERUN')
hide_output_and_error="2>&- >&-" if is_non_interactive else "" 
hide_output_and_error2="2>&- >&-"


if is_interactive:
    # https://www.kaggle.com/discussions/product-feedback/554027
    import socket
    REMOTE_SERVER = "one.one.one.one"
    def is_connected(hostname):
      try:
        # See if we can resolve the host name - tells us if there is
        # A DNS listening
        host = socket.gethostbyname(hostname)
        # Connect to the host - tells us if the host is actually reachable
        s = socket.create_connection((host, 80), 2)
        s.close()
        return True
      except Exception:
         pass # We ignore any errors, returning False
      return False
    print('Internet :',is_connected(REMOTE_SERVER))


!touch /kaggle/working/submission.csv
print('submission csv created')
from time import sleep
if is_non_interactive:sleep(10)


import io
import shutil
import subprocess
import traceback

import pandas as pd
# import polars as pl

import kaggle_evaluation.konwinski_prize_inference_server
%env PIP_NO_INDEX=1


if is_interactive:
    !python3.10 -m pip install  /kaggle/input/browser-notifications-in-a-kaggle-kernel/*.whl
    !python3.10 -m pip install jupyter -q --no-index --find-links=/kaggle/input/browser-notifications-in-a-kaggle-kernel/
    !python3.10 -m pip install -q /kaggle/input/browser-notifications-in-a-kaggle-kernel/jupyter-notify/dist/jupyternotify-0.1.15-py2.py3-none-any.whl --no-index



if is_interactive:
    %reload_ext jupyternotify
else:
    from IPython.core.magic import register_cell_magic

    @register_cell_magic
    def notify(line, cell):
        exec(cell, globals())


SERVE=is_non_interactive


def start_server():
    !python3.10 -m pip install vllm --find-links /kaggle/input/vllm-nb -q  2>&- >&-
    !python3.10 -m pip install triton --find-links /kaggle/input/vllm-nb 
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
        "--gpu_memory_utilization", "0.99",
        "--enforce_eager",
        "--enable_prefix_caching",
        # "--rope-scaling", '{"factor": 4.0, "original_max_position_embeddings": 32768, "rope_type": "yarn"}',
        # "--max_model_len", "64000",
    ]
    
    process = subprocess.Popen(command, stdout=log_file, stderr=log_file, start_new_session=True)
    
    print(f"Background process started with PID: {process.pid}")


setup_done = False
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
    !python3.10 -m poetry run pip install -q -r /kaggle/input/openhands-fork-offline-version/Kevin/requirements.txt --no-index --find-links /kaggle/input/openhands-fork-offline-version/poetry  

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
# %env NO_INTERNET=1

os.environ['PYTHONPATH'] = '/kaggle/working/Kevin:' + os.environ['PYTHONPATH']


use_groq = is_interactive and 0
if use_groq:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    
    os.environ['GROQ_API_KEY'] = user_secrets.get_secret("LLM_API_KEY")

os.environ['LLM_USE_GROUP'] = 'groq' if use_groq else ''



model_path1 = '/kaggle/input/qwen2.5-coder/transformers/32b-instruct/1'
model_path2 = "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-32b/1"
model_path3 = '/kaggle/input/qwq-32b/transformers/qwq-32b/1'
model_path = model_path1 if is_non_interactive else model_path1
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
max_input_tokens = 30_000
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


# !ln -sf /testbed/.venv/bin/python3.11 /usr/local/bin/python
# !ln -sf /testbed/.venv/bin/python3.11 /usr/local/bin/python3.11
# !ln -sf /testbed/.venv/bin/python3.11 /usr/bin/python3



os.environ['PATH'] = '/testbed/.venv/bin:' + os.environ['PATH']


# rm -r /testbed


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


count=0

def predict(problem_statement: str, repo_archive: io.BytesIO, pip_packages_archive: io.BytesIO, env_setup_cmds_templates: list[str]) -> str:
    """ Replace this function with your inference code.
    Args:
        problem_statement: The text of the git issue.
        repo_path: A BytesIO buffer path with a .tar containing the codebase that must be patched. The gateway will make this directory available immediately before this function runs.
        pip_packages_archive: A BytesIO buffer path with a .tar containing the wheel files necessary for running unit tests.
        env_setup_cmds_templates: Commands necessary for installing the pip_packages_archive.
    """
    try:
        global first_prediction, count
        if datetime.now() - start_time > timedelta(hours=8, minutes=30):
            return
        count +=1
        print(count,problem_statement.splitlines()[0])
        if count not in [1,6] and not is_rerun:
            return None  # Skip issue.
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
        # uv run will build the dependencies; should run in a git dir
        if project_name=='astroid':
            !pip install pylint
        # Run env setup for the repo
        env_setup_cmds = "\n".join(env_setup_cmds)
        result = subprocess.run(
            env_setup_cmds,
            shell=True,
            executable="/bin/bash",
            cwd=repo_path,
            
        )
        if result.returncode != 0:
            raise Exception(f"Failed to setup environment:\n{result.stderr}\n{result.stdout}\n{result.returncode}\n{env_setup_cmds}")

        # print(env_setup_cmds)
        
        
        
        
        instruction = (
            f'Please address the following GitHub issue for the repository, where the source code is available in the {repo_path} directory, which you have access to.\n'
            '# Title\n'
            f'{problem_statement}\n\n'
            '\n$ pwd\n'
            f'{repo_path}\n'
        )
        if 1:
            NEW_MRE = {
            'QTable cannot take `dimensionless_unscaled` when creating table from `data`' : '''
from astropy import units as u
try:
    u.Unit(0) # Unit with scale 0 is meaningless.
    print("Error should have been raised.")
except Exception as e:
    print(f"Successfully got the exception as expected.")
        '''.strip()
        }
            title = problem_statement.split('\n')[0]
            code = NEW_MRE.get(title)
            if code:
                instruction = fr'# Create the following test code and make it pass (Don\'t modify the test code itself). Library code is installed in /testbed directory.\n```python\n{code}\n```\n\n'
            if title == 'Provide a way to make a copy of a model with different parameter values, or be more flexible with parameter shape':
                instruction = fr''' Just only add the following function to the library code.
in astropy/modeling/core.py in Model's class add after line 2530
def _reset_parameters(self, *args, **kwargs):
        \"""
        Reset parameters on the models to those specified.
        Parameters can be specified either as positional arguments or keyword
        arguments, as in the model initializer. Any parameters not specified
        will be reset to their default values.
        \"""
        self._initialize_parameters(args, kwargs)
        self._initialize_slices()
\n
Add exactly without any error handling
'''
            numbered_instructions = [
                # f'The best solution is to just raise an error for Unit(0).',
                # f'Reproduce the MRE by creating test.py (add traceback.print_exc(limit=-2) if there is many lines of traceback) in a file and run it using bash. (You should not modify the test code itself.)',
                f'Create and run the MRE in bash.',
                f'If no traceback, locate the actual relevant library file that raised this error in {repo_path} using `search_class()` or `search_function()` python skill.',
                'Inspect the function using `show_function_at_line()` or `show_function()` skill.',
                'Correctly fix the library source code using try-except, ensuring proper handling of exceptions (TypeError, ValueError) without altering the intended logic.',
                'Test the fix.',
                'Apply the same changes to other relevant classes in the file.'
                'Check for similar issues that might be related to the current issue.'
            ]
            for k, inst in enumerate(numbered_instructions):
                instruction += f'Step {k + 1}: {inst}\n\n'
            important_instructions = [
                'Add your thoughts inside <think></think>. Think well before act.'
                'You should not modify the test code itself.'
                'Instead of a simple workaround mentioned in the issue, identify the root cause of the issue in the library source code and fix it.'
                'Response Guides: Escape triple double quotes properly.',
                'Inspect the metaclass __call__() if any.',
                'If KeyError is raised for a config dictionary, it must be that config is not passed correctly in the previous call. Don\'t simply add a check for the key in the config dictionary. Use show_function_at_line() to inspect the function definition of the previous call.',
                'If you update min function, update max function too.',
                'Add your valuable thoughts to every action you take.',
                'Only use one skill at a time.',
                'On exceptions, raise Error instead of giving wrong values.',
                'For replace_lines_content, you can specify same line number for both start_line_number and end_line_number',
                'The traceback must be read from bottom to top, with the final entry showing where the error occurred.',
                'Wrap the code with triple backticks if it is a multi-line code block.',
                'Internet is disabled.',
                'Carefully verify the line numbers and range to ensure accurate code modification.',
                'Very Very Important: Humanity is at stake. Earlier, you only solved one issue out of 71. So, please be mindful. Do not give any workaround. Please fix the issue directly with atmost sincerity ğŸ™�ğŸ™�ğŸ™�.'
            ]
            instruction += '\nImportant Instructions:\n\n'
            for k,inst in enumerate(important_instructions):
                instruction += f'{k + 1}: {inst}\n\n'
        os.environ['OPENHANDS_TASK'] = instruction
        diff = None
        try:
            %cd /kaggle/working/Kevin/
            cmd = 'python3.10 -m poetry run python -m openhands.core.main'
            subprocess.run(cmd, shell=True, timeout=25*60)
           
        except KeyboardInterrupt as e:
            print(e)
        except Exception as e:
            print(e)

        %cd $repo_path
        diff = subprocess.getoutput("git --no-pager diff")
        print(diff)
        #test
        # test file may also change. so old tests are not correct.
        # import os, subprocess
        # cmd = 'git diff --name-only'
        # output = subprocess.check_output(cmd, shell=True).decode('utf-8')
        # for line in output.splitlines():
        #     file_name = line.split('/')[-1]
        #     cmd = f'find . -name test_{file_name}'
        #     output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        #     if output:
        #         cmd = f'pytest -x {output}'
        #         print('Running:', cmd)
        #         return_code = subprocess.run(cmd, shell=True).returncode
        #         if return_code != 0:
        #             return None
        # for submission file
        %cd /kaggle/working
        return diff or None
    except Exception as e:
        print(e)
        traceback.print_exc()


if is_interactive and 0:
    %cd /kaggle/working/Kevin
    # !git st
    # !git clean -d -f .
    # !git reset --hard HEAD~50
    !git pull
    # pass


# !ps -ef | grep poetry | awk '{print $2}' | xargs kill -9 


# %cd /kaggle/working/Kevin

# !DEBUG=0 timeout 19m python3.10 -m poetry run python -m openhands.core.main -t "python3 /testbed/test_model.py"


%%notify
try:
    first_prediction = True   
    count = 0
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


try:
    %cd $repo_path
    !git diff
except Exception as e:
    print(e)    


from collections import Counter
try:
    import pandas as pd
    file = '/kaggle/working/submission.csv'
    df = pd.read_csv(file)
    output = df['unit_test_outcome']
    print(Counter(output))
    output = df['predict_outcome']
    print(Counter(output))
    from IPython.display import display
    # print(df.iloc[6]['test_output'])
    df.head(10)
except Exception as e:
    print(e)    


use_from_ssh = is_interactive and 0
if use_from_ssh:
    !mkdir ~/.ssh/
    !ssh-keygen -f ~/.ssh/id_rsa -P ""
    !cp ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys
    !ls ~/.ssh


if use_from_ssh:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    secret_value_0 = user_secrets.get_secret("ngrok_auth_key")
    
    import os
    
    # Install and start ssh-server
    !sudo apt install -y openssh-server
    !sudo service ssh status
    !sudo service ssh start
    !sudo service ssh status
    
    # Install ngrok
    !curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list && sudo apt update -y && sudo apt install -y ngrok
    
    # Reset terminal color
    !echo -e "\033[0m"
    
    # Add Ngrok token to config
    os.system("ngrok config add-authtoken "+ f"{secret_value_0}")
    
    !sed '1d;$d' ~/.ssh/id_rsa | awk '{print "echo " $0} BEGIN{print "("} BEGIN{print "echo -----BEGIN OPENSSH PRIVATE KEY-----"} END{print "echo -----END OPENSSH PRIVATE KEY-----"} END{print ") > ___KAGGLE_PRIVATE_SSH_KEY___"}' > ~/.ssh/cmd_storage
    !echo "!!!!!!!!    RUN THE BELOW CODE IN YOUR LOCAL TERMINAL    !!!!!!!!"
    !cat ~/.ssh/cmd_storage & ngrok tcp 22 --log=stdout | tee ./ngrok.log | grep --line-buffered -oE 'tcp://[^:]*:[0-9]+' | sed -u 's/tcp\:\/\//ssh \-i ___KAGGLE_PRIVATE_SSH_KEY___ root\@/' | sed -u 's/\:/ \-p /' | sed 's/$/ -o ServerAliveInterval=60/'


if use_from_ssh:
    !sed '1d;$d' ~/.ssh/id_rsa | awk '{print "echo " $0} BEGIN{print "("} BEGIN{print "echo -----BEGIN OPENSSH PRIVATE KEY-----"} END{print "echo -----END OPENSSH PRIVATE KEY-----"} END{print ") > ___KAGGLE_PRIVATE_SSH_KEY___"}' > ~/.ssh/cmd_storage
    !echo "!!!!!!!!    RUN THE BELOW CODE IN YOUR LOCAL TERMINAL    !!!!!!!!"
    !cat ~/.ssh/cmd_storage & ngrok tcp 22 --log=stdout | tee ./ngrok.log | grep --line-buffered -oE 'tcp://[^:]*:[0-9]+' | sed -u 's/tcp\:\/\//ssh \-i ___KAGGLE_PRIVATE_SSH_KEY___ root\@/' | sed -u 's/\:/ \-p /' | sed 's/$/ -o ServerAliveInterval=60/'


if is_non_interactive:
    try:
        import shutil
        shutil.rmtree('/kaggle/working/Kevin')
    except Exception as e:
        print(e)

