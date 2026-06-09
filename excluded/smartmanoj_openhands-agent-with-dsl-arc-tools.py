import os
import time
from datetime import datetime, timedelta
import subprocess

os.environ['TZ'] = 'Asia/Kolkata'
time.tzset()
start_time = datetime.now()
print('Started')

is_interactive_mode = os.getenv('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive'
is_rerun_mode = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))
is_save_mode = not is_interactive_mode and not is_rerun_mode
is_non_rerun_mode =  not is_rerun_mode
is_non_interactive_mode = not is_interactive_mode
print(f'{is_interactive_mode = }; {is_rerun_mode = }; {is_save_mode = }')


!touch submission.json
print('Dummy submission file created')
from time import sleep
if is_save_mode:sleep(10)


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
internet_enabled = is_connected(REMOTE_SERVER)
print('Internet :', internet_enabled)


!ln -sf /usr/bin/python3.12 /usr/bin/python3


hide_output = '> /dev/null'
hide_output_in_rerun_only = '> /dev/null' if is_rerun_mode else ''


!sudo dpkg -i  $(ls /kaggle/input/openhands-agent-wheels/apt/*.deb) $hide_output

!python /kaggle/input/openhands-agent-wheels/apt/get-pip.py  --no-index --find-links /kaggle/input/openhands-agent-wheels/apt
!pip install /kaggle/input/openhands-agent-wheels/apt/pip*.whl $hide_output
!pip install /kaggle/input/openhands-agent-wheels/apt/*.whl $hide_output
!pip install --no-index --no-build-isolation  /kaggle/input/openhands-agent-wheels/openhands_tools_wheels/func_timeout-4.3.5.tar.gz $hide_output
!pip install --no-index --ignore-installed  /kaggle/input/openhands-agent-wheels/openhands_sdk_wheels/*.whl $hide_output
!pip install --no-deps --no-index --ignore-installed /kaggle/input/openhands-agent-wheels/openhands_tools_wheels/*.whl $hide_output


import os
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"


os.environ['PYTHONPATH'] = '/kaggle/lib/kagglegym:/kaggle/lib:/kaggle/usr/lib:/kaggle/input/arc-prize-2025:/kaggle/usr/lib/vllm_us'


is_cloud=internet_enabled
is_local= not is_cloud
os.environ['IS_CLOUD'] = str(int(is_cloud))
if is_cloud:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    os.environ['GROQ_API_KEY'] = user_secrets.get_secret("GROQ_API_KEY")
else:
    log_file = open("/kaggle/working/vllm_output.log", "w")
    model_path = "/kaggle/input/qwen3-coder/transformers/30b-a3b-instruct/1"
    os.environ['VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT'] = '1'
    # background task
    command = [
        "python3.11", 
        "-m", 
        "vllm.entrypoints.openai.api_server", 
        "--model",
        model_path,
        "--tensor_parallel_size", "4",
        "--gpu_memory_utilization", "0.80",
        "--enforce_eager",
        "--dtype","half",
        "--seed", "42",
        "--enable_prefix_caching",
        "--rope-scaling", '{"factor": 4.0, "original_max_position_embeddings": 32768, "rope_type": "yarn"}',
        "--max_model_len", "54000",
        # "--enable-auto-tool-choice", "--tool-call-parser", "hermes"
        
    ]
    
    process = subprocess.Popen(command, stdout=log_file, stderr=log_file, start_new_session=True)
    print(" ".join(command))
    print(f"Background process started with PID: {process.pid}")


if is_local:
    !tail -100 /kaggle/working/vllm_output.log


if is_local:
    import requests
    import time
    last=''
    try:
        while True:
            try:
                with open('/kaggle/working/vllm_output.log') as f:
                    a=(f.readlines()[-1])
                    print(a)
                    if 'failed to be inspected' in a or a==last or "warnings.warn('resource_tracker: There appear to be %d" in a:
                        break
                    last=a
                requests.get('http://localhost:8000/v1/models')
                break
            except Exception as e:
                print(end='.')
                time.sleep(60)
    except KeyboardInterrupt:
        print('KeyboardInterrupt')


if is_local:
    # !pkill -f VLLM
    !nvidia-smi


os.environ['PYTHONPATH'] = '/kaggle/lib/kagglegym:/kaggle/lib:/kaggle/usr/lib:/kaggle/input/arc-prize-2025'


os.environ['IS_AGENT_TERMINAL']= '1'


import json

path=r'/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
with open(path, 'r') as file:
    test_data = json.load(file)

submission_data = {}
dummy_data = [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}]
for task_id in (list(test_data.keys())):
    number_of_test_inputs = len(test_data[task_id]['test'])
    submission_data[task_id] = dummy_data * number_of_test_inputs



!cp -r /kaggle/input/arc-tools-source/ARC-Tools .
%cd /kaggle/working/ARC-Tools
!git log -1 --pretty=format:"%h"


# removed when removing vllm utlity script from path
!pip install /kaggle/input/openhands-agent-wheels/apt/setuptools*.whl


!python -m pip install --no-build-isolation  --no-index -e /kaggle/working/ARC-Tools $hide_output
!python -m pip install --no-index /kaggle/input/arc-tools-source/wheels/* $hide_output


%%file /kaggle/working/agent.py
print('Starting')
import os
from openhands.sdk import LLM, Conversation, Agent
from openhands.sdk.tool import Tool, register_tool
from openhands.tools.execute_bash import BashTool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.preset.default import get_default_condenser
# Configure LLM
is_cloud=os.environ.get('IS_CLOUD','').lower() in ['1', 'true']
if is_cloud:
    base_url=None
    model="groq/qwen/qwen3-32b"
    max_input_tokens=None
    max_output_tokens=16_000
else:
    model = 'hosted_vllm//kaggle/input/qwen3-coder/transformers/30b-a3b-instruct/1'
    base_url = 'http://localhost:8000/v1/'
    max_input_tokens=48_000
    max_output_tokens=16_000
llm = LLM(model=model, base_url=base_url, max_input_tokens=max_input_tokens, max_output_tokens=max_output_tokens, seed=42)

# Register tools
register_tool("BashTool", BashTool)
register_tool("FileEditorTool", FileEditorTool)
register_tool("TaskTrackerTool", TaskTrackerTool)

# Create agent with custom bash tool configuration
agent = Agent(
    llm=llm,
    tools=[
        Tool(name="BashTool", params={"terminal_type": "subprocess"}),
        Tool(name="FileEditorTool"),
        Tool(name="TaskTrackerTool"),
    ],
    condenser=get_default_condenser(
            llm=llm.model_copy(update={"usage_id": "condenser"})
        ),
)

# Start a conversation
conversation = Conversation(agent=agent, workspace=".")
conversation.send_message(open('/kaggle/working/prompt.txt').read())
conversation.run()


def create_prompt(data):
    prompt = f"""
class GridPoint:
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value

class GridRegion:
    attr x1, y1, x2, y2

class Grid(SafeList):
    def __init__(self, grid: list[list[int]], background_color: int | None = None,...)

class SubGrid(Grid):
    def __init__(self, region: GridRegion, parent_grid: Grid, obj_color: int | None = None, points: list[GridPoint] | None = None):
        attr region, 
        points : list[GridPoint] -> list of points in the subgrid

Find the transformation rule for the following input and output.
Input: {data['train'][0]['input']}
Output: {data['train'][0]['output']}

API reference:
- Grid class: Grid(2d_list) - create a grid from a 2D list
- grid.get(x, y) - get value at position (x, y)
- detect_objects(grid) - return list of SubGrids (detects connected components)
- SubGrid.points - list of GridPoint objects in order

CRITICAL RULES:
0. Don't view this file because it is too large to fit in the context.
"/kaggle/working/ARC-Tools/arc_tools/grid.py"
1. Do NOT hardcode dimensions or positions
2. ALWAYS return Grid object: Grid([[val1], [val2], ...])
3. Keep solution SIMPLE - prefer 2-3 lines
4. Look for spatial patterns (straighten, align) not just color counting
5. For shape transformations, use object.points directly
6. All points in the SubGrid may not have the same value.
7. Input and Output are Grid Objects (2D list)

Core Rules:
1) Rows correspond to the y-axis and columns to the x-axis
grid.get(x, y) is equivalent to grid[y][x] (grid[row][col])

Steps:
1) Complete the function in the following file and run it
/kaggle/working/ARC-Tools/workspace/task.py

2) Run the function and make it pass the train tasks.

Important: Your workspace is /kaggle/working/ARC-Tools/workspace and not /workspace.
                   
"""
    with open('/kaggle/working/prompt.txt', 'w') as f:
        f.write(prompt)


WORKSPACE_DIR='/kaggle/working/ARC-Tools/workspace'


def solve():
    for task_id, data in list(test_data.items())[:120]:
        try:
            if datetime.now() - start_time > timedelta(hours=11, minutes=00):
                return
            print(f'{task_id =}')
            %cd /kaggle/working/ARC-Tools
            !git clean -fd
            with open(f'{WORKSPACE_DIR}/data.json', 'w') as file:
                json.dump(data, file)
            create_prompt(data)
            !timeout -k 1s 35m python /kaggle/working/agent.py $hide_output_in_rerun_only
            if os.path.exists(( output_file := f'{WORKSPACE_DIR}/output.json')):
                with open(output_file, 'r') as f:
                    submission_data[task_id] = json.load(f)
            print('='*50)
            if is_non_rerun_mode:
                break
        except Exception as e:
            print(e)

solve()


if is_non_interactive_mode:
    try:
        import shutil
        shutil.rmtree('/kaggle/working/ARC-Tools')
    except Exception as e:
        print(e)


with open('/kaggle/working/submission.json', 'w') as file:
    json.dump(submission_data, file)


!ln -sf /usr/bin/python3.11 /usr/bin/python
!ln -sf /usr/bin/python3.11 /usr/bin/python3

