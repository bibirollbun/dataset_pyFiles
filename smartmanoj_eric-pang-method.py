2


import os
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    !touch submission.json
    print('Exiting')
    exit()



print('Copying')
!cp -r /kaggle/input/arc-agi-by-eric-pang/ARC_AGI_By_Eric_Pang /kaggle/working


%cd /kaggle/working/ARC_AGI_By_Eric_Pang


import os
os.environ['PYTHONPATH'] = "/kaggle/working/ARC_AGI_By_Eric_Pang/venv/lib/python3.11/site-packages:" + os.environ['PYTHONPATH']


llm_base_url = 'http://localhost:8000/v1/'
cloud = 0
if cloud:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    llm_base_url = user_secrets.get_secret("LLM_BASE_URL")
    


os.environ['LLM_MODEL']='/kaggle/input/qwen-3/transformers/30b-a3b-thinking-2507/1'
os.environ['LLM_BASE_URL']= llm_base_url
os.environ['OPENROUTER_API_KEY']='dummy'


import os
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
!export CUDA_HOME=/usr/local/cuda
os.environ['PATH']="/usr/bin:${CUDA_HOME}/bin:"+os.environ['PATH']


import subprocess
log_file = open("/kaggle/working/vllm_output.log", "w")
model_path = "/kaggle/input/qwen2.5-coder/transformers/32b-instruct/1"
model_path = "/kaggle/input/qwen-3/transformers/30b-a3b-thinking-2507/1"

# background task
command = [
    "python3.11", 
    "-m", 
    "vllm.entrypoints.openai.api_server", 
    "--model",
    model_path,
    "--tensor_parallel_size", "4",
    "--gpu_memory_utilization", "0.90",
    "--enforce_eager",
    "--dtype","half",
    "--seed", "42",
    "--enable_prefix_caching",
    # "--rope-scaling", '{"factor": 4.0, "original_max_position_embeddings": 32768, "rope_type": "yarn"}',
    "--max_model_len", "180000",
    # "--enable-auto-tool-choice", "--tool-call-parser", "hermes"
    
]

process = subprocess.Popen(command, stdout=log_file, stderr=log_file, start_new_session=True)

print(f"Background process started with PID: {process.pid}")


# !tail /kaggle/working/vllm_output.log


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


# !python test_single_task.py 7b5033c1 


!python -m src.submission -p /kaggle/input/arc-prize-2025/arc-agi_test_challenges.json

