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


!python -m pip install  /kaggle/input/browser-notifications-in-a-kaggle-kernel/*.whl 
!python -m pip install jupyter -q --no-index --find-links=/kaggle/input/browser-notifications-in-a-kaggle-kernel/ 
!python -m pip install -q /kaggle/input/browser-notifications-in-a-kaggle-kernel/jupyter-notify/dist/jupyternotify-0.1.15-py2.py3-none-any.whl --no-index



%reload_ext jupyternotify


import os
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"


import subprocess
log_file = open("/kaggle/working/vllm_output.log", "w")
model_path = "/kaggle/input/qwen-3/transformers/30b-a3b-thinking-2507/1"
model_path = "/kaggle/input/qwen-3/transformers/30b-a3b-instruct-2507/1"
os.environ['VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT'] = '1'
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


!tail -n 10 vllm_output.log


%%notify
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



!pip install pyngrok -q
!pip install pause -q


from pyngrok import ngrok
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("ngrok_auth_key")
secret_value_1 = user_secrets.get_secret("ngrok_domain_name")


ngrok.set_auth_token(secret_value_0)
public_url = ngrok.connect(8000, domain=secret_value_1).public_url
public_url + '/v1'


%%notify
try:
    from datetime import datetime, timedelta
    from pause import until
    dt = datetime.now()
    dt = dt + timedelta(minutes=39, seconds=30)
    print(dt + timedelta(hours=5, minutes=30))
    until(dt)
except KeyboardInterrupt:
    print('KeyboardInterrupt')

