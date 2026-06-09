import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
import requests
import subprocess
import time
import torch
import vllm
import warnings

# Summary
warnings.simplefilter('ignore')
print('PyTorch version:', torch.__version__)
print('vLLM:', vllm.__version__)


def start_model(gpu_ids, model_name, model_path, port, max_model_len):
    command = f"""
    CUDA_VISIBLE_DEVICES={gpu_ids} python -m vllm.entrypoints.openai.api_server \
    --served-model-name {model_name} \
    --model {model_path} \
    --port {port} \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95
    """
    
    # Start the process without blocking
    process = subprocess.Popen(
        ["bash", "-c", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process

def check_server_ready(api_url, timeout = 600):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(api_url + "/health", timeout=5)
            if response.status_code == 200:
                print(f"âœ… Server is ready: {api_url}")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"â�³ Waiting for {api_url} to be ready...")
        time.sleep(15)
    
    print(f"â�Œ Timeout: Server {api_url} did not start in time.")
    return False

# Start Models
p1 = start_model(gpu_ids = "0,1",
                 model_name = "qwen2.5-1.5b-instruct", 
                 model_path = "/kaggle/input/qwen2.5/transformers/1.5b-instruct-awq/1", 
                 port = 8000,
                 max_model_len = 4096)
p2 = start_model(gpu_ids = "2,3", 
                 model_name = "qwen2.5-3b-instruct", 
                 model_path = "/kaggle/input/qwen2.5/transformers/3b-instruct-awq/1", 
                 port = 8001,
                 max_model_len = 4096)

# Model Urls
MODEL1_URL = "http://localhost:8000"
MODEL2_URL = "http://localhost:8001"

# Check Availability
if check_server_ready(MODEL1_URL) and check_server_ready(MODEL2_URL):
    print("ğŸš€ Both models are running and ready to accept requests!")
else:
    print("âš ï¸� One or both servers failed to start.")


def get_response(api_url, model_name, question):
    message = [{"role": "system", "content": "You're a helpful AI assistant."},{"role": "user", "content": question}]
    
    try:
        data = {"messages": message,
                "n": 1,
                "model": model_name,
                "temperature": 1.0,
                "top_p": 0.99,
                "max_tokens": 128}
        
        response = requests.post(f'{api_url}/v1/chat/completions', 
                                 json = data, 
                                 timeout = 120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"error: {e}")
        return {'text': ['error']}


questions = ["What is the capital of France?",
             "What is the capital of Germany?"]

for question in questions:
    print('\n### Inference Model 1')
    response = get_response(MODEL1_URL, "qwen2.5-1.5b-instruct", question)
    print(response)

    print('\n### Inference Model 2')
    response = get_response(MODEL2_URL, "qwen2.5-3b-instruct", question)
    print(response)

