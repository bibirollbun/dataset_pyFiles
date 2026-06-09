# IMPORTANT: Before running this file, please toggle Internet on

import time
import socket

def is_connected(host="8.8.8.8", port=53, timeout=3):
    """Check internet connectivity by attempting to connect to a public DNS server."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False

print("Checking internet connection...")
while not is_connected():
    print("No internet connection. Retrying in 1 second...")
    time.sleep(0.5)

print("Internet connection is active.")


# !pip install -q --upgrade pip
# !pip install -q vllm huggingface_hub
# !pip install flask pyngrok flask_cors
# !pip install -q bitsandbytes accelerate
!curl -LsSf https://astral.sh/uv/install.sh | sh
!source ~/.bashrc
!uv pip install --system --no-cache --quiet vllm huggingface_hub flask pyngrok flask_cors bitsandbytes accelerate


import os
import shutil

working_dir = '/kaggle/working/'

for item in os.listdir(working_dir):
    path = os.path.join(working_dir, item)
    try:
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)              # xÃ³a file hoáº·c link
        elif os.path.isdir(path):
            shutil.rmtree(path)          # xÃ³a thÆ° má»¥c vÃ  toÃ n bá»™ ná»™i dung
    except Exception as e:
        print(f"KhÃ´ng thá»ƒ xÃ³a {path}. Lá»—i: {e}")

from huggingface_hub import login

# Biáº¿n HF_TOKEN do Kaggle Secrets táº¡o sáºµn
hf_token = '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'
if hf_token is None:
    raise RuntimeError("Thiáº¿u HF_TOKEN. Vui lÃ²ng thÃªm vÃ o Kaggle Secrets.")

login(token=hf_token)
print("âœ… Ä�Äƒng nháº­p Hugging Face thÃ nh cÃ´ng!")
# Ã”Â 2b: Táº£i model safetensors tá»« repo gated
from huggingface_hub import snapshot_download

MODEL_REPO = "ktam204/Qwen3-32B-AWQ-r16-lora-all-Pentest-swift"
# MODEL_REPO = "openai/gpt-oss-20b"
# MODEL_REPO = "LGAI-EXAONE/EXAONE-4.0-32B-AWQ"
LOCAL_DIR   = "/kaggle/tmp/Qwen3-32B"

# MODEL_REPO = "cpatonn/XBai-o4-AWQ-4bit"
# LOCAL_DIR = "/kaggle/tmp/XBai-o4"

snapshot_download(
    repo_id=MODEL_REPO,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False
)
print("âœ… Ä�Ã£ download xong vÃ o:", LOCAL_DIR)




import subprocess
import time
import requests
import re

# Log file names
vllm_logfile = 'vllm.log'

# 1) Launch vLLM server in background
vllm_cmd = [
    "python3", "-m", "vllm.entrypoints.openai.api_server",
    "--model", LOCAL_DIR,#"/kaggle/input/qwen-3/transformers/32b/1",
    "--served-model-name", "qwen/qwen3-32B",
    "--dtype", "bfloat16",             # â†� use float16 on T4
    "--gpu-memory-utilization", "0.9",
    "--tensor-parallel-size", "4",
    "--enforce-eager",
    "--max-model-len", "32768", # Qwen3-32B
    # "--reasoning-parser", "deepseek_r1",
    "--enable-auto-tool-choice", 
    "--tool-call-parser", "hermes", # https://docs.vllm.ai/en/stable/features/tool_calling.html#qwen-models
    "--host", "0.0.0.0",
    "--port", "8000",
]

with open(vllm_logfile, "w") as logf:
    proc = subprocess.Popen(vllm_cmd, stdout=logf, stderr=subprocess.STDOUT)

# 2) Poll /v1/models up to 360 times, waiting 10s between tries (max 1 hour)
for attempt in range(360):
    try:
        resp = requests.get("http://127.0.0.1:8000/v1/models", timeout=10)
        if resp.status_code == 200:
            print(f"âœ… vLLM is up on attempt {attempt+1}!")
            break
    except requests.RequestException as e:
       print(f"Error on attempt {attempt+1}: {e}")
       with open(vllm_logfile, "r") as f:
           lines = f.readlines()
       print("ğŸ“„ Last 20 lines of log:")
       print("".join(lines[-20:]))
    print(f"Waiting for vLLM startup... ({attempt+1}/360)")
    time.sleep(10)
else:
    print("â�Œ Still not up after 1 hour. Check", vllm_logfile)
    proc.terminate()
    raise RuntimeError("vLLM startup timeout")



from pyngrok import ngrok
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
# ngrok.set_auth_token(secrets.get_secret("NGROK2"))
ngrok.set_auth_token("2uQlObp7GLKKDnY1pswagdeod2w_2J6wcvAkuSvAW6zCTsdQH") # 2252????@gm.uit.edu.vn

# 3) Open an HTTP tunnel on port 8000
tunnel = ngrok.connect(8000, "http")
public_url = tunnel.public_url
# 4) Print the public URL
print("ğŸš€ Public URL for your vLLM API:", public_url)

inspector_tunnel = ngrok.connect(4040, "http")
inspector_url = inspector_tunnel.public_url
print("ğŸ”� Ngrok Inspector URL (logs):", inspector_url)


import time, os
from openai import OpenAI

# Configuration
api_base   = public_url.rstrip("/") + "/v1"
# client     = OpenAI(base_url=api_base, api_key=os.getenv("OPENAI_API_KEY", "dummy"))

# t0 = time.perf_counter()

# Pick the first model
models = requests.get(f"{api_base}/models").json()["data"]
model_id = models[0]["id"]

print("Using model:", model_id)

# # Prepare and send the chat request
# messages = [
#     {"role": "system", "content": "You are a helpful Penetration assistant."},
#     {
#         "role":    "user",
#         "content": (
#             "write a 10 words about UIT  /no_think "
#             # "Keep reasoning â‰¤700 tokens."
#         )
#     }
# ]

# resp = client.chat.completions.create(
#     model=model_id,
#     messages=messages,
#     max_tokens=2048,
#     temperature=0.7,
#     top_p=0.8,
#     extra_body={
#         "top_k": 8,
#         "min_p": 0.0,    
#         "chat_template_kwargs": {"thinking": True},
#     }
# )

# # Print timings
# print("â�±ï¸� Total time:", time.perf_counter() - t0)

# # Extract and print outputs
# msg = resp.choices[0].message
# print("\n--- Reasoning ---")
# print(getattr(msg, "reasoning_content", "<none>"))
# print("\n--- Answer ---")
# print(msg.content)



import time, os
from openai import OpenAI

# Configuration
# api_base   = public_url.rstrip("/") + "/v1"
# client     = OpenAI(base_url=api_base, api_key=os.getenv("OPENAI_API_KEY", "dummy"))

# t0 = time.perf_counter()

# Pick the first model
models = requests.get(f"{api_base}/models").json()["data"]
model_id = models[0]["id"]

print("Using model:", model_id)

# # Prepare and send the chat request
# messages = [
#     {"role": "system", "content": """
# You are a cybersecurity expert specializing in payload generation and evasion techniques for penetration testing purposes. Your task is to take an existing payload and a CVE vulnerability description and generate an enhanced version of the payload that can potentially bypass basic filtering mechanisms such as WAFs, IDS, or input sanitization routines.
# You must apply modern obfuscation, encoding, or logic manipulation strategies such as:
# - URL encoding / hex encoding
# - Case manipulation or splitting
# - JavaScript or shell command wrapping
# - Insertion of benign comments or whitespace
# - Use of alternate syntax or Unicode-based bypasses
# Do not invent a new exploit; your goal is only to modify the existing payload to make it stealthier while keeping its original intention intact. Always make sure the payload remains logically valid and executable.
# You must NEVER suggest malicious activity outside authorized security testing. All outputs must be used only in controlled, ethical environments.
# Your output format **must be exactly** a JSON array of strings, no keys, no comments, no explanations.
# Example (exact format):
# ["payload1","payload2","payload3"]
# """},
#     {
#         "role":    "user",
#         "content": (
#             """
# Given the following:
#  - Original Payload: {"cat${IFS}/etc/passwd"}
#  - CVE Description: { CVE-1999-1556": "Microsoft SQL Server 6.5 uses weak encryption for the password for the SQLExecutiveCmdExec account and stores it in an accessible portion of the registry, which could allow local users to gain privileges by reading and decrypting the CmdExecAccount value."}
# Please generate a modified version of the payload that maintains its original function but applies obfuscation or evasion techniques to help bypass common security mechanisms.
# \no_think
# """
#             # "Keep reasoning â‰¤700 tokens."
#         )
#     }
# ]

# resp = client.chat.completions.create(
#     model=model_id,
#     messages=messages,
#     temperature=0.2,
#     top_p=0.8,
#     extra_body={
#         "top_k": 20,
#         "min_p": 0.0,    
#         "chat_template_kwargs": {"thinking": True},
#     }
# )

# # Print timings
# print("â�±ï¸� Total time:", time.perf_counter() - t0)

# # Extract and print outputs
# msg = resp.choices[0].message
# print("\n--- Reasoning ---")
# print(getattr(msg, "reasoning_content", "<none>"))
# print("\n--- Answer ---")
# print(msg.content)
# # print("\n--- Raw message ----")
# # print(msg)



print("OPENAI_URL_BASE:", api_base)
print("CAI_MODEL:", model_id)


import time

# while True:
#     print("âš™ï¸� Still running...")  # Hoáº·c báº¥t ká»³ dÃ²ng nÃ o\
#     time.sleep(60)

# limit to 5h
time.sleep(5*3600)

print("Your session is interrupted.")

