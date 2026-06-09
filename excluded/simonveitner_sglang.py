from sglang.utils import (
    execute_shell_command,
    wait_for_server,
    terminate_process,
    print_highlight,
)

import os


os.environ["TRITON_PTXAS_PATH"]="/usr/local/cuda/bin/ptxas"


server_process = execute_shell_command(
    "python -m sglang.launch_server --model-path /kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-14b/1 --port 30020 --host 0.0.0.0 --tp-size 4"
)

wait_for_server("http://localhost:30020")


import openai

problem = "Solve the integral $sin(x)/x$ from 0 to infinity using Feynmans trick."

client = openai.Client(base_url="http://127.0.0.1:30020/v1", api_key="None")

response = client.chat.completions.create(
    model="/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-14b/1",
    messages=[
        {"role": "user", "content": problem},
    ],
    temperature=0,
    max_tokens=4096,
)


print_highlight(response.choices[0].message.content)


terminate_process(server_process)

