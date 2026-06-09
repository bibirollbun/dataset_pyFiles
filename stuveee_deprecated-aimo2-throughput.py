import time
nb_start_time = time.time()


# !pip install sgl-kernel --force-reinstall --no-deps --no-index --find-links file:///kaggle/input/sglang-download/packages
!pip install triton --no-deps --no-index --find-links file:///kaggle/input/sglang-download/packages

!pip install "sglang[all]" --no-index --find-links file:///kaggle/input/sglang-download/packages -U


print(f'time install packages: {time.time() - nb_start_time}')


test_sglang = False
test_vllm = False
test_lmdeploy = True


llm_model = '/kaggle/input/deepseek-r1/transformers/14b-awq-v2-g128-s64/1'


from sglang.utils import (
    execute_shell_command,
    wait_for_server,
    terminate_process,
    print_highlight,
)

import os


os.environ["TRITON_PTXAS_PATH"]="/usr/local/cuda/bin/ptxas"


# sglang


if test_sglang:
    cmd = f"python -m sglang.launch_server --model-path {llm_model} --port 30020 --host 0.0.0.0 --tp-size 4 --mem-fraction-static 0.8"
    if 'fp8' in llm_model or 'FP8' in llm_model:
        cmd += ' --quantization w8a8_fp8'
        
    server_process = execute_shell_command(
        cmd
    )
    
    wait_for_server("http://localhost:30020")


if test_sglang:
    # bs 11, seqlen 7500
    ! python3 -m sglang.bench_serving --dataset-name random --dataset-path /kaggle/input/sharegpt/transformers/default/1/ShareGPT_V3_unfiltered_cleaned_split.json --random-input-len 200 --random-output-len 7500 --random-range-ratio 1 --request-rate inf --num-prompts 11 --port 30020
    # bs 11, seqlen 7500
    ! python3 -m sglang.bench_serving --dataset-name random --dataset-path /kaggle/input/sharegpt/transformers/default/1/ShareGPT_V3_unfiltered_cleaned_split.json --random-input-len 200 --random-output-len 7500 --random-range-ratio 1 --request-rate inf --num-prompts 11 --port 30020 
    # bs 15, seqlen 10000
    ! python3 -m sglang.bench_serving --dataset-name random --dataset-path /kaggle/input/sharegpt/transformers/default/1/ShareGPT_V3_unfiltered_cleaned_split.json --random-input-len 200 --random-output-len 10000 --random-range-ratio 1 --request-rate inf --num-prompts 15 --port 30020
    pass


if test_sglang:
    terminate_process(server_process)
    ! sleep 30


# vllm


if test_vllm:

    server_process = execute_shell_command(
        f"python3 -m vllm.entrypoints.openai.api_server --model {llm_model} --gpu-memory-utilization 0.9 --tensor-parallel-size 4 --port 30021"
    )

    wait_for_server("http://localhost:30021")


if test_vllm:
    # bs 11, seqlen 7500
    ! python3 -m sglang.bench_serving --dataset-name random --dataset-path /kaggle/input/sharegpt/transformers/default/1/ShareGPT_V3_unfiltered_cleaned_split.json --random-input-len 200 --random-output-len 7500 --random-range-ratio 1 --request-rate inf --num-prompts 11 --port 30021 --backend vllm 
    # bs 11, seqlen 7500
    ! python3 -m sglang.bench_serving --dataset-name random --dataset-path /kaggle/input/sharegpt/transformers/default/1/ShareGPT_V3_unfiltered_cleaned_split.json --random-input-len 200 --random-output-len 7500 --random-range-ratio 1 --request-rate inf --num-prompts 11 --port 30021 --backend vllm 
    # bs 15, seqlen 10000
    ! python3 -m sglang.bench_serving --dataset-name random --dataset-path /kaggle/input/sharegpt/transformers/default/1/ShareGPT_V3_unfiltered_cleaned_split.json --random-input-len 200 --random-output-len 10000 --random-range-ratio 1 --request-rate inf --num-prompts 15 --port 30021 --backend vllm
    pass


if test_vllm:
    terminate_process(server_process)
    ! sleep 30


# lmdeploy


if test_lmdeploy:
    ! pip install lmdeploy --no-index --find-links file:///kaggle/input/lmdeploy-download/packages


if test_lmdeploy:
    server_process = execute_shell_command(
        f"lmdeploy serve api_server {llm_model} --server-port 30022 --tp 4 --cache-max-entry-count 0.9"
    )
    
    wait_for_server("http://localhost:30022")


if test_lmdeploy:
    # bs 11 seqlen 7500
    ! python3 -m sglang.bench_serving --dataset-name random --dataset-path /kaggle/input/sharegpt/transformers/default/1/ShareGPT_V3_unfiltered_cleaned_split.json --random-input-len 200 --random-output-len 7500 --random-range-ratio 1 --request-rate inf --num-prompts 11 --port 30022 --backend lmdeploy
    # bs 11 seqlen 7500
    ! python3 -m sglang.bench_serving --dataset-name random --dataset-path /kaggle/input/sharegpt/transformers/default/1/ShareGPT_V3_unfiltered_cleaned_split.json --random-input-len 200 --random-output-len 7500 --random-range-ratio 1 --request-rate inf --num-prompts 11 --port 30022 --backend lmdeploy
    # bs 15 seqlen 10000
    ! python3 -m sglang.bench_serving --dataset-name random --dataset-path /kaggle/input/sharegpt/transformers/default/1/ShareGPT_V3_unfiltered_cleaned_split.json --random-input-len 200 --random-output-len 10000 --random-range-ratio 1 --request-rate inf --num-prompts 15 --port 30022 --backend lmdeploy
    pass


if test_lmdeploy:
    terminate_process(server_process)

