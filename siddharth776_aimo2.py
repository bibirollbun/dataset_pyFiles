!pip uninstall -y torch
!pip install -U --no-index --find-links=/kaggle/input/vllm-whl -U vllm
!pip install -U --upgrade /kaggle/input/vllm-t4-fix/grpcio-1.62.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install -U --upgrade /kaggle/input/vllm-t4-fix/ray-2.11.0-cp310-cp310-manylinux2014_x86_64.whl



import os
import logging
import re
import gc
import warnings
import random
import scipy as sp
import numpy as np
import pandas as pd
import math
from glob import glob
from pathlib import Path
import joblib
import pickle
import itertools
from tqdm.auto import tqdm
import polars as pl
import kaggle_evaluation.aimo_2_inference_server
import torch  # For GPU detection

# -------------------------------------------------------
# Setup Logging and Environment
# -------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set visible GPUs (adjust as needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

warnings.filterwarnings('ignore')

# Detect available GPUs and adjust tensor_parallel_size accordingly
available_gpus = torch.cuda.device_count()
logger.info("Available GPUs: %d", available_gpus)
# Use at most 4 GPUs if available, otherwise use the available count (or 1 if none are found)
tensor_parallel_size = min(4, available_gpus) if available_gpus > 0 else 1
logger.info("Using tensor_parallel_size: %d", tensor_parallel_size)

# -------------------------------------------------------
# Import vLLM (after pip installs)
# -------------------------------------------------------
import vllm

# -------------------------------------------------------
# Initialize the LLM Model with Enhanced Configuration
# -------------------------------------------------------
llm = vllm.LLM(
    "/kaggle/input/deepseek-math-7b-instruct/transformers/main/1",  # Path to model or model identifier
    tensor_parallel_size=tensor_parallel_size,  # Adjusted based on available GPUs
    gpu_memory_utilization=0.95,
    trust_remote_code=True,
    dtype="half",
    enforce_eager=True,
    swap_space=2,  # Allocate 2GB CPU swap space per GPU to prevent OOM errors
)
tokenizer = llm.get_tokenizer()

# -------------------------------------------------------
# Function: Generate Text using vLLM
# -------------------------------------------------------
def generate_text_vllm(requests: list, tokenizer, model) -> list:
    """
    Generates text outputs using the vLLM model.
    """
    sampling_params = vllm.SamplingParams(
        temperature=0.00,
        seed=42,
        max_tokens=1024
    )
    responses = model.generate(requests, sampling_params=sampling_params, use_tqdm=False)
    return [response.outputs[0].text for response in responses]

# -------------------------------------------------------
# Function: Extract Answer from Generated Text
# -------------------------------------------------------
def extract_answer(text: str) -> int:
    """
    Extracts an integer answer enclosed in \\boxed{} from the provided text.
    Returns None if no valid answer is found.
    """
    matches = re.findall(r'\\boxed\{(\d+)\}', text)
    if matches:
        try:
            answer = int(matches[0])
            if answer < 0 or answer > 999:
                logger.warning("Extracted answer %d is out of expected range. Adjusting modulo 1000.", answer)
                answer = answer % 1000
            return answer
        except Exception as e:
            logger.error("Error converting extracted text to integer: %s", e)
            return None
    return None

tool_instruction = '\nPlease solve the problem above, and put your final answer within \\boxed{}.'

# -------------------------------------------------------
# Function: Prediction with Enhanced Accuracy
# -------------------------------------------------------
def predict(id_: pl.Series, question: pl.Series) -> pl.DataFrame | pd.DataFrame:
    """
    Generates a prediction by creating a prompt from the input question,
    generating text with vLLM, and extracting the answer within \\boxed{}.
    Retries up to 3 times if a valid answer is not extracted.
    """
    id_val = id_.item(0)
    question_val = question.item(0)
    prompt = question_val + tool_instruction
    
    max_attempts = 3
    answer = None
    
    for attempt in range(max_attempts):
        logger.info("Attempt %d for id: %s", attempt + 1, id_val)
        generated_text = generate_text_vllm([prompt], tokenizer, llm)[0]
        logger.debug("Generated text: %s", generated_text)
        answer = extract_answer(generated_text)
        if answer is not None:
            logger.info("Successfully extracted answer: %d", answer)
            break
        else:
            logger.warning("No valid answer extracted in attempt %d. Retrying...", attempt + 1)
    
    if answer is None:
        logger.error("Failed to extract a valid answer after %d attempts for id: %s", max_attempts, id_val)
        answer = 0  # Default fallback answer
    
    return pl.DataFrame({'id': id_val, 'answer': answer})

# -------------------------------------------------------
# Start the Inference Server
# -------------------------------------------------------
inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv',))





