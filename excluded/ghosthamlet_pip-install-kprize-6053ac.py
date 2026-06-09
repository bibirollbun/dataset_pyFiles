!pip install --target=/kaggle/working datasets==3.1.0 vllm==0.6.3.post1 bitsandbytes==0.44.1


!rm -rf /kaggle/working/ray*


import kaggle_evaluation.konwinski_prize_inference_server


import importlib
import gc, torch

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

