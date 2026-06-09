# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install --target=/kaggle/working vllm bitsandbytes -U


import sys
sys.path.append("/kaggle/working")
with open("/kaggle/working/Custom_path.txt", "w") as f:
    f.write("\n".join(sys.path))



# import os
# from vllm import LLM, SamplingParams

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# MAX_NUM_SEQS = 16
# MAX_MODEL_LEN = 8192
# model_path = "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-14b/1"
# # model_path = '/kaggle/input/deepseek-r1/transformers/deepseek-aideepseek-r1-distill-qwen-14b-awq-neody/1'


# # import torch
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# llm = LLM(
#     model_path,
#     max_num_seqs = MAX_NUM_SEQS,
#     max_model_len = MAX_MODEL_LEN,
#     trust_remote_code = True,
#     tensor_parallel_size = 2,
#     gpu_memory_utilization = 0.95,
#     seed = 2024,
#     device = "cuda",
#     dtype="float16"
# )










