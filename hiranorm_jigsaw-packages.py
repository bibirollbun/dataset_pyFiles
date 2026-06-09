!mkdir whls
!pip download -d '/kaggle/working/whls' 'vllm'
!pip download -d '/kaggle/working/whls' 'triton==3.2.0'
!pip download -d '/kaggle/working/whls' 'logits-processor-zoo==0.1.10'
!pip download -d '/kaggle/working/whls' 'clean-text' 'bitsandbytes' 'peft' 'accelerate' 'datasets' 'emoji' 'setuptools>=40.8.0' 'numpy<2'


!uv pip uninstall tensorflow
!uv pip install --system --no-index --find-links='/kaggle/working/whls/' 'vllm'
!uv pip install --system --no-index --find-links='/kaggle/working/whls/' 'triton==3.2.0'
!uv pip install --system --no-index --find-links='/kaggle/working/whls/' 'logits-processor-zoo==0.1.10'
!uv pip install -U --system --no-index --find-links='/kaggle/working/whls/' 'clean-text' 'bitsandbytes' 'peft' 'accelerate' 'datasets' 'emoji' 'setuptools>=40.8.0' 'numpy<2'


import os
import pandas as pd
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
import torch
import vllm
import numpy as np
from vllm.lora.request import LoRARequest
import argparse
from scipy.special import softmax


! mkdir -p /tmp/src


%%writefile /tmp/src/infer_qwen.py

import os
import pandas as pd
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
import torch
import vllm
import numpy as np
from vllm.lora.request import LoRARequest
import argparse
from scipy.special import softmax


%cd /tmp
!python src/infer_qwen.py


!uv pip check


!uv pip list | grep vllm




