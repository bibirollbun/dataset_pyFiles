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


import torch
assert torch.cuda.is_available(), "WARNING! You are running on a non-GPU instance. A GPU is highly recommended."
REQUIRED_VERSION = "2.2.1+cu121"
TORCH_VERSION = torch.__version__
if TORCH_VERSION != REQUIRED_VERSION:
    print(f"Detected torch version {TORCH_VERSION}, but notebook was created for {REQUIRED_VERSION}.")
    print(f"Attempting installation of {REQUIRED_VERSION}")
    !pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
print("Correct version of torch detected. You are running on a machine with GPU.")


!pip install -q torch_geometric==2.3.1
!pip install -q torch_scatter==2.1.1
!pip install -q torch_cluster==1.6.1
!pip install -q biopython==1.80 fair_esm==2.0.0 ml_collections==0.1.1


!gdown 1TNab2MVPT0MXIxqizYfpd6YSAa5i8d4T
!gdown 1-IfWkLa5asu4SeeZAQ09oWm4KlpBMPmq


import os
if not os.path.exists("/content/RiboDiffusion"):
    !git clone --depth 1 -b main https://github.com/GRAPH-0/RiboDiffusion.git
%cd /content/RiboDiffusion/


!mkdir ckpts
!mv ../exp_inf.pth ./ckpts
!mv ../exp_inf_large.pth ./ckpts


import torch
from tqdm import tqdm
import numpy as np
import random
from models import *
from utils import *
from diffusion import NoiseScheduleVP
from sampling import get_sampling_fn
from datasets import utils as du
import functools
import tree
from configs.inference_ribodiffusion import get_config




config = get_config()
# Choose heckpoint name
checkpoint_path = './ckpts/exp_inf.pth'
# checkpoint_path = './ckpts/exp_inf_large.pth'


config.eval.sampling_steps = 50
# config.eval.sampling_steps = 100




NUM_TO_LETTER = np.array(['A', 'G', 'C', 'U'])

def get_optimizer(config, params):
  """Return a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
      optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps, weight_decay=config.optim.weight_decay)
  elif config.optim.optimizer == 'AdamW':
      optimizer = torch.optim.AdamW(params, lr=config.optim.lr, amsgrad=True, weight_decay=1e-12)
  else:
      raise NotImplementedError(f'Optimizer {config.optim.optimizer} not supported yet!')
  return optimizer



# Initialize model
model = create_model(config)
ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
params = model.parameters()
optimizer = get_optimizer(config, model.parameters())
state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

model_size = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
print('model size: {:.1f}MB'.format(model_size))

# Load checkpoint
state = restore_checkpoint(checkpoint_path, state, device=config.device)
ema.copy_to(model.parameters())

# Initialize noise scheduler
noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                  continuous_beta_1=config.sde.continuous_beta_1)
# Obtain data scalar and inverse scalar
inverse_scaler = get_data_inverse_scaler(config)

# Setup sampling function
test_sampling_fn = get_sampling_fn(config, noise_scheduler, config.eval.sampling_steps, inverse_scaler)
pdb2data = functools.partial(du.PDBtoData, num_posenc=config.data.num_posenc, num_rbf=config.data.num_rbf, knn_num=config.data.knn_num)

