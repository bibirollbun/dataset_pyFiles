!pip install -q -U --upgrade pip
!pip install -q -U --upgrade tensorflow-hub
!pip install -q -U --upgrade tensorflow-cpu 
!pip install -q -U --upgrade tf-keras
!pip install -q -U --upgrade keras-hub tensorflow-text
!pip install -q -U --upgrade keras>=3


import os


print("\n... ENVIRONMENT SETUP (LOGGING/KERAS-BACKEND) ...\n")
# Set backend
# Pre-allocate 90% of TPU memory to minimize memory fragmentation and allocation overhead
# Disable logging/warning that may bloat the output

# keras_backend = "tensorflow"
keras_backend = 'jax'
allocation_fraction = 0.9
tf_min_log_level = '3'
os.environ["KERAS_BACKEND"] = keras_backend
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(allocation_fraction)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf_min_log_level)

# Finish imports
import jax
import keras
import keras_hub

# Other imports
import time
import copy

print(jax.devices())
print(keras.backend.backend())

# Run at half precision.
# keras.config.set_floatx("bfloat16")


# Note we use jax.devices() instead of keras.distribution.list_devices()
def initialize_device_mesh(
    shape: tuple[int, int] = (1, 8), 
    batch_axis_name: str = "batch",
    model_axis_name: str = "model"
) -> keras.distribution.DeviceMesh:
    """Initializes and returns a DeviceMesh for distributing computation across devices.
    
    To load the model with the weights and tensors distributed across TPUs, we first create a new DeviceMesh. 
        - DeviceMesh represents a collection of hardware devices configured for distributed computation.
        - DeviceMesh was introduced in Keras 3 as part of the unified distribution API.
    
    The distribution API enables data and model parallelism.
        - This allows for efficient scaling of deep learning models on multiple accelerators and hosts. 
        - The API leverages the underlying framework (e.g. JAX) to distribute the program and tensors according to the sharding directives.
            - This is done through a procedure called single program, multiple data (SPMD) expansion. 
            - Check out more details in the new Keras 3 distribution API guide.
                --> https://keras.io/guides/distribution/
    
    Args:
        shape: A tuple specifying the shape of the overall `DeviceMesh` 
            - `(8,)` for a data parallel only distribution,
            - `(4, 2)` for a model+data parallel distribution.
        batch_axis_name: A string indicating the axis name for the batch axis for DeviceMesh
        model_axis_name: The logical name of the model axis for the `DeviceMesh`

    Returns:
        A configured DeviceMesh instance. 
            - Defaults to (1, 8) shape so that the weights are sharded across all 8 TPUs (v3-8).
        NOTE: This API is aligned with `jax.sharding.Mesh` and `tf.dtensor.Mesh`
            - i.e. It represents the computation devices in the global context.
    """
    return keras.distribution.DeviceMesh(
        shape=shape, 
        axis_names=[batch_axis_name, model_axis_name], 
        devices=jax.devices()
    )


def configure_layout_map(
    device_mesh: keras.distribution.DeviceMesh,
    model_axis_name: str = "model"
) -> keras.distribution.LayoutMap:
    """Configures and returns a LayoutMap for model weight distribution.
    
    LayoutMap from the distribution API specifies how the weights and tensors should be sharded or replicated, using the string keys. 
        - For example: 'token_embedding/embeddings' below, which are treated like regex to match tensor paths. 
        - Matched tensors are sharded with model dimensions (8 TPUs); others will be fully replicated.

    Args:
        device_mesh: The `DeviceMesh` that is used to populate the `TensorLayout.device_mesh`
        axis_name: The logical name of the model axis for the `DeviceMesh`

    Returns:
        A LayoutMap instance with predefined sharding configurations.
    """
    layout_map = keras_hub.models.GemmaBackbone.get_layout_map(
        device_mesh, model_parallel_dim_name="model")
    
    return layout_map


def update_distribution_strategy(
    layout_map: keras.distribution.LayoutMap
) -> None:
    """Loads a distributed Gemma model based on the provided device mesh and layout map.
    
    ModelParallel allows you to shard model weights or activation tensors across all devcies on the DeviceMesh.
    In this case, some of the Gemma model weights are sharded across 8 TPU chips according the layout_map.
    
    Args:
        model_name: The name of the Gemma model to load.
        device_mesh: A DeviceMesh instance for model distribution.
        layout_map: A LayoutMap instance defining how to distribute model weights.

    Returns:
        None; The keras backend is updated with the appropriate distribution strategy
    """
    
    # Shard across devices in the mesh and update distribution strategy accordingly
    model_parallel = keras.distribution.ModelParallel(layout_map=layout_map, batch_dim_name="batch")
    keras.distribution.set_distribution(model_parallel)
    
    
def get_distributed_gemma(model_name: str) -> keras_hub.models.GemmaCausalLM:
    """Obtain the TPU compatible Gemma model of your choice.
    
    Args:
        model_name: The name of the Gemma model to load.
        
    Returns:
        A loaded Gemma model instance configured for distributed computation.
    """
    # Return the model 
    return keras_hub.models.GemmaCausalLM.from_preset(model_name)


def do_gemma_prep(return_device_mesh=True, return_layout_map=True):
    """Does the necessary steps so that we can instantiate a model properly
    
    Args:
        return_*: Whether to return the respective object as part of a dictionary
            - The key is the name (*) and the value is the object itself
    """
    # device_mesh = initialize_device_mesh(shape=(4,2))
    device_mesh = initialize_device_mesh()
    layout_map = configure_layout_map(device_mesh)
    update_distribution_strategy(layout_map)
    
    return_map = {}
    if not (return_device_mesh or return_layout_map):
        pass
    else:
        if return_device_mesh:
            return_map["device_mesh"] = device_mesh
        if return_layout_map:
            return_map["layout_map"] = layout_map
    return return_map

setup_objects = do_gemma_prep()
model = get_distributed_gemma("/kaggle/input/gemma2/keras/gemma2_9b_en/2")

for layer in model._backbone.layers:
    layer.trainable = False

tokenizer = keras_hub.models.GemmaTokenizer.from_preset("/kaggle/input/gemma2/keras/gemma2_9b_en/2",add_bos=True, add_eos=True)

print(f"SETUP ARTIFACTS: \n{setup_objects}\n\nMODEL OBJECT:\n{model}")
print("\n\n\n... MODEL SUMMARY ...\n\n")
model.summary()


# Check that it works...
model.generate("Best comedy movies in the 90s ", max_length=200)


"""TPU Evaluation metric for Santa 2024."""

import gc
import os
import math
import random
from tqdm import tqdm
from math import exp
from collections import Counter, OrderedDict
from typing import List, Optional, Union
from functools import partial

import numpy as np
import pandas as pd
import transformers
import torch

os.environ['OMP_NUM_THREADS'] = '96'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
PAD_TOKEN_ID = 0


class LRUCache:
    def __init__(self, capacity=10**11):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def __len__(self):
        return len(self.cache)


class PerplexityCalculator:
    """
    Calculates perplexity of text using a pre-trained language model.

    Adapted from https://github.com/asahi417/lmppl/blob/main/lmppl/ppl_recurrent_lm.py

    Parameters
    ----------
    model_path : str
        Path to the pre-trained language model

    load_in_8bit : bool, default=False
        Use 8-bit quantization for the model. Requires CUDA.

    device_map : str, default="auto"
        Device mapping for the model.
    """

    def __init__(
        self,
        model,
        tokenizer,
        capacity=10**11
    ):
        self.tokenizer = tokenizer
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        self.model = model
        self.cache = LRUCache(capacity=capacity)

    def pad_func(self, seqs):
        return keras.utils.pad_sequences(
                seqs,
                maxlen=None,
                dtype='int32',
                padding='post',
                truncating='post',
                value=PAD_TOKEN_ID
            )

    def get_perplexity(
        self, input_texts: Union[str, List[str]], use_cache=True, batch_size=32) -> Union[float, List[float]]:

        single_input = isinstance(input_texts, str)
        input_texts = [input_texts] if single_input else input_texts

        results = [None] * len(input_texts)
        if use_cache:
            text_to_process = []
            for i, text in enumerate(input_texts):
                cached_val = self.cache.get(text)
                if cached_val is not None:
                    results[i] = cached_val
                else:
                    text_to_process.append(text)
        else:
            text_to_process = input_texts.copy()
        
        loss_list = []

        batches = len(text_to_process)//batch_size + (len(text_to_process)%batch_size != 0)
        for j in range(batches):

            a = j*batch_size
            b = (j+1)*batch_size
            input_batch = text_to_process[a:b]

            tk_ids = self.pad_func(self.tokenizer(input_batch))
            model_inputs = {
                "token_ids": tk_ids,
                "padding_mask": tk_ids!=PAD_TOKEN_ID
            }
            
            logits = self.model.score(**model_inputs)

            label = np.array(model_inputs['token_ids'],dtype=int)
            label[label == PAD_TOKEN_ID] = PAD_TOKEN_LABEL_ID

            # Shift logits and labels for calculating loss
            shift_logits = torch.from_numpy(np.array(logits,dtype='float32'))[..., :-1, :].contiguous() # Drop last prediction
            shift_labels = torch.from_numpy(label)[..., 1:].contiguous()  # Drop first input

            # Calculate token-wise loss
            loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            loss = loss.view(len(logits), -1)
            valid_length = (shift_labels != PAD_TOKEN_LABEL_ID).sum(dim=-1)
            loss = torch.sum(loss, -1) / valid_length

            loss_list += loss.tolist()

        ppl = [exp(i) for i in loss_list]

        index_ppl = 0
        for index_el, el in enumerate(results):
            if el is None:
                results[index_el] = ppl[index_ppl]
                self.cache.set(text_to_process[index_ppl], ppl[index_ppl])
                index_ppl += 1
        return results[0] if single_input else results


scorer = PerplexityCalculator(model,tokenizer)


samples = pd.read_csv("/kaggle/input/fast-ensemble-of-multi-solutions-scores-analysis/submission.csv")
texts = samples['text'].to_list()
init_scores = scorer.get_perplexity(texts, batch_size=8)

for t,s in zip(texts, init_scores):
    print(t)
    print(f"Score: {s}\n")

print(f"Starting mean score: {np.mean(init_scores)}")

