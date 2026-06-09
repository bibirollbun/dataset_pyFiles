# Installing offline dependencies
!pip install -U --no-deps /kaggle/input/faiss-gpu-173-python310/faiss_gpu-1.7.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install -U --no-deps /kaggle/input/datasets-214/datasets-2.14.5-py3-none-any.whl
!pip install -U --no-deps /kaggle/input/optimum-113/optimum-1.13.2-py3-none-any.whl
!pip install -U --no-deps /kaggle/input/transformers-432/transformers-4.32.1-py3-none-any.whl


import gc
import logging
from time import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Condition
import ctypes
from functools import partial

import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# For RAG
import faiss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset

NUM_TITLES = 5
MAX_SEQ_LEN = 512
MODEL_PATH = "/kaggle/input/bge-small-faiss/"

# For LLM
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from safetensors.torch import load_file
from optimum.bettertransformer import BetterTransformer

N_BATCHES = 5
MAX_LENGTH = 4096
MAX_CONTEXT = 1200
# With NUM_TITLES = 5, the median lenght of a context if 1100 tokens (Q1: 900, Q3: 1400)


import pandas as pd



df = pd.read_parquet('/kaggle/input/wikipedia-20230701/c.parquet')


# Function to clean RAM & vRAM
def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

# Load data
df = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/test.csv", index_col="id")

# Variable used to avoid running the notebook for 3 hours when submitting. Credit : CPMP
IS_TEST_SET = len(df) != 200

# Uncomment this to see results on the train set
# df = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/train.csv", index_col="id")
# IS_TEST_SET = True
# N_BATCHES = 1


# New SentenceTransformer class similar to the one used in @Mgöksu notebook but relying on the transformers library only

class SentenceTransformer:
    def __init__(self, checkpoint, device="cuda:0"):
        self.device = device
        self.checkpoint = checkpoint
        self.model = AutoModel.from_pretrained(checkpoint).to(self.device).half()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def transform(self, batch):
        tokens = self.tokenizer(batch["text"], truncation=True, padding=True, return_tensors="pt", max_length=MAX_SEQ_LEN)
        return tokens.to(self.device)  

    def get_dataloader(self, sentences, batch_size=32):
        sentences = ["Represent this sentence for searching relevant passages: " + x for x in sentences]
        dataset = Dataset.from_dict({"text": sentences})
        dataset.set_transform(self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader

    def encode(self, sentences, show_progress_bar=False, batch_size=32):
        dataloader = self.get_dataloader(sentences, batch_size=batch_size)
        pbar = tqdm(dataloader) if show_progress_bar else dataloader

        embeddings = []
        for batch in pbar:
            with torch.no_grad():
                e = self.model(**batch).pooler_output
                e = F.normalize(e, p=2, dim=1)
                embeddings.append(e.detach().cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings


if IS_TEST_SET:
    # Load embedding model
    start = time()
    print(f"Starting prompt embedding, t={time() - start :.1f}s")
    model = SentenceTransformer(MODEL_PATH, device="cuda:0")

    # Get embeddings of prompts
    f = lambda row : " ".join([row["prompt"], row["A"], row["B"], row["C"], row["D"], row["E"]])
    inputs = df.apply(f, axis=1).values # better results than prompt only
    prompt_embeddings = model.encode(inputs, show_progress_bar=False)

    # Search closest sentences in the wikipedia index 
    print(f"Loading faiss index, t={time() - start :.1f}s")
    faiss_index = faiss.read_index(MODEL_PATH + '/faiss.index')
    # faiss_index = faiss.index_cpu_to_all_gpus(faiss_index) # causes OOM, and not that long on CPU

    print(f"Starting text search, t={time() - start :.1f}s")
    search_index = faiss_index.search(np.float32(prompt_embeddings), NUM_TITLES)[1]

    print(f"Starting context extraction, t={time() - start :.1f}s")
    dataset = load_from_disk("/kaggle/input/all-paraphs-parsed-expanded")
    for i in range(len(df)):
        df.loc[i, "context"] = "-" + "\n-".join([dataset[int(j)]["text"] for j in search_index[i]])

    # Free memory
    faiss_index.reset()
    del faiss_index, prompt_embeddings, model, dataset
    clean_memory()
    print(f"Context added, t={time() - start :.1f}s")


# Create symlinks from kaggle datasets to fake cached model

checkpoint_path = Path("/root/.cache/")
checkpoint_path.mkdir(exist_ok=True, parents=True)

for part in [1, 2, 3]:
    source_dir = Path(f'/kaggle/input/platypus2-chuhac2-part{part}')
    for path in source_dir.glob("*"):
        (checkpoint_path / path.name).symlink_to(path)


class WeightsLoader:
    """
    Thread-safe class to load the weights of the model.
    The weights are loaded in the background and can be accessed with get_state_dict().
    All devices must call set_state_dict() before the weights are loaded.
    """
    
    def __init__(self, checkpoint_path, devices):
        self.checkpoint_path = Path(checkpoint_path)
        self.states = {device: None for device in devices}
        self.state_dict = None
        self.condition = Condition()
        
    def get_state_dict(self, device):
        with self.condition:
            while self.states[device] is not None:
                self.condition.wait()
            
            result = self.state_dict
            self.states[device] = None
            
            if not any(self.states.values()):
                self.condition.notify_all()

        return result

    def set_state_dict(self, layer_name, device):
        with self.condition:
            self.states[device] = layer_name
            if all(self.states.values()):
                assert len(set(self.states.values())) == 1, "All devices should load the same layer"
                self.state_dict = load_file(self.checkpoint_path / (layer_name + ".safetensors"), device="cpu")
                for d in self.states:
                    self.states[d] = None
                self.condition.notify_all()


# Class for sharded llama
class ShardedLlama:
    def __init__(self, checkpoint_path, weights_loader, device="cuda:0", dtype=torch.float16):
        """
        Sharded version of LlamaForCausalLM : the model is splitted into layer shards to reduce GPU memory usage.
        During the forward pass, the inputs are processed layer by layer, and the GPU memory is freed after each layer.
        To avoid loading the layers multiple times, we could save all the intermediate activations in RAM, but
        as Kaggle accelerators have more GPU memory than CPU, we simply batch the inputs and keep them on the GPU.

        Parameters
        ----------
        checkpoint_path : str or Path
            path to the checkpoint
        weights_loader : WeightsLoader
            object to load the weights
        device : str, optional
            device, by default "cuda:0"
        dtype : torch.dtype, optional
            dtype, by default torch.float16
        """
        
        # Save parameters
        self.checkpoint_path = Path(checkpoint_path)
        self.weights_loader = weights_loader
        self.device = device 
        self.dtype = dtype

        # Create model
        self.config = AutoConfig.from_pretrained(self.checkpoint_path)   
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.init_model()
        self.layer_names = ["model.embed_tokens"] + [f"model.layers.{i}" for i in range(len(self.model.model.layers))] + ["model.norm", "value_head"]

    def init_model(self):
    
        # Load meta model (no memory used)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.config)
            self.model.lm_head = torch.nn.Linear(8192, 8, bias=False) # originally 32k
            self.model.eval()
            self.model = BetterTransformer.transform(self.model) # enable flash attention
            self.model.tie_weights()
            
        self.layers = [self.model.model.embed_tokens] + list(self.model.model.layers) + [self.model.model.norm, self.model.lm_head]

        # Move buffers to device (not that much GPU memory used)
        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(self.model, buffer_name, self.device, value=buffer, dtype=self.dtype)

    def load_layer_to_cpu(self, layer_name):
        self.weights_loader.set_state_dict(layer_name, self.device)
        state_dict = self.weights_loader.get_state_dict(self.device)
        if "value_head.weight" in state_dict:
            state_dict = {"lm_head.weight" : state_dict["value_head.weight"]}
        return state_dict
        
    def move_layer_to_device(self, state_dict):
        for param_name, param in state_dict.items():
            assert param.dtype != torch.int8, "int8 not supported (need to add fp16_statistics)"
            set_module_tensor_to_device(self.model, param_name, self.device, value=param, dtype=self.dtype)

    def __call__(self, inputs):
        # inputs = [(prefix, suffix), ...] with prefix.shape[0] = 1 and suffix.shape[0] = 5
        
        # Reboot the model to make sure buffers are loaded and memory is clean
        del self.model
        clean_memory()
        self.init_model()
        
       # Send batch to device
        batch = [(prefix.to(self.device), suffix.to(self.device)) for prefix, suffix in inputs]
        n_suffixes = len(batch[0][1])
        suffix_eos = [(suffix != self.tokenizer.pad_token_id).sum(1) - 1 for _, suffix in inputs]

        # Create attention mask for the largest input, and position ids to use KV cache
        attention_mask = torch.ones(MAX_LENGTH, MAX_LENGTH)
        attention_mask = attention_mask.triu(diagonal=1)[None, None, ...] == 0
        attention_mask = attention_mask.to(self.device)
        position_ids = torch.arange(MAX_LENGTH, dtype=torch.long, device=self.device)[None, :]

        with ThreadPoolExecutor() as executor, torch.inference_mode():

            # Load first layer
            future = executor.submit(self.load_layer_to_cpu, "model.embed_tokens")

            for i, (layer_name, layer) in tqdm(enumerate(zip(self.layer_names, self.layers)), desc=self.device, total=len(self.layers)):

                # Load current layer and prepare next layer
                state_dict = future.result()
                if (i + 1) < len(self.layer_names):
                    future = executor.submit(self.load_layer_to_cpu, self.layer_names[i + 1])
                self.move_layer_to_device(state_dict)
                
                # Run layer
                for j, (prefix, suffix) in enumerate(batch):
                    if layer_name == "model.embed_tokens":
                        batch[j] = (layer(prefix), layer(suffix))
                    elif layer_name == "model.norm":
                        # Only keep the last token at this point
                        batch[j] = (None, layer(suffix[torch.arange(n_suffixes), suffix_eos[j]][:, None]))
                    elif layer_name == "value_head":
                        batch[j] = layer(suffix)[:, 0].mean(1).detach().cpu().numpy()
                    else:
                        # Run prefix
                        len_p, len_s = prefix.shape[1], suffix.shape[1]
                        new_prefix, (k_cache, v_cache) = layer(prefix, use_cache=True, attention_mask=attention_mask[:, :, -len_p:, -len_p:])
                        
                        # Run suffix
                        pos = position_ids[:, len_p:len_p + len_s].expand(n_suffixes, -1)
                        attn = attention_mask[:, :, -len_s:, -len_p - len_s:].expand(n_suffixes, -1, -1, -1)
                        kv_cache = (k_cache.expand(n_suffixes, -1, -1, -1), v_cache.expand(n_suffixes, -1, -1, -1))
                        new_suffix = layer(suffix, past_key_value=kv_cache, position_ids=pos, attention_mask=attn)[0]
                        batch[j] = (new_prefix, new_suffix)

                # Remove previous layer from memory (including buffers)
                layer.to("meta")
                clean_memory() # proposed by CPMP

        # Get scores
        return batch


# Run model on the 2 GPUs

def get_tokens(row, tokenizer): 
        system_prefix = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\nContext:\n{context}"
        instruction = "Your task is to analyze the question and answer below. If the answer is correct, respond yes, if it is not correct respond no. As a potential aid to your answer, background context from Wikipedia articles is at your disposal, even if they might not always be relevant."

        # max length : MAX_LENGTH
        prompt_suffix = [f"{row[letter]}\n\n### Response:\n" for letter in "ABCDE"]
        suffix = tokenizer(prompt_suffix, return_tensors="pt", return_attention_mask=False, truncation=True, max_length=MAX_LENGTH, padding=True)["input_ids"][:, 1:]

        # max length : max(0, MAX_LENGTH - len(suffix))
        prompt_question = f"\nQuestion: {row['prompt']}\nProposed answer: "
        question = tokenizer(prompt_question, return_tensors="pt", return_attention_mask=False, truncation=True, max_length=max(0, MAX_LENGTH - suffix.shape[1]))["input_ids"][:, 1:]

        # max length : min(MAX_CONTEXT, max(0, MAX_LENGTH - len(suffix) - len(question)))
        prompt_context = system_prefix.format(instruction=instruction, context=row["context"])
        max_length = min(MAX_CONTEXT, max(0, MAX_LENGTH - question.shape[1] - suffix.shape[1]))
        context = tokenizer(prompt_context, return_tensors="pt", return_attention_mask=False, truncation=True, max_length=max_length)["input_ids"]

        prefix = torch.cat([context, question], dim=1)
        return prefix, suffix

def run_model(device, df, weights_loader):
    model = ShardedLlama(checkpoint_path, weights_loader, device=device)
    f = partial(get_tokens, tokenizer=model.tokenizer)
    inputs = df.apply(f, axis=1).values
    batches = np.array_split(inputs, N_BATCHES)
    outputs = []
    for i, batch in enumerate(batches):
        outputs += model(batch)
    return outputs

# Run model
if IS_TEST_SET:
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    weights_loader = WeightsLoader(checkpoint_path, devices)
    f = partial(run_model, weights_loader=weights_loader) # added by treesky
    with ThreadPoolExecutor() as executor:
        outputs = list(executor.map(f, devices, np.array_split(df, 2)))
        outputs = sum(outputs, [])
        
    # Save results
    n = len(df)
    for i, scores in enumerate(outputs):
        top3 = np.argsort(scores)[::-1]
        df.loc[i, "prediction"] = " ".join(["ABCDE"[j] for j in top3])
    
    # Display performances if train set is used 
    if "answer" in df.columns:
        for i in range(n):
            df.loc[i, "top_1"] = df.loc[i, "prediction"][0]
            df.loc[i, "top_2"] = df.loc[i, "prediction"][2]
            df.loc[i, "top_3"] = df.loc[i, "prediction"][4]

        top_i = [(df[f"top_{i}"] == df["answer"]).sum() for i in [1, 2, 3]]
        print(f"top1 : {top_i[0]}/{n}, top2 : {top_i[1]}/{n}, top3 : {top_i[2]}/{n} (total={sum(top_i)} / {n})")
        print(f"Accuracy: {100*top_i[0]/n:.1f}%, map3: {100*(top_i[0] + top_i[1]*1/2 + top_i[2]*1/3).sum()/n:.1f}%")
else:
    df["prediction"] = "A B C"

df[["prediction"]].to_csv("submission.csv")

