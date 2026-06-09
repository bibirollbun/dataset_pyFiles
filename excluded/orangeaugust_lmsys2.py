%%writefile config.yaml

testing: false

ab:
  model_path: "/kaggle/input/qwen14b-merge-useall-nopvhs-nopv"
  max_length: 4096
  method: "pp"

ba:
  model_path: "/kaggle/input/qwen14b-useall-nopvhs-clean31"
  max_length: 4096
  method: "pp"
  frac_samples: 0.2
  weight: 1.0


!cp -r /kaggle/input/packing/packing .

if 'vllm' in open("config.yaml").read():
    !pip uninstall -y pynvml
    !pip install nvidia-ml-py triton vllm==0.7.0 logits-processor-zoo -U --no-index --find-links /kaggle/input/vllm-070
else:
    !pip install triton xformers -U --no-index --find-links /kaggle/input/vllm-070


%%writefile infer.py
import time
import sys
import yaml
import torch
from torch import nn
import numpy as np
from tqdm import tqdm

class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

def time_to_str(t):
    mode = 'min' if t > 60 * 60 else 'sec'
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2dh:%02dm' % (hr, min)
    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2dm:%02ds' % (min, sec)
    else:
        raise NotImplementedError

do_swap = sys.argv[1] == "ba"
config = dotdict(yaml.safe_load(open("config.yaml"))["ab" if not do_swap else "ba"])

def distribute_lengths(lengths, max_length):
    # Make a copy to avoid modifying the original list
    remaining = lengths.copy()
    result = [0] * len(lengths)

    # Take full length for small elements first
    threshold = max_length // 3
    for i, length in enumerate(lengths):
        if length <= threshold:
            result[i] = length
            remaining[i] = 0

    # Distribute remaining space proportionally
    total_remaining = sum(remaining)
    remaining_space = max_length - sum(result)

    if total_remaining > 0:
        for i, length in enumerate(remaining):
            if length > 0:
                # Calculate proportional share of remaining space
                share = int((length / total_remaining) * remaining_space)
                result[i] = min(lengths[i], share)

    return result

def fmt_v5_multi(truncate_side, p, a, b, max_length, tokenizer):
    name = tokenizer.name_or_path.lower()
    if "qwen" in name or "7b" in name or "14b" in name or "32b" in name or "72" in name:
        start_p = tokenizer("<|im_start|>user\n", add_special_tokens=False).input_ids
        start_a = tokenizer("<|im_start|>model\n", add_special_tokens=False).input_ids
        start_b = tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
        eot = tokenizer("<|im_end|>\n", add_special_tokens=False).input_ids
        cut = tokenizer("\n........\n", add_special_tokens=False).input_ids
    else:
        start_p = tokenizer("<start_of_turn>user\n", add_special_tokens=False).input_ids
        start_a = tokenizer("<start_of_turn>model\n", add_special_tokens=False).input_ids
        start_b = tokenizer("<start_of_turn>assistant\n", add_special_tokens=False).input_ids
        eot = tokenizer("<end_of_turn>\n", add_special_tokens=False).input_ids
        cut = tokenizer("[...]", add_special_tokens=False).input_ids

    if tokenizer.bos_token_id is not None:
        bos = [tokenizer.bos_token_id]
    else:
        bos = []
    if tokenizer.eos_token_id is not None:
        eos = [tokenizer.eos_token_id]
    else:
        eos = []

    token_overhead = len(bos + start_p + eot + start_a + eot + start_b + eot + eos) + len(cut) * 3

    tok = tokenizer(sum(map(list, list(zip(p, a, b))), []), add_special_tokens=False).input_ids

    all_input_ids = []
    tok = [(tok[i], tok[i+1], tok[i+2]) for i in range(0,len(tok),3)]
    for pt, at, bt in tok:
        total_len = len(pt) + len(at) + len(bt) + token_overhead
        if total_len <= max_length:
            input_ids = bos + start_p + pt + eot + start_a + at + eot + start_b + bt + eot + eos
            all_input_ids.append(input_ids)
            continue

        new_len_p, new_len_a, new_len_b = distribute_lengths([len(pt), len(at), len(bt)], max_length - token_overhead)

        half_p = new_len_p // 2
        half_a = new_len_a // 2
        half_b = new_len_b // 2

        if len(pt) > new_len_p:
            pt_trunc = pt[:half_p-1] + cut + pt[-half_p:]
        else:
            pt_trunc = pt
        if len(at) > new_len_a:
            at_trunc = at[:half_a-1] + cut + at[-half_a:]
        else:
            at_trunc = at
        if len(bt) > new_len_b:
            bt_trunc = bt[:half_b-1] + cut + bt[-half_b:]
        else:
            bt_trunc = bt

        input_ids = bos + start_p + pt_trunc + eot + start_a + at_trunc + eot + start_b + bt_trunc + eot + eos
        all_input_ids.append(input_ids)

    return {"input_ids": all_input_ids}

if config.method == 'pp':
    from packing.data.dataset import LMSYSDataset
    from packing.data.collators import VarlenCollator, ShardedMaxTokensCollator
    from packing.utils import to_device
    from sklearn.metrics import log_loss, accuracy_score
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    if 'gemma' in config.model_path.lower() or '9b' in config.model_path.lower():
        from packing.models.modeling_gemma2 import Gemma2ForSequenceClassification
        model_fn = Gemma2ForSequenceClassification
        num_hidden_layers = 42
        device_map = {
            "model.embed_tokens": "cuda:0",
            "model.norm": "cuda:1",
            "score": "cuda:1",
        }
        for i in range(num_hidden_layers // 2):
            device_map[f"model.layers.{i}"] = "cuda:0"
        for i in range(num_hidden_layers // 2, num_hidden_layers):
            device_map[f"model.layers.{i}"] = "cuda:1"

    if 'qwen' in config.model_path.lower() or '14b' in config.model_path.lower():
        from packing.models.modeling_qwen2 import Qwen2ForSequenceClassification
        model_fn = Qwen2ForSequenceClassification
        num_hidden_layers = 48
        device_map = {
            "model.embed_tokens": "cuda:0",
            "model.norm": "cuda:1",
            "score": "cuda:1",
        }
        for i in range(num_hidden_layers // 2 - 1):
            device_map[f"model.layers.{i}"] = "cuda:0"
        for i in range(num_hidden_layers // 2 - 1, num_hidden_layers):
            device_map[f"model.layers.{i}"] = "cuda:1"

    class ProcessorPABfmtv5:
        LABEL_COLS = ["winner_model_a", "winner_model_b"]

        def __init__(self, tokenizer, max_length, swap=False):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.swap = swap

        def build_input(self, data):
            if self.swap:
                input_ids = fmt_v5_multi(
                    "both",
                    [data["prompt"]],
                    [data["response_b"]],
                    [data["response_a"]],
                    self.max_length,
                    self.tokenizer
                )["input_ids"][0]
            else:
                input_ids = fmt_v5_multi(
                    "both",
                    [data["prompt"]],
                    [data["response_a"]],
                    [data["response_b"]],
                    self.max_length,
                    self.tokenizer
                )["input_ids"][0]

            input_ids = torch.tensor(input_ids)
            return dict(input_ids=input_ids)

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    processor = ProcessorPABfmtv5(
        tokenizer=tokenizer,
        max_length=config.max_length,
        swap=False,
    )

    dataset = LMSYSDataset(
        csv_file=f"test_{'ab' if not do_swap else 'ba'}.parquet",
        query=None,
        processor=[processor],
        include_swap=False,
        is_parquet=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=80,
        num_workers=4,
        collate_fn=ShardedMaxTokensCollator(
            max_tokens=8192,
            base_collator=VarlenCollator(),
        ),
    )

    model = model_fn.from_pretrained(
        config.model_path,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    # inv_freq clones for each device
    dim = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    inv_freq = 1.0 / (
        model.config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )
    inv_freq0 = inv_freq.to("cuda:0")
    inv_freq1 = inv_freq.to("cuda:1")

    def inf_pp(dataloader, model):
        is_first = True
        hidden_states = None
        outs = []
        for batch in tqdm(dataloader):
            for micro_batch in batch:
                input_ids = to_device(micro_batch["input_ids"], "cuda:0")
                seq_info = dict(
                    cu_seqlens=micro_batch["cu_seqlens"],
                    position_ids=micro_batch["position_ids"],
                    max_seq_len=micro_batch["max_seq_len"],
                )
                seq_info = to_device(seq_info, "cuda:0")

                if is_first:
                    with torch.inference_mode(), torch.amp.autocast("cuda"):
                        prev_hidden_states = model.forward_part1(input_ids, seq_info, inv_freq0)
                    is_first = False
                    prev_seq_info, prev_hidden_states = to_device(
                        [seq_info, prev_hidden_states], "cuda:1"
                    )
                    continue

                with torch.inference_mode(), torch.amp.autocast("cuda"):
                    logits = model.forward_part2(prev_hidden_states, prev_seq_info, inv_freq1)
                    hidden_states = model.forward_part1(input_ids, seq_info, inv_freq0)

                    prev_seq_info, prev_hidden_states = to_device(
                        [seq_info, hidden_states], "cuda:1"
                    )
                    outs.append(logits.cpu())

        # last micro-batch
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            logits = model.forward_part2(prev_hidden_states, prev_seq_info, inv_freq1)
            outs.append(logits.cpu())

        pred = torch.cat(outs, dim=0)
        prob = pred.softmax(-1)
        return prob

    start_time = time.time()
    prob = inf_pp(dataloader, model).numpy()
elif config.method == 'vllm':
    import os
    import numpy as np
    import torch
    from vllm import LLM, SamplingParams
    from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
    from functools import partial
    import datasets
    datasets.disable_caching()

    def softmax(x, temp=1.0):
        x = np.nan_to_num(x, nan=0.0, posinf=1e10, neginf=-1e10)
        x = np.array(x) / temp
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x)

    llm = LLM(
        config.model_path,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        max_num_seqs=35,
        cpu_offload_gb=8.5,
        swap_space=1,
        dtype=torch.float16,
        enable_prefix_caching=True,
        enforce_eager=True,
        disable_log_stats=True,
    )

    tokenizer = llm.get_tokenizer()

    dataset = datasets.Dataset.from_parquet(f"test_{'ab' if not do_swap else 'ba'}.parquet")
    dataset = dataset.map(partial(fmt_v5_multi, "both"), num_proc=1, batched=True, input_columns=["prompt", "response_a", "response_b"], fn_kwargs={"max_length": config.max_length, "tokenizer": tokenizer}, batch_size=1000)
    prompts = [{"prompt_token_ids": x} for x in dataset["input_ids"]]

    token_a = tokenizer(" A", return_tensors="pt").input_ids.item()
    token_b = tokenizer(" B", return_tensors="pt").input_ids.item()

    params = SamplingParams(max_tokens=1, logprobs=20)
    start_time = time.time()
    outputs = llm.generate(prompts, params)
    logprobs = [[x.outputs[0].logprobs[0].get(token_a), x.outputs[0].logprobs[0].get(token_b)] for x in outputs]
    logprobs = [[x.logprob if x is not None else -np.inf for x in xs] for xs in logprobs]
    prob = np.array([softmax(x) for x in logprobs])

np.save(f"probs_{'ab' if not do_swap else 'ba'}.npy", prob)
print(f"Took: {time_to_str(time.time() - start_time)}, {time.time() - start_time:.1f}")
print(f"Samples per second: {len(prob) / (time.time() - start_time):.2f}")


import os
import yaml
from datasets import Dataset
import numpy as np
import pandas as pd
from time import time

start_time = time()

config = yaml.safe_load(open("config.yaml"))
if config['testing']:
    xs = pd.read_parquet("/kaggle/input/wsdm-cup-multilingual-chatbot-arena/train.parquet").head(500)
else:
    xs = pd.read_parquet("/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet")

print(f"Length of first pass: {len(xs)}")

xs.to_parquet("test_ab.parquet")
!python infer.py ab
probs_ab = np.load("probs_ab.npy")

xs["probs_ab"] = [xs.tolist() for xs in probs_ab]
xs["confidence"] = probs_ab.max(-1) - 0.5
n_samples_for_second_pass = max(1, int(len(xs) * config['ba']['frac_samples']))
xs_ba = xs.sort_values("confidence", ascending=True).head(n_samples_for_second_pass)
xs_ba["response_a"], xs_ba["response_b"] = xs_ba["response_b"], xs_ba["response_a"]

print(f"Length of second pass: {len(xs_ba)}")

xs_ba.to_parquet("test_ba.parquet")
!python infer.py ba
probs_ba = np.load("probs_ba.npy")

xs_ba["probs_ba"] = [xs.tolist() for xs in probs_ba]
xs.loc[xs_ba.index, "probs_ba"] = xs_ba["probs_ba"]

def combine_probs(prob_ab, prob_ba, weight_ba=1):
    if prob_ba is None:
        return prob_ab

    arr_ba = np.flip(prob_ba)
    if np.isnan(arr_ba).all():
        return prob_ab

    arr_ab = np.array(prob_ab, dtype=float)
    combined = (arr_ab + weight_ba * arr_ba) / (1 + weight_ba)
    return combined.tolist()

xs["probs_mean"] = [combine_probs(ab, ba, weight_ba=config['ba']['weight']) for ab, ba in zip(xs["probs_ab"], xs["probs_ba"])]

if config['testing'] and 'winner' in xs.columns:
    accuracy_ab = np.mean([['model_a', 'model_b'][np.argmax(x)] == y for x, y in zip(xs["probs_ab"], xs["winner"])])
    accuracy_ba = np.mean([['model_b', 'model_a'][np.argmax(x)] == y for x, y in zip(xs_ba["probs_ba"], xs_ba["winner"])])
    accuracy = np.mean([['model_a', 'model_b'][np.argmax(x)] == y for x, y in zip(xs["probs_mean"], xs["winner"])])

    print(f"Accuracy AB: {accuracy_ab:.4f}")
    print(f"Accuracy BA: {accuracy_ba:.4f}")
    print(f"Accuracy <>: {accuracy:.4f}")

    print(f"Samples: {len(xs) + len(xs_ba)}")
    print(f"Took: {time() - start_time:.1f}")
    print(f"Samples per second: {(len(xs) + len(xs_ba)) / (time() - start_time)}")


probs_mean = np.array(xs["probs_mean"].tolist())
xs["winner"] = np.where(probs_mean[:, 0] > probs_mean[:, 1], "model_a", "model_b")
submission = xs[["id", 'winner']]
submission.to_csv('submission.csv', index=False)

if len(xs) == 3:
    print(xs)
    print(submission)

