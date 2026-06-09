%pip install /kaggle/input/lmsys-packages/triton-2.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
%pip install --no-index --find-links=/kaggle/input/xformers/xformers_dependencies xformers


!cp -r /kaggle/input/new-lmsys-085/new-lmsys-085 human_pref


%%writefile prepare_test_file.py
import pandas as pd
import os

IS_SUBMISSION=bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))


df = pd.read_parquet("/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet") if IS_SUBMISSION else pd.read_parquet("/kaggle/input/wsdm-cup-multilingual-chatbot-arena/train.parquet").head(n=100)
df["winner_model_a"] = 1
df["winner_model_b"] = 0
df["winner_tie"] = 0
df.to_parquet("test.parquet", index=False)
if not IS_SUBMISSION:
    df[['id', 'language', 'winner']].to_parquet('gt.parquet', index=False)

df["response_a"], df["response_b"] = df["response_b"], df["response_a"]
df.to_parquet("test_swap.parquet", index=False)


%%writefile predict_m0.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask
from human_pref.models.modeling_gemma2 import Gemma2ForSequenceClassification
from human_pref.data.processors import ProcessorPAB
from human_pref.data.dataset import LMSYSDataset
from human_pref.data.collators import VarlenCollator, ShardedMaxTokensCollator
from human_pref.utils import to_device


model_name_or_path = "/kaggle/input/lmsys-checkpoints-0-0805"
csv_path = "test.parquet"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
processor = ProcessorPAB(
    tokenizer=tokenizer,
    max_length=4096,
    support_system_role=False,
)

dataset = LMSYSDataset(
    csv_file=csv_path,
    query=None,
    processor=processor,
    include_swap=False,
    is_parquet=True,
)
dataloader = DataLoader(
    dataset,
    batch_size=80,
    num_workers=4,
    collate_fn=ShardedMaxTokensCollator(
        max_tokens=8192, base_collator=VarlenCollator()
    ),
)

# model for pipelined inference
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

model = Gemma2ForSequenceClassification.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    device_map=device_map,
)

# inv_freq clones for each device
config = model.config
dim = config.head_dim
inv_freq = 1.0 / (
    config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
)
inv_freq0 = inv_freq.to("cuda:0")
inv_freq1 = inv_freq.to("cuda:1")

# pipeline parallelism with two GPUs
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
            # attn_bias=BlockDiagonalCausalMask.from_seqlens(micro_batch["seq_lens"]),
        )
        seq_info = to_device(seq_info, "cuda:0")
        if is_first:
            with torch.no_grad(), torch.cuda.amp.autocast():
                prev_hidden_states = model.forward_part1(input_ids, seq_info, inv_freq0)
            is_first = False
            prev_seq_info, prev_hidden_states = to_device(
                [seq_info, prev_hidden_states], "cuda:1"
            )
            continue
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits = model.forward_part2(prev_hidden_states, prev_seq_info, inv_freq1)
            hidden_states = model.forward_part1(input_ids, seq_info, inv_freq0)

            prev_seq_info, prev_hidden_states = to_device(
                [seq_info, hidden_states], "cuda:1"
            )
            outs.append(logits.cpu())

# last micro-batch
with torch.no_grad(), torch.cuda.amp.autocast():
    logits = model.forward_part2(prev_hidden_states, prev_seq_info, inv_freq1)
    outs.append(logits.cpu())

pred = torch.cat(outs, dim=0)
prob = pred.softmax(-1)
print(dataset.evaluate(prob.numpy()))

np.save('prob_m0.npy', prob)


%%writefile predict_m3.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask
from human_pref.models.modeling_llama import LlamaForSequenceClassification
from human_pref.data.processors import ProcessorPAB
from human_pref.data.dataset import LMSYSDataset
from human_pref.data.collators import VarlenCollator, ShardedMaxTokensCollator
from human_pref.utils import to_device


model_name_or_path = "/kaggle/input/lmsys-checkpoints-3-0805"
csv_path = "test_swap.parquet"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.deprecation_warnings[
    "sequence-length-is-longer-than-the-specified-maximum"
] = True
processor = ProcessorPAB(
    tokenizer=tokenizer,
    max_length=4096,
    support_system_role=True,
)
dataset = LMSYSDataset(
    csv_file=csv_path,
    query=None,
    processor=processor,
    include_swap=False,
    is_parquet=True,
)
dataloader = DataLoader(
    dataset,
    batch_size=80,
    num_workers=4,
    collate_fn=ShardedMaxTokensCollator(
        max_tokens=8192, base_collator=VarlenCollator()
    ),
)

# model for pipelined inference
num_hidden_layers = 32
device_map = {
    "model.embed_tokens": "cuda:0",
    "model.norm": "cuda:1",
    "score": "cuda:1",
}
for i in range(num_hidden_layers // 2):
    device_map[f"model.layers.{i}"] = "cuda:0"
for i in range(num_hidden_layers // 2, num_hidden_layers):
    device_map[f"model.layers.{i}"] = "cuda:1"

model = LlamaForSequenceClassification.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    device_map=device_map,
)

# inv_freq clones for each device
config = model.config
dim = config.hidden_size // config.num_attention_heads
inv_freq = 1.0 / (
    config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
)
inv_freq0 = inv_freq.to("cuda:0")
inv_freq1 = inv_freq.to("cuda:1")

# pipeline parallelism with two GPUs
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
            attn_bias=BlockDiagonalCausalMask.from_seqlens(micro_batch["seq_lens"]),
        )
        seq_info = to_device(seq_info, "cuda:0")
        if is_first:
            with torch.no_grad(), torch.cuda.amp.autocast():
                prev_hidden_states = model.forward_part1(input_ids, seq_info, inv_freq0)
            is_first = False
            prev_seq_info, prev_hidden_states = to_device(
                [seq_info, prev_hidden_states], "cuda:1"
            )
            continue
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits = model.forward_part2(prev_hidden_states, prev_seq_info, inv_freq1)
            hidden_states = model.forward_part1(input_ids, seq_info, inv_freq0)

            prev_seq_info, prev_hidden_states = to_device(
                [seq_info, hidden_states], "cuda:1"
            )
            outs.append(logits.cpu())

# last micro-batch
with torch.no_grad(), torch.cuda.amp.autocast():
    logits = model.forward_part2(prev_hidden_states, prev_seq_info, inv_freq1)
    outs.append(logits.cpu())

pred = torch.cat(outs, dim=0)
prob = pred.softmax(-1)
print(dataset.evaluate(prob.numpy()))

np.save('prob_m3.npy', prob)


%%writefile make_submission.py
import numpy as np
import pandas as pd
import os

IS_SUBMISSION=bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))


df = pd.read_parquet("test.parquet")
preds = np.average(
    [
        np.load("prob_m0.npy"),
        np.load("prob_m3.npy")[:, [1, 0, 2]],
    ],
    axis=0,
    weights=[2, 1],
)
sub = pd.DataFrame({
    "id": df["id"],
    "winner_model_a": preds[:, 0],
    "winner_model_b": preds[:, 1],
    "winner_tie": preds[:, 2],
})
sub['winner'] = (sub['winner_model_a'] >= sub['winner_model_b']).map({True: 'model_a', False: 'model_b'})
sub[['id', 'winner']].to_csv("submission.csv", index=False)
if not IS_SUBMISSION:
    gt = pd.read_parquet("gt.parquet")
    df_merged = pd.merge(sub, gt, on='id', suffixes=('_sub', '_gt'))
    df_merged['correct'] = df_merged['winner_sub'] == df_merged['winner_gt']
    # # Compare the 'winner' columns
    # df_merged['correct'] = df_merged['winner_submission'] == df_merged['winner_reference']
    # # Calculate overall accuracy
    overall_accuracy = df_merged['correct'].mean()
    # # Print results
    print(f"\nOverall Accuracy: {overall_accuracy:.2%}")
    language_stats = df_merged.groupby('language').agg(
        count=('id', 'size'),
        mean_accuracy=('correct', 'mean')
    ).sort_values(by='count', ascending=False)
    language_stats.to_csv('language_breakdown.csv')
    print(language_stats.head(n=10))


%%bash
python prepare_test_file.py
python predict_m0.py
python predict_m3.py
python make_submission.py

