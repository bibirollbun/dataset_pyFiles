%pip install /kaggle/input/lmsys-packages/triton-2.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


%pip install /kaggle/input/lmsys-packages/xformers-0.0.24042abc8.d20240802-cp310-cp310-linux_x86_64.whl


!cp -r /kaggle/input/lmsys-modules-0805 human_pref


import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/kaggle/input/lmsys-chatbot-arena/train.csv")

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

train_df.to_parquet("train.parquet")
test_df.to_parquet("eval.parquet")



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


model_name_or_path = "princeton-nlp/gemma-2-9b-it-SimPO"
# model_name_or_path = "/kaggle/input/lmsys-checkpoints-0-0805"
csv_path = "train.parquet"

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


# for name, p in model.named_parameters():
#     print(name, p.device)
# for name, b in model.model.named_buffers():
#     print(name, b.device)

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

# np.save('prob_m0.npy', prob)

print("Finished")


# import torch
# import numpy as np
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from transformers import AutoTokenizer

# from human_pref.models.modeling_gemma2 import Gemma2ForSequenceClassification
# from human_pref.data.processors import ProcessorPAB
# from human_pref.data.dataset import LMSYSDataset
# from human_pref.data.collators import VarlenCollator, ShardedMaxTokensCollator
# from human_pref.utils import to_device


# model_name_or_path = "/kaggle/input/lmsys-checkpoints-0-0805"
# csv_path = "eval.parquet"  # changed to evaluation dataset

# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# processor = ProcessorPAB(
#     tokenizer=tokenizer,
#     max_length=4096,
#     support_system_role=False,
# )
# dataset = LMSYSDataset(
#     csv_file=csv_path,
#     query=None,
#     processor=processor,
#     include_swap=False,
#     is_parquet=True,
# )
# dataloader = DataLoader(
#     dataset,
#     batch_size=80,
#     num_workers=4,
#     collate_fn=ShardedMaxTokensCollator(
#         max_tokens=8192, base_collator=VarlenCollator()
#     ),
# )

# # model for pipelined inference
# num_hidden_layers = 42
# device_map = {
#     "model.embed_tokens": "cuda:0",
#     "model.norm": "cuda:1",
#     "score": "cuda:1",
# }
# for i in range(num_hidden_layers // 2):
#     device_map[f"model.layers.{i}"] = "cuda:0"
# for i in range(num_hidden_layers // 2, num_hidden_layers):
#     device_map[f"model.layers.{i}"] = "cuda:1"

# model = Gemma2ForSequenceClassification.from_pretrained(
#     model_name_or_path,
#     torch_dtype=torch.float16,
#     device_map=device_map,
# )

# # inv_freq clones for each device
# config = model.config
# dim = config.head_dim
# inv_freq = 1.0 / (
#     config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
# )
# inv_freq0 = inv_freq.to("cuda:0")
# inv_freq1 = inv_freq.to("cuda:1")

# # pipeline parallelism with two GPUs
# is_first = True
# hidden_states = None
# outs = []
# for batch in tqdm(dataloader):
#     for micro_batch in batch:
#         input_ids = to_device(micro_batch["input_ids"], "cuda:0")
#         seq_info = dict(
#             cu_seqlens=micro_batch["cu_seqlens"],
#             position_ids=micro_batch["position_ids"],
#             max_seq_len=micro_batch["max_seq_len"],
#             # attn_bias=BlockDiagonalCausalMask.from_seqlens(micro_batch["seq_lens"]),
#         )
#         seq_info = to_device(seq_info, "cuda:0")
#         if is_first:
#             with torch.no_grad(), torch.cuda.amp.autocast():
#                 prev_hidden_states = model.forward_part1(input_ids, seq_info, inv_freq0)
#             is_first = False
#             prev_seq_info, prev_hidden_states = to_device(
#                 [seq_info, prev_hidden_states], "cuda:1"
#             )
#             continue
#         with torch.no_grad(), torch.cuda.amp.autocast():
#             logits = model.forward_part2(prev_hidden_states, prev_seq_info, inv_freq1)
#             hidden_states = model.forward_part1(input_ids, seq_info, inv_freq0)

#             prev_seq_info, prev_hidden_states = to_device(
#                 [seq_info, hidden_states], "cuda:1"
#             )
#             outs.append(logits.cpu())

# # last micro-batch
# with torch.no_grad(), torch.cuda.amp.autocast():
#     logits = model.forward_part2(prev_hidden_states, prev_seq_info, inv_freq1)
#     outs.append(logits.cpu())

# pred = torch.cat(outs, dim=0)
# prob = pred.softmax(-1)

# # Evaluation on the eval dataset (if labels exist)
# if hasattr(dataset, "evaluate"):
#     print(dataset.evaluate(prob.numpy()))

# # save probabilities
# np.save('prob_eval.npy', prob)



results = prob.numpy()

# evaluate
metrics = dataset.evaluate(results)
print("Evaluation metrics:")
print(metrics)


import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# LABEL_COLS
LABEL_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]

# Ground-truth labels
y_true = dataset.df[LABEL_COLS].values  # shape (num_samples, 3)

# Convert one-hot labels to integer class indices
y_true_idx = np.argmax(y_true, axis=1)

# Predicted probabilities from the model
y_pred_prob = prob.numpy()  # shape (num_samples, 3)
y_pred_idx = np.argmax(y_pred_prob, axis=1)  # predicted class

# Compute classification metrics
accuracy = accuracy_score(y_true_idx, y_pred_idx)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true_idx, y_pred_idx, average="weighted"
)
cm = confusion_matrix(y_true_idx, y_pred_idx)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print("Confusion Matrix:")
print(cm)



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# LABEL_COLS
LABEL_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]

# Ground-truth labels
y_true = dataset.df[LABEL_COLS].values
y_true_idx = np.argmax(y_true, axis=1)

# Predicted labels
y_pred_idx = np.argmax(prob.numpy(), axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true_idx, y_pred_idx)

# Plot
plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=LABEL_COLS,
    yticklabels=LABEL_COLS
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


