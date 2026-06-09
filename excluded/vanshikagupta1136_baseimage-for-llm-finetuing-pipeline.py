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


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import json

from tqdm.auto import tqdm
from datasets import load_dataset
from huggingface_hub import login

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("HF_TOKEN")

login(secret_value_0)


train = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/train.csv')
test = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')


print('-----------train-----------')
print(train.head(1))
print(train.shape)
print(train.columns)


train.info()


skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
group_id = train['prompt']
label_id = train['winner_model_a winner_model_b winner_tie'.split()].values.argmax(1)
splits = list(skf.split(train, label_id, group_id))


print(f"group_id: {group_id}")
print(f"label: {label_id}, len {len(label_id)}")


splits


print(f"number of splits {len(splits)}, each splits contain {len(splits[0])}, and within it {len(splits[0][0])}: {len(splits[0][1])}, ideally {len(train)//5}")


splits[0][1]


group_id


label_id


print(len(group_id.unique()))


train["fold"] = -1
for fold, (_, valid_idx) in enumerate(splits):
    train.loc[valid_idx, "fold"] = fold

train.head(1)


train['fold'].value_counts()


tqdm.pandas()

train_original = pd.read_csv("/kaggle/input/lmsys-chatbot-arena/train.csv")
external_data = load_dataset("lmsys/chatbot_arena_conversations")["train"].to_pandas()

print('-----------external_data-----------')
print(external_data.head(1))
print(external_data.shape)
print(external_data.columns)


def separate_conv(conv):
    try:
        user_texts = [x["content"] for x in conv if x["role"] == "user"]
        assistant_texts = [x["content"] for x in conv if x["role"] == "assistant"]
        
        return user_texts, json.dumps(assistant_texts)
    except:
        print(conv)

external_data["prompt_a"], external_data["response_a"] = zip(*external_data.conversation_a.progress_apply(separate_conv))
external_data["prompt_b"], external_data["response_b"] = zip(*external_data.conversation_b.progress_apply(separate_conv))
external_data["prompt"] = external_data["prompt_a"].progress_apply(json.dumps)
print(external_data.winner.value_counts())

def one_hot_encode(winner):
    return pd.Series([int("model_a" == winner), int("model_b" == winner), int("tie" == winner or "tie (bothbad)" == winner)])

external_data[["winner_model_a", "winner_model_b", "winner_tie"]] = (external_data.winner.progress_apply(one_hot_encode))
cols = ["question_id","model_a","model_b","prompt","response_a","response_b","winner_model_a","winner_model_b","winner_tie"]
external_data = pd.DataFrame(external_data[cols].copy().values, columns=train_original.columns)

superset = pd.concat([external_data, train_original]).reset_index(drop=True)
external_data_deduplicated = superset.drop_duplicates(subset=["prompt"], keep="last")
external_data_deduplicated = external_data_deduplicated[external_data_deduplicated.index.isin(external_data.index)]
print(len(external_data_deduplicated))
print(external_data_deduplicated.head(1))

# external_data.to_csv("output/lmsys-33k.csv", index=False)
external_data_deduplicated = external_data_deduplicated.reset_index(drop=True)
# external_data_deduplicated.to_csv("output/lmsys-33k-deduplicated.csv", index=False)


print('-----------external_data_deduplicated-----------')
print(external_data_deduplicated.head(1))
print(external_data_deduplicated.shape)
print(external_data_deduplicated.columns)

print('-----------train-----------')
print(train.head(1))
print(train.shape)
print(train.columns)


# testing of tokenizer process 
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer_testing = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")

# Example text
text_testing = "This is an example sentence model path when have acces to dowload weights from HF"
max_length1 = 10
# tokenizer(text, add_special_tokens=False, max_length=max_length, truncation=True).input_ids

# Tokenize the text
inputs_testing = tokenizer_testing(text_testing, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
inputs_testing1 = tokenizer_testing(text_testing, add_special_tokens=False, max_length=max_length1, truncation=True)

# Print the tokenized input
print(inputs_testing)
print(inputs_testing1)
print(len(text_testing))
print(len(inputs_testing1.input_ids))
# print(len(inputs_testing['input_ids'][0]))
# print(len(inputs_testing['attention_mask']))


from transformers import AutoTokenizer

model_name_or_path = "google/gemma-2-9b-it" # model path when have acces to dowload weights from HF
tokenizer1 = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer1


idx = 1
data1 = train

data = data1.iloc[idx].to_dict()

print(f"data: {data}\n\n")

prompts = json.loads(data["prompt"])
responses_a = json.loads(data["response_a"])
responses_b = json.loads(data["response_b"]),


import torch
max_length1 = 4096
support_system_role1 = False

tokenizer = tokenizer1
max_length = max_length1
support_system_role = support_system_role1

PROMPT_PREFIX = """Please act as an impartial judge and evaluate the quality of the responses provided by two
AI assistants to the user question displayed below. You should choose the assistant that
follows the user’s instructions and answers the user’s question better. Your evaluation
should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of their responses. Begin your evaluation by comparing the two
responses and provide a short explanation. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow
the length of the responses to influence your evaluation. Do not favor certain names of
the assistants. Be as objective as possible. After providing your explanation, output your
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
if assistant B is better, and "[[C]]" for a tie."""

PROMPT_SUFFIX = "verdict is: [["

LABEL_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]

head = "<|The Start of Conversation between a User and two Assistants|>"
tail = "<|The End of Conversation between a User and two Assistants|>\n"
parts = []

for prompt, response_a, response_b in zip(prompts, responses_a, responses_b):
    if prompt is None:
        prompt = "null"
    if response_a is None:
        response_a = "null"
    if response_b is None:
        response_b = "null"
    parts.append(f"\n### User:\n{prompt}\n\n### Assistant A:\n{response_a}\n\n### Assistant B:\n{response_b}\n")

text = "".join(parts)
print(text)
print(tokenizer(text, add_special_tokens=False, max_length=max_length, truncation=True))
input_ids = tokenizer(text, add_special_tokens=False, max_length=max_length, truncation=True).input_ids
truncated_text = tokenizer.decode(input_ids)
# print(truncated_text)

conversation = head + truncated_text + tail
print(f"input_ids length with truncation: {len(input_ids)}\n\n")
print(f"truncated_text length: {len(truncated_text.split())}\n\n")
print(f"conversation: {conversation}\n\n")
print(f"conversation words length: {len(conversation.split())}\n\n")

if support_system_role:
    messages = [{"role": "system", "content": PROMPT_PREFIX}, {"role": "user", "content": conversation}]
else:
    messages = [{"role": "user", "content": f"{PROMPT_PREFIX}\n{conversation}"}]

input_text = (tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + PROMPT_SUFFIX)
input_ids = tokenizer(input_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
label = torch.tensor([data[col] for col in LABEL_COLS]).float()

print(f"after apply_chat_template input_ids length: {len(input_ids)}\n\n")
print(f"after apply_chat_template input_text words length: {len(input_text.split())}\n\n")
print(f"label length: {len(label)}\n\n")

# import numpy as np
# np.set_printoptions(threshold=np.inf)
# print(input_ids.numpy())

processorPAB = dict(input_ids=input_ids, input_text=input_text, label=label)

print(f"\n\nprocessorPAB: {processorPAB}")


import torch
max_length1 = 4096 // 2
support_system_role1 = False

tokenizer = tokenizer1
max_length = max_length1
support_system_role = support_system_role1

head = "<|The Start of Assistant A’s Conversation with User|>"
sep = "<|The End of Assistant A’s Conversation with User|>\n\n<|The Start of Assistant B’s Conversation with User|>"
tail = "<|The End of Assistant B’s Conversation with User|>\n"
parts_a = []
parts_b = []

for prompt, response_a, response_b in zip(prompts, responses_a, responses_b):
    if prompt is None:
        prompt = "null"
    if response_a is None:
        response_a = "null"
    if response_b is None:
        response_b = "null"
    parts_a.append(f"\n### User:\n{prompt}\n\n### Assistant A:\n{response_a}\n")
    parts_b.append(f"\n### User:\n{prompt}\n\n### Assistant B:\n{response_b}\n")

text_a = "".join(parts_a)
text_b = "".join(parts_b)

input_ids_a = tokenizer(text_a, add_special_tokens=False, max_length=max_length, truncation=True).input_ids
input_ids_b = tokenizer(text_b, add_special_tokens=False, max_length=max_length, truncation=True).input_ids

truncated_text_a = tokenizer.decode(input_ids_a)
truncated_text_b = tokenizer.decode(input_ids_b)

conversation = head + truncated_text_a + sep + truncated_text_b + tail
print(f"conversation: {conversation}\n\n")

if support_system_role:
    messages = [{"role": "system", "content": PROMPT_PREFIX}, {"role": "user", "content": conversation}]
else:
    messages = [{"role": "user", "content": f"{PROMPT_PREFIX}\n{conversation}"}]
    
input_text = (tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + PROMPT_SUFFIX)
input_ids = tokenizer(input_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
label = torch.tensor([data[col] for col in LABEL_COLS]).float()

ProcessorPAPB = dict(input_ids=input_ids, input_text=input_text, label=label)

print(f"ProcessorPAPB: {ProcessorPAPB}")


# processor = [ProcessorPAB, ProcessorPAPB]

import pandas as pd
import copy

# Define the processors
class ProcessorPAB:
    def __init__(self, tokenizer, max_length, support_system_role):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.support_system_role = support_system_role

    def build_input(self, data):
        # Example processing (you need to implement the actual processing logic)
        return {
            'input_ids': self.tokenizer.encode(data['prompt'], max_length=self.max_length, truncation=True),
            'labels': data['winner_model_a']
        }

class ProcessorPAPB:
    def __init__(self, tokenizer, max_length, support_system_role):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.support_system_role = support_system_role

    def build_input(self, data):
        # Example processing (you need to implement the actual processing logic)
        return {
            'input_ids': self.tokenizer.encode(data['prompt'], max_length=self.max_length, truncation=True),
            'labels': data['winner_model_b']
        }

# Initialize processors
processor = [
    ProcessorPAB(tokenizer=tokenizer, max_length=max_length, support_system_role=False),
    ProcessorPAPB(tokenizer=tokenizer, max_length=max_length // 2, support_system_role=False)
]

# Load the DataFrame
# df = pd.read_csv("../output/dtrainval.csv")
df = train
include_swap = True

# Apply the query
query = f"fold != {fold}"
df = df.query(query).reset_index(drop=True)

# Process each sample
dataset0 = []
for idx in range(len(df[0:20])):
    data = df.iloc[idx].to_dict()
    ret = [proc.build_input(data) for proc in processor]
    
    if include_swap:
        data_swap = copy.deepcopy(data)
        data_swap["model_a"], data_swap["model_b"] = data_swap["model_b"], data_swap["model_a"]
        data_swap["response_a"], data_swap["response_b"] = data_swap["response_b"], data_swap["response_a"]
        data_swap["winner_model_a"], data_swap["winner_model_b"] = data_swap["winner_model_b"], data_swap["winner_model_a"]
        ret.extend([proc.build_input(data_swap) for proc in processor])
    
    dataset0.append(ret)

# Now dataset0 contains the processed data
dataset0


# Initialize processors
processor = [
    ProcessorPAB(tokenizer=tokenizer, max_length=max_length, support_system_role=False),
    ProcessorPAPB(tokenizer=tokenizer, max_length=max_length // 2, support_system_role=False)
]

# Load the DataFrame
# df = pd.read_csv("../output/dtrainval.csv")
df = external_data
include_swap = True

# Apply the query
query = f"fold != {fold}"
# df = df.query(query).reset_index(drop=True)

# Process each sample
dataset1 = []
for idx in range(len(df[0:20])):
    data = df.iloc[idx].to_dict()
    ret = [proc.build_input(data) for proc in processor]
    
    if include_swap:
        data_swap = copy.deepcopy(data)
        data_swap["model_a"], data_swap["model_b"] = data_swap["model_b"], data_swap["model_a"]
        data_swap["response_a"], data_swap["response_b"] = data_swap["response_b"], data_swap["response_a"]
        data_swap["winner_model_a"], data_swap["winner_model_b"] = data_swap["winner_model_b"], data_swap["winner_model_a"]
        ret.extend([proc.build_input(data_swap) for proc in processor])
    
    dataset1.append(ret)

# Now dataset1 contains the processed data
dataset1


print(type(dataset0[0][0].keys()))
print(len(dataset0[0][0]['input_ids']))
print(len(dataset1[0][0]['input_ids']))


dataset = torch.utils.data.ConcatDataset([dataset0, dataset1])
dataset


# torch.utils.data.DataLoader(dataset,
#         shuffle=training,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         drop_last=training,
#         collate_fn=ShardedMaxTokensCollator(
#             max_tokens=max_tokens,
#             base_collator=VarlenCollator(),
#             sort_samples=training,
#         ),
#     )


batch_size =80
num_workers=4
training=True

max_tokens = 1024 * 16


import os
import torch
import torch.distributed as dist

# Manually set environment variables
os.environ['RANK'] = '1'
os.environ['WORLD_SIZE'] = '2'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'

# Initialize the distributed environment
dist.init_process_group(backend='nccl', init_method='tcp://localhost:12345', rank=0, world_size=1)

# Get the rank and world size
rank = dist.get_rank()
world_size = dist.get_world_size()


import torch 

samples = [  
{"input_ids":  torch.tensor([3, 4]), "label": 1},  
{"input_ids": torch.tensor([5, 6, 7, 8]), "label": 0},  
{"input_ids": torch.tensor([9, 10, 11]), "label": 1},  
{"input_ids": torch.tensor([12, 13]), "label": 0},  
{"input_ids": torch.tensor([14, 15, 16, 17, 18]), "label": 1}  
]
sort_samples = True
max_tokens = 10
world_size = 2 # (2 processes)
rank = 0 # then 1 on paralllel one 

print(f"Rank: {rank}, World Size: {world_size}\n\n")


# flatten if a 'sample' is a list of samples
if isinstance(samples[0], list):
    samples = [sample for s in samples for sample in s]
print(f'flatten samples {samples}\n\n')

# sorting on basis of size of input_ids within sample
if sort_samples:
    samples = sorted(samples, key=lambda x: x["input_ids"].size(0))
print(f'sort_samples {samples}\n\n')

# to handle future split samples into shards with padding till following conditions is true 
while len(samples) % world_size != 0:
    samples.append(samples[-1])
    print(f"Padding samples to make them divisible by num_shards={world_size}")
print(f'padding {samples}\n\n')

sample_index_matrix = torch.arange(len(samples)).reshape(-1, world_size)
size_matrix = torch.tensor([sample["input_ids"].size(0) for sample in samples]).reshape(-1, world_size)
micro_batch_segments = []
# (start, end) that (size_matrix[start:end].sum(0) <= max_tokens).all()
print(f'sample_index_matrix {sample_index_matrix}\n\n')
print(f'size_matrix {size_matrix}\n\n')

start = 0
for end in range(size_matrix.size(0)):
    # look ahead
    if (size_matrix[start : end + 1].sum(0) > max_tokens).any() and end > start:
        micro_batch_segments.append((start, end))
        start = end
print(f'micro_batch_segments {micro_batch_segments}\n\n')

if start < size_matrix.size(0):
    micro_batch_segments.append((start, size_matrix.size(0)))
print(f'micro_batch_segments {micro_batch_segments}\n\n')

micro_batches = []
for start, end in micro_batch_segments:
    micro_batch_samples = [samples[i] for i in sample_index_matrix[start:end, rank]]
    print(f'{start}, {end}, {micro_batch_samples}')
    # samples = micro_batch_samples
    seq_lens = []
    cu_seqlens = [0]
    end = 0
    input_idss = []
    position_idss = []
    
    for sample in micro_batch_samples:
        seq_len = sample["input_ids"].size(0)
        seq_lens.append(seq_len)
        end += seq_len
        cu_seqlens.append(end)
        input_idss.append(sample["input_ids"])
        position_idss.append(torch.arange(seq_len))
    
    input_ids = torch.cat(input_idss, dim=0)
    position_ids = torch.cat(position_idss, dim=0)
    
    data = dict(batch_size=len(micro_batch_samples),
        input_ids=input_ids.unsqueeze(0),
        position_ids=position_ids.unsqueeze(0),
        seq_lens=seq_lens,
        cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32),
        max_seq_len=max(seq_lens))
    print(f"micro_batch_samples, {micro_batch_samples}\n")
    print([sample["label"] for sample in micro_batch_samples])
    # data["label"] = torch.stack([sample["label"] for sample in micro_batch_samples], dim=0)
    data["label"] = torch.stack([torch.tensor(sample["label"]) for sample in micro_batch_samples], dim=0)

    
    for key in micro_batch_samples[0]:
        if key not in data.keys():
            data[key] = [sample[key] for sample in micro_batch_samples]
    micro_batches.extend([data])
    print(f'micro_batches {micro_batches}\n\n')

print(f'micro_batch_samples {micro_batch_samples}\n\n')
print(f'micro_batches {micro_batches}\n\n')

# Clean up
# dist.destroy_process_group()


VAL_FOLD = 0
max_tokens = 1024 * 16

max_length = 4096


# gemma2 model config

vocab_size=256000,
hidden_size=2304,
intermediate_size=9216,
num_hidden_layers=26,
num_attention_heads=8,
num_key_value_heads=4,
head_dim=256,
hidden_activation="gelu_pytorch_tanh",
max_position_embeddings=8192,
initializer_range=0.02,
rms_norm_eps=1e-6,
use_cache=True,
pad_token_id=0,
eos_token_id=1,
bos_token_id=2,
tie_word_embeddings=True,
rope_theta=10000.0,
attention_bias=False,
attention_dropout=0.0,
query_pre_attn_scalar=256,
sliding_window=4096,
final_logit_softcapping=30.0,
attn_logit_softcapping=50.0,
cache_implementation="hybrid",


from torch import nn
import torch

base_model_prefix = "model"
supports_gradient_checkpointing = True
_no_split_modules = ["Gemma2DecoderLayer"]
_skip_keys_device_placement = ["past_key_values"]
_supports_flash_attn_2 = True
_supports_sdpa = True
_supports_cache_class = False
_supports_quantized_cache = False
_supports_static_cache = True
_is_stateful = True


# _init_weights for both type of module, in a gaussian distribution
# from gemma2config
std =  0.2 # initializer_range

# i added whichever module we have nn.Linear or nn.Embedding
module = nn.Linear(10, 30)
if isinstance(module, nn.Linear):
    module.weight.data.normal_(mean=0.0, std=std)
    if module.bias is not None:
        module.bias.data.zero_()

elif isinstance(module, nn.Embedding):
    module.weight.data.normal_(mean=0.0, std=std)
    if module.padding_idx is not None:
        module.weight.data[module.padding_idx].zero_()

module


import torch
from torch import nn

# for creating sample model
input_id = torch.randint(0, 10000, (1020,)) # in model design 16 * 1024 is max of sequence of tokens # (T)
input_id = input_id.unsqueeze(0)
print(input_id)

vocab_size = 256000 # from gemma2 config
padding_idx = 0 # from gemma2 config
hidden_size = 2304

input_id = input_id.squeeze(0)
print(input_id)
embed_tokens = nn.Embedding(num_embeddings = vocab_size, embedding_dim = hidden_size, padding_idx = padding_idx) #(V,C)
inputs_embeds = embed_tokens(input_id) 
print(inputs_embeds) # (T,C)


print(inputs_embeds.shape) #(T, C)


hidden_states = inputs_embeds

print(hidden_size**0.5)
normalizer = torch.tensor(hidden_size**0.5, dtype=hidden_states.dtype)
hidden_states = hidden_states * normalizer
normalizer


import torch
head_dim = 256
max_position_embeddings = 8192
dim = head_dim
inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
seq = torch.arange(max_position_embeddings, dtype=inv_freq.dtype)
freqs = torch.einsum("i , j -> i j", seq, inv_freq)
emb = torch.cat((freqs, freqs), dim=-1)
emb = emb.reshape(emb.size(0), 1, 1, emb.size(1))
rotary_emb = emb
print(rotary_emb)
print(rotary_emb.shape)
print(len(rotary_emb))


num_hidden_layers=26
is_last_decoder_layer = False
head_dim=256 # 32 * 4 * 2
attention_bias = False
num_heads = 8
num_key_value_heads = 4
hidden_size = 2304

q_len, _ = hidden_states.size() # T 
# q_proj : 
q_proj = nn.Linear(in_features = hidden_size, out_features = num_heads * head_dim, bias = attention_bias)  
k_proj = nn.Linear(in_features = hidden_size, out_features = num_key_value_heads * head_dim, bias = attention_bias)
v_proj = nn.Linear(in_features = hidden_size, out_features = num_key_value_heads * head_dim, bias = attention_bias)
o_proj = nn.Linear(in_features = hidden_size, out_features = num_heads * head_dim, bias = attention_bias)

print(f"q_proj : {q_proj}, k_proj : {k_proj}, v_proj : {v_proj}, o_proj : {o_proj}\n\n")

query_states = q_proj(hidden_states)
key_states = k_proj(hidden_states)
value_states = v_proj(hidden_states)

# T, head*dim   | T, kvhead*dim  |  T, kvhead*dim
print(f"query_states : {query_states.shape}, key_states : {key_states.shape}, value_states : {value_states.shape}\n\n")


query_states = query_states.view(q_len, num_heads, head_dim)
key_states = key_states.view(q_len, num_key_value_heads, head_dim)
value_states = value_states.view(q_len, num_key_value_heads, head_dim)

print(f"query_states : {query_states.shape}, key_states : {key_states.shape}, value_states : {value_states.shape}\n\n")
# print(f"query_states : {query_states}, key_states : {key_states}, value_states : {value_states}\n\n")





# Above i considered layer_index = 1 for understanding 
# for i in range(num_hidden_layers):

# Gemma2Decoder > P Rotation > Attention > MLP > RMSNorm


# model = FSDP(
#     model,
#     auto_wrap_policy=fsdp.auto_wrap_policy,
#     sharding_strategy=fsdp.sharding_strategy,
#     device_id=torch.cuda.current_device(),
#     mixed_precision=fsdp.mixed_precision
# )

