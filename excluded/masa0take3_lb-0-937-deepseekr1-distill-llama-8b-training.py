# Install libs, run once and then comment
!pip uninstall -qq tensorflow -y
#!pip install flash_attn
!pip install -qq transformers==4.45.0
!pip install -qq accelerate==1.10.1 #not required for tpu runtime, keeping to not have transformer dependency problem
!pip install -qq peft==0.10.0
!pip install -qq torch==2.6.0
!pip install -qq 'torch_xla[tpu]==2.6.0' -f https://storage.googleapis.com/libtpu-releases/index.html


import os
import gc
import re
import types
from time import time
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder
import time
import os
import glob
from tqdm.auto import tqdm

import torch
from sklearn.model_selection import train_test_split
import transformers
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                        AutoTokenizer, Qwen2Config, get_cosine_schedule_with_warmup)
from transformers import LlamaConfig  # Import LlamaConfig for DeepSeek
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import  PeftModel,  get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F
from IPython.display import display, Math, Latex

# import torch_xla.debug.profiler as xp
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

xr.use_spmd()
import torch_xla.distributed.spmd as xs
from torch.optim import AdamW
from torch_xla.distributed.spmd import Mesh

import torch.nn as nn
import re
tqdm.pandas()

print(f'Torch Version: {torch.__version__}')


class CFG:
    SEED = 42 
    MODEL_NAME = '/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-llama-8b/2'
    NUM_EPOCHS = 2
    BATCH_SIZE = 8
    EVAL_BATCH_SIZE = 8
     
    MAX_LENGTH = 256
    WARMUP_RATIO = 0.01
    LR = 4e-4 #lora likes high
    
    # Lora configs
    NUM_LABELS = 65 
    LORA_RANK = 8
    LORA_ALPHA = 16
    DROPOUT = 0.005
    LORA_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"] 
    
    # Gradient accumulation
    GRADACCUM = 1
    
    # Evaluation strategy
    VAL_SPLIT = 0.1
    EVAL_STRATEGY = "steps"  # "epoch", "steps", "no"
    EVAL_STEPS = 4000  # Only used if EVAL_STRATEGY = "steps"
    
    # Save strategy  
    SAVE_STRATEGY = "epoch"  # "epoch", "steps", "no"
    SAVE_STEPS = 1000  # Only used if SAVE_STRATEGY = "steps"
    SAVE_TOTAL_LIMIT = 2  # Keep only 2 most recent checkpoints
    
    # Output directory
    OUT_DIR = "checkpoints"
    
    # Additional training parameters
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0  
    
DEVICE = xm.xla_device() # Initialize TPU Device


def set_seeds(seed):
    """Set seeds for reproducibility """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Set seed for all TPU cores
    xm.set_rng_state(seed, device=xm.xla_device())  

set_seeds(seed=CFG.SEED)


def get_token_lengths(texts):
    # tokenize and receive input_ids for reach text
    input_ids = tokenizer(texts.tolist(), return_tensors='np')['input_ids']
    # return length of inputs_ids for each text
    return [len(t) for t in input_ids]

def create_dataset(input_ids, attention_masks, labels, batch_size, shuffle=True):
    N_SAMPLES = labels.shape[0]
    IDXS = np.arange(N_SAMPLES - (N_SAMPLES % batch_size))
    while True:
        if shuffle:
            np.random.shuffle(IDXS)
        
        for idxs in IDXS.reshape(-1, batch_size):
            input_ids_batch = torch.tensor(input_ids[idxs]).to(DEVICE)
            attention_mask_batch = torch.tensor(attention_masks[idxs]).to(DEVICE)
            labels_batch = torch.tensor(labels[idxs], dtype=torch.long).to(DEVICE)  # torch.long for classification
            
            # Shard Over TPU Nodes if applicable
            xs.mark_sharding(input_ids_batch, mesh, (0, 1))
            xs.mark_sharding(attention_mask_batch, mesh, (0, 1))
            xs.mark_sharding(labels_batch, mesh, (0,))  # Labels are 1D
            
            yield input_ids_batch, attention_mask_batch, labels_batch

def compute_map3(y_true, y_pred_probs):
    """Compute MAP@3 metric"""
    top3 = np.argsort(-y_pred_probs, axis=1)[:, :3]
    match = (top3 == y_true[:, None])
    map3 = 0
    for i in range(len(y_true)):
        if match[i, 0]:
            map3 += 1.0
        elif match[i, 1]:
            map3 += 1.0 / 2
        elif match[i, 2]:
            map3 += 1.0 / 3
    return map3 / len(y_true)

def evaluate_model(model, val_dataset, steps_per_val_epoch):
    """Evaluate model on validation set"""
    model.eval()
    val_losses = []
    val_y_true = []
    val_y_probs = []
    
    with torch.no_grad():
        for step in range(steps_per_val_epoch):
            input_ids, attention_mask, labels = next(val_dataset)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.to(dtype=torch.float32)
            
            loss = LOSS_FN(logits, labels)
            val_losses.append(float(loss))
            
            probs = F.softmax(logits, dim=-1)
            val_y_true.extend(labels.cpu().tolist())
            val_y_probs.extend(probs.detach().cpu().numpy())
    
    val_loss = np.mean(val_losses)
    val_map3 = compute_map3(np.array(val_y_true), np.array(val_y_probs))
    
    model.train()
    return val_loss, val_map3

def manage_checkpoints(save_dir, limit):
    """Keep only the most recent checkpoints based on SAVE_TOTAL_LIMIT"""
    if limit <= 0:
        return
        
    checkpoints = glob.glob(os.path.join(save_dir, "model_step_*.pth"))
    checkpoints.extend(glob.glob(os.path.join(save_dir, "model_epoch_*.pth")))
    
    if len(checkpoints) > limit:
        # Sort by modification time
        checkpoints.sort(key=os.path.getmtime)
        # Remove oldest checkpoints
        for checkpoint in checkpoints[:-limit]:
            try:
                os.remove(checkpoint)
                # Also remove corresponding optimizer file
                opt_file = checkpoint.replace("model_", "optimizer_")
                if os.path.exists(opt_file):
                    os.remove(opt_file)
            except:
                pass

def save_checkpoint(epoch, step, val_map3=None):
    """Save model and optimizer checkpoints"""
    if CFG.SAVE_STRATEGY == "no":
        return
        
    if val_map3 is not None:
        model_path = os.path.join(CFG.OUT_DIR, f'model_epoch_{epoch}_map3_{val_map3:.4f}.pth')
        opt_path = os.path.join(CFG.OUT_DIR, f'optimizer_epoch_{epoch}.pth')
    else:
        model_path = os.path.join(CFG.OUT_DIR, f'model_step_{step}.pth')
        opt_path = os.path.join(CFG.OUT_DIR, f'optimizer_step_{step}.pth')
    
    xm.save({k: v.cpu() for k, v in model.named_parameters() if v.requires_grad}, model_path)
    xm.save(OPTIMIZER.state_dict(), opt_path)
    
    # Manage checkpoint limit
    manage_checkpoints(CFG.OUT_DIR, CFG.SAVE_TOTAL_LIMIT)
    
    return model_path

def plot_metrics(metrics):
    """Simple training metrics visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(metrics['train_loss'], 'b-', alpha=0.7, label='Train')
    if metrics['val_loss']:
        val_steps = [i * len(metrics['train_loss']) // len(metrics['val_loss']) 
                    for i in range(len(metrics['val_loss']))]
        axes[0].plot(val_steps, metrics['val_loss'], 'r-', label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Step')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAP@3 plot
    if metrics['train_map3']:
        epoch_steps = [i * len(metrics['train_loss']) // len(metrics['train_map3']) 
                      for i in range(len(metrics['train_map3']))]
        axes[1].plot(epoch_steps, metrics['train_map3'], 'g-', label='Train')
    if metrics['val_map3']:
        val_steps = [i * len(metrics['train_loss']) // len(metrics['val_map3']) 
                    for i in range(len(metrics['val_map3']))]
        axes[1].plot(val_steps, metrics['val_map3'], 'orange', label='Val')
    axes[1].set_title('MAP@3')
    axes[1].set_xlabel('Step')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Quick summary
    print(f"Final - Train Loss: {metrics['train_loss'][-1]:.4f}")
    if metrics['val_loss']: 
        print(f"Final - Val Loss: {metrics['val_loss'][-1]:.4f}")
    if metrics['train_map3']: 
        print(f"Final - Train MAP@3: {metrics['train_map3'][-1]:.4f}")
    if metrics['val_map3']: 
        print(f"Final - Val MAP@3: {metrics['val_map3'][-1]:.4f}")


le = LabelEncoder()
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category+":"+train.Misconception
train['label'] = le.fit_transform(train['target'])
target_classes = le.classes_
n_classes = len(target_classes)
print(f"Train shape: {train.shape} with {n_classes} target classes")


train.head()


idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)


def format_input(row):
    x = "This is Correct answer."
    if not row['is_correct']:
        x = "This is Incorrect answer."
    return (
        f"• Question: {row['QuestionText']}\n"
        f"• Answer: {row['MC_Answer']}\n"
        f"• Correctness: {x}\n"
        f"• Student Explanation: {row['StudentExplanation']}"
    )

train['text'] = train.apply(format_input,axis=1)
print("Example prompt for our LLM:")
print()
print( train.text.values[0] )


"""
# ===== PREPARE DATA (replace this whole cell) =====
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 1) Load
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv').copy()

# 2) Fix known noise (for QuestionId=31778, 6 is correct, 9 is incorrect)
QID_NOISY = 31778

def _fix_tf_prefix(cat: str, want_true: bool) -> str:
    if not isinstance(cat, str) or "_" not in cat:
        suffix = "Neither:NA"
    else:
        suffix = cat.split("_", 1)[1]
    return ("True_" if want_true else "False_") + suffix

mask_31778 = train["QuestionId"] == QID_NOISY
m6 = mask_31778 & (train["MC_Answer"].astype(str) == "6")
m9 = mask_31778 & (train["MC_Answer"].astype(str) == "9")
train.loc[m6, "Category"] = train.loc[m6, "Category"].apply(lambda c: _fix_tf_prefix(c, True))
train.loc[m9, "Category"] = train.loc[m9, "Category"].apply(lambda c: _fix_tf_prefix(c, False))

# 3) Determine the correct MC_Answer per QuestionId by majority vote among True_* rows
true_only = train[train["Category"].str.startswith("True", na=False)].copy()
true_only["_MC_"] = true_only["MC_Answer"].astype(str)
true_only["cnt_true"] = true_only.groupby(["QuestionId", "_MC_"])["_MC_"].transform("count")

true_ranked = (
    true_only.sort_values(["QuestionId", "cnt_true", "_MC_"], ascending=[True, False, True])
             .drop_duplicates(["QuestionId"])[["QuestionId", "_MC_", "cnt_true"]]
             .rename(columns={"_MC_": "_MC_correct"})
)

# Force 31778 → correct answer is "6" just to be safe
if (true_ranked["QuestionId"] == QID_NOISY).any():
    true_ranked.loc[true_ranked["QuestionId"] == QID_NOISY, "_MC_correct"] = "6"
else:
    true_ranked = pd.concat([
        true_ranked,
        pd.DataFrame({"QuestionId": [QID_NOISY], "_MC_correct": ["6"], "cnt_true": [np.nan]})
    ], ignore_index=True)

# 4) Re-align True/False prefix in train["Category"] to match the decided correct option
train["_MC_"] = train["MC_Answer"].astype(str)
train = train.merge(true_ranked[["QuestionId", "_MC_correct"]], on="QuestionId", how="left")

def _rewrite_tf_prefix_row(row):
    want_true = (row["_MC_"] == row["_MC_correct"])
    return _fix_tf_prefix(row["Category"], want_true)

train["Category"] = train.apply(_rewrite_tf_prefix_row, axis=1)

# 5) Add is_correct flag (used later in the prompt)
#    Build lookup while preserving dtype of MC_Answer
lookup = true_ranked[["QuestionId", "_MC_correct"]].copy()
if pd.api.types.is_integer_dtype(train["MC_Answer"]):
    lookup["MC_Answer"] = pd.to_numeric(lookup["_MC_correct"], errors="coerce").astype(train["MC_Answer"].dtype, errors="ignore")
else:
    lookup["MC_Answer"] = lookup["_MC_correct"].astype(str)
lookup = lookup.drop(columns=["_MC_correct"]).drop_duplicates()

train = train.merge(lookup, on=["QuestionId", "MC_Answer"], how="left", indicator="__m")
train["is_correct"] = (train["__m"] == "both").astype(int)
train = train.drop(columns=["__m", "_MC_"])

# 6) Rebuild target AFTER corrections
train["Misconception"] = train["Misconception"].fillna("NA")
train["target"] = train["Category"] + ":" + train["Misconception"]

# 7) LabelEncoder → n_classes
le = LabelEncoder()
train["label"] = le.fit_transform(train["target"])
target_classes = le.classes_
n_classes = len(target_classes)
print(f"Train shape: {train.shape} with {n_classes} target classes")

# 8) Build prompt text (same style as before)
def format_input(row):
    x = "This is Correct answer." if row['is_correct'] else "This is Incorrect answer."
    return (
        f"• Question: {row['QuestionText']}\n"
        f"• Answer: {row['MC_Answer']}\n"
        f"• Correctness: {x}\n"
        f"• Student Explanation: {row['StudentExplanation']}"
    )

train['text'] = train.apply(format_input, axis=1)
print("Example prompt for our LLM:\n")
print(train.text.values[0])

# (Optional) Keep CFG.NUM_LABELS in sync with the corrected number of classes
try:
    CFG.NUM_LABELS = n_classes
except NameError:
    pass

np.save("label_classes.npy", le.classes_)
"""


tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# save tokenizer to load offline during inference
tokenizer.save_pretrained('tokenizer')

# train, test
train_df, val_df = train_test_split(train, test_size=CFG.VAL_SPLIT, random_state=CFG.SEED)

# we train full this time, comment this if u want no leak val
train_df = train.copy()

# Tokenize both splits
train_tokens = tokenizer(train_df['text'].tolist(), padding='max_length', max_length=CFG.MAX_LENGTH, 
                        truncation=True, return_tensors='np')
val_tokens = tokenizer(val_df['text'].tolist(), padding='max_length', max_length=CFG.MAX_LENGTH, 
                        truncation=True, return_tensors='np')


print(f"TRAIN - INPUT_IDS: {train_tokens['input_ids'].shape}, LABELS: {train_df['label'].shape}")
print(f"VAL - INPUT_IDS: {val_tokens['input_ids'].shape}, LABELS: { val_df['label'].values}")
print(f"Number of unique classes: {len(np.unique(train_df['label'].values))}")


# Create datasets
TRAIN_DATASET = create_dataset(train_tokens['input_ids'], train_tokens['attention_mask'], 
                               train_df['label'].values , CFG.BATCH_SIZE, shuffle=True)
VAL_DATASET = create_dataset(val_tokens['input_ids'], val_tokens['attention_mask'], 
                             val_df['label'].values, CFG.EVAL_BATCH_SIZE, shuffle=False)


# Load base CausalLM
base_lm = transformers.AutoModelForCausalLM.from_pretrained(
    CFG.MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# Wrapper: add a simple classification head on top of CausalLM
class CausalLMSequenceClassifier(nn.Module):
    def __init__(self, base_lm, num_labels: int):
        super().__init__()
        self.base = base_lm
        self.config = base_lm.config          # Required: PEFT may reference this
        self.num_labels = num_labels          # Required: some utilities may reference this

        hidden_size = getattr(self.base.config, "hidden_size", None) \
                      or getattr(self.base.config, "hidden_dim", None)
        if hidden_size is None:
            raise ValueError("Could not determine hidden_size from config.")
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        # Follow standard behavior for return_dict (PEFT relies on it)
        return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", True)

        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,   # Always get hidden states for pooling
            use_cache=False,
            return_dict=True,            # Force True (will reshape later)
            **kwargs
        )

        last_hidden = outputs.hidden_states[-1]  # [B, T, H]

        if attention_mask is None:
            pooled = last_hidden.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # [B,T,1]
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        logits = self.classifier(pooled)

        if not return_dict:
            # Minimal tuple-compatible output for Hugging Face Transformers
            out = (logits,)
            if output_hidden_states:
                out = out + (outputs.hidden_states,)
            if output_attentions:
                out = out + (outputs.attentions,)
            return out

        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )

# Wrap into final model
base_model = CausalLMSequenceClassifier(base_lm, num_labels=CFG.NUM_LABELS)

# Set padding token
base_model.base.config.pad_token_id = tokenizer.pad_token_id



lora_config = LoraConfig(
    r=CFG.LORA_RANK,
    lora_alpha=CFG.LORA_ALPHA,
    lora_dropout=CFG.DROPOUT,
    bias='none',
    inference_mode=False,
    task_type=TaskType.SEQ_CLS,
    target_modules=CFG.LORA_MODULES
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()



# deepseek is distil of llama
llama_rule = (
    ("model\\.embed_tokens", ("mp", "fsdp")),
    ("self_attn\\.(q_proj|k_proj|v_proj)", ("fsdp", "mp")),
    ("self_attn\\.o_proj", ("mp", "fsdp")),
    ("mlp\\.gate_proj", ("fsdp", "mp")),
    ("mlp\\.down_proj", ("mp", "fsdp")),
    ("mlp\\.up_proj", ("fsdp", "mp")),
    ("lm_head", ("fsdp", "mp")),
    )

def find_rule(_model):
    return llama_rule

strkey2id = {
    "dp": 0,
    "fsdp": 1,
    "mp": 2
}

def partition_module(model, mesh, device=xm.xla_device(), verbose=False):
    partition_specs = find_rule(model)
    rule = [(k, tuple([strkey2id[x] for x in v])) for k, v in partition_specs]
        
    # print(rule)

    for name, module in model.named_modules():
        module.to(device)
        # print(name, module.__class__.__name__)
        if isinstance(module, (nn.Embedding, nn.Linear)):
            for rule_pattern, spec in rule:
                if re.findall(rule_pattern, name):
                    if verbose:
                        print("match", rule_pattern, name)
                    
                    xs.mark_sharding(module.weight, mesh, spec)
                    break
        
def partition_module_dp(model, mesh, device=xm.xla_device(), verbose=False):
    spec = (1, 2)

    for name, module in model.named_modules():
        module.to(device)
        if isinstance(module, (nn.Embedding, nn.Linear)):
            xs.mark_sharding(module.weight, mesh, spec)


# Number of TPU Nodes
num_devices = xr.global_runtime_device_count()
mesh_shape = (1, num_devices, 1)
device_ids = np.array(range(num_devices))
mesh = Mesh(device_ids, mesh_shape, ('dp', 'fsdp', 'mp'))
partition_module(model, mesh)


print(f'Num devices: {num_devices}')


def get_device_of(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")

print("Model device:", get_device_of(model))


# Verfy The Trainable Layers
MODEL_LAYERS_ROWS = []
TRAINABLE_PARAMS = []
N_TRAINABLE_PARAMS = 0

for name, param in model.named_parameters():
    # Layer Parameter Count
    n_parameters = int(torch.prod(torch.tensor(param.shape)))
    # Only Trainable Layers
    if param.requires_grad:
        # Add Layer Information
        MODEL_LAYERS_ROWS.append({
            'param': n_parameters,
            'name': name,
            'dtype': param.data.dtype,
        })
        # Append Trainable Parameter
        TRAINABLE_PARAMS.append({ 'params': param })
        # Add Number Of Trainable Parameters"
        N_TRAINABLE_PARAMS += n_parameters
        
display(pd.DataFrame(MODEL_LAYERS_ROWS))

print(f"""
===============================
N_TRAINABLE_PARAMS: {N_TRAINABLE_PARAMS:,}
N_TRAINABLE_LAYERS: {len(TRAINABLE_PARAMS)}
===============================
""")


def check_model_device(model):
    """Check which device each model parameter is on"""
    print("Model Parameter Devices:")
    print("-" * 50)
    
    for name, param in model.named_parameters():
        print(f"{name}: {param.device}")
        if 'xla' not in str(param.device):
            print(f"  WARNING: {name} is NOT on XLA device!")
    
    print("-" * 50)
    print(f"Current XLA device: {xm.xla_device()}")
    
    # Check if any parameters are on XLA
    xla_params = [p for p in model.parameters() if 'xla' in str(p.device)]
    cpu_params = [p for p in model.parameters() if p.device.type == 'cpu']
    
    print(f"Parameters on XLA: {len(xla_params)}")
    print(f"Parameters on CPU: {len(cpu_params)}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# check more if u wish 
# check_model_device(model)


input_ids, attention_mask, labels = next(TRAIN_DATASET)

print(f'input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}')
print(f'attention_mask shape: {attention_mask.shape}, dtype: {attention_mask.dtype}')
print(f'labels shape: {labels.shape}, dtype: {labels.dtype}')


%%time
# Dummy Prediction
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
print(f'logits: {outputs.logits[:1]}, dtype: {outputs.logits.dtype}')


# LR & Optimizer(try a different too)
N_SAMPLES = len(train_df)  # Use train_df after the split
STEPS_PER_EPOCH = N_SAMPLES // CFG.BATCH_SIZE
OPTIMIZER = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
NUM_WARMUP_STEPS = int(STEPS_PER_EPOCH * CFG.NUM_EPOCHS * CFG.WARMUP_RATIO)
VAL_STEPS_PER_EPOCH = len(val_df) // CFG.EVAL_BATCH_SIZE

print(f'BATCH_SIZE: {CFG.BATCH_SIZE}, N_SAMPLES: {N_SAMPLES}, STEPS_PER_EPOCH: {STEPS_PER_EPOCH}')
print(f'NUM_WARMUP_STEPS: {NUM_WARMUP_STEPS}, TOTAL_STEPS: {STEPS_PER_EPOCH * CFG.NUM_EPOCHS}')


# Try out more differnt schedules, cosine, cyclic, linear, etc...
# Cosine Learning Rate With Warmup
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=OPTIMIZER,
    num_warmup_steps=NUM_WARMUP_STEPS,
    num_training_steps=STEPS_PER_EPOCH * CFG.NUM_EPOCHS)

# Set the data type for the optimizer's state (e.g., momentum buffers)
for state in OPTIMIZER.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and state[k].dtype is not torch.float32:
            state[v] = v.to(dtype=torch.float32)


# Put Model In Train Mode
model.train()

# Loss Function, Cross Entropy
LOSS_FN = torch.nn.CrossEntropyLoss().to(dtype=torch.float32)


# Create output directory
os.makedirs(CFG.OUT_DIR, exist_ok=True)

st = time.time()
warnings.filterwarnings("error")

METRICS = {
    'train_loss': [],
    'val_loss': [],
    'train_map3': [],
    'val_map3': []
}

# if true, shows progress bar, in saved notebook it looks cryptic, so set to false
INTERACTIVE_MODE = False  

global_step = 0
best_val_map3 = -1.0
bad_epochs = 0
best_ckpt = None

# Training loop
for epoch in range(CFG.NUM_EPOCHS):
    epoch_start = time.time()
    model.train()
    
    print(f"Starting Epoch {epoch+1}/{CFG.NUM_EPOCHS} - {STEPS_PER_EPOCH} steps")
    
    # Progress bar only in interactive mode
    if INTERACTIVE_MODE:
        epoch_pbar = tqdm(range(STEPS_PER_EPOCH), 
                         desc=f"Epoch {epoch+1}/{CFG.NUM_EPOCHS}", 
                         leave=True,
                         mininterval=1.0)
    
    epoch_losses = []
    epoch_y_true = []
    epoch_y_probs = []
    
    for step in range(STEPS_PER_EPOCH):
        global_step += 1
        
        # Gradient accumulation loop
        accumulated_loss = 0
        OPTIMIZER.zero_grad()
        
        for accum_step in range(CFG.GRADACCUM):
            input_ids, attention_mask, labels = next(TRAIN_DATASET)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.to(dtype=torch.float32)
            
            loss = LOSS_FN(logits, labels) / CFG.GRADACCUM
            loss.backward()
            accumulated_loss += float(loss) * CFG.GRADACCUM
            
            if accum_step == CFG.GRADACCUM - 1:
                probs = F.softmax(logits, dim=-1)
                epoch_y_true.extend(labels.cpu().tolist())
                epoch_y_probs.extend(probs.detach().cpu().numpy())
        
        # Move gradient clipping here
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.MAX_GRAD_NORM)
        OPTIMIZER.step()
        xm.mark_step()
        lr_scheduler.step()
        
        epoch_losses.append(accumulated_loss)
        METRICS['train_loss'].append(accumulated_loss)
        
        # Update progress bar in interactive mode
        if INTERACTIVE_MODE:
            current_lr = OPTIMIZER.param_groups[0]['lr']
            
            # Calculate recent MAP@3 for progress bar
            if len(epoch_y_true) >= 100:
                recent_true = epoch_y_true[-100:]
                recent_probs = epoch_y_probs[-100:]
                recent_map3 = compute_map3(np.array(recent_true), np.array(recent_probs))
            else:
                recent_map3 = 0.0
                
            epoch_pbar.set_postfix({
                'loss': f'{accumulated_loss:.4f}',
                'avg_loss': f'{np.mean(epoch_losses[-100:]):.4f}',
                'map3': f'{recent_map3:.4f}',
                'lr': f'{current_lr:.2e}'
            })
            epoch_pbar.update(1)
        
        # Step-based evaluation
        should_eval_step = (CFG.EVAL_STRATEGY == "steps" and 
                           global_step % CFG.EVAL_STEPS == 0)
        
        # Step-based saving
        should_save_step = (CFG.SAVE_STRATEGY == "steps" and 
                           global_step % CFG.SAVE_STEPS == 0)
        
        if should_eval_step:
            if INTERACTIVE_MODE:
                epoch_pbar.write(f"Evaluating at step {global_step}...")
            else:
                print(f"Evaluating at step {global_step}...")

            patience = 2

            val_loss, val_map3 = evaluate_model(model, VAL_DATASET, VAL_STEPS_PER_EPOCH)
            if val_map3 > best_val_map3 + 1e-4:
                best_val_map3 = val_map3
                best_ckpt = save_checkpoint(epoch+1, global_step, val_map3)
                bad_epochs = 0
            else:
                bad_epochs += 1
                if CFG.EVAL_STRATEGY == "epoch" and bad_epochs >= patience:
                    print(f"Early stop at epoch {epoch+1}. Best MAP@3 = {best_val_map3:.4f}")
                    break
            METRICS['val_loss'].append(val_loss)
            METRICS['val_map3'].append(val_map3)
            if INTERACTIVE_MODE:
                epoch_pbar.write(f"Val Loss: {val_loss:.4f} | Val MAP@3: {val_map3:.4f}")
            else:
                print(f"Val Loss: {val_loss:.4f} | Val MAP@3: {val_map3:.4f}")
        
        if should_save_step:
            checkpoint_path = save_checkpoint(epoch+1, global_step)
            if INTERACTIVE_MODE:
                epoch_pbar.write(f"Checkpoint saved: {checkpoint_path}")
            else:
                print(f"Checkpoint saved: {checkpoint_path}")
        
        # Milestone logging
        milestone_steps = [STEPS_PER_EPOCH // 10 * i for i in range(1, 11)]  # 10%, 20%, etc.
        if (step + 1) in milestone_steps or (step + 1) % 500 == 0:
            # Calculate ETA
            elapsed = time.time() - epoch_start
            progress = (step + 1) / STEPS_PER_EPOCH
            eta_seconds = (elapsed / progress) * (1 - progress) if progress > 0 else 0
            eta_minutes = eta_seconds / 60
            
            # Calculate metrics
            avg_loss = np.mean(epoch_losses[-min(500, len(epoch_losses)):])
            if len(epoch_y_true) >= 100:
                recent_true = epoch_y_true[-min(500*CFG.BATCH_SIZE, len(epoch_y_true)):]
                recent_probs = epoch_y_probs[-min(500*CFG.BATCH_SIZE, len(epoch_y_probs)):]
                train_map3 = compute_map3(np.array(recent_true), np.array(recent_probs))
            else:
                train_map3 = 0.0
            
            progress_pct = progress * 100
            current_lr = OPTIMIZER.param_groups[0]['lr']
            
            log_msg = (f"Step {step+1}/{STEPS_PER_EPOCH} ({progress_pct:.1f}%) | "
                      f"Loss: {accumulated_loss:.4f} | Avg: {avg_loss:.4f} | "
                      f"MAP@3: {train_map3:.4f} | LR: {current_lr:.2e} | "
                      f"ETA: {eta_minutes:.1f}min")
            
            if INTERACTIVE_MODE:
                epoch_pbar.write(log_msg)
            else:
                print(log_msg)
    
    if INTERACTIVE_MODE:
        epoch_pbar.close()
    
    # End of epoch processing
    epoch_train_loss = np.mean(epoch_losses)
    epoch_train_map3 = compute_map3(np.array(epoch_y_true), np.array(epoch_y_probs))
    METRICS['train_map3'].append(epoch_train_map3)
    
    val_loss, val_map3 = None, None
    
    # Epoch-based evaluation
    if CFG.EVAL_STRATEGY == "epoch":
        print(f"Evaluating at end of epoch {epoch+1}...")
        val_loss, val_map3 = evaluate_model(model, VAL_DATASET, VAL_STEPS_PER_EPOCH)
        METRICS['val_loss'].append(val_loss)
        METRICS['val_map3'].append(val_map3)
    
    epoch_time = time.time() - epoch_start
    
    # Clean epoch summary
    print(f"\n{'='*50}")
    print(f"EPOCH {epoch+1} COMPLETE")
    print(f"Train Loss: {epoch_train_loss:.4f} | Train MAP@3: {epoch_train_map3:.4f}")
    if val_loss is not None and val_map3 is not None:
        print(f"Val Loss: {val_loss:.4f} | Val MAP@3: {val_map3:.4f}")
    print(f"Time: {epoch_time:.0f}s | Steps: {global_step}")
    print(f"{'='*50}")
    
    # Epoch-based saving
    if CFG.SAVE_STRATEGY == "epoch":
        checkpoint_path = save_checkpoint(epoch+1, global_step, val_map3)
        print(f"Saved: {checkpoint_path}")

print(f"\nTraining completed in {time.time() - st:.0f}s")


# skip if not necessary
model = model.cpu()
torch.save(dict([(k,v) for k, v in model.named_parameters() if v.requires_grad]), 'deepseekrllama_v1_epoch_2.pth')


plot_metrics(METRICS)

