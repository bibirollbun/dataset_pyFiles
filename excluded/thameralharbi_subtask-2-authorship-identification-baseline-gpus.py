import pandas as pd


train_df = pd.read_csv("/kaggle/input/subtask-2-authorshipclassficiation/AuthorshipClassficiationTrain.csv")  # update filename
val_df = pd.read_csv("/kaggle/input/subtask-2-authorshipclassficiation/AuthorshipClassficiationVal.csv")
test_df = pd.read_csv("/kaggle/input/subtask-2-authorshipclassficiation/PublicDataFinalPhaseTask2.csv")


train_df.info()
val_df.info()
test_df.info()


train_df.head()


val_df.head()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda_summary(df, name='DataFrame'):
    print(f'ğŸ“‹ Summary of {name}')
    print(df.info())
    print('\nğŸ“Œ Basic stats:')
    print(df.describe(include='object'))

    print('\nğŸ”� Null values:')
    print(df.isnull().sum())

    print('\nğŸ‘¥ Number of unique authors:', df['author'].nunique())
    print('ğŸ§¾ Sample authors:', df['author'].unique()[:10])

    # Distribution of samples per author
    author_counts = df['author'].value_counts()
    print('\nğŸ“Š Top 10 authors by number of samples:')
    print(author_counts.head(10))

    plt.figure(figsize=(12, 4))
    author_counts.plot(kind="bar")
    # sns.histplot(author_counts, bins=len(author_counts), kde=True)
    plt.title(f'{name}: Samples per Author')
    plt.xlabel('Number of Samples')
    plt.ylabel('Author Count')
    plt.show()

    # Text length analysis
    df['text_length'] = df['text_in_author_style'].apply(len)
    df['word_count'] = df['text_in_author_style'].apply(lambda x: len(str(x).split()))

    print('\nğŸ“� Text Length Statistics:')
    print(df[['text_length', 'word_count']].describe())

    # Histograms
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.histplot(df['text_length'], bins=50, ax=axes[0])
    axes[0].set_title(f'{name}: Text Length (chars)')
    sns.histplot(df['word_count'], bins=50, ax=axes[1])
    axes[1].set_title(f'{name}: Word Count')
    plt.tight_layout()
    plt.show()

    return df

train_df = eda_summary(train_df, 'Train')
val_df = eda_summary(val_df, 'Validation')


!pip install bitsandbytes -q


%%writefile train.py

import os
import platform
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.amp import GradScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder


from transformers import AutoModel, AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig

from peft import (
    get_peft_config, 
    get_peft_model, 
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Constants
model_path = 'ALLaM-AI/ALLaM-7B-Instruct-preview'
num_folds = 3
num_epochs = 3
batch_size = 6

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token

# 1. Read and combine the data
train_df = pd.read_csv("/kaggle/input/subtask-2-authorshipclassficiation/AuthorshipClassficiationTrain.csv")
val_df   = pd.read_csv("/kaggle/input/subtask-2-authorshipclassficiation/AuthorshipClassficiationVal.csv")
df = pd.concat([train_df, val_df], ignore_index=True)

# 2. Standardize column names
df.columns = ['id', 'text_in_author_style', 'author']
num_classes = len(df["author"].unique())

# # Example
# print("Prompt example:\n", prompts[0])
# print("Target example:", targets[0])
# print("Mapping author_idâ†’name:", dict(enumerate(le.classes_)))


class TextDataset(Dataset):
    def __init__(self, prompts, targets):
        self.prompts = prompts
        self.targets = targets

    def __getitem__(self, idx):
        return self.prompts[idx], self.targets[idx]

    def __len__(self):
        return len(self.targets)

class Net(nn.Module):
    def __init__(self, model_path, rank):
        super(Net, self).__init__()
        self.config = AutoConfig.from_pretrained(model_path)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        self.backbone = AutoModel.from_pretrained(
            model_path,
            use_cache=False,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            device_map=rank
        )

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules='all-linear',
            bias='none',
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05
        )

        # self.backbone.gradient_checkpointing_enable()

        # self.backbone = prepare_model_for_kbit_training(self.backbone, use_gradient_checkpointing = True)

        self.backbone = get_peft_model(self.backbone, peft_config)

        
        self.head = nn.Linear(self.config.hidden_size, num_classes, bias=False)

    def forward(self, x):
        x = self.backbone(**x).last_hidden_state[:, -1, :]
        return self.head(x)

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if platform.system() == 'Windows':
        os.environ['USE_LIBUV'] = '0'
        init_process_group(backend='gloo', rank=rank, world_size=world_size)
    else:
        init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def get_optimizer(model, learning_rate=0.0001, diff_lr=0.00001, weight_decay=0.01):

	no_decay = ['bias', 'LayerNorm.weight']
	differential_layers = ['backbone']

	optimizer = torch.optim.AdamW(
			[
				{
					"params": [
						param
						for name, param in model.named_parameters()
						if (not any(layer in name for layer in differential_layers))
						and (not any(nd in name for nd in no_decay))
					],
					"lr": learning_rate,
					"weight_decay": weight_decay,
				},
				{
					"params": [
						param
						for name, param in model.named_parameters()
						if (not any(layer in name for layer in differential_layers))
						and (any(nd in name for nd in no_decay))
					],
					"lr": learning_rate,
					"weight_decay": 0,
				},
				{
					"params": [
						param
						for name, param in model.named_parameters()
						if (any(layer in name for layer in differential_layers))
						and (not any(nd in name for nd in no_decay))
					],
					"lr": diff_lr,
					"weight_decay": weight_decay,
				},
				{
					"params": [
						param
						for name, param in model.named_parameters()
						if (any(layer in name for layer in differential_layers))
						and (any(nd in name for nd in no_decay))
					],
					"lr": diff_lr,
					"weight_decay": 0,
				},
			],
			lr=learning_rate,
			weight_decay=weight_decay,
	)

	return optimizer

def train_model(rank, world_size, num_epochs, fold, train_index, val_index, all_prompts, all_targets):
    ddp_setup(rank, world_size)

    train_prompts = [all_prompts[i] for i in train_index]
    val_prompts = [all_prompts[i] for i in val_index]
    train_targets = [all_targets[i] for i in train_index]
    val_targets = [all_targets[i] for i in val_index]

    class_weights = 1 / (np.unique(train_targets, return_counts=True)[1] / len(train_targets))
    class_weights = torch.tensor(class_weights, dtype=torch.half)

    train_dataset = TextDataset(train_prompts, train_targets)
    val_dataset = TextDataset(val_prompts, val_targets)

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, shuffle=False, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model = Net(model_path, rank).to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = get_optimizer(model, learning_rate=2e-4, diff_lr=2e-4, weight_decay=0.01)
        
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=0, 
                                                num_training_steps=len(train_loader) * num_epochs)
    scaler = GradScaler()

    best_f1 = 0.0  # Track best F1
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()

        for batch_prompts, batch_targets in tqdm(train_loader):
            max_len = max(len(x) for x in tokenizer(batch_prompts).input_ids)

            if max_len > 300:
                encodings = tokenizer(batch_prompts,
                  return_tensors='pt', 
                  padding='max_length', 
                  truncation=True,
                  max_length=300).to(rank)
            else:
                encodings = tokenizer(batch_prompts,
                  return_tensors='pt', 
                  padding='longest').to(rank)            
            
            batch_targets = batch_targets.long().to(rank)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(encodings)
                loss = F.cross_entropy(logits, batch_targets, weight=class_weights.to(rank))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_prompts, batch_targets in tqdm(val_loader, total=len(val_loader)):
                max_len = max(len(x) for x in tokenizer(batch_prompts).input_ids)

                if max_len > 300:
                    encodings = tokenizer(batch_prompts,
                      return_tensors='pt', 
                      padding='max_length', 
                      truncation=True,
                      max_length=300).to(rank)
                else:
                    encodings = tokenizer(batch_prompts,
                      return_tensors='pt', 
                      padding='longest').to(rank)
                    
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    
                    logits = model(encodings)
                    preds = torch.argmax(logits, dim=1).cpu().tolist()
    
                    all_preds.extend(preds)
                    all_labels.extend(batch_targets)

        f1 = f1_score(all_labels, all_preds, average='micro')
        print(f'[GPU {rank}] Fold {fold+1} | Epoch {epoch+1}/{num_epochs} | Val F1-micro: {f1:.4f}')
    
        if rank == 0 and f1 > best_f1:
            best_f1 = f1
            model.eval()
            model.module.backbone.save_pretrained(f'backbone_fold_{fold}_best')
            torch.save(model.module.head.state_dict(), f'head_fold_{fold}_best.pt')
            
    destroy_process_group()

def run_ddp(rank, world_size, num_epochs, splits, fold, all_prompts, all_targets):
    train_index, val_index = splits[fold]
    train_model(rank, world_size, num_epochs, fold, train_index, val_index, all_prompts, all_targets)

if __name__ == '__main__':
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs available:", torch.cuda.device_count())
    
    torch.manual_seed(1)
    
    # Encoding author names into numeric labels.
    le = LabelEncoder()
    df['author_id'] = le.fit_transform(df['author'])
    
    # Constructing the prompts.
    prompts = []
    for text in df['text_in_author_style']:
        prompts.append(
            f"""<|im_start|>user
    ### Instruction:
    ØµÙ†Ù� Ø§Ù„Ù†Øµ Ø§Ù„ØªØ§Ù„ÙŠ Ø­Ø³Ø¨ Ù…Ø¤Ù„Ù�Ù‡ Ù…Ù† Ø§Ù„Ù‚Ø§Ø¦Ù…Ø© Ø§Ù„Ù…Ø±Ù�Ù‚Ø© Ø£Ø¯Ù†Ø§Ù‡. Ø£Ø¬Ø¨ Ø¨Ø±Ù‚Ù… Ø§Ù„Ù…Ø¤Ù„Ù� Ù�Ù‚Ø·.
    
    Ù‚Ø§Ø¦Ù…Ø© Ø§Ù„Ù…Ø¤Ù„Ù�ÙŠÙ† (Ù…Ø¹ Ø§Ù„Ø±Ù‚Ù… Ø§Ù„Ù…Ù‚Ø§Ø¨Ù„):
    {chr(10).join(f"{i}: {name}" for i, name in enumerate(le.classes_))}
    
    ### Input:
    {text}
    
    ### Response:"""
        )
    
    # List of targets (numeric labels).
    targets = df['author_id'].tolist()

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    splits = list(skf.split(prompts, targets))

    world_size = torch.cuda.device_count()

    for fold in range(num_folds):
        mp.spawn(run_ddp, args=(world_size, num_epochs, splits, fold, prompts, targets), nprocs=world_size)


!python train.py


# !pip install /kaggle/input/bitsandbytes-pip-download/bitsandbytes-0.43.1-py3-none-manylinux_2_24_x86_64.whl -q
# !pip install /kaggle/input/accelerate-pip-download/accelerate-0.30.1-py3-none-any.whl -q
# !pip install --no-index --find-links=/kaggle/input/install-peft peft -q


!pip install bitsandbytes -q


import os
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModel, AutoModelForCausalLM

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from peft import (
    get_peft_config, 
    get_peft_model, 
    LoraConfig,
    TaskType,
     prepare_model_for_kbit_training
)

from transformers import BitsAndBytesConfig
from transformers import get_cosine_schedule_with_warmup

from sentence_transformers import SentenceTransformer
from peft import AutoPeftModelForFeatureExtraction

from sklearn.preprocessing import LabelEncoder

import ctypes
import os

from threading import Thread
from tqdm.notebook import tqdm

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def clean_memory(deep=True):
    gc.collect()
    if deep:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()


test = pd.read_csv('/kaggle/input/subtask-2-authorshipclassficiation/PublicDataFinalPhaseTask2.csv')


# Reading and merging the data
train_df = pd.read_csv("/kaggle/input/subtask-2-authorshipclassficiation/AuthorshipClassficiationTrain.csv")
val_df   = pd.read_csv("/kaggle/input/subtask-2-authorshipclassficiation/AuthorshipClassficiationVal.csv")
df = pd.concat([train_df, val_df], ignore_index=True)

# Standardizing column names
df.columns = ['id', 'text_in_author_style', 'author']
num_classes = len(df["author"].unique())

# Encoding author names into numerical values
le = LabelEncoder()
le.fit(df['author'])
test.columns = ['id', 'text_in_author_style']

# Constructing the prompts
prompts = []
for text in test['text_in_author_style']:
    prompts.append(
        f"""<|im_start|>user
### Instruction:
ØµÙ†Ù� Ø§Ù„Ù†Øµ Ø§Ù„ØªØ§Ù„ÙŠ Ø­Ø³Ø¨ Ù…Ø¤Ù„Ù�Ù‡ Ù…Ù† Ø§Ù„Ù‚Ø§Ø¦Ù…Ø© Ø§Ù„Ù…Ø±Ù�Ù‚Ø© Ø£Ø¯Ù†Ø§Ù‡. Ø£Ø¬Ø¨ Ø¨Ø±Ù‚Ù… Ø§Ù„Ù…Ø¤Ù„Ù� Ù�Ù‚Ø·.

Ù‚Ø§Ø¦Ù…Ø© Ø§Ù„Ù…Ø¤Ù„Ù�ÙŠÙ† (Ù…Ø¹ Ø§Ù„Ø±Ù‚Ù… Ø§Ù„Ù…Ù‚Ø§Ø¨Ù„):
{chr(10).join(f"{i}: {name}" for i, name in enumerate(le.classes_))}

### Input:
{text}

### Response:"""
    )

# List of targets (numerical labels)
test['text_in_author_style'] = prompts
test['author_id'] = -100
test


model_path = 'ALLaM-AI/ALLaM-7B-Instruct-preview'


tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token


test['text_in_author_style'].apply(lambda x:len( tokenizer(x).input_ids)).hist()


class TextDataset(Dataset):
    def __init__(self, prompts, targets):
        self.prompts = prompts
        self.targets = targets

    def __getitem__(self, idx):
        return self.prompts[idx], self.targets[idx]

    def __len__(self):
        return len(self.targets)
        
# Split the test set into 4 parts
length = len(test)
split_size = length // 4

# Create dataloaders for each GPU
test_datasets = []
test_dataloaders = []

for i in range(4):
    start_idx = i * split_size
    end_idx = (i + 1) * split_size if i < 3 else length  # last chunk takes the remainder

    dataset = TextDataset(
        prompts=test['text_in_author_style'].iloc[start_idx:end_idx].to_numpy(),
        targets=test['author_id'].iloc[start_idx:end_idx].to_numpy()
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False
    )
    
    test_dataloaders.append(dataloader)


def get_preds(model, tokenizer, test_dataloader, device, results):
    y_pred = torch.tensor([])
    y_true = torch.tensor([])

    with torch.no_grad():
        model.eval()
        for batch_idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

            batch_prompts, batch_targets = batch
            
            encodings = tokenizer(batch_prompts, return_tensors='pt').to(device)            
        
            batch_targets = batch_targets.long().to(device)

            with autocast(device_type=device):
                logits = model(encodings)

                y_pred = torch.cat([y_pred, logits.cpu()])
        
    
    results[device] = y_pred


class Net(nn.Module):
    def __init__(self, base_model_path, trained_backbone_path, load_in_device):
        super(Net, self).__init__()
        self.config = AutoConfig.from_pretrained(base_model_path)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        self.backbone = AutoPeftModelForFeatureExtraction.from_pretrained(
            trained_backbone_path,
            use_cache=False,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            device_map=load_in_device
        )

        self.head = nn.Linear(self.config.hidden_size, num_classes, bias=False)

    def forward(self, x):
        x = self.backbone(**x).last_hidden_state[:, -1, :]
        return self.head(x)


!mkdir best-fold-0-backbone
!cp /kaggle/input/allam-adapter_aid/pytorch/default/1/adapter_config.json /kaggle/input/allam-adapter_aid/pytorch/default/1/adapter_model.safetensors best-fold-0-backbone


# ALLaM-7B-Instruct-preview-fold-1
trained_backbone_path = 'best-fold-0-backbone'
trained_head_path = '/kaggle/input/allam-adapter_aid/pytorch/default/1/head_fold_0_best.pt'

model_1 = Net(base_model_path=model_path, 
              trained_backbone_path=trained_backbone_path,
              load_in_device='cuda:0')

model_2 = Net(base_model_path=model_path, 
              trained_backbone_path=trained_backbone_path,
              load_in_device='cuda:1')

model_3 = Net(base_model_path=model_path, 
              trained_backbone_path=trained_backbone_path,
              load_in_device='cuda:2')

model_4 = Net(base_model_path=model_path, 
              trained_backbone_path=trained_backbone_path,
              load_in_device='cuda:3')

model_1.head.load_state_dict(torch.load(trained_head_path, weights_only=True))
model_2.head.load_state_dict(torch.load(trained_head_path, weights_only=True))
model_3.head.load_state_dict(torch.load(trained_head_path, weights_only=True))
model_4.head.load_state_dict(torch.load(trained_head_path, weights_only=True))

model_1.head.to('cuda:0')
model_2.head.to('cuda:1')
model_3.head.to('cuda:2')
model_4.head.to('cuda:3')

results = {}

t0 = Thread(target=get_preds, args=(model_1, tokenizer, test_dataloaders[0], 'cuda:0', results))
t1 = Thread(target=get_preds, args=(model_2, tokenizer, test_dataloaders[1], 'cuda:1', results))
t2 = Thread(target=get_preds, args=(model_3, tokenizer, test_dataloaders[2], 'cuda:2', results))
t3 = Thread(target=get_preds, args=(model_4, tokenizer, test_dataloaders[3], 'cuda:3', results))

t0.start()
t1.start()
t2.start()
t3.start()

t0.join()
t1.join()
t2.join()
t3.join()

logits_fold_1 = torch.cat([results['cuda:0'], results['cuda:1'], results['cuda:2'], results['cuda:3']])
pred_probs_fold_1 = F.softmax(logits_fold_1, dim=-1)

del model_1, model_2, model_3, model_4
clean_memory()


pred_probs_fold_1.size()


pred_probs_fold_1.argmax(dim=-1).tolist()[:10]


pred_probs_fold_1


le.classes_


sub = test[['id']].copy()
sub['label'] = le.inverse_transform(pred_probs_fold_1.argmax(dim=-1).tolist())



sub['label']


sub.to_csv('predictions.csv', index=False)


!zip predictions.zip predictions.csv


# sub = test[['id']].copy()
# sub['label'] = ensembled_pred_probs.argmax(dim=-1).tolist()
# sub.to_csv('submission.csv', index=False)

