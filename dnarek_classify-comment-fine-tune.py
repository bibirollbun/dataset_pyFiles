import pandas as pd
from datasets import Dataset
import random
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from torch import nn
from datasets import Dataset
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType
import kagglehub
from torch.amp import GradScaler
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


def build_prompt(row):
    prompt = f'''<start_of_turn>user
                You are given a Reddit comment and a specific rule. Your task is to decide if the comment violates the rule. Respond only with "Yes" or "No".
                Rule: {row["rule"].strip()}
                Examples:
                1) {row["positive_example_1"].strip()}
                Answer: Yes
                2) {row["negative_example_1"].strip()}
                Answer: No
                Now, here is the comment to classify:
                "{row["body"].strip()}"
                Answer "Yes" if it violates the rule, otherwise "No".<end_of_turn>
                <start_of_turn>model
                Answer:'''
    return prompt


train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')

cols_to_integrate = ['negative_example_2','positive_example_2']
for violation_idx, example in enumerate(cols_to_integrate):
    train_df_temp = train_df.copy()
    train_df_temp['body'] = train_df_temp[example]
    train_df_temp['rule_violation'] = violation_idx #0 for negative and 1 for positive
    train_df = pd.concat([train_df, train_df_temp], axis=0)

train_df = train_df.drop(cols_to_integrate, axis=1)
train_df = train_df.drop_duplicates(ignore_index=True)
train_df = train_df.sample(frac=1)
train_df = train_df.reset_index(drop=True)

train_df["prompt"] = train_df.apply(build_prompt, axis=1)
train_df['label'] = train_df['rule_violation'].map({1:'yes',0: 'no',})
train_df = train_df[['prompt','label']]

train_dataset = Dataset.from_pandas(train_df)





model_path = "/kaggle/input/gemma/transformers/2b-it/3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = 'left'


class TrainDataset(Dataset):
    def __init__(self, prompt, label):
        self.prompt = prompt
        self.label = label

    def __getitem__(self, idx):
        return self.prompt[idx], self.label[idx]

    def __len__(self):
        return len(self.label)





class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.config = AutoConfig.from_pretrained(model_path)

        self.backbone = AutoModel.from_pretrained(
            model_path,
            use_cache=False,
            torch_dtype=torch.float16,
        )

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules='all-linear',
            bias='none',
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.
        )
        self.backbone = get_peft_model(self.backbone, peft_config)

        self.head = nn.Linear(self.config.hidden_size, 2, bias=False)

    def forward(self, encodings):
        out = self.backbone(**encodings).last_hidden_state  
        x = out[:, -1, :]  
        return self.head(x) 


batch_size = 4
lr = 1e-5
epochs = 4


d_generator = torch.Generator(device=device)
train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        generator=d_generator
    )
def target_idx(value):
    if value == 'yes':
        return 1
    elif value == 'no':
        return 0
    raise InvalidInputError("should be either yes or no")


max_len = int(np.quantile([len(tokenizer(x).input_ids) for x in train_df.prompt.to_list()], q=0.99))


model = Net()
scaler = GradScaler()
opt = torch.optim.AdamW(model.parameters(),lr=lr)


for _ in range(epochs):
    model.train()

    for batch in train_loader:
        batch_prompts = batch['prompt']
        batch_targets = batch['label']
        enc = tokenizer(
            batch_prompts,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=max_len
        )

        labels = torch.tensor(list(map(target_idx,batch_targets)), dtype=torch.long)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(enc)
            ce = F.cross_entropy(logits, labels)
  

            loss = ce 
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            scaler.step(opt)
            opt.zero_grad(set_to_none=True)
            scaler.update()

            print(ce.item())#will properly log later










