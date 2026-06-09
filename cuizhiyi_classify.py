import torch


print(torch.cuda.memory_summary())
# 或者
print(torch.cuda.memory_allocated())  # 已分配显存
print(torch.cuda.memory_reserved())   # 已缓存显存


torch.cuda.empty_cache()

print(torch.cuda.memory_summary())
# 或者
print(torch.cuda.memory_allocated())  # 已分配显存
print(torch.cuda.memory_reserved())   # 已缓存显存


!pip install --upgrade transformers


from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-1.7B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
print(model)


from peft import LoraConfig, get_peft_model
lora_config=LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","v_proj","o_proj"]
)

model=get_peft_model(model,lora_config)
model.print_trainable_parameters()


import pandas as pd
data=pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
data.info()


data['rule_violation'].value_counts()


# from sklearn.model_selection import train_test_split

# df_train, df_test = train_test_split(
#     df,
#     test_size=0.2,     # 20% 测试集
#     random_state=42,   # 固定随机种子，保证可复现
#     shuffle=True       # 是否打乱
# )

# print(len(df_train), len(df_test))



new_data=[]
for _,row in data.iterrows():
    message = [
            {'role':'system','content': 'You are a content reviewer. After careful consideration, please finally respond with "Yes" or "No".'},
            {'role':'user','content':f"""
            Rule:{row['rule']},
            Violation1:{row['positive_example_1']},
            Violation2:{row['positive_example_2']},
            Non-violation1:{row['negative_example_1']},
            Non-violation2:{row['negative_example_2']},
            Comment:{row['body']},
            Violates
            """},
            {'role':'assistant','content':'Yes' if row['rule_violation']==1 else 'No'}
        ]
    full_texts = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        enable_thinking=True
)
    full_tokens = tokenizer([full_texts],truncation=True, max_length=512, padding='max_length')

    assistant_response = 'Yes' if row['rule_violation'] == 1 else 'No'
    response_tokens = tokenizer.encode(assistant_response, add_special_tokens=False)
    response_len = len(response_tokens)
    labels = full_tokens['input_ids'][0].copy()

    for i in range(len(labels)-response_len,-1,-1):
        if labels[i:i+response_len] == response_tokens:
           
            labels[i+response_len:] = [-100]*(len(labels)-i-response_len)
            labels[:i] = [-100]*(i)
            break
    else:
        labels = [-100]*len(labels)
        
    
    new_row={
        'input_ids': full_tokens['input_ids'][0],
        'attention_mask': full_tokens['attention_mask'][0],
        'labels': labels
    }    
    new_data.append(new_row)



from datasets import Dataset

dataset = Dataset.from_list(new_data)
split = dataset.train_test_split(
    test_size=0.2,
    seed=42
)

train_dataset = split["train"]
test_dataset  = split["test"]
print(len(train_dataset),train_dataset[0])


yes_id = tokenizer.encode("Yes", add_special_tokens=False)
no_id  = tokenizer.encode("No",  add_special_tokens=False)
print(yes_id,no_id)


print(torch.cuda.memory_summary())
# 或者
print(torch.cuda.memory_allocated())  # 已分配显存
print(torch.cuda.memory_reserved())   # 已缓存显存



import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

def collate_fn(batch):
    return {
        'input_ids': torch.tensor([item['input_ids'] for item in batch], dtype=torch.long),
        'attention_mask': torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long),
        'labels': torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    }
batch_size=2
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,collate_fn=collate_fn)


optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.01)
epochs=3
GRADIENT_ACCUMULATION=4
WARMUP_STEPS=75
total_steps = len(train_loader) * epochs // (GRADIENT_ACCUMULATION*batch_size)
scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)




for step, batch in enumerate(train_loader):
    print(batch)
    break


device = next(model.parameters()).device  # 获取模型所在设备
device


import torch.nn as nn
from tqdm import tqdm

criterion = nn.CrossEntropyLoss()
for i in range(epochs):
    model.train()
    optimizer.zero_grad()
    train_correct_num = 0
    test_correct_num = 0
    train_total_loss=0
    test_total_loss=0
    train_pbar=tqdm(train_loader,desc="Training...")
    for step,batch in enumerate(train_pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        valid_indices = (labels != -100).long().argmax(dim=1) - 1 # 对齐
        token_values = labels[torch.arange(labels.size(0)), valid_indices]
        binary_labels = (token_values == yes_id[0]).long() # 真实label
        
        output_ids = model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
        logits = output_ids.logits
        
        B = logits.size(0)
        batch_idx = torch.arange(B, device=logits.device)
        
        pre_yes = logits[batch_idx, valid_indices, yes_id[0]]  # [B]
        pre_no  = logits[batch_idx, valid_indices, no_id[0]]   # [B]
        pred_labels = (pre_yes > pre_no).long()
        
        correct_num = (pred_labels == binary_labels).sum().item()
        train_correct_num += correct_num
        batch_train_accuracy = correct_num / batch_size
        
        train_total_loss+=output_ids.loss*batch_size
        loss = output_ids.loss/GRADIENT_ACCUMULATION
        loss.backward()
        
        if (step+1)%GRADIENT_ACCUMULATION ==0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_avg_accuracy = train_correct_num/((step+1)*batch_size)
            train_avg_loss = train_total_loss/((step+1)*batch_size)
            
            train_pbar.set_postfix({'batch_train_loss': f"{loss:.4f}","train_avg_loss":f"{train_avg_loss:.4f}",
                                   'batch_train_accuracy': f"{batch_train_accuracy:.4f}","train_avg_accuracy":f"{train_avg_accuracy:.4f}"})
    avg_accuracy = train_correct_num/len(train_dataset)
    avg_loss = train_total_loss/len(train_dataset)
    print(f"train epoch:{i+1}\t\t avg_loss:{avg_loss}\t\t avg_accuracy:{avg_accuracy}")

    # test_pbar=tqdm(test_loader,desc="Testing...")
    # model.eval()
    # for step, batch in enumerate(test_pbar):
    #     input_ids = batch["input_ids"].to(device)
    #     attention_mask = batch["attention_mask"].to(device)
    #     output_ids = model(input_ids=input_ids,attention_mask=attention_mask)
    #     logits = output_ids.logits
    #     last_logits = logits[:, -1, :]        # [B, vocab_size]
    
    #     device = last_logits.device
    #     target_token_ids = torch.tensor([yes_id, no_id], device=device)
    #     selected_logits = last_logits[:, target_token_ids].squeeze(-1)  # [B,2]
        
    #     gold_tokens = input_ids[:, -1]             # [B]
    #     labels = (gold_tokens == yes_id[0]).long()    
        
    #     loss = criterion(selected_logits, labels)
        
    #     test_total_loss +=loss.item()*batch_size
        
    
    #     preds = torch.argmax(selected_logits, dim=-1)
    #     test_correct_num += (preds == labels).sum().item()

    #     temp_accuracy = test_correct_num/(batch_size*(step+1))
    #     temp_avg_loss = test_total_loss/(batch_size*(step+1))
    #     test_pbar.set_postfix({'test_avg_loss': f"{temp_avg_loss:.4f}",'temp_accuracy': f"{temp_accuracy:.2f}"})
    #     # accuracy = correct_num/(batch_size*(i+1))
    # test_accuracy = test_correct_num/len(test_dataset)
    # print(f"test epoch:{i+1}\t\t test_accuracy:{test_accuracy}")


'''
1、增强泛化能力：数据增强
2、提高计算效率：类似于DPO选择偏好，改写loss,score = logp(Yes) - logp(No),loss = BCEWithLogitsLoss(score, label)
    不需要对所有可能的字符做归一化，只需要利用logits中Yes和No对应的值即可
'''












from transformers import AutoModelForCausalLM, AutoTokenizer

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)






test_pbar=tqdm(test_loader,desc="Testing...")
model.eval()
for step, batch in enumerate(test_loader):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    output_ids = model.generate(input_ids=input_ids,attention_mask=attention_mask)
    logits = output_ids.logits
    last_logits = logits[:, -1, :]        # [B, vocab_size]

    device = last_logits.device
    target_token_ids = torch.tensor([yes_id, no_id], device=device)
    selected_logits = last_logits[:, target_token_ids].squeeze(-1)  # [B,2]
    
    gold_tokens = input_ids[:, -1]             # [B]
    labels = (gold_tokens == yes_id[0]).long()    
    
    loss = criterion(selected_logits, labels)
    
    test_total_loss +=loss.item()*batch_size
    

    preds = torch.argmax(selected_logits, dim=-1)
    correct_num += (preds == labels).sum().item()
    
    # accuracy = correct_num/(batch_size*(i+1))
accuracy = correct_num/len(test_dataset)
print(f"test epoch:{i+1}\t\t accuracy:{accuracy}")






from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4B-Thinking-2507"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content) # no opening <think> tag
print("content:", content)

















