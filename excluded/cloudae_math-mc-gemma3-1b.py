from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
from torch import nn
from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder
import joblib


train_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
train_df['Misconception'] = train_df['Misconception'].fillna('NA')


text = [
    f"Question: {q}\nAnswer: {a}\nExplanation: {e}"
    for q,a,e in zip(train_df['QuestionText'],train_df['MC_Answer'],train_df['StudentExplanation'])
]


train_df['text'] = text


train_df['labels'] = [
    f"{c}:{m}" for c,m in zip(train_df['Category'],train_df['Misconception'])
]


train_df = train_df.drop(['QuestionText','MC_Answer','StudentExplanation','Category','Misconception','row_id','QuestionId'],axis=1)


le = LabelEncoder()
train_df['labels'] = le.fit_transform(train_df['labels'])


ds = Dataset.from_pandas(train_df)


ds = ds.train_test_split(test_size=0.15,seed=123)


ds = ds.with_format('torch')


#model
model_name = "/kaggle/input/gemma-3/transformers/gemma-3-1b-it/1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:0", attn_implementation='eager')


#label_encoder, NUM_CLASSES, Custom classification head...
NUM_CLASSES = len(train_df['labels'].unique())
class Gemma3_Head(nn.Module):
    def __init__(self,base_model,num_classes):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size,num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self,labels,input_ids,attention_mask):
        outputs = self.base_model(input_ids=input_ids,attention_mask=attention_mask,output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        pooled_output = last_hidden_state[:, -1, :]
        logits = self.classifier(pooled_output)

        loss = self.loss_fn(logits,labels)
        return loss,logits


def compute_map3(eval_pred):
    """Compute MAP@3 metric for evaluation"""
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    top3 = np.argsort(-probs, axis=1)[:, :3]

    map3 = 0.0
    for i in range(len(labels)):
        if top3[i, 0] == labels[i]:
            map3 += 1.0
        elif top3[i, 1] == labels[i]:
            map3 += 1.0 / 2
        elif top3[i, 2] == labels[i]:
            map3 += 1.0 / 3
    map3 /= len(labels)

    acc = accuracy_score(labels, np.argmax(probs, axis=1))
    return {"accuracy": acc, "map@3": map3}


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj","k_proj","o_proj"], #"gate_proj","up_proj","down_proj"
    lora_dropout=0.14,
    bias="none",
    task_type="CAUSAL_LM"
)

peft_Gemma3model = get_peft_model(model, lora_config)


Gemma3model = Gemma3_Head(peft_Gemma3model,NUM_CLASSES)


from torchinfo import summary
summary(Gemma3model)


def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
    )

tokenized_ds = ds.map(
    tokenize,
    batched=True,
    num_proc=4,
    remove_columns='text',
)


EPOCHS = 8
total_training_steps = len(tokenized_ds['train']) * EPOCHS
warmup_steps = 0.07 * total_training_steps
from torch.optim import AdamW
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup,DataCollatorWithPadding


train_dataloader = DataLoader(
    tokenized_ds['train'],
    batch_size=8,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest"),
)
val_dataloader = DataLoader(
    tokenized_ds['test'],
    batch_size=16,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest"),
)


optimizer = AdamW(Gemma3model.parameters(),lr=1e-4,weight_decay=0.1)
scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps,
    num_cycles=1,
)


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
Gemma3model.to(device)
Gemma3model.base_model.to(device)
best_val_loss = float('inf')
PATIENCE = 3
scaler = GradScaler()
patience = PATIENCE

for epoch in range(EPOCHS):
    Gemma3model.train()
    train_pbar = tqdm(train_dataloader,desc='Training',leave=False)
    val_pbar = tqdm(val_dataloader,desc='Evaluating',leave=False)
    
    print(f"Epoch {epoch+1}/{EPOCHS}:")

    train_loss,train_total = 0.0,0.0
    for batch in train_pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss,logits = Gemma3model(**batch)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        optimizer.zero_grad(set_to_none=True)

        train_loss += loss * len(batch['labels'])
        train_total += len(batch['labels'])
        train_pbar.set_postfix(loss=train_loss / train_total)

    print(f'Training Loss: {train_loss/train_total}')

    val_loss,val_total,correct = 0.0,0.0,0.0
    Gemma3model.eval()
    with torch.inference_mode():
        for batch in val_pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss,logits = Gemma3model(**batch)
    
            pred = torch.argmax(logits,dim=1)
            label = batch['labels']
    
            correct += (pred == label).sum().cpu().numpy()
            val_total += len(batch['labels'])

            val_loss += loss.cpu().numpy() * len(batch['labels'])

            val_pbar.set_postfix(loss = val_loss/val_total,accuracy = correct/val_total)

        accuracy = correct / val_total
        val_loss /= val_total

    print(f"Validation Accuracy: {accuracy}, Validation Loss: {val_loss}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        Gemma3model.base_model.save_pretrained("Gemma_Lora_Adapter")
        torch.save(Gemma3model.classifier.state_dict(),'Gemma_Classifier_Head.pth')
        patience = PATIENCE
        #patience -= 1 # debugging
        best_epoch = epoch + 1
    else:
        patience -= 1

    if patience == 0:
        print(f"Early stopping triggered, using model from Epoch {best_epoch}")
        break


joblib.dump(le,'label_encoder_Gemma3.pkl')


#Is LoRa working?
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:0", attn_implementation='eager')
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
Gemma3model.to(device)
test = 'What is the capital of Japan?'
test = tokenizer(test,return_tensors='pt')
test.to(device)
torch.allclose(model(**test).logits, Gemma3model.base_model(**test).logits)




