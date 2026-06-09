import warnings 
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from wordcloud import WordCloud


df = pd.read_csv('/kaggle/input/mercor-ai-detection/train.csv')
df.head()


sns.countplot(data=df,x='is_cheating')


def get_sentences(text):
    return len(text.split('\n'))

def get_words(text):
    return len(text.strip().split(' '))

def get_avg_sentence_length(text):
    sentences = text.split('\n')
    n_sentences = 0
    length = 0
    for sentence in sentences:
        n_sentences+=1
        length += len(sentence.strip().split(' '))
    return length/n_sentences

def get_stopwords(text):
    stop_english_words = set(stopwords.words('english'))
    sentences = text.split('\n')
    count = 0
    for sentence in sentences:
        words = sentence.split(' ')
        for word in words:
            if word in stop_english_words:
                count+=1
    return count
    

df['n_sentences'] = df.answer.apply(get_sentences)
df['n_words'] = df.answer.apply(get_words)
df['avg_sentence_length'] = df.answer.apply(get_avg_sentence_length)


df['n_stopwords'] = df.answer.apply(get_stopwords)
df['ratio_of_stopwords'] = df['n_stopwords']/df['n_words']


fig,ax = plt.subplots(1,5,figsize=(13,5))
ax = ax.flatten()
for idx,feat in enumerate(['n_sentences','n_words','avg_sentence_length','n_stopwords','ratio_of_stopwords']):
    sns.histplot(
        data =df,
        x=feat,
        hue='is_cheating',
        ax= ax[idx],
        kde=True
    )
plt.tight_layout()


ai_generated_text = '\n'.join(df[df.is_cheating==1].answer)
human_text = '\n'.join(df[df.is_cheating==0].answer)


ai_cloud = WordCloud(width=800, height=400, background_color='white').generate(ai_generated_text)
human_cloud = WordCloud(width=800, height=400, background_color='white').generate(human_text)


fig,ax = plt.subplots(1,2,figsize=(12,12))
ax[0].imshow(ai_cloud)
ax[0].axis('off')
ax[0].set_title('Ai Generated Text')

ax[1].imshow(human_cloud)
ax[1].axis('off')
ax[1].set_title('Human Written Text')


from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding
from typing import List, Tuple, Dict


##applying the custom prompt 
def prompt(topic,answer):
    return f'''Predict if AI generated text was used:
Topic:{topic}
Answer:{answer}
    '''

df['input'] = df.apply(
    lambda row: prompt(row['topic'],row['answer']),
    axis=1
)

df.head()


import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


from sklearn.utils.class_weight import compute_class_weight

class getData(Dataset):
    def __init__(self,df:pd.DataFrame,tokenizer,max_len=512):
        self.data = df
        self.texts = self.data.input.tolist()
        self.labels = self.data.is_cheating.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,       
            truncation=True,               
            return_token_type_ids=False,   
            return_attention_mask=True,    
        )


        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': label
        }



def get_kfold_loaders(df: pd.DataFrame, tokenizer, batch_sizes:list , num_splits: int = 5,target:str='is_cheating',device=DEVICE) -> List[Tuple[DataLoader, DataLoader]]:
    """
    Sets up 5-fold stratified cross-validation DataLoaders using dynamic padding.
    Returns a list of (train_loader, val_loader) pairs, one for each fold.
    """

    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    X = df.drop(columns=[target]) 
    y = df[target] 
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    all_loaders = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f"--- Fold {fold+1}/{num_splits} ---")

        train_df = df.iloc[train_index].reset_index(drop=True)
        val_df = df.iloc[val_index].reset_index(drop=True)

        weights = compute_class_weight('balanced',classes=np.array([0,1]),y=train_df.is_cheating)
        
        train_dataset = getData(train_df, tokenizer)
        val_dataset = getData(val_df, tokenizer)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_sizes[0],
            shuffle=True,
            collate_fn=data_collator,
            num_workers =2,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_sizes[1], 
            shuffle=False,
            collate_fn=data_collator,
            num_workers=2,
            pin_memory=True
        )
        weights_tensor = torch.tensor(weights,dtype=torch.float,device=device)
        all_loaders.append((train_loader, val_loader,weights_tensor))

    return all_loaders


import torch 
LR = 5e-5 #a good place to start - lowering it for further experimentation
NUM_EPOCHS = 50 # will use early stopping 
WARMUPS = 500

batches = [8,16] #16 for training and 32 for validation
best_metric = float('inf')
trigger=0
patience = 5


from transformers import RobertaForSequenceClassification
import torch 

def train_one_epoch(
    model, 
    loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    loss_fn: torch.nn.Module, 
    DEVICE
) -> float:
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE).long()

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        

        logits = outputs.logits 


        loss = loss_fn(logits, labels) 

        loss.backward()

        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(loader)



def evaluate_one_epoch(
    model, 
    loader: DataLoader, 
    loss_fn: torch.nn.Module, 
    DEVICE) -> Tuple[float, float, float]:
    """
    Performs one full validation pass over the provided DataLoader.

    Returns:
        A tuple containing: (average_validation_loss, accuracy, auc_roc_score)
    """
    
    model.eval()
    total_loss = 0.0
    
    all_labels = []
    all_probabilities = []
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE).long()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            
            _, predicted_classes = torch.max(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities)
            all_predictions.extend(predicted_classes.cpu().numpy())

    avg_loss = total_loss / len(loader)
    
    all_probabilities = np.array(all_probabilities)
    
    auc_roc = roc_auc_score(all_labels, all_probabilities[:, 1])
    
    acc = accuracy_score(all_labels, all_predictions)

    return avg_loss, acc, auc_roc


from transformers import get_cosine_schedule_with_warmup,RobertaTokenizer
model_id = 'FacebookAI/roberta-large'
tokenizer = RobertaTokenizer.from_pretrained(model_id)


all_loaders = get_kfold_loaders(df,tokenizer,batches)


from tqdm.notebook import tqdm, trange

# oof_predictions = 
for idx,loaders in enumerate(tqdm((all_loaders))):
    torch.cuda.empty_cache()
    print(f'Training Model-{idx+1}')
    train_loader,val_loader,weights = loaders
    # weights.to(DEVICE)
    model = RobertaForSequenceClassification.from_pretrained(model_id)
    model.to(DEVICE)

    all_steps = len(train_loader)*NUM_EPOCHS
    
    optimizer = torch.optim.AdamW(model.parameters(),lr=LR)
    scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps = WARMUPS,num_training_steps=all_steps)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    best_metric = float('inf')
    trigger = 0
    patience = 5
    
    for epochs in trange(NUM_EPOCHS):
        training_loss = train_one_epoch(model,train_loader,optimizer,loss_fn,DEVICE)
        scheduler.step()
        avg_val_loss,val_acc,aur_roc = evaluate_one_epoch(model,val_loader,loss_fn,DEVICE)

        print(f'EPOCH: {epochs}/{NUM_EPOCHS}, TrainingLoss: {training_loss}, ValidationLoss: {avg_val_loss}, AUC_ROC: {aur_roc}, Acc: {val_acc}')

        if avg_val_loss < best_metric:
            best_metric = avg_val_loss
            trigger = 0
            print(f'Saving Best {idx+1} model')
            torch.save(model.state_dict(),f'BERT{idx+1}Tuned.pth')
        else:
            trigger +=1
        if trigger >= patience:
            print('-----EARLY STOPPING-----')
            break


model_list = []

for idx in range(5):
    # model = DistilBertForSequenceClassification.from_pretrained(model_id)
    torch.cuda.empty_cache()
    model = RobertaForSequenceClassification.from_pretrained(model_id)
    state_dict=torch.load(f'BERT{idx+1}Tuned.pth',map_location=DEVICE)
    model.load_state_dict(state_dict)
    model_list.append(model)


def predict_test_set(
    model_list,
    test_df: pd.DataFrame,
    tokenizer,
    batch_size: int,
    DEVICE: str
) -> np.ndarray:
    
    test_dataset = getData(test_df, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, 
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True
    )
    
    all_model_predictions = []
    
    for fold_idx, model in enumerate(model_list):
        torch.cuda.empty_cache()
        print(f"Generating predictions using Model {fold_idx + 1}...")
        
        model.to(DEVICE)
        model.eval()
        
        fold_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Move inputs to device
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                
                logits = outputs.logits
                
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                fold_probabilities.append(probabilities)

        all_model_predictions.append(np.concatenate(fold_probabilities, axis=0))


    stacked_predictions = np.stack(all_model_predictions, axis=0)
    
    avg_probabilities = np.mean(stacked_predictions, axis=0)
    
    print("\nâœ… Final test set probabilities generated via ensemble averaging.")
    
    return avg_probabilities


torch.cuda.empty_cache()



test = pd.read_csv('/kaggle/input/mercor-ai-detection/test.csv')
test['input'] = test.apply(
    lambda row: prompt(row['topic'],row['answer']),
    axis=1
)
test['is_cheating'] = -1
preds = predict_test_set(model_list,test,tokenizer,4,DEVICE)


sub = pd.DataFrame({
    'id':test.id,
    'is_cheating':preds[:,1]
})
sub.head()


sub.to_csv('SUB_TOM_08_lower_learning_rate.csv',index=False)




