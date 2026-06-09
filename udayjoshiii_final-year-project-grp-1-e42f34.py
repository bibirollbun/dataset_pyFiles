import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModel, AutoConfig

import sys
sys.path.append('/kaggle/input/ess-utilities')
import utilities
from tqdm import tqdm


class Extra_Head_1(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()  

        self.backbone = nn.Sequential(nn.Linear(embedding_size, embedding_size), nn.GELU())

        self.head = nn.Linear(embedding_size, 2)
        self.aux_head = nn.Linear(embedding_size, 6)

    def forward(self, x):

        x = self.backbone(x)

        x_preds = self.head(x)

        return x_preds


class Weighted_Linear(nn.Module):
    def __init__(self, hidden_size, model_instance):
        super().__init__()
        self.hidden_size = hidden_size
        self.cat_size = hidden_size*3

        self.layer_pooler = utilities.WeightedLayerPooling(model_instance.n_layers) 
        self.sequence_pooler = utilities.MeanPooling(.0 if model_instance.use_prompt_text else 1e-9)  
 
        self.head = nn.Sequential(nn.Linear(self.hidden_size, CFG.n_classes))
        
        if model_instance.aux:
            self.aux_head = nn.Sequential(nn.Linear(self.hidden_size, 6))
        
        self.dropout = utilities.Multisample_Dropout()
                                
    def forward(self, x, mask):
        
        x = self.layer_pooler(x.hidden_states) 

        x = self.sequence_pooler(x, mask).half()

        #x = self.dropout(x, self.head) 

        return self.head(x)

class Cat_LSTM(nn.Module):
    def __init__(self, hidden_size, model_instance):
        super().__init__()
        self.hidden_size = hidden_size
        self.cat_size = hidden_size*model_instance.n_layers
        self.n_layers = model_instance.n_layers

        #self.layer_pooler = utilities.WeightedLayerPooling(6) 
        self.sequence_pooler = utilities.MeanPooling(.0 if model_instance.use_prompt_text else 1e-9)  
        self.rnn = utilities.Bi_RNN_FOUT(self.cat_size, self.cat_size//2)   
 
        self.head = nn.Sequential(nn.Linear(self.cat_size, CFG.n_classes)) 
        if model_instance.aux:
            self.aux_head = nn.Sequential(nn.Linear(self.cat_size, 6))
        
        #self.dropout = utilities.Multisample_Dropout()
        self.extra_heads = nn.ModuleList()
        if model_instance.extra_head_instances:
            for extra_head_instance in model_instance.extra_head_instances:
                
                head_path = extra_head_instance.folds[model_instance.current_fold]
            
                head = extra_head_instance.head(extra_head_instance.emb_size)
                head.load_state_dict(torch.load(head_path))
                #head = head.to(CFG.device).half()
                #head = nn.DataParallel(head).half()
                self.extra_heads.append(head)
                                
    def forward(self, x, mask): 
        
        x = torch.cat(x.hidden_states[-self.n_layers:], dim=-1)  
        
        hidden_mask = mask.unsqueeze(-1).expand(x.size()).float()
        x = (x * hidden_mask).half() 
        
        x = self.rnn(x) 
        x = self.sequence_pooler(x, mask).half() 

        #x_preds = self.dropout(x, self.head) 
        #aux = self.dropout(x, self.aux_head) 
        
        output = self.head(x) 
        if self.extra_heads:
            for head in self.extra_heads:
                output += head(x)
            
            output = output / (len(self.extra_heads) + 1)
        
        return output
    
class Pool_LSTM(nn.Module):
    def __init__(self, hidden_size, model_instance):
        super().__init__()
        self.hidden_size = hidden_size
        self.cat_size = hidden_size*model_instance.n_layers
        self.n_layers = model_instance.n_layers

        self.pooler = utilities.LSTMPooling(hidden_size, num_hidden_layers=self.n_layers) 
        #self.rnn = utilities.Bi_RNN_FOUT(self.cat_size, self.cat_size//2)   
 
        self.head = nn.Sequential(nn.Linear(self.hidden_size, 2)) 
    
        if model_instance.aux:
            self.aux_head = nn.Sequential(nn.Linear(self.hidden_size, 6)) 

        self.extra_heads = nn.ModuleList()
        if model_instance.extra_head_instances:
            for extra_head_instance in model_instance.extra_head_instances:
                
                head_path = extra_head_instance.folds[model_instance.current_fold]
            
                head = extra_head_instance.head(extra_head_instance.emb_size)
                head.load_state_dict(torch.load(head_path))
                #head = head.to(CFG.device).half()
                #head = nn.DataParallel(head).half()
                self.extra_heads.append(head)
                                
    def forward(self, x, mask): 
        
        
        x = self.pooler(x.hidden_states, mask) 

        output = self.head(x) 
        if self.extra_heads:
            for head in self.extra_heads:
                output += head(x)
            
            output = output / (len(self.extra_heads) + 1) 

        return output


class CFG:

    n_classes = 2

    n_workers = 2

    device = torch.device('cuda')
    #autocast = True
        


class Extra_Head_Instance(nn.Module):
    def __init__(self, head, emb_size, folds):
        super().__init__()
        self.folds = folds
        self.head = head
        self.emb_size = emb_size



class Model_Instance(nn.Module):
    def __init__(self, batch_size, max_len, model_name, tokenizer, config, folds, weight, head, extra_head_instances=None, n_layers=6, aux=False, use_prompt_text=True):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.config = config
        self.folds = folds
        self.n_layers = n_layers
        self.use_prompt_text = use_prompt_text
        self.aux=aux
        self.weight = weight
        self.head = head
        self.extra_head_instances = extra_head_instances
        self.current_fold = 0




deberta_v3_base_long_v3 = []
deberta_v3_base_long_v3.append(Extra_Head_Instance(
    head=Extra_Head_1,
    emb_size=1536,
    folds=[#'/kaggle/input/deberta-v3-base-long-v3-h3/deberta-v3-base-long-v3-h3/microsoft-deberta-v3-base-0-0.433',
           #'/kaggle/input/deberta-v3-base-long-v3-h3/deberta-v3-base-long-v3-h3/microsoft-deberta-v3-base-1-0.496',
           '/kaggle/input/deberta-v3-base-long-v3-h3/deberta-v3-base-long-v3-h3/microsoft-deberta-v3-base-2-0.419',
           '/kaggle/input/deberta-v3-base-long-v3-h3/deberta-v3-base-long-v3-h3/microsoft-deberta-v3-base-3-0.502',
    ]
))


deberta_v3_large_long_v9 = []
deberta_v3_large_long_v9.append(Extra_Head_Instance(
    head=Extra_Head_1,
    emb_size=1024,
    folds=[#'/kaggle/input/deberta-v3-large-long-v3-h2/deberta-v3-large-long-v3-h2/microsoft-deberta-v3-large-0-0.421',
           #'/kaggle/input/deberta-v3-large-long-v3-h2/deberta-v3-large-long-v3-h2/microsoft-deberta-v3-large-1-0.488',
           '/kaggle/input/deberta-v3-large-long-v3-h2/deberta-v3-large-long-v3-h2/microsoft-deberta-v3-large-2-0.415',
           '/kaggle/input/deberta-v3-large-long-v3-h2/deberta-v3-large-long-v3-h2/microsoft-deberta-v3-large-3-0.504',
          ]
))


OpenAssistant_v1 = []
OpenAssistant_v1.append(Extra_Head_Instance(
    head=Extra_Head_1,
    emb_size=1024,
    folds=['/kaggle/input/openassistant-large-v2-long-v1-h1/OpenAssistant--large-v2-long-v1-h1/microsoft-deberta-v3-large-0-0.421',
           '/kaggle/input/openassistant-large-v2-long-v1-h1/OpenAssistant--large-v2-long-v1-h1/microsoft-deberta-v3-large-1-0.482',
           #'/kaggle/input/openassistant-large-v2-long-v1-h1/OpenAssistant--large-v2-long-v1-h1/microsoft-deberta-v3-large-2-0.424',
           #'/kaggle/input/openassistant-large-v2-long-v1-h1/OpenAssistant--large-v2-long-v1-h1/microsoft-deberta-v3-large-3-0.516',
          ]
))


model_instances = []
model_instances.append(Model_Instance(batch_size=32,
                                      max_len=1792,  
                                      model_name='microsoft/deberta-v3-large', 
                                      tokenizer='/kaggle/input/deberta-v3-large-long-v9/deberta-v3-large-long-v9/microsoft-deberta-v3-large-tokenizer',
                                      config='/kaggle/input/deberta-v3-large-long-v9/deberta-v3-large-long-v9/microsoft-deberta-v3-large-config',
                                      folds = [#'/kaggle/input/deberta-v3-large-long-v9/deberta-v3-large-long-v9/microsoft-deberta-v3-large-0-0.424',
                                               #'/kaggle/input/deberta-v3-large-long-v9/deberta-v3-large-long-v9/microsoft-deberta-v3-large-1-0.489',  
                                               '/kaggle/input/deberta-v3-large-long-v9/deberta-v3-large-long-v9/microsoft-deberta-v3-large-2-0.417',
                                               '/kaggle/input/deberta-v3-large-long-v9/deberta-v3-large-long-v9/microsoft-deberta-v3-large-3-0.513',
                                              ],
                                      weight=0.2,
                                      aux=True,
                                      head=Pool_LSTM,
                                      extra_head_instances=deberta_v3_large_long_v9,
                                      ))
model_instances.append(Model_Instance(batch_size=32, 
                                      max_len=2048,  
                                      model_name='microsoft/deberta-v3-base', 
                                      tokenizer='/kaggle/input/deberta-v3-base-long-v3/deberta-v3-base-long-v3/microsoft-deberta-v3-base-tokenizer',
                                      config='/kaggle/input/deberta-v3-base-long-v3/deberta-v3-base-long-v3/microsoft-deberta-v3-base-config',
                                      folds = [#'/kaggle/input/deberta-v3-base-long-v3/deberta-v3-base-long-v3/microsoft-deberta-v3-base-0-0.436',
                                               #'/kaggle/input/deberta-v3-base-long-v3/deberta-v3-base-long-v3/microsoft-deberta-v3-base-1-0.502',  
                                               '/kaggle/input/deberta-v3-base-long-v3/deberta-v3-base-long-v3/microsoft-deberta-v3-base-2-0.422',
                                               '/kaggle/input/deberta-v3-base-long-v3/deberta-v3-base-long-v3/microsoft-deberta-v3-base-3-0.513',
                                              ],  
                                      n_layers=2,  
                                      weight=.2,
                                      aux=True,
                                      head=Cat_LSTM,
                                      extra_head_instances=deberta_v3_base_long_v3,
                                      ))
model_instances.append(Model_Instance(batch_size=32, 
                                      max_len=1792,  
                                      model_name='OpenAssistant/reward-model-deberta-v3-large-v2', 
                                      tokenizer='/kaggle/input/openassistant-large-v2-long-v1/OpenAssistant--large-v2-long-v1/OpenAssistant-reward-model-deberta-v3-large-v2-tokenizer',
                                      config='/kaggle/input/openassistant-large-v2-long-v1/OpenAssistant--large-v2-long-v1/OpenAssistant-reward-model-deberta-v3-large-v2-config',
                                      folds = ['/kaggle/input/openassistant-large-v2-long-v1/OpenAssistant--large-v2-long-v1/OpenAssistant-reward-model-deberta-v3-large-v2-0-0.42',
                                               '/kaggle/input/openassistant-large-v2-long-v1/OpenAssistant--large-v2-long-v1/OpenAssistant-reward-model-deberta-v3-large-v2-1-0.485', 
                                               #'/kaggle/input/openassistant-large-v2-long-v1/OpenAssistant--large-v2-long-v1/OpenAssistant-reward-model-deberta-v3-large-v2-2-0.426',
                                               #'/kaggle/input/openassistant-large-v2-long-v1/OpenAssistant--large-v2-long-v1/OpenAssistant-reward-model-deberta-v3-large-v2-3-0.526',
                                              ],
                                      weight=0.2,
                                      aux=True,
                                      head=Pool_LSTM,
                                      extra_head_instances=OpenAssistant_v1,
                                      ))
# model_instances.append(Model_Instance(batch_size=32,
#                                       max_len=1792,  
#                                       model_name='microsoft/deberta-large', 
#                                       tokenizer='/kaggle/input/deberta-large-long-v5/deberta-large-long-v5/microsoft-deberta-large-tokenizer',
#                                       config='/kaggle/input/deberta-large-long-v5/deberta-large-long-v5/microsoft-deberta-large-config',
#                                       folds = [#'/kaggle/input/deberta-large-long-v5/deberta-large-long-v5/microsoft-deberta-large-0-0.425',
#                                                #'/kaggle/input/deberta-large-long-v5/deberta-large-long-v5/microsoft-deberta-large-1-0.492', 
#                                                '/kaggle/input/deberta-large-long-v5/deberta-large-long-v5/microsoft-deberta-large-2-0.432',
#                                                '/kaggle/input/deberta-large-long-v5/deberta-large-long-v5/microsoft-deberta-large-3-0.515',
#                                               ],
#                                       aux=True,
#                                       weight=0.2,
#                                       head=Weighted_Linear,
#                                       ))
model_instances.append(Model_Instance(batch_size=32, 
                                      max_len=1792,  
                                      model_name='microsoft/deberta-v3-base', 
                                      tokenizer='/kaggle/input/deberta-v3-large-long-v7/deberta-v3-large-long-v7/microsoft-deberta-v3-large-tokenizer',
                                      config='/kaggle/input/deberta-v3-large-long-v7/deberta-v3-large-long-v7/microsoft-deberta-v3-large-config',
                                      folds = ['/kaggle/input/deberta-v3-large-long-v7/deberta-v3-large-long-v7/microsoft-deberta-v3-large-0-0.426',
                                               '/kaggle/input/deberta-v3-large-long-v7/deberta-v3-large-long-v7/microsoft-deberta-v3-large-1-0.494',  
                                               #'/kaggle/input/deberta-v3-large-long-v7/deberta-v3-large-long-v7/microsoft-deberta-v3-large-2-0.421',
                                               #'/kaggle/input/deberta-v3-large-long-v7/deberta-v3-large-long-v7/microsoft-deberta-v3-large-3-0.504',
                                              ],  
                                      n_layers=2,  
                                      weight=.2,
                                      aux=True,
                                      head=Cat_LSTM,
                                      ))



#prompts_path = '/kaggle/input/commonlit-evaluate-student-summaries/prompts_test.csv'
#summaries_path = '/kaggle/input/commonlit-evaluate-student-summaries/summaries_test.csv' 

prompts_path = '/kaggle/input/commonlit-evaluate-student-summaries/prompts_train.csv'
summaries_path = '/kaggle/input/commonlit-evaluate-student-summaries/summaries_train.csv'


prompts_df = pd.read_csv(prompts_path)
summaries_df = pd.read_csv(summaries_path)


merged_df = summaries_df.merge(prompts_df, how='inner', on=None, left_on='prompt_id', right_on='prompt_id')
merged_df = summaries_df.merge(prompts_df, how='inner', on='prompt_id')

# Optional: keep only first 20%
keep_rows = int(len(merged_df) * 0.2)
merged_df = merged_df.iloc[:keep_rows].reset_index(drop=True)

true_labels = merged_df[['student_id', 'content', 'wording']].copy()
merged_df = merged_df.drop(columns=['content', 'wording'])


merged_df.head()


merged_df['concat'] = merged_df.prompt_question + merged_df.text + merged_df.prompt_text


merged_df['concat_len'] = merged_df['concat'].apply(lambda x: len(x))


merged_df = merged_df.sort_values(by=['concat_len'], ascending=True)


merged_df.head(20)


class Summary_DS(Dataset):
    def __init__(self, df, tokenizer, use_prompt_text):
        self.use_prompt_text = use_prompt_text
        self.df = df
        self.tokenizer = tokenizer
        self.seperator = " " + self.tokenizer.sep_token + " "
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]
        
        prompt_text = (self.seperator + row.prompt_text) if self.use_prompt_text else ''

        input_text = 'Think through this step by step : ' + row.prompt_question + self.seperator + 'Pay attention to the content and wording : ' + row.text + prompt_text

        tokenized_dict = self.tokenizer(input_text, add_special_tokens = False) 

        input_ids = tokenized_dict.input_ids
        attention_mask = tokenized_dict.attention_mask 

        head_mask = []
        use_full = False
        for token in tokenized_dict.input_ids:
            
            if token == self.tokenizer.sep_token_id:
                use_full = not use_full  

            head_mask.append(1 if use_full else .0) 

        return {'ids':input_ids,'mask':attention_mask, 'head_mask':head_mask}


class Collate:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        output = dict()
        output["ids"] = [sample["ids"] for sample in batch]
        output["mask"] = [sample["mask"] for sample in batch]
        output["head_mask"] = [sample["head_mask"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["ids"]]) 

        batch_max = min(batch_max, self.max_len) 

        output["ids"] = [s[:batch_max] for s in output["ids"]]
        output["mask"] = [s[:batch_max] for s in output["mask"]] 
        output["head_mask"] = [s[:batch_max] for s in output["head_mask"]] 

        output["ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["ids"]]
        output["mask"] = [s + (batch_max - len(s)) * [0] for s in output["mask"]]
        output["head_mask"] = [s + (batch_max - len(s)) * [0] for s in output["head_mask"]]


        # convert to tensors
        output["ids"] = torch.tensor(output["ids"], dtype=torch.long)
        output["mask"] = torch.tensor(output["mask"], dtype=torch.long)
        output["head_mask"] = torch.tensor(output["head_mask"], dtype=torch.float)

        return output


class Model(nn.Module):
    def __init__(self, model_instance):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_instance.config)
        self.transformer = AutoModel.from_config(
            config=self.config
        )
        self.use_prompt_text = model_instance.use_prompt_text
        self.head = model_instance.head(self.config.hidden_size, model_instance)

    def forward(self, input_ids, attention_mask, head_mask):

        x = self.transformer(input_ids, attention_mask = attention_mask)

        x = self.head(x, head_mask if self.use_prompt_text else attention_mask)
        
        return x



def eval(model, valid_loader, fold):
    with torch.no_grad():
        model.eval() 

        all_preds=torch.tensor([],dtype=torch.float)

        bar = tqdm(valid_loader)
        for i, data in enumerate(bar): 

            input_ids = data['ids'].to(CFG.device)
            attention_mask = data['mask'].to(CFG.device)
            head_mask = data['head_mask'].to(CFG.device)

            preds = model(input_ids, attention_mask, head_mask)
            
            preds = preds.cpu().detach()
            
            all_preds = torch.cat([all_preds, preds], dim=0)

            bar.set_postfix(fold=fold)

    return all_preds


preds_list = []
oof_total = 0  
for model_instance in model_instances:
    tokenizer = AutoTokenizer.from_pretrained(model_instance.tokenizer)
    for fold,path in enumerate(model_instance.folds):  
        model_instance.current_fold = fold
        model = Model(model_instance) 
        model.load_state_dict(torch.load(path),strict=False)
        model = model.to(CFG.device).half()
        model = nn.DataParallel(model).half()
        
        ds = Summary_DS(merged_df, tokenizer, model_instance.use_prompt_text)
        loader = DataLoader(ds, batch_size = model_instance.batch_size, num_workers=CFG.n_workers, shuffle=False, drop_last=False, collate_fn=Collate(tokenizer, model_instance.max_len))

        preds = eval(model, loader, fold)
        preds = preds * (model_instance.weight/len(model_instance.folds))
        preds_list.append(preds)
        
        del model
        torch.cuda.empty_cache()




preds_np = np.stack(preds_list, 0)


final_preds = preds_np.sum(0)


final_preds.shape


# Create DataFrame with predicted scores
preds_df = pd.DataFrame({
    'student_id': merged_df['student_id'],
    'prompt_id': merged_df['prompt_id'],
    'predicted_content_score': final_preds[:, 0],
    'predicted_wording_score': final_preds[:, 1]
})

# Now merge with true_labels to get the actual content and wording scores
results_df = preds_df.merge(true_labels, on='student_id', how='left')

# Rearrange columns as needed
results_df = results_df[['prompt_id', 
                         'predicted_content_score', 
                         'predicted_wording_score', 
                         'content',  # actual content score
                         'wording']]  # actual wording score

# Rename content and wording for clarity
results_df.rename(columns={'content': 'actual_content_score',
                            'wording': 'actual_wording_score'}, inplace=True)

# Save to CSV
results_df.to_csv('/kaggle/working/predictions_vs_actuals.csv', index=False)



from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_predictions(df):
    results = {}
    for col in ['content', 'wording']:
        y_true = df[f'actual_{col}_score']
        y_pred = df[f'predicted_{col}_score']
        
        results[f'{col}_mse'] = mean_squared_error(y_true, y_pred)
        results[f'{col}_rmse'] = mean_squared_error(y_true, y_pred, squared=False)
        results[f'{col}_mae'] = mean_absolute_error(y_true, y_pred)
        results[f'{col}_r2'] = r2_score(y_true, y_pred)
    
    return results

metrics = evaluate_predictions(results_df)
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")



import matplotlib.pyplot as plt
import seaborn as sns

def scatter_plot(df, col):
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=f'actual_{col}_score', y=f'predicted_{col}_score', data=df)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{col.capitalize()} - Actual vs Predicted")
    plt.grid()
    plt.show()

scatter_plot(results_df, 'content')
scatter_plot(results_df, 'wording')



def residual_plot(df, col):
    df[f'{col}_residual'] = df[f'predicted_{col}_score'] - df[f'actual_{col}_score']
    plt.figure(figsize=(6, 4))
    sns.histplot(df[f'{col}_residual'], kde=True)
    plt.title(f'{col.capitalize()} Residuals Distribution')
    plt.xlabel("Residual (Prediction - Actual)")
    plt.ylabel("Count")
    plt.grid()
    plt.show()

residual_plot(results_df, 'content')
residual_plot(results_df, 'wording')



def error_by_prompt(df, col):
    df[f'{col}_abs_error'] = abs(df[f'predicted_{col}_score'] - df[f'actual_{col}_score'])
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='prompt_id', y=f'{col}_abs_error', data=df)
    plt.xticks(rotation=90)
    plt.title(f'{col.capitalize()} Absolute Error by Prompt')
    plt.xlabel("Prompt ID")
    plt.ylabel("Absolute Error")
    plt.grid()
    plt.show()

error_by_prompt(results_df, 'content')
error_by_prompt(results_df, 'wording')



def per_prompt_stats(df):
    grouped = df.groupby('prompt_id').apply(
        lambda x: pd.Series({
            'content_rmse': mean_squared_error(x['actual_content_score'], x['predicted_content_score'], squared=False),
            'wording_rmse': mean_squared_error(x['actual_wording_score'], x['predicted_wording_score'], squared=False),
            'n_samples': len(x)
        })
    ).reset_index()
    return grouped

prompt_stats = per_prompt_stats(results_df)
print(prompt_stats.head())



plt.figure(figsize=(12,6))
sns.barplot(x='prompt_id', y='content_rmse', data=prompt_stats)
plt.xticks(rotation=90)
plt.title("Content RMSE per Prompt")
plt.grid()
plt.show()



submission_df = pd.DataFrame({'student_id':merged_df.student_id,'content':final_preds[:,0],'wording':final_preds[:,1]})


submission_df.head()


submission_df.to_csv("/kaggle/working/submission.csv", index=False)




