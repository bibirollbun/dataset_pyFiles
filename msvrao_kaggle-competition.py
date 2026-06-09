import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel


df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
df.head()


df['rule_violation'].value_counts()


df['rule_violation_text'] = df['rule_violation'].map({1: "Yes", 0: "No"})


df = df.loc[:, ['body',  'positive_example_1','rule', 
                'positive_example_2', 'negative_example_1', 'negative_example_2', 'rule_violation_text']]


df['Text'] = df.apply(lambda x: f'''Text: {x.loc["body"]}

Rules: 
{x.loc["rule"]}
Allowed Examples:
1. {x.loc["positive_example_1"]}
2. {x.loc["positive_example_2"]}

Violation Examples:
1. {x.loc["negative_example_1"]}
2. {x.loc["negative_example_2"]}''', axis=1)


df['Text'].head()


df['Text'].apply(lambda x: len(x)).max()


from transformers import AutoTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
import wandb
from torch.utils.data import Dataset


from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, train_size=0.8, random_state=30)




import seaborn as sns
tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/bart/transformers/bart/1", 
                                         padding_side='right', truncation_side='right')

out_tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/bart/transformers/bart/1", 
                                         padding_side='right', truncation_side='right')


sns.distplot(df_train['Text'].apply(lambda x: len(tokenizer(x)['input_ids'])))


tokenizer.decode(tokenizer(df_train['Text'].iloc[5])['input_ids'])


df_train.columns


df_train['rule_violation_text'].value_counts()


tokenizer.decode(tokenizer(df_train['rule_violation_text'].iloc[5], padding="max_length", 
                truncation=True, max_length=5)['input_ids'])


class MyDataSet(Dataset):
    def __init__(self, df, col, out_col):
        self.df = df
        self.col = col
        self.out_col = out_col

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        tokens = tokenizer(self.df[self.col].iloc[idx], padding="max_length", truncation=True,
                          max_length=512, return_tensors='pt')
        
        out_tokens = out_tokenizer(self.df[self.out_col].iloc[idx], padding="max_length", 
                                  truncation=True, max_length=5, return_tensors='pt')
        return {
            "input_ids": tokens['input_ids'][0,:],
            "attention_mask": tokens['attention_mask'][0,:],
            "labels": out_tokens['input_ids'][0,:]
        }


train_ds = MyDataSet(df_train, "Text", 'rule_violation_text')
val_ds = MyDataSet(df_test, "Text", 'rule_violation_text')


train_ds.__getitem__(0)


wandb.init(mode="disabled")


model = BartForConditionalGeneration.from_pretrained("/kaggle/input/bart/transformers/bart/1")

args = TrainingArguments(output_dir="models", eval_strategy="epoch", 
                        num_train_epochs=3, learning_rate=3e-5, per_device_train_batch_size=8,
                        per_device_eval_batch_size=8, logging_strategy='epoch', save_strategy='epoch')

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
trainer.train()


x = df_train['Text'].iloc[100]



tokens = tokenizer(x, padding="max_length", truncation=True, max_length=512, return_tensors='pt')['input_ids'].cuda()
l = len(tokens) + 10
y = model.generate(tokens, temperature=0.01, max_length=l)

tokenizer.decode(y[0])


from transformers.utils import logging
logging.set_verbosity_error()
def predict(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)['input_ids'].cuda()
    y = model.generate(tokens, temperature=0.01)
    res = out_tokenizer.decode(y[0], skip_special_tokens=True)
    return res


df_train["Pred"] = df_train['Text'].apply(lambda x: predict(x))

df_train.head()


from sklearn.metrics import confusion_matrix
y_true = df_train['rule_violation_text'].apply(lambda x: 1 if 'yes' in x.lower() else 0)
y_pred = df_train['Pred'].apply(lambda x: 1 if 'yes' in x.lower() else 0)

cf = confusion_matrix(y_true, y_pred)
cf


def metrics(cf):
    precision = cf[1,1]/(cf[1,1]+ cf[0,1])
    recall = cf[1,1]/(cf[1,1]+ cf[1,0])
    acc = (cf[0,0]+ cf[1,1])/sum(sum(cf))
    f1 = 2*precision*recall/(precision+recall)

    return pd.DataFrame({
        "metric": ['Precision', "Recall", "Accuracy", "F1"],
        "Value": [precision, recall, acc, f1]
        
    })

metrics(cf)


df_test["Pred"] = df_test['Text'].apply(lambda x: predict(x))

df_test.head()


y_true = df_test['rule_violation_text'].apply(lambda x: 1 if 'yes' in x.lower() else 0)
y_pred = df_test['Pred'].apply(lambda x: 1 if 'yes' in x.lower() else 0)

cf = confusion_matrix(y_true, y_pred)
cf


metrics(cf)


df_testing = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

df_testing.head()


df_testing['Text'] = df.apply(lambda x: f'''Text: {x.loc["body"]}

Rules: 
{x.loc["rule"]}
Allowed Examples:
1. {x.loc["positive_example_1"]}
2. {x.loc["positive_example_2"]}

Violation Examples:
1. {x.loc["negative_example_1"]}
2. {x.loc["negative_example_2"]}''', axis=1)

df_testing.head()


df_testing.shape


df_testing['Pred'] = df_testing["Text"].apply(lambda x: predict(x))
df_testing['Pred']


y_pred = df_testing['Pred'].apply(lambda x: 1 if 'yes' in x.lower() else 0)
y_pred


df_out = pd.DataFrame({
    "row_id": df_testing['row_id'],
    "rule_violation": y_pred
})
df_out


df_out.to_csv("submission.csv", index=False)




