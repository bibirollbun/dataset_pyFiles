!pip install -q condacolab
import condacolab
condacolab.install()


import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizer, DistilBertModel


MAX_LEN = 320
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 1e-05
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


train_data = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')

label_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train_data['labels'] = train_data[label_columns].apply(lambda x: list(x), axis=1)

train_data.drop(['id'], inplace=True, axis=1)
train_data.drop(label_columns, inplace=True, axis=1)

train_data.head()


class MultiLabelDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len, new_data=False):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.comment_text
        self.new_data = new_data
        
        if not new_data:
            self.targets = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        out = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }
        
        if not self.new_data:
            out['targets'] = torch.tensor(self.targets[index], dtype=torch.float)

        return out


train_size = 1.0

train_df = train_data.sample(frac=train_size, random_state=123)
val_df = train_data.drop(train_df.index).reset_index(drop=True)
train_df = train_df.reset_index(drop=True)


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)
training_set = MultiLabelDataset(train_df, tokenizer, MAX_LEN)
val_set = MultiLabelDataset(val_df, tokenizer, MAX_LEN)


train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 4
                }

val_params = {'batch_size': VALID_BATCH_SIZE,
               'shuffle': False,
               'num_workers': 4
                }

training_loader = DataLoader(training_set, **train_params)



class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(768, 6)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        out = hidden_state[:, 0]
        out = self.classifier(out)
        return out

model = DistilBERTClass()
model.to(DEVICE);


optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


def train(epoch):
    model.train()
    
    for _, data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(DEVICE, dtype=torch.long)
        mask = data['mask'].to(DEVICE, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(DEVICE, dtype=torch.long)
        targets = data['targets'].to(DEVICE, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
        
        if _ % 5000 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        loss.backward()
        optimizer.step()


for epoch in range(EPOCHS):
    train(epoch)


test_data = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
test_data.head()


test_set = MultiLabelDataset(test_data, tokenizer, MAX_LEN, new_data=True)
test_loader = DataLoader(test_set, **val_params)


all_test_pred = []

def test(epoch):
    model.eval()
    
    with torch.inference_mode():
    
        for _, data in tqdm(enumerate(test_loader, 0)):


            ids = data['ids'].to(DEVICE, dtype=torch.long)
            mask = data['mask'].to(DEVICE, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(DEVICE, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids)
            probas = torch.sigmoid(outputs)

            all_test_pred.append(probas)
            
            
    return probas
probas = test(model)


all_test_pred = torch.cat(all_test_pred)


submit_df = test_data.copy()
submit_df.drop("comment_text", inplace=True, axis=1)


label_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


for i,name in enumerate(label_columns):

    submit_df[name] = all_test_pred[:, i].cpu()
    submit_df.head()


submit_df.to_csv('submission.csv', index=False)


submit_df.head()

