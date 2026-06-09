# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

from sklearn.utils import shuffle

import torch
import torch.nn as nn

!pip -q install torchmetrics # Colab doesn't come with torchmetrics
from torchmetrics import Accuracy

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Data size
train_df = pd.read_csv('/kaggle/input/playground-series-s4e11/train.csv')

print (f'Number of data samples: {len(train_df)}')


# Shuffle training dataset
train_df = shuffle(train_df)
train_df.info()


# Sample Data
train_df.head()


# id and Name columns are unique, so we will drop it
train_df = train_df.drop(['id', 'Name'], axis=1)


# Check how many null values are there for each field
train_df.isnull().sum()


# Value counts for Working Professionals and Students
train_df['Working Professional or Student'].value_counts()


# For Working Professionals, set Academic Pressure, CGPA, and Study Satisfaction to 0
train_df.loc[((train_df['Working Professional or Student'] == 'Working Professional') & 
              (train_df['Academic Pressure'].isnull())), 'Academic Pressure'] = 0

train_df.loc[((train_df['Working Professional or Student'] == 'Working Professional') & 
              (train_df['CGPA'].isnull())), 'CGPA'] = 0

train_df.loc[((train_df['Working Professional or Student'] == 'Working Professional') & 
              (train_df['Study Satisfaction'].isnull())), 'Study Satisfaction'] = 0


# Check how many null values are there for each field
train_df.isnull().sum()


# For Students, set Work Pressure, Job Satisfaction to 0, and Profession to 'No Profession'
train_df.loc[((train_df['Working Professional or Student'] == 'Student') & 
              (train_df['Work Pressure'].isnull())), 'Work Pressure'] = 0

train_df.loc[((train_df['Working Professional or Student'] == 'Student') & 
              (train_df['Job Satisfaction'].isnull())), 'Job Satisfaction'] = 0

train_df.loc[((train_df['Working Professional or Student'] == 'Student') & 
              (train_df['Profession'].isnull())), 'Profession'] = 'No Profession'


# Check how many null values are there for each field
train_df.isnull().sum()


# Remove the rest of the rows with null values
train_df = train_df.dropna()


# Check how many null values are there for each field
train_df.isnull().sum()


# Data distribution based on the target column
train_df['Depression'].value_counts()


# Continuous columns statistics
train_df.select_dtypes('number').describe().transpose()


# Categorical columns statistics
train_df.select_dtypes('object').describe().transpose()


categorical_columns = ['Gender', 'City', 'Working Professional or Student', 'Profession',
                       'Sleep Duration', 'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?', 
                       'Family History of Mental Illness']

eda_df = pd.concat([
    train_df.select_dtypes('number'),
    pd.get_dummies(train_df[categorical_columns], drop_first=True)
], axis=1)

corr_ser = eda_df.corr()['Depression'].sort_values()[:-1]
corr_ser


# Distribution for the Academic Pressure field separated by Depression
sns.countplot(train_df, x='Academic Pressure', hue='Depression')
plt.show()

# As the Academic Pressure increases, the probability of Depression is high


# Distribution for the Working Professional field separated by Depression
sns.countplot(train_df, x='Working Professional or Student', hue='Depression')
plt.show()

# The probability of Depression is high for Students rather than working professionals


# Distribution for the Age field separated by Segmentation
sns.histplot(train_df, x='Age', hue='Depression', multiple='stack')
plt.show()

# Individuals with a lower age have a high probability of Depression


# Distribution for the Job Satisfaction field separated by Depression
sns.countplot(train_df, x='Job Satisfaction', hue='Depression')
plt.show()

# The probability of Depression is high for working professionals with lower job satisfaction


# Separate categorical, continuous, and target columns
category_columns = ['Gender', 'City', 'Working Professional or Student', 'Profession',
                    'Sleep Duration', 'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?', 
                    'Family History of Mental Illness']
continuous_columns = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 
                      'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours', 'Financial Stress']
target_column = ['Depression']

print(f'There are total {len(category_columns)} category columns')
print(f'There are total {len(continuous_columns)} continuous columns')
print(f'There are total {len(target_column)} target column')


# Convert category columns from object to category
for col in category_columns:
    train_df[col] = train_df[col].astype('category')
    
train_df.info()


# Create tuples with category and embedding size
cat_szs = [len(train_df[col].cat.categories) for col in category_columns]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
emb_szs


# Category columns
cats = [train_df[col].cat.codes.values for col in category_columns]
cats = np.stack(cats, 1)
cats = torch.tensor(cats, dtype=torch.int64)

cats, cats.dtype


# Continuous columns
conts = [train_df[col].values for col in continuous_columns]
conts = np.stack(conts, 1)
conts = torch.tensor(conts, dtype=torch.float32)

conts, conts.dtype


# Target column
y = torch.tensor(train_df['Depression'].values).unsqueeze(dim=1)


# Train and eval sets
batch_size = 131815
eval_size = 20000

cats_train = cats[:batch_size-eval_size]
cats_eval = cats[batch_size-eval_size:batch_size]
conts_train = conts[:batch_size-eval_size]
conts_eval = conts[batch_size-eval_size:batch_size]
y_train = y[:batch_size-eval_size]
y_eval = y[batch_size-eval_size:batch_size]


# Define a model
class TabularModel(nn.Module):
    
    def __init__(self, emb_szs, n_cont, out_sz, layers, p, emb_p):
        super().__init__()
        self.emb_drop = nn.Dropout(emb_p)
        self.batch_norm = nn.BatchNorm1d(n_cont)
        self.embeds = nn.ModuleList([
            nn.Embedding(unique_category_size, emb_dim) for (unique_category_size, emb_dim) in emb_szs
        ])
        
        input_size = n_cont + sum((emb_dim for unique_category_size, emb_dim in emb_szs))
        
        layerlist = []
        for i in layers:
            layerlist.append(nn.Linear(input_size, i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            input_size = i
            
        layerlist.append(nn.Linear(layers[-1], out_sz))
        
        self.nn_layers = nn.Sequential(*layerlist)
        
    def forward(self, x_cat, x_conts):
        
        x_conts = self.batch_norm(x_conts)
        
        embeddings = []
        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)
        
        x = torch.cat([x, x_conts], 1)
        
        return self.nn_layers(x)


# Instantiate model
tabular_model = TabularModel(emb_szs, len(continuous_columns), 1, [200, 100], p=0.1, emb_p=0.1)
tabular_model


# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(tabular_model.parameters(), lr=0.01)


# Accuracy function
acc_fn = Accuracy(task="multiclass", num_classes=2)
acc_fn


# Training loop

epochs = 100
train_losses = []
train_accs = []
eval_losses = []
eval_accs = []

for i in range(epochs):
    
    tabular_model.train()
    
    y_pred_logits = tabular_model(cats_train, conts_train)
    y_pred_probs = torch.sigmoid(y_pred_logits)
    y_preds = torch.round(y_pred_probs)

    train_loss = criterion(y_pred_logits, y_train.float())
    train_losses.append(train_loss.item())
    train_acc = acc_fn(y_preds.long(), y_train.long())
    train_accs.append(train_acc.item())

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    tabular_model.eval()
    with torch.inference_mode():

        y_pred_eval_logits = tabular_model(cats_eval, conts_eval)
        y_pred_eval_probs = torch.sigmoid(y_pred_eval_logits)
        y_eval_preds = torch.round(y_pred_eval_probs)
    
        eval_loss = criterion(y_pred_eval_logits, y_eval.float())
        eval_losses.append(eval_loss.item())
        eval_acc = acc_fn(y_eval_preds.long(), y_eval.long())
        eval_accs.append(eval_acc.item())

    if i%25 == 0:
        print(f'epoch: {i+1:4} Train_Loss: {train_loss.item():4.4f} Train_Acc: {train_acc.item():4.4f} Eval_Loss: {eval_loss.item():4.4f} Eval_Acc: {eval_acc.item():4.4f}')

print(f'epoch: {i+1:4} Train_Loss: {train_loss.item():4.4f} Train_Acc: {train_acc.item():4.4f} Eval_Loss: {eval_loss.item():4.4f} Eval_Acc: {eval_acc.item():4.4f}')


# Train/Eval loss vs epoch
plt.plot(range(epochs), train_losses, label='Training Loss')
plt.plot(range(epochs), eval_losses, label='Eval Loss')
plt.xlabel('epoch')
plt.legend()
plt.show()


# Train/Eval Accuracy vs epoch
plt.plot(range(epochs), train_accs, label='Training Accuracy')
plt.plot(range(epochs), eval_accs, label='Eval Accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()


# Load data and apply data transformations as training data
test_df = pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv')
print (f'Number of data samples: {len(test_df)}')

# For Working Professionals, set Academic Pressure, CGPA, and Study Satisfaction to 0
test_df.loc[((test_df['Working Professional or Student'] == 'Working Professional') & 
              (test_df['Academic Pressure'].isnull())), 'Academic Pressure'] = 0

test_df.loc[((test_df['Working Professional or Student'] == 'Working Professional') & 
              (test_df['CGPA'].isnull())), 'CGPA'] = 0

test_df.loc[((test_df['Working Professional or Student'] == 'Working Professional') & 
              (test_df['Study Satisfaction'].isnull())), 'Study Satisfaction'] = 0
			  
			  
# For Students, set Work Pressure, Job Satisfaction to 0, and Profession to 'No Profession'
test_df.loc[((test_df['Working Professional or Student'] == 'Student') & 
              (test_df['Work Pressure'].isnull())), 'Work Pressure'] = 0

test_df.loc[((test_df['Working Professional or Student'] == 'Student') & 
              (test_df['Job Satisfaction'].isnull())), 'Job Satisfaction'] = 0

test_df.loc[((test_df['Working Professional or Student'] == 'Student') & 
              (test_df['Profession'].isnull())), 'Profession'] = 'No Profession'

# Fill with default values
test_df.loc[test_df['Profession'].isnull(), 'Profession'] = 'No Profession'
test_df = test_df.fillna(0)
print (f'Number of data samples after feature engineering: {len(test_df)}')


# Convert category columns from object to category
for col in category_columns:
    test_df[col] = test_df[col].astype('category')

# Category columns
cats = [test_df[col].cat.codes.values for col in category_columns]
cats = np.stack(cats, 1)
cats = torch.tensor(cats, dtype=torch.int64)

# Continuous columns
conts = [test_df[col].values for col in continuous_columns]
conts = np.stack(conts, 1)
conts = torch.tensor(conts, dtype=torch.float32)


# Predictions
with torch.inference_mode():

    y_pred_test_logits = tabular_model(cats, conts)
    y_pred_test_probs = torch.sigmoid(y_pred_test_logits)
    y_test_preds = torch.round(y_pred_test_probs)

print(y_test_preds.int().shape)


# Submission
submission = pd.DataFrame()
submission['id'] = test_df['id']
submission['Depression'] = y_test_preds.numpy()
submission.to_csv('submission.csv', index=False)

