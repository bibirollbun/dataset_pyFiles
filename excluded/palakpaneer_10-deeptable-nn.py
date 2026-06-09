import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from category_encoders import TargetEncoder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from itertools import combinations


import warnings 
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
submit_df = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


# Deal with big test_df['Episode_Length_minutes']
#test_df['Episode_Length_minutes'].sort_values(ascending=False)
max_epi_len = test_df[test_df['Episode_Length_minutes'] <= 2000]['Episode_Length_minutes'].max()
test_df.loc[test_df['Episode_Length_minutes'] >= 2000, 'Episode_Length_minutes'] = max_epi_len


# Deal with float of number od ads
#train_df['Number_of_Ads'].value_counts()
#train_df['Number_of_Ads'].value_counts()
for df in [train_df, test_df]:
    mode_num_ads = df['Number_of_Ads'].mode()[0]
    df.loc[df['Number_of_Ads'] >=13, 'Number_of_Ads'] = mode_num_ads


# imputing missing data
for df in [train_df, test_df]:
    median_Episode_Length_minutes = df['Episode_Length_minutes'].median()
    median_Guest_Popularity_percentage = df['Guest_Popularity_percentage'].median()
    median_Number_of_Ads = df['Number_of_Ads'].median()

    df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(median_Episode_Length_minutes)
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(median_Guest_Popularity_percentage)
    df['Number_of_Ads'] = df['Number_of_Ads'].fillna(median_Number_of_Ads)


# Remove id
train_df = train_df.drop(columns=['id'])
test_df = test_df.drop(columns=['id'])


# Check
print(f'train_isnull {train_df.isnull().sum()}')
print(f'train_isnull {test_df.isnull().sum()}')


# Mapping dictionaries
def feature_eng(df):
    podc_dict = {'Mystery Matters': 0, 'Joke Junction': 1, 'Study Sessions': 2, 'Digital Digest': 3, 'Mind & Body': 4, 'Fitness First': 5, 'Criminal Minds': 6, 'News Roundup': 7, 'Daily Digest': 8, 'Music Matters': 9, 'Sports Central': 10, 'Melody Mix': 11, 'Game Day': 12, 'Gadget Geek': 13, 'Global News': 14, 'Tech Talks': 15, 'Sport Spot': 16, 'Funny Folks': 17, 'Sports Weekly': 18, 'Business Briefs': 19, 'Tech Trends': 20, 'Innovators': 21, 'Health Hour': 22, 'Comedy Corner': 23, 'Sound Waves': 24, 'Brain Boost': 25, "Athlete's Arena": 26, 'Wellness Wave': 27, 'Style Guide': 28, 'World Watch': 29, 'Humor Hub': 30, 'Money Matters': 31, 'Healthy Living': 32, 'Home & Living': 33, 'Educational Nuggets': 34, 'Market Masters': 35, 'Learning Lab': 36, 'Lifestyle Lounge': 37, 'Crime Chronicles': 38, 'Detective Diaries': 39, 'Life Lessons': 40, 'Current Affairs': 41, 'Finance Focus': 42, 'Laugh Line': 43, 'True Crime Stories': 44, 'Business Insights': 45, 'Fashion Forward': 46, 'Tune Time': 47}
    genr_dict = {'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 'Health': 4, 'News': 5, 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9}
    week_dict = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    time_dict = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    sent_dict = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    
    df['Episode_Num'] = df['Episode_Title'].str[8:].astype('category')
    
    df['Genre'] = df['Genre'].replace(genr_dict)
    df['Podcast_Name'] = df['Podcast_Name'].replace(podc_dict)
    df['Publication_Day'] = df['Publication_Day'].replace(week_dict)
    df['Publication_Time'] = df['Publication_Time'].replace(time_dict)
    df['Episode_Sentiment'] = df['Episode_Sentiment'].replace(sent_dict)
    
    df['Genre'] = df['Genre'].astype('category')
    df['Podcast_Name'] = df['Podcast_Name'].astype('category')
    df['Publication_Day'] = df['Publication_Day'].astype('category')
    df['Publication_Time'] = df['Publication_Time'].astype('category')
    df['Episode_Sentiment'] = df['Episode_Sentiment'].astype('category')

    df['Has_Ads'] = (df['Number_of_Ads'] > 0).astype('int8')
    df['Is_Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype('int8')
    
    df = df.drop(columns=['Episode_Title'])
    return df


# Apply
train_df = feature_eng(train_df)
test_df = feature_eng(test_df)


# Check
train_df.dtypes


# Feature Combinations
encode_columns = ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment', 'Publication_Day', 'Publication_Time', 'Guest_Popularity_percentage', 'Genre']
#pair_size = [2, 3, 4]
pair_size = [2]

for r in pair_size:
    for cols in tqdm(list(combinations(encode_columns, r))):
        new_col_name = '_'.join(cols)
        
        train_df[new_col_name] = train_df[list(cols)].astype(str).agg('_'.join, axis=1)
        train_df[new_col_name] = train_df[new_col_name].astype('category')
        
        test_df[new_col_name] = test_df[list(cols)].astype(str).agg('_'.join, axis=1)
        test_df[new_col_name] = test_df[new_col_name].astype('category')


# Check
train_df.head()


# Check
test_df.head()


# Check
train_df.dtypes


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


torch.manual_seed(86)
num_batches = 100


# Make Dataset
class TrainDataset(Dataset):
    def __init__(self, df):
        self.X = df.drop('Listening_Time_minutes', axis=1).values
        self.y = df['Listening_Time_minutes'].values
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


train_dataset = TrainDataset(train_df)
# train_dataloader = DataLoader(train_dataset, batch_size=num_batches, shuffle=True)


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Sequential(
         nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        output = self.classifier(x)
        return output


# Setting
n_splits = 5
kf = KFold(n_splits = n_splits)

num_epochs = 10
losses = []
rmses = []

target_col = 'Listening_Time_minutes'


# KFold
for fold, (train_index, valid_index) in enumerate(kf.split(train_df)):
    print(f'Fold: {fold+1}/{n_splits}')
    
    
    '''Data split'''
    train_fold_df = train_df.iloc[train_index].copy()
    valid_fold_df = train_df.iloc[valid_index].copy()
    
    
    '''Target encoder using train_subset'''
    encoded_columns = train_fold_df.select_dtypes(include=['category']).columns.tolist()
    
    encoder = TargetEncoder(cols=encoded_columns)

    train_fold_df[encoded_columns] = encoder.fit_transform(train_fold_df[encoded_columns], train_fold_df[target_col])
    valid_fold_df[encoded_columns] = encoder.transform(valid_fold_df[encoded_columns])
    
    
    '''Scaler'''
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(train_fold_df[[col for col in train_fold_df.columns if col != target_col]])
    y_train = train_fold_df[target_col]
    
    X_val = scaler.transform(valid_fold_df[[col for col in valid_fold_df.columns if col != target_col]])
    y_val = valid_fold_df[target_col]
    
    
    '''Make dataloader'''
    # train
    # from pandas to tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # yは2Dに整形

    # from tensor to TensorDataset
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    # from TensorDataset to DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # val
    # from pandas to tensor
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)  # yは2Dに整形

    # from tensor to TensorDataset
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # from TensorDataset to DataLoader
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        

    '''Initialise the model'''
    input_dim = train_df.drop('Listening_Time_minutes', axis=1).shape[1]
    model = MLP(input_dim = input_dim)
    
    
    '''Set loss function and optimizer'''
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    
    '''Training loop'''
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_rmse = 0.0
        
        for X_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            
        '''Val loop'''
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_dataloader:
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                running_loss += loss.item()
              
                rmse = torch.sqrt(loss)
                running_rmse += rmse.item()
                running_loss /= len(val_dataloader)
            
        running_loss /= len(val_dataloader)
        running_rmse /= len(val_dataloader)
            
        print('epoch: {}, loss: {}, rmse: {}'.format(epoch, running_loss, running_rmse))
        losses.append(running_loss)
        rmses.append(running_rmse)


plt.plot(losses)


plt.plot(rmses)

