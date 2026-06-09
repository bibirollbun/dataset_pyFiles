import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device


training = pd.read_csv('/kaggle/input/facial-keypoints-detection/training.zip')
id_lookup_table = pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')
test = pd.read_csv('/kaggle/input/facial-keypoints-detection/test.zip')
SampleSubmission = pd.read_csv('/kaggle/input/facial-keypoints-detection/SampleSubmission.csv')


training.head(2)


img_data = training['Image'].apply(lambda x : x.split(' ')).to_list() # Convert to array
img_data = np.array(img_data, dtype='float32')
# Normalize
img_datanorm = img_data/np.max(img_data) 
# reshape from 9216 to (data_size, channel, height, width)
img_datanorm = img_datanorm.reshape(img_datanorm.shape[0], 1, 96, 96) 

labels = np.array(list(training[training.columns[0:-1]].values), dtype='float32')

# fill null values with mean (even a single null can cause entire loss function to become 0)
coord_means = np.nanmean(labels, axis=0)
for n in range(30):
    labels[:,n] = np.nan_to_num(labels[:,n], nan=coord_means[n])


data_t = torch.tensor(img_datanorm)
label_t = torch.tensor(labels)

train_data, val_data, train_label, val_label = train_test_split(data_t, label_t, test_size=0.1, random_state=32)

train_data = TensorDataset(train_data, train_label)
val_data = TensorDataset(val_data, val_label)

batchsize = 16
train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size=val_data.tensors[0].shape[0])


class cnn_net(nn.Module):
    def __init__(self):
        super().__init__()

        # initial size (batch_size, channel, height, width) = (16, 1, 96, 96)
        # output = (Input+2XPadding−Kernel)/stride + 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, 
                               kernel_size=5) # (16,1,96,96) to (16,4,92,92)
        # pooling1 (16,4,46,46)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=64, 
                               kernel_size=3) # (16,4,46,46) to (16,64,44,44)
        # pooling2 (16,64,22,22)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, 
                               kernel_size=3) # (16,64,22,22) to (16,128,20,20)
        # pooling3 (16,128,10,10)

        
        # Prepare output before feeding it to linear layer as input
        expected_size = expected_size = np.floor(((10+2*0)-1)/1 +1)
        expected_size = 128*int(expected_size**2)

        self.fc1 = nn.Linear(expected_size, 250)
        self.fc2 = nn.Linear(250, 128)
        # final output should be 30, same as number of labels
        self.out = nn.Linear(128, 30)

    
    def forward(self, x):

        # comment all print statements out during training
        # print('0 : ', x.shape)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # print('1 : ', x.shape)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # print('2 : ', x.shape)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        # print('3 : ', x.shape)

        # reshape for linear
        n_units = x.shape.numel()/x.shape[0]
        x = x.view(-1, int(n_units))
        # print('4 : ', x.shape)

        x = F.relu(self.fc1(x))
        # print('5 : ', x.shape)
        x = F.relu(self.fc2(x))
        # print('6 : ', x.shape)

        return self.out(x)
        


# test model
net = cnn_net()
y = net(torch.randn(16,1,96,96))
print(y.shape)


net = cnn_net()

loss_func = nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


# test with one batch
x,y = next(iter(train_loader))
x=x.to(device)
y=y.to(device)
net.to(device)

yhat = net(x)
loss = loss_func(yhat, y)
print(loss.item())


epoch = 50

train_loss = []
val_loss = []

net.to(device)

for i in tqdm(range(epoch)):

    net.train()

    # batchacc = []
    batchloss = []
    for x,y in train_loader:

        x = x.to(device)
        y = y.to(device)
        
        yhat = net(x)

        yhat = yhat.cpu()
        y = y.cpu()

        loss = loss_func(yhat, y)
        batchloss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss.append(np.mean(batchloss))

    net.eval()
    x,y = next(iter(val_loader))
    x = x.to(device)
    y = y.to(device)
    yhat = net(x)
    yhat = yhat.cpu()
    y = y.cpu()
    loss = loss_func(yhat, y)
    val_loss.append(loss.item())


plt.plot(train_loss, label = "train loss")
plt.plot(val_loss, label = "val loss")
plt.legend()
plt.show()


test.head(2)


test_img_data = test['Image'].apply(lambda x : x.split(' ')).to_list()
test_img_data = np.array(test_img_data, dtype='float32')

test_img_datanorm = test_img_data/np.max(test_img_data)
test_img_datanorm = test_img_datanorm.reshape(test_img_datanorm.shape[0], 1, 96, 96)

test_img_data_t = torch.tensor(test_img_datanorm)
test_img_data_t = test_img_data_t.to(device)

preds = net(test_img_data_t)
preds = preds.cpu()
predictions = preds.detach().reshape(-1)


raw_fearure_names = training.columns[0:-1]

FeatureName = np.tile(raw_fearure_names, (len(preds),1)).reshape(-1)
ImageId = np.arange(1,len(preds)+1).repeat(30)

pred_test_final_df = pd.DataFrame({'ImageId': ImageId
                                  ,'FeatureName': FeatureName
                                  ,'Location': predictions})


SampleSubmission['Location'] = pred_test_final_df['Location']


print(SampleSubmission['Location'].min())
print(SampleSubmission['Location'].max())

# set feature above threhold of 96 to 96 
threhold_mask = SampleSubmission['Location'] > 96
SampleSubmission.loc[threhold_mask, 'Location'] = 96

print(SampleSubmission['Location'].min())
print(SampleSubmission['Location'].max())


SampleSubmission.to_csv('submission_format.csv', index=False)
print("Your submission was successfully saved!")


SampleSubmission




