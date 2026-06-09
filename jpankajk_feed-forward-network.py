import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')


df


pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import numpy as np
class RainFallDatasetTrain(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.df_x = self.df[['pressure','maxtemp','temparature','mintemp','dewpoint','humidity','cloud','sunshine','winddirection','windspeed']]
        self.df_y = self.df[['rainfall']]
        
        self.data = torch.tensor(self.df_x.values, dtype=torch.float32)
        self.targets = torch.tensor(self.df_y.values, dtype=torch.float32).view(-1, 1)
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        return sample, target
        
        data = self.df_x.iloc[idx]
        data = np.array(data, dtype=float)

        label = self.df_y.iloc[idx]
        label = np.array(label, dtype=float)
        
        return data, label




training_data = RainFallDatasetTrain('/kaggle/input/playground-series-s5e3/train.csv')

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 1)

    def forward(self, input):
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        f3 = F.relu(self.fc3(f2))
        f4 = F.relu(self.fc4(f3))
        f5 = F.relu(self.fc5(f4))
        output = self.fc6(f5)
        m = nn.Sigmoid()
        return m(output) # nn.Sigmoid(output)
net = Net()
print(net)


iter = 0
learning_rate=0.001
num_epochs = 100
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_dataloader):
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = net(inputs)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        
        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        if iter % 100 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for data, labels in train_dataloader:
                outputs = net(data)
                correct += (outputs == labels).sum()
                total += len(labels)
            accuracy = 100 * correct / total
            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))


class RainFallDatasetTest(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.df = self.df.fillna(0)
        
        self.df_x = self.df[['pressure','maxtemp','temparature','mintemp','dewpoint','humidity','cloud','sunshine','winddirection','windspeed']]
        self.id = self.df[['id']]
        
        self.data = torch.tensor(self.df_x.values, dtype=torch.float32)
        self.id = torch.tensor(self.id.values, dtype=torch.float32).view(-1, 1)
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.id[idx]
        return sample, target
        
        data = self.df_x.iloc[idx]
        data = np.array(data, dtype=float)

        id = self.id.iloc[idx]
        id = np.array(id, dtype=float)
        
        return data, id


test_data = RainFallDatasetTest('/kaggle/input/playground-series-s5e3/test.csv')
test_dataloader = DataLoader(test_data, batch_size=730)

out, ids_arr = [], []
for i, (inputs, ids) in enumerate(test_dataloader):
    output = net(inputs)
    out = torch.flatten(output).detach().numpy()
    ids_arr = torch.flatten(ids).detach().numpy()
ids_arr = pd.DataFrame( ids_arr, columns=["id"])
pred_df = pd.DataFrame( out, columns=["rainfall"])
pred_df = pd.concat([ids_arr, pred_df ], axis=1)
pred_df['id'] = pred_df['id'].astype(int)

pred_df


pred_df.to_csv(path_or_buf="submission.csv", index=False)




