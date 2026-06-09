# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from collections import Counter




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_FOLDER = '/kaggle/input/arc-prize-2025'
cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=10)
class ARCDataset(Dataset):
    def __init__(self, file_path, mode='train', dim={}, keys={}):
        self.mode = mode
        self.dim = {}
        self.keys = {}
        self.data = self.extract_file(file_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        task = self.data[idx]
        return task[1], task[2]
    
    def extract_file(self, file_path):
        data = []
        if isinstance(file_path, list):
            for path in file_path:
                d = self.load_data(path)
                data = [*data, *d]
        else:
            data = self.load_data(file_path)
        return data
    
    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return self.parse_data(data)
    
    def parse_data(self, data):
        ndata = []
        for key in data.keys():
            task = data[key][self.mode]
            dim_inp = []
            dim_out = []
            for i in range(len(task)):
#                 index = list(self.dim).index(key) if key in self.dim else len(self.dim)
                input_tensor, i_shape = self.expand_data(task[i]['input'])
                output_tensor, o_shape = self.expand_data(task[i]['output']) if self.mode == 'train' else [[], [0,0]]
                input_tensor = torch.tensor(input_tensor)
                output_tensor = torch.tensor(output_tensor)
                ndata.append([key, input_tensor, output_tensor])
                if self.mode == 'train':
                    dim_inp.append(tuple(i_shape))
                    dim_out.append(tuple(o_shape))
            if self.mode == 'train':
                counter_inp = Counter(dim_inp)
                counter_out = Counter(dim_out)
                self.dim[key] = {
                    "inp_dim": list(counter_inp.most_common(1)[0][0]),
                    "out_dim": list(counter_out.most_common(1)[0][0]),
                }
        return ndata

    def expand_data(self, data):
        # convert from matrix to 1D array
        data = np.array(data)
        shape = data.shape
        data = data.flatten()
        ndata = []
        while len(ndata) < 100:
            ndata = [*ndata, *data]
        data = np.array(ndata[:100]) / 10
        return data, shape
    
    def set_dataset(self):
        keys = [[d[0]] for d in self.data]
        ndata = []
        for i in range(len(self.data)):
            d = self.data[i]
            if self.mode == 'train':
                if d[0] not in self.keys:
                    self.keys[d[0]] = len(self.keys)
                encoded = self.keys[d[0]]
            else:
                encoded = self.keys[d[0]] if d[0] in self.keys else -1
            ndata.append([
                keys[i][0],
                torch.tensor([torch.tensor(encoded), *d[1]]),
#                 torch.tensor(self.data[i][1]),
                d[2]
            ])
        self.data = ndata
file_list = [
    f'{BASE_FOLDER}/arc-agi_training_challenges.json',
    f'{BASE_FOLDER}/arc-agi_evaluation_challenges.json',
    f'{BASE_FOLDER}/arc-agi_test_challenges.json'

]

train_dataset = ARCDataset(file_list)
train_dataset.set_dataset()
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
Dimension = train_dataset.dim
Keys = train_dataset.keys
class AutoEncoder(nn.Module):
    def __init__(self, input_size, output_size, latent_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, latent_size),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)
class LSTM(nn.Module):
    def __init__(self, INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)
        self.linear = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, input_data):
        lstm_out, _ = self.lstm(input_data)
        predictions = self.linear(lstm_out)
        return predictions
class Training:
    def __init__(self, model, train_loader, criterion, optimizer):
        self.model = model 
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
    
    def _train_one(self, model, data, criterion, optimizer):
        model.train()
        input_data, target = data
        input_data, target = input_data.to(device).float(), target.to(device).float()

        output = model(input_data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
    
    def _train_loop(self, model, train_loader, criterion, optimizer):
        model.train()
        history = {'train_loss': []}
        loss = 1
        epoch = 0

        while True:
            epoch += 1
            train_loss = 0
            for data in train_loader:
                ls = self._train_one(model, data, criterion, optimizer)
                train_loss += ls
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)

            print(f'\rEpoch : {epoch}, Loss: {train_loss:.5f}, Lowest Loss: {loss:.5f}', end='')

            if train_loss < loss:
                loss = train_loss
                torch.save(model.state_dict(), 'model.pth')
            if train_loss < 0.01:
                break
            
            # Waste too much time and loss is ok
            if epoch > 100 and train_loss < 0.01:
                break
            # F
            if epoch > 200:
                break


        return history
    
    def train(self):
        history = self._train_loop(self.model, self.train_loader, self.criterion, self.optimizer)
        self._plot_loss(history)
        
    def _plot_loss(self, history):
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], 'o-', label='train_loss')
        plt.legend()
        plt.title('Loss Plot')
        plt.show()
IN_DIM = 100 + 1
print(IN_DIM)
OUT_DIM = 100
LATENT_DIM = 1024
model = LSTM(IN_DIM, OUT_DIM, LATENT_DIM).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
training = Training(model, train_loader, criterion, optimizer)
training.train()


model.load_state_dict(torch.load('model.pth'))


class Prediction:
    def __init__(self, model, data):
        self.model = model
        self.data = data
    
    def predict(self, model, data):
        model.eval()
        input_data, target = data
        input_data, target = torch.tensor(input_data).to(device).float(), torch.tensor(target).to(device).float()

        with torch.no_grad():
            input_data = input_data.unsqueeze(0)
            output = model(input_data)

        return output[0]
    
    def remove_tail_zeros(self, key, data, mode = 'inp'):
        data = data.cpu().numpy()
        data = data.astype(int)
        ndata = []
        for i in range(len(data)):
            ndata.append(data[i] - 1 if data[i] > 0 else 0)
        data = np.array(ndata)
        dim = Dimension[key]['inp_dim' if mode == 'inp' else 'out_dim']
        return data[:dim[0] * dim[1]].reshape(dim[0], dim[1])
    
    def plot_result(self, key, inputs, target, output):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(self.remove_tail_zeros(key, inputs * 9, 'inp'), cmap=cmap, norm=norm)
        ax[0].set_title('Input')
        ax[1].imshow(self.remove_tail_zeros(key, target * 9, 'out'), cmap=cmap, norm=norm)
        ax[1].set_title('Target')
        ax[2].imshow(self.remove_tail_zeros(key, torch.round(output * 9), 'out'), cmap=cmap, norm=norm)
        ax[2].set_title('Output')
        plt.show()
        
    def sanity(self,idx):
        data = self.data.data[idx]
        output = self.predict(self.model, [data[1], data[2]])
        self.plot_result(data[0],data[1][1:], data[2], output)
        
    def pred_all(self, show_step = 10):
        pred_data = {}
        model = self.model
        i = 0
        for data in tqdm(self.data.data):
            i += 1
            output1 = self.predict(model, [data[1], data[2]])
            output2 = self.predict(model, [data[1], data[2]])
            if data[0] not in pred_data:
                pred_data[data[0]] = []
            output1 = self.remove_tail_zeros(data[0], torch.round(output1 * 9), 'out')
            output2 = self.remove_tail_zeros(data[0], torch.round(output2 * 9), 'out')
            pred_data[data[0]].append({
                "attempt_1": output1.tolist(),
                "attempt_2": output2.tolist()
            })
            if i % show_step == 0:
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(self.remove_tail_zeros(data[0], torch.tensor(data[1][1:]*9), 'out'), cmap=cmap, norm=norm)
                ax[0].set_title('Input')
                ax[1].imshow(output1, cmap=cmap, norm=norm)
                ax[1].set_title('Output 1')
                ax[2].imshow(output2, cmap=cmap, norm=norm)
                ax[2].set_title('Output 2')
                plt.show()
        return pred_data


sanity = Prediction(model, train_dataset)


sanity.sanity(1)
sanity.sanity(2)
sanity.sanity(4)
sanity.sanity(5)
sanity.sanity(6)


test_dataset = ARCDataset(f'{BASE_FOLDER}/arc-agi_test_challenges.json', 'test', dim=Dimension, keys=Keys)
test_dataset.set_dataset()


pred = Prediction(model, test_dataset)


base_path='/kaggle/input/arc-prize-2024/'
# Loading JSON data
def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


def plot_task(task, task_solutions, i, t):
    """    Plots the train and test pairs of a specified task,
    using same color scheme as the ARC app    """    
    
    num_train = len(task['train'])
    num_test  = len(task['test'])
    
    w=num_train+num_test
    fig, axs  = plt.subplots(2, w, figsize=(3*w ,3*2))
    plt.suptitle(f'Set #{i}, {t}:', fontsize=20, fontweight='bold', y=1)
    
    for j in range(num_train):     
        plot_one(axs[0, j], j,'train', 'input')
        plot_one(axs[1, j], j,'train', 'output')        
    
    plot_one(axs[0, j+1], 0, 'test', 'input')

    cmap = colors.ListedColormap(['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
                                  '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    answer = task_solutions
    input_matrix = answer
    
    axs[1, j+1].imshow(input_matrix, cmap=cmap, norm=norm)
    axs[1, j+1].grid(True, which = 'both',color = 'lightgrey', linewidth = 0.5)
    axs[1, j+1].set_yticks([x-0.5 for x in range(1 + len(input_matrix))])
    axs[1, j+1].set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])     
    axs[1, j+1].set_xticklabels([])
    axs[1, j+1].set_yticklabels([])
    axs[1, j+1].set_title('Test output')

    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('black')  # substitute 'k' for black
    fig.patch.set_facecolor('#dddddd')
   
    plt.tight_layout()
    plt.show()  
    
    print()
    print()
    
    
def plot_one(ax, i, train_or_test, input_or_output):
    cmap = colors.ListedColormap(['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
                                  '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 0.5)
    
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])     
    ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])   
    ax.set_title(train_or_test + ' ' + input_or_output)


submission = pd.read_csv('/kaggle/input/arc-prize-2025/sample_submission.json')
submission.to_csv('sample_submission.json', index=False)

