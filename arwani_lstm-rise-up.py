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


import json
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.colors as colors


CMAP = colors.ListedColormap(
    ['#ffffff','#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
NORM = colors.Normalize(vmin=-1, vmax=10)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


BASE_DIR = '/kaggle/input/arc-prize-2025'
SUB_TARGET = './submission.json'



def generate_submission(count=1):
    sub = []
    for i in range(count):
        sub.append({
            "attempt_1": [[0]],
            "attempt_2": [[0]]
        })
    return sub

def create_submission():
    with open(f'{BASE_DIR}/arc-agi_test_challenges.json','r') as file:
        sub = json.load(file)
    subkey = list(sub.keys())
    submission = {}
    for key in subkey:
        submission[key] = generate_submission(len(sub[key]['test']))
    json_object = json.dumps(submission, indent=4)
 
    # Writing to sample.json
    with open(SUB_TARGET, "w") as outfile:
        outfile.write(json_object)

create_submission()


class Dataset:
    def __init__(self, path):
        self.path = path
        self.data = None
        self.key = []
        self.load_data()

    def load_data(self):
        try:
            with open(self.path, 'r') as file:
                self.data = json.load(file)
                self.key = list(self.data.keys())
        except FileNotFoundError:
            print(f"File {self.path} not found.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from the file {self.path}.")




class ARCDataset:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.inp_dim = (0,0)
        self.out_dim = (0,0)
        self.input_length = 0
        self.output_length = 0

        self.train_dataset = []
        self.test_dataset = []
        self.get_dim()
        self.process_get_data()

    def get_dim(self):
        temp_input = (0,0)
        temp_output = (0,0)
        for data in self.train_data:
            inp_dim = np.array(data['input']).shape
            out_dim = np.array(data['output']).shape
            temp_input = (max(temp_input[0], inp_dim[0]), max(temp_input[1], inp_dim[1]))
            temp_output = (max(temp_output[0], out_dim[0]), max(temp_output[1], out_dim[1]))
        self.inp_dim = temp_input
        self.out_dim = temp_output

    def get_samples(self, total=10, exclude=0):
        candidate = list(range(total))
        candidate.remove(exclude)
        return random.sample(candidate, total - 1)

    def flat(self, data):
        return np.array(data).flatten()
    
    def process_data(self, data):
        temp = []
        for d in data:
            temp.append([self.flat(d['input']), self.flat(d['output']) if 'output' in d else None])
        return temp
    
    def pad_data(self, data, max_len, pos='start'):
        # if pos == 'start':
        #     return torch.tensor([*np.zeros(max_len - len(data)), *data]) if len(data) < max_len else torch.tensor(data[-max_len:])
        return torch.tensor([*data, *np.zeros(max_len - len(data))]) if len(data) < max_len else torch.tensor(data[:max_len])

    def process_get_data(self):
        temp_train = self.process_data(self.train_data)
        temp_test = self.process_data(self.test_data)

        temp_dataset_train = []
        temp_dataset_test = []
        temp_dataset_test_input = []
        max_len = 0
        for i, data in enumerate(temp_train):
            sample = self.get_samples(len(temp_train), i)
            max_len = (np.dot(*self.inp_dim) + 1) * (len(sample) + 1) + (np.dot(*self.out_dim) + 1) * len(sample)
            temp_data = []
            for j in sample:
                temp_data = [*temp_data, *temp_train[j][0], 10, *temp_train[j][1], 11]
            temp_dataset_test_input.append(temp_data)
            temp_train_data = [*temp_data, *data[0], 10]
            temp_train_data = self.pad_data(temp_train_data, max_len, 'start')
            temp_dataset_train.append([temp_train_data, self.pad_data(data[1], np.dot(*self.out_dim), 'end')])

        for i, data in enumerate(temp_test):
            test_case = []
            for inp in temp_dataset_test_input:
                temp_data = [*inp, *data[0], 10]
                temp_data = self.pad_data(temp_data, max_len, 'start')
                test_case.append(temp_data)
                if len(test_case) >= 2:
                    break
            temp_dataset_test.append(test_case)

        self.train_dataset = DataLoader(temp_dataset_train, batch_size=len(self.train_data) // 2, shuffle=True)
        self.test_dataset = temp_dataset_test
        self.input_length = max_len
        self.output_length = np.dot(*self.out_dim)
        
        print(f"Input dimension: {self.inp_dim}")
        print(f"Output dimension: {self.out_dim}")
        print(f"Input length: {self.input_length}")
        print(f"Output length: {self.output_length}")





class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            # nn.ReLU(),
            nn.Hardtanh(-10, 10),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Hardtanh(-5, 5),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


class Training:
    def __init__(self, model, train_loader, criterion, optimizer, device, loss = 100):
        self.model = model 
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.loss = loss
    
    def _train_one(self, model, data, criterion, optimizer):
        # declare model for train mode
        model.train()
        
        # data is on cpu, transfer to gpu if gpu is available
        input_data, target = data
        input_data, target = input_data.to(self.device).float(), target.to(self.device).float()

        # get the output
        output = model(input_data)
        
        # calculate the loss
        loss = criterion(output, target)
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
    
    def _train_loop(self, model, train_loader, criterion, optimizer):
        model.train()
        history = {'train_loss': []}
        loss = self.loss
        epoch = 0
        patient = 0
        while True:
            epoch += 1
            train_loss = 0
            for data in train_loader:
                ls = self._train_one(model, data, criterion, optimizer)
                train_loss += ls
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)

            print(f'\rEpoch : {epoch}, Loss: {train_loss:.5f}, Lowest Loss: {loss:.5f}, Patient: {patient}', end='')

            # if loss is smaller than before, save the model
            if train_loss < loss:
                loss = train_loss
                torch.save(model.state_dict(), 'model.pth')
                patient = 0
            else:
                patient += 1
            if patient >= 50:
                break
                

        self.loss = loss
        return history
    
    def train(self, plot=False):
        history = self._train_loop(self.model, self.train_loader, self.criterion, self.optimizer)
        if plot:
            self.plot(history)
        self.model.load_state_dict(torch.load('./model.pth',map_location=device, weights_only=True))
        return self.model

    def plot(self, history):
        plt.plot(history['train_loss'], label='train loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()



def predict(model, data, inp_dim, out_dim):
    model.eval()
    pred = {}
    for i, d in enumerate(data):
        ipdata = d.to(device).float()
        prediction = {}
        with torch.no_grad():
            input_data = ipdata.unsqueeze(0)
            output = F.relu(model(input_data)).abs().floor().int().cpu()
            output = np.array(output).reshape(out_dim).tolist()
            pred[f"attempt_{i+1}"] = output
        ipdata = ipdata.cpu().numpy()[-(np.dot(*inp_dim)+1):-1]
        
        # Plot the input data and the first prediction
        plt.figure(figsize=(18, 9))
        
        # Plot input data
        plt.subplot(1, 2, 1)
        plt.title("Input Data")
        plt.imshow(ipdata.reshape(inp_dim), cmap=CMAP, norm=NORM)
        plt.colorbar()
        
        # Plot real output prediction
        plt.subplot(1, 2, 2)
        plt.title("Prediction")
        plt.imshow(output, cmap=CMAP, norm=NORM)
        plt.colorbar()

        
        plt.show()
    
    return pred


def create_final_submission():
    dataset = Dataset(f'{BASE_DIR}/arc-agi_test_challenges.json')
    keys = dataset.key
    criterion = nn.MSELoss()
    for i, key in enumerate(keys):
        datatrain = dataset.data[key]
        train_dataset = ARCDataset(datatrain['train'], datatrain['test'])

        input_dim = train_dataset.input_length
        output_dim = train_dataset.output_length
        hidden_dim = int((input_dim + output_dim) // 2)

        model = LSTM(input_dim, output_dim, hidden_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print(f"{i+1}/{len(keys)} {key}")
        
        training = Training(model, train_dataset.train_dataset, criterion, optimizer, device)
        new_model = training.train(True)

        predictions = []
        for test in train_dataset.test_dataset:
            prediction = predict(new_model, test, train_dataset.inp_dim, train_dataset.out_dim)
            predictions.append(prediction)

        with open(SUB_TARGET,'r') as f:
            submission = json.load(f)
        submission[key] = predictions
        json_object = json.dumps(submission, indent=4)
        
        # Writing to sample.json
        with open(SUB_TARGET, "w") as outfile:
            outfile.write(json_object)


create_final_submission()




