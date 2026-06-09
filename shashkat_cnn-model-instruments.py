# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


### imports
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from tqdm import tqdm
from torchvision import transforms
from pytorch_lightning.loggers import WandbLogger

# set working directory and move to it (then the automatic logs also get there)
wd = '/kaggle/input/musical-instrumemts-sound-classification/Melspectogram_split'


### Useful functions
# function to get just the non-hidden files in a director
def ListDirectory(dir):
    files = []
    for item in os.listdir(dir):
        # if the item doesnt start with . then it non-hidden
        if not item.startswith('.'):
            files.append(item)
    return (files)

# standardize an image, by calculating its mean and std and creating a transform, and 
# then performing the standardization. 
# NOTE: assumes the tensor to have dimensions: c,h,w
def StandardizeImage(input_tensor):
    mean = input_tensor.mean(axis = (1,2))
    std = input_tensor.std(axis = (1,2))
    transformation = transforms.Normalize(mean, std)
    return transformation(input_tensor)

# go through a location where data is stored (like train, val, test), and read and process the 
# images there and return the appropriate objects
def ReadAndProcessImages(location, dim_reducer):
    
    '''
    go through a location where data is stored (like train, val, test), and read and process the images there and return the appropriate objects
    Args:
        location: location where there are multiple folders, each with the title of an instrument, and within each folder, there are images for the sound-captures of that instrument
        dim_reducer: nn.AdaptiveAvgPool2d() instance according to which, each image's dimensions are reduced
    Returns:
        a tensor with the images stacked, shape: (num_images, c, h, w) and a list of int labels, 
        according to label_to_int 
    '''

    # load the training images and finally make a training dataloader
    directories = ListDirectory(location)

    # a dictionary storing which label maps to which int
    label_to_int = {label:idx for idx, label in enumerate(directories)}

    # loop through all folders (instruments) in the directory
    img_tensors = [] # list storing the image tensors (permuted appropriately)
    ys = [] # list storing the labels for the images in order
    for directory in tqdm(directories):
        # directory = 'BOM'
        # read the images in given directory and store in a list
        for file_name in ListDirectory(f'{location}/{directory}'):
            # file_name = 'Bom (444).jpg'
            # read in the image from the appropriate directory
            img = Image.open(f'{location}/{directory}/{file_name}')
            # convert the image to np format
            img_np = np.array(img)
            # convert to tensor, permute dimensions appropriately, perform standardization on individual level, and reduce dimensions (for less load on memory)
            img_tensor = torch.tensor(img_np, dtype = torch.float).permute(2,0,1)
            standardized_img_tensor = StandardizeImage(img_tensor)
            final_img_tensor = dim_reducer(standardized_img_tensor)
            img_tensors.append(final_img_tensor)
            # append the label (directory) in the appropriate object
            ys.append(label_to_int[directory])

    # obtain a single image tensor by concatenating all the image tensors in img_tensors
    img_tensors_stacked = torch.stack(img_tensors, dim = 0)

    return img_tensors_stacked, ys


### Setting up data
# nn Module which will be used to reduce the dimensions of each image
dim_reducer = nn.AdaptiveAvgPool2d(output_size=(20,50))

# get the train and validation data, after processing
train_img_tensors, train_ys = ReadAndProcessImages(location=f'{wd}/train', dim_reducer=dim_reducer)
val_img_tensors, val_ys = ReadAndProcessImages(location=f'{wd}/val', dim_reducer=dim_reducer)

# Checked and the overall means were very close to 0, and std around 0.8. So its fine, and probably don't need to standardize again.

# Now, lets loop through the stacked tensor, and store the labels also in each datapoint in the form of a list. 
# Then, I can proceed with making it into a dataset object
train_data = []
for i in range(train_img_tensors.shape[0]):
    train_data.append([train_img_tensors[i], train_ys[i]])
val_data = []
for i in range(val_img_tensors.shape[0]):
    val_data.append([val_img_tensors[i], val_ys[i]])

# now, we need to make a dataloader class for our data, so that we can do batch treatment with it
class InstrumentsDataset(Dataset):
    # initialize the instance of this class
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# make the dataset and the dataloader object using that
train_dataset = InstrumentsDataset(train_data)
train_loader = DataLoader(dataset = train_dataset, batch_size = 25, shuffle = True)
val_dataset = InstrumentsDataset(val_data)
val_loader = DataLoader(dataset = val_dataset, batch_size = 25, shuffle = True)


### Setting up the model classes
# now, we can create the pl.LightningModule subset class, where we will define the things which 
# our model will do
class InstrumentsModule(pl.LightningModule):
    def __init__(self, model_object, loss_module):
        super().__init__()
        self.model = model_object
        self.loss_module = loss_module

    def forward(self, x):
        # just return the result of passing the data as it is, through the model
        return self.model(x)

    def configure_optimizers(self):
        # this function should return an optimizer object
        optimizer = torch.optim.AdamW(params = self.model.parameters(), lr = 0.01)
        return optimizer

    def training_step(self, batch, batch_index):
        # we dont need to call things like optimizer.zero_grad or optimizer.step here
        # returns the loss tensor
        X, y = batch
        preds = self(X)
        loss = self.loss_module(preds, y)
        acc = torch.sum(preds.argmax(dim = 1) == y)/len(y)
        self.log('training_accuracy', acc)
        self.log('training_loss', loss)
        return loss

    def validation_step(self, batch, batch_index):
        # same thing as training step, just that it won't be used to change model parameters 
        # (which will happen automatically, using the method's name). We don't have to return 
        # anything in this method too. Just have to log the accuracy
        X, y = batch
        preds = self(X)
        acc = torch.sum(preds.argmax(dim = 1) == y)/len(y)
        self.log('validation_accuracy', acc)

    def test_step(self, batch, batch_index):
        # same thing as validation step, just that its on a test set. We don't have to return 
        # anything in this method too. Just have to log the accuracy
        X, y = batch
        preds = self(X)
        acc = torch.sum(preds.argmax(dim = 1) == y)/len(y)
        self.log('test_accuracy', acc)

# an initial, basic model architecture
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels = 10, kernel_size=3, padding = 1), # a first convolution to scale up the channel size
            nn.BatchNorm2d(num_features=10),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels = 10, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features=10),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(10, 6)
            )

    def forward(self, x):
        return (self.model(x))


# Actually training model
simple_model = CNN()

# lets make a basic instance of the InstrumentsModule class and try passing the data through it
instruments_module = InstrumentsModule(model_object = simple_model, loss_module = nn.CrossEntropyLoss())

# set the wandb logger
# wandb_logger = WandbLogger(name = 'simple_model_second', project = 'instruments_classification_cnn')

# train
trainer = pl.Trainer(accelerator='cpu', enable_progress_bar=True, max_epochs=10)
trainer.fit(model = instruments_module, train_dataloaders=train_loader, val_dataloaders=val_loader)


### make predictions

# NOTE: since we already have the images in the true-label folders, this function works for the test data. 
# Else, it would require some redesigning. I just want to see the model performance on the test data hence 
# using it, and will ignore the true labels.
test_img_tensors, test_ys = ReadAndProcessImages(location=f'{wd}/test', dim_reducer=dim_reducer) 

test_data = []
for i in range(test_img_tensors.shape[0]):
    test_data.append([test_img_tensors[i], test_ys[i]])

test_dataset = InstrumentsDataset(test_data)
test_loader = DataLoader(dataset = test_dataset)

trainer.test(instruments_module, dataloaders=test_loader) # Got accuracy around 99%


# make the submission file

submission = pd.DataFrame()
# obtain the names of all the images in order they were loaded into the test_data
file_names = []
test_directories = os.listdir(f'{wd}/test')
for test_directory in test_directories:
    for image in os.listdir(f'{wd}/test/{test_directory}'):
        file_names.append(image)

test_dataset_only_X = InstrumentsDataset(test_img_tensors)
test_loader_only_X = DataLoader(dataset = test_dataset_only_X, batch_size=len(test_dataset_only_X))

predictions = trainer.predict(model = instruments_module, dataloaders = test_loader_only_X, return_predictions=True)
predictions_int = predictions[0].argmax(dim = 1)
int_to_label = {i:name for i, name in enumerate(test_directories)} # the dict mapping integers to the actual labels of instruments
predictions_string = [int_to_label[int(prediction_int)] for prediction_int in predictions_int]

submission['Filename'] = file_names
submission['Instrument_name'] = predictions_string

submission.to_csv('submission.csv', index = False)

