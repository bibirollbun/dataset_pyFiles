import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import pydicom

import os
import torch
import torch.nn as nn
import numpy as np

from pathlib import Path
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import time

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


BASE_FOLDER = '/kaggle/input'
DETECTION_FOLDER = BASE_FOLDER + '/rsna-intracranial-aneurysm-detection'
SERIES_FOLDER = DETECTION_FOLDER + '/series'
TRAINING_FILE = DETECTION_FOLDER + '/train.csv'

CATEGORIES = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present'
]

# Model selection - Change this to select which model to use for inference
# Options: 'tf_efficientnetv2_s', 'convnext_small', 'swin_small_patch4_window7_224', 'ensemble'
SELECTED_MODEL = 'tf_efficientnetv2_s' 

# Model paths configuration
MODEL_PATHS = {
    'tf_efficientnetv2_s': '/kaggle/input/rsna-iad-trained-models/models/tf_efficientnetv2_s_fold0_best.pth',
    'convnext_small': '/kaggle/input/rsna-iad-trained-models/models/convnext_small_fold0_best.pth',
    'swin_small_patch4_window7_224': '/kaggle/input/rsna-iad-trained-models/models/swin_small_patch4_window7_224_fold0_best.pth'
}

BATCH_SIZE = 32
NUM_EPOCHS = 20
FILE_NAME = 'best_vit_rsna_model'
TARGET_SIZE = (224,224)
NUM_TRAINING_ROWS = 100


df_train = pd.read_csv(TRAINING_FILE)

df_train.shape


df_train.head()


df_train.columns


gender_counts = df_train['PatientSex'].value_counts()

colors = sns.color_palette('Set1')[0:len(gender_counts)]

plt.figure(figsize=(15,8))
fig, ax = plt.subplots()
ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=colors, textprops={'fontsize': 12})
ax.set_title('Patients by Gender', fontsize=14)
plt.axis('equal')
plt.show()


unique_modalities = df_train['Modality'].unique()

print("Unique values in 'Modality' column:")
print(unique_modalities)


modality_counts = df_train['Modality'].value_counts()

colors = sns.color_palette('Set1')[0:len(modality_counts)]

plt.figure(figsize=(15,8))
fig, ax = plt.subplots()
ax.pie(modality_counts, labels=modality_counts.index, autopct='%1.1f%%', colors=colors, textprops={'fontsize': 12})
ax.set_title('Modality Breakdown', fontsize=14)
plt.axis('equal')
plt.show()


bins = [19, 36, 50, 65, 100]
labels = ['19-35', '36-49', '50-64', '65+']

df = df_train.copy()

df['AgeGroup'] = pd.cut(df['PatientAge'], bins=bins, labels=labels, right=False)

age_group_counts = df['AgeGroup'].value_counts().sort_index()

colors = sns.color_palette('Set1')[0:len(age_group_counts)]

plt.figure(figsize=(15,8))
fig, ax = plt.subplots()
ax.pie(age_group_counts, labels=age_group_counts.index, autopct='%1.1f%%', colors=colors, textprops={'fontsize': 12})
ax.set_title('Age Group Breakdown', fontsize=14)
plt.axis('equal')
plt.show()


def convert_binary_to_yn(ax):
    legend = ax.get_legend()

    new_labels = ['No', 'Yes']

    for t, l in zip(legend.get_texts(), new_labels):
        t.set_text(l)


plt.figure(figsize=(10,6))
ax = sns.histplot(
    data=df_train,
    x='PatientAge',
    hue='Aneurysm Present',
    bins=30,
    kde=True,
    palette={0: '#00BFC4', 1: '#C77CFF'}
)

convert_binary_to_yn(ax)

plt.title("Age Distribution by Aneurysm Presence", fontsize=14)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


df = df_train.copy()
location_cols = df.columns[4:-1]  # skip UID, Age, Sex, Modality, and skip final label
location_df = df[location_cols].astype(int) 

# Co-occurrence matrix
co_matrix = location_df.T.dot(location_df)

# Plot heatmap
plt.figure(figsize=(12, 10))
ax = sns.heatmap(co_matrix, cmap="magma", annot=True, fmt=".0f", linewidths=0.5)
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
plt.title("Aneurysm Co-occurrence Matrix", fontsize=16, color='white')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.gca().set_facecolor('black')
plt.gcf().set_facecolor('#111111')
plt.tight_layout()
plt.show()


class RSNADataset(Dataset):
    def __init__(self, csv_file, series_dir=SERIES_FOLDER, incoming_df=None, transform=None):
        
        self.series_dir = series_dir
        self.transform = transform
        
        if incoming_df is None:
            self.df = pd.read_csv(csv_file)
        else:
            self.df = incoming_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        series_path = Path(self.series_dir) / self.df.iloc[idx, 0]
        images = list(series_path.glob('**/*.dcm')) 
        imgs = [pydicom.dcmread(str(f)).pixel_array for f in sorted(images)]
        volume = np.stack(imgs) 

        labels = self.df.iloc[idx, 4:]
        sample = {'images': volume, 'labels': labels}
        
        return sample
        


train_ds = RSNADataset(TRAINING_FILE)
sample = train_ds[0]
print(sample['images'].shape)


print (len(train_ds))


print(sample['labels'])


fig = plt.figure(figsize=(15, 10))
columns = 5; rows = 4
for i in range(20):
    fig.add_subplot(rows, columns, i + 1)
    plt.imshow(sample['images'][i], cmap=plt.cm.bone)
    plt.axis('off')
plt.show()


#Use the most power possible
device = "cuda" if torch.cuda.is_available() else "cpu"


train_transform = transforms.Compose([
    #Resize to the target size defined above, if necessary
    transforms.Resize(TARGET_SIZE),

    #Flip it horizontally... maybe
    transforms.RandomHorizontalFlip(),

    #More random transformation
    #Read more here: https://medium.com/@MarkAiCode/random-affine-transformations-in-pytorch-c45a290e44d0
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),

    #Have fun with the color brightness
    transforms.ColorJitter(brightness=(0.8, 1.2)),

    #Rotate the image by a max of 10 degrees in either direction
    transforms.RandomRotation(10),

    #Tranform it to a tensor
    transforms.ToTensor(),

    #Standard normalization value here, feel free to experiment
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        labels = self.dataframe.iloc[idx, 1]
        labels_tensor = torch.tensor(labels, dtype=torch.float32) 
        img_array = pydicom.dcmread(img_path).pixel_array

        img = Image.fromarray(img_array).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, labels_tensor


class ImageDataLoader():
    #Note that this class allows you to send in custom transformers, here, we're just using
    #the transformers we created above
    #Also, we pass in the BASE series directory here
    def __init__(self, df_train, series_dir, train_transform=train_transform, test_transform=test_transform):

        self.df_train = df_train
        
        #Set the images dir as an instance variable
        self.images_dir = series_dir

        #Gotta create the datasets before we create the loaders
        #Note that we datasets here are instance variables as a convenience for debugging
        self.create_datasets(train_transform, test_transform)

        #And, finally, the loaders
        self.create_data_loaders()

    #Creates all datasets
    def create_datasets(self, train_transform, test_transform):
        #Create the training dataframe
        full_dataset = self.create_dataset()

        train, test_full = train_test_split(full_dataset)
        test, val = train_test_split(test_full)
        
        # Reset indices
        test = test.reset_index(drop=True)
        val = val.reset_index(drop=True)

        #Now create three (3) datasets and store them as instance variables so
        #developers can debug
        self.train_dataset = self.create_image_dataset(train, transform=train_transform)
        self.val_dataset = self.create_image_dataset(val, transform=test_transform)
        self.test_dataset = self.create_image_dataset(test, transform=test_transform)

    def create_data_loaders(self):

        #We shuffule this one because it's used for training, don't want to send everything in
        #in the same order
        self.train_loader = self.create_data_loader(self.train_dataset, shuffle=True)
        
        self.val_loader = self.create_data_loader(self.val_dataset)
        self.test_loader = self.create_data_loader(self.test_dataset)

    #Convenience function for creating the data loader. It's small now but left
    #in its own function for scalability purposes.
    def create_data_loader(self, dataset, batch_size=BATCH_SIZE, shuffle=False):
        return DataLoader(dataset, batch_size, shuffle)

    #Convenience function for creating the dataset. Uses the ImageDataset class defined above. 
    #What's the difference between the standard dataset and the ImageDataSet?
    #The standard dataset stores the class plus the path to the image.
    #The ImageDataSet allows retrieval of the image itself.
    def create_image_dataset(self, df, transform=None):
        return ImageDataset(df, transform)

    #Here's where we create the standard dataset.
    #Note: this only stores the class plus the path to the image, not the image itself.
    def create_dataset(self):
        my_list = []
        
        #for i,row in self.df_train.iterrows():
        for i in range(0,NUM_TRAINING_ROWS):
            labels = self.df_train.iloc[i, 4:]
            full_path = os.path.join(self.images_dir, self.df_train.iloc[i,0])
            
            for file_name in os.listdir(full_path):
                file_path = os.path.join(full_path, file_name)
                if file_path.endswith('.dcm'):
                    try:
                        img_array = pydicom.dcmread(file_path).pixel_array
                        img = Image.fromarray(img_array).convert('RGB')
                        my_list.append([file_path, labels])
                    except TypeError as e:
                        print(f"Caught TypeError for file: {file_path}")
                        
        return pd.DataFrame(my_list, columns=['file_path', 'labels'])



image_loader = ImageDataLoader(df_train, SERIES_FOLDER)


print(len(image_loader.train_dataset))


first_batch_data, first_batch_labels = next(iter(image_loader.train_loader))

print(f"First batch data shape: {first_batch_data.shape}")
print(f"First batch labels shape: {first_batch_labels.shape}")


    # Note that model is not optional here. You must pass in a model.
    # Note that optimizer is also not optional. It usually requires the model as input
    # so it's best to pass it in this way.
    # Gotta add the data loader as well.
    # Note that we default the optimizer to CrossEntropyLoss here. But you can use any loss function
    # That suits your fancy.
    # The num_epochs parameters defaults to 10 and tells the trainer how many times to run
    # through the training.
    # The patience parameter tells the model how many epochs to go through when it's not getting
    # any better.
    def train_model(model, optimizer, image_loader, 
                    criterion=nn.CrossEntropyLoss(), num_epochs=100, patience=6, file_name='best_model'):

        #Routine code here
        model.to(device)

        #Start with best value loss of infinity
        best_val_loss = float("inf")

        #We'll use to check for early stoppage
        tolerance = 0

        #This is what we'll actually return here, giving developers insight into training success
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(num_epochs):
            # Boilerplate code again - training mode
            model.train()

            # Set counters
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for images, labels in image_loader.train_loader:
                images, labels = images.to(device), labels.to(device)

                #This resets the gradients of optimized tensors
                optimizer.zero_grad()

                #This is where we get actual output from training
                outputs = model(images)

                #This is what determins the loss based on the outputs
                loss = criterion(outputs, labels)

                #Initiates back propagation
                #As the name implies, the logic traverses through the computational
                #graph that it created during the forward pass.
                #It computes the loss with respect to each intermediate tensor and weights and biases.
                #Note: this just calculates but does not optimize. The next line optimizes.
                loss.backward()

                #Handles the process of minimizing the loss function.
                #This is, in fact, the optimization step.
                optimizer.step()

                #This gets the scalar value of the loss
                running_loss += loss.item()

                predicted = outputs > 0.5

                #This just increases the total_train value by the batch size
                total_train += (len(CATEGORIES) * BATCH_SIZE)

                # The (predicted == labels) part performs an element-wise tensor comparison
                # It creates a new boolean tensor
                # The sum() method calculates the number of True values in the tensor
                # item() gets the boolean value as a number
                # Basically it's the number of right answers
                correct_train += (predicted == labels).sum().item()

            train_loss = running_loss / len(image_loader.train_loader)
            train_acc = 100 * correct_train / total_train

            # Now that we've trained, let's evaluate
            # First, set the model to eval() because we're no longer training
            model.eval()

            # Once again, establish the counters
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            # No need for gradient here because we're not correcting/training
            with torch.no_grad():
                for images, labels in image_loader.val_loader:
                    images, labels = images.to(device), labels.to(device)

                    # Once again, the actual output
                    outputs = model(images)
                    
                    # Now let's see how right we were (or weren't)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    predicted = outputs > 0.5
                    
                    total_val += (len(CATEGORIES) * BATCH_SIZE)
                    correct_val += (predicted == labels).sum().item()

            val_loss = val_loss / len(image_loader.val_loader)
            val_acc = 100 * correct_val / total_val

            # Again, this is what this function returns
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Print out some helpful info to the console
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
            print(f"Evaluation Loss: {val_loss:.4f}, Evaluation Accuracy: {val_acc:.2f}%")
            print("#" * 80)

            # Save the model if we got the best score so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print('Saving model')
                torch.save(model.state_dict(), f'{file_name}.pth')
                tolerance = 0
            else:
                tolerance += 1
                if tolerance >= patience:
                    print(f"Jumping out early, we can't do any better after {epoch + 1} epochs.")
                    break

        return history


    def test_model(model, image_loader, file_name='best_model'):
        #According to PyTorch docs, loading state_dict is the preferred way to load the model from disk
        model.load_state_dict(torch.load(f'{file_name}.pth'))
        
        #Moves all parameters and buffers to a specific device
        #On my laptop, it's always CPU
        model.to(device)

        #Setting to eval disables dropout that's used during training to prevent overfitting
        #It also adjusts batch normalization
        model.eval()

        correct = 0
        total = 0

        all_preds = []
        all_labels = []
        all_images = []

        #no_grad() is for memory efficiency
        #It disables gradient calculation so PyTorch won't store the computation graph
        with torch.no_grad():
            for images, labels in image_loader.test_loader:
                images, labels = images.to(device), labels.to(device)

                #Run the inputs through the model to get the outputs
                outputs = model(images)
                predicted = outputs > 0.5
                
                total += (len(CATEGORIES) * BATCH_SIZE)
                
                #The (predicted == labels) part performs an element-wise tensor comparison
                #It creates a new boolean tensor
                #The sum() method calculates the number of True values in the tensor
                #item() gets the boolean value as a number
                correct += (predicted == labels).sum().item()
        
        test_acc = 100 * correct / total

        print(f"Test Accuracy: {test_acc:.2f}%\n")



class VitModel(nn.Module):
    def __init__(self, num_classes):
        super(VitModel, self).__init__()
        self.pretrained = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.pretrained.head = nn.Identity()
        self.new_head = nn.Sequential(
            nn.Linear(1000, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.pretrained(x)
        x = self.new_head(x)
        return x


vit_model = VitModel(num_classes=len(CATEGORIES))


optimizer = torch.optim.SGD(vit_model.parameters(), lr=0.001, momentum=0.9)

file_name = 'best_vit_rsna_model'

start_time = time.perf_counter()

print("Launching ViT training...")
history = train_model(vit_model, optimizer, image_loader, criterion=torch.nn.BCELoss(), num_epochs=NUM_EPOCHS, file_name=FILE_NAME)

end_time = time.perf_counter()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time:.1f} seconds")


test_model(vit_model, image_loader, file_name)

