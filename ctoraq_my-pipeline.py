# Standard library
import copy
import glob
import multiprocessing
import os
import time
import zipfile

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# Related third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io, transform
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

input_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

num_classes = 10

table_for_convert = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
#Need only when multiclass, this snippet will help you to extract this var:
'''
temp = CIFAR10Dataset(train_list, transform=data_transforms['train'])
temp.idx_to_class
'''

batch_size = 32
num_epochs = 5

finetune = True

num_workers = multiprocessing.cpu_count()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    history = {'accuracy': [],
               'val_accuracy': [],
               'loss': [],
               'val_loss': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'train':
                history['accuracy'].append(epoch_acc.item())
                history['loss'].append(epoch_loss)
            else:
                history['val_accuracy'].append(epoch_acc.item())
                history['val_loss'].append(epoch_loss) 

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


class CIFAR10Dataset(Dataset):
  
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform
        
        # Load the labels file
        self.labels_df = pd.read_csv("/kaggle/working/label.csv") # or throw labels directly
        
        # Create a mapping from class names ('cat', 'dog') to integer indices (0, 1)
        # This is essential for training with loss functions like CrossEntropyLoss
        self.classes = sorted(self.labels_df['label'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

        # Set the 'id' column as the index for fast lookup later
        self.labels_df = self.labels_df.set_index('id')
    
    def __len__(self):
        return len(self.file_list)
  
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
       
        img_name = self.file_list[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
    
        # 1. Extract the image ID from the filename.
        img_id_str = os.path.basename(img_name).split('.')[0]
        img_id = int(img_id_str)
        
        # 2. Look up the string label (e.g., 'cat') from the DataFrame.
        label_name = self.labels_df.loc[img_id, 'label']
        
        # 3. Convert the string label to its corresponding integer index.
        label = self.class_to_idx[label_name]
    
        return image, label

all_train_files = glob.glob(os.path.join("/kaggle/working/train", '*.jpg'))
train_list, val_list = train_test_split(all_train_files, random_state=42)

print(len(train_list))
print(len(val_list))

data_transforms = {
    'train': transforms.Compose([
        #transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.RandomResizedCrop(input_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        #transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

image_datasets = {
    'train': CIFAR10Dataset(train_list,
                             transform=data_transforms['train']),
    'val': CIFAR10Dataset(val_list,
                           transform=data_transforms['val'])
}

dataloaders_dict = {x: DataLoader(image_datasets[x],
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers) for x in ['train', 'val']}


# weights=None if without pretrained


model_ft = models.vgg16(pretrained=True)
model_ft.classifier[6] = nn.Linear(4096, num_classes)

'''
model_ft = models.vit_b_16(pretrained=True)
model_ft.heads = nn.Sequential(
    nn.Linear(768, 4096),
    nn.ReLU(),
    nn.Linear(4096, num_classes)
) # or nn.Linear(768, num_classes)
'''

model_ft = model_ft.to(device)

params_to_update = model_ft.parameters()
print("Params to learn:")
if finetune:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)


test_list = glob.glob(os.path.join("/kaggle/working/test", '*.png'))
test_data_transform = data_transforms['val']

labels = []
submission_ids = []

with torch.no_grad():
    # Loop through all the test images
    for test_path in tqdm(test_list):
        # 1. Get the image ID from the filename
        # e.g., from '/path/to/123.png', get 123
        img_id = int(os.path.basename(test_path).split('.')[0])
        submission_ids.append(img_id)
        
        # 2. Load and transform the image
        img = Image.open(test_path).convert("RGB") # Ensure it's RGB
        img = test_data_transform(img)
        
        # Add a batch dimension (C, H, W) -> (1, C, H, W) and send to device
        img = img.unsqueeze(0)
        img = img.to(device)

        # 3. Get model predictions (logits)
        outputs = model_ft(img)

        # 4. Get the predicted class index by finding the max logit
        _, pred_idx_tensor = torch.max(outputs, 1)
        pred_idx = pred_idx_tensor.item() # .item() gets the integer from the tensor

        labels.append(pred_idx)


test_list = glob.glob(os.path.join("/kaggle/working/test", '*.jpg'))
test_data_transform = data_transforms['val']

ids = []
labels = []

with torch.no_grad():
    for test_path in tqdm(test_list):
        img = Image.open(test_path)
        img = test_data_transform(img)
        img = img.unsqueeze(0)
        img = img.to(device)

        model_ft.eval()
        outputs = model_ft(img)
        preds = F.softmax(outputs, dim=1)[:, 1].tolist()

        test_id = extract_class_from(test_path)
        ids.append(int(test_id))
        labels.append(preds[0])


class_labels = list(map(table_for_convert.get, labels))


# Core
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.4)
import itertools
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import time

# Sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, PredefinedSplit
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.utils import resample

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB


# Load data
train = pd.read_csv("/kaggle/input/playground-series-s3e3/train.csv", index_col = "id")
test = pd.read_csv("/kaggle/input/playground-series-s3e3/test.csv", index_col = "id")
sub = pd.read_csv("/kaggle/input/playground-series-s3e3/sample_submission.csv", index_col = "id")
train.shape


train_full = train
train_full.shape


# Preview data
train_full.head(3)


# Drop constant columns
train_full.drop(["EmployeeCount", "Over18", "StandardHours"], axis=1, inplace=True)
test.drop(["EmployeeCount", "Over18", "StandardHours"], axis=1, inplace=True)


# Split features and labels
y = train_full["Attrition"]
X = train_full.drop("Attrition", axis=1)
X_test = test


# Categorical columns
cat_cols = [col for (col, d) in zip(X.columns,X.dtypes) if d == "object"]
cat_cols


# Encode cat columns
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)


# Scale data
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)


# PCA
pca = PCA()
X_pca = pca.fit_transform(X)
pca_df = pd.DataFrame(data = X_pca)

# Scatterplot
plt.figure(figsize=(8,6))
plt.scatter(pca_df.iloc[:,0], pca_df.iloc[:,1], c=y, cmap="brg", s=40)
plt.title('PCA plot in 2D')
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.show()


# Plot
plt.figure(figsize=(14,5))
xi = np.arange(1,1+X.shape[1], step=1)
yi = np.cumsum(pca.explained_variance_ratio_)
plt.plot(xi, yi, marker='o', linestyle='--', color='b')

# Aesthetics
plt.ylim(0.0,1.1)
plt.xlabel('Number of Components')
plt.xticks(np.arange(1, 1+X.shape[1], step=2))
plt.ylabel('Cumulative variance (%)')
plt.title('Explained variance by each component')
plt.axhline(y=1, color='r', linestyle='-')
plt.gca().xaxis.grid(False)
plt.show()


classifiers = {
    "LogisticRegression" : LogisticRegression(random_state=0),
    "KNN" : KNeighborsClassifier(),
    "SVC" : SVC(random_state=0, probability=True),
    "RandomForest" : RandomForestClassifier(random_state=0),
    "ExtraTrees" : ExtraTreesClassifier(random_state=0),
    "XGBoost" : XGBClassifier(random_state=0, use_label_encoder=False, eval_metric='logloss'),
    "LGBM" : LGBMClassifier(random_state=0),
    "CatBoost" : CatBoostClassifier(random_state=0, verbose=False),
    "NaiveBayes": GaussianNB()
}


# Grids for grid search
LR_grid = {'penalty': ['l1','l2'],
           "solver": ["liblinear"],
           'C': [0.25, 0.5, 0.75, 1, 1.25, 1.5]}

KNN_grid = {'n_neighbors': [3, 5, 7, 9],
            'p': [1, 2]}

SVC_grid = {'C': [0.25, 0.5, 0.75, 1, 1.25, 1.5],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']}

RF_grid = {'n_estimators': [50, 100, 150, 200, 250, 300],
        'max_depth': [4, 6, 8, 10, 12]}

boosted_grid = {'n_estimators': [50, 100, 150, 200],
        'max_depth': [4, 8, 12],
        'learning_rate': [0.05, 0.1, 0.15]}

NB_grid={'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7]}


# Dictionary of all grids
grid = {
    "LogisticRegression" : LR_grid,
    "KNN" : KNN_grid,
    "SVC" : SVC_grid,
    "RandomForest" : RF_grid,
    "ExtraTrees" : RF_grid,
    "XGBoost" : boosted_grid,
    "LGBM" : boosted_grid,
    "CatBoost" : boosted_grid,
    "NaiveBayes": NB_grid
}


n_folds = 5

# Train models
for key, classifier in classifiers.items():
    # Initialise outputs
    test_preds = np.zeros(len(X_test))
    oof_full = y.copy()
    
    # Start timer
    start = time.time()
    
    # k-fold cross validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    
    score=0
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Get training and validation sets
        X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]
        
        # Start timer
        start = time.time()
        
        # Specify predefined validation set (-1 = train, 0 = valid)
        #split_idx = np.zeros(len(y))
        #split_idx[train_idx] = -1
        #ps = PredefinedSplit(split_idx)
        
        # Tune hyperparameters
        clf = RandomizedSearchCV(estimator=classifier, param_distributions=grid[key], n_iter=20, scoring='roc_auc', n_jobs=-1, cv=5)
        
        # Train using PredefinedSplit
        clf.fit(X_train, y_train)
        
        # Out-of-fold predictions
        oof_preds = clf.predict_proba(X_valid)[:,1]
        score += roc_auc_score(y_valid, oof_preds)/n_folds
        oof_full[val_idx] = oof_preds
        
        # Test set predictions
        test_preds += clf.predict_proba(X_test)[:,1]/n_folds
    
    # Stop timer
    stop = time.time()
    
    # Print score and time
    print('Model:', key)
    print('Average validation AUC:', np.round(100*score,2))
    print('Training time (mins):', np.round((stop - start)/60,2))
    print('')
    
    # Plot ROC curve
    #plot_roc_curve(clf, X, y)
    #plt.legend([key])
    #plt.xlabel("False Positive Rate")
    #plt.ylabel("True Positive Rate")
    #plt.show()
    
    # Save oof and test set preds
    oof_full.to_csv(f"{key}_oof_preds.csv", index=False)
    ss = sub.copy()
    ss["Attrition"] = test_preds
    ss.to_csv(f"{key}_test_preds.csv", index=False)


# Join oof preds
oof_df = pd.DataFrame(index=np.arange(len(y)))
for i in classifiers.keys():
    df = pd.read_csv(f"/kaggle/working/{i}_oof_preds.csv")
    df.rename(columns={"Attrition": i}, inplace=True)
    oof_df = pd.concat([oof_df,df], axis=1)
    
# Join test preds
test_preds = pd.DataFrame(index=np.arange(len(X_test)))
for i in classifiers.keys():
    df = pd.read_csv(f"/kaggle/working/{i}_test_preds.csv")
    df.rename(columns={"Attrition": i}, inplace=True)
    test_preds = pd.concat([test_preds,df], axis=1)
    
oof_df.head(3)


# Evaluate oof preds
scores = {}
for col in oof_df.columns:
    scores[col] = roc_auc_score(y, oof_df[col])

# Sort scores
scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}

# Sort oof_df and test_preds
oof_df = oof_df[list(scores.keys())]
test_preds = test_preds[list(scores.keys())]

scores


# Initialise
STOP = False
current_best_ensemble = oof_df.iloc[:,0]
current_best_test_preds = test_preds.iloc[:,0]
MODELS = oof_df.iloc[:,1:]
weight_range = np.arange(0.01,0.51,0.01)   # or with negative weights: np.arange(-0.5,0.51,0.01)
history = [roc_auc_score(y, current_best_ensemble)]
i=0

# Hill climbing
while not STOP:
    i+=1
    potential_new_best_cv_score = roc_auc_score(y, current_best_ensemble)
    k_best, wgt_best = None, None
    for k in MODELS:
        for wgt in weight_range:
            potential_ensemble = (1-wgt) * current_best_ensemble + wgt * MODELS[k]
            cv_score = roc_auc_score(y, potential_ensemble)
            if cv_score > potential_new_best_cv_score:
                potential_new_best_cv_score = cv_score
                k_best, wgt_best = k, wgt
            
    if k_best is not None:
        current_best_ensemble = (1-wgt_best) * current_best_ensemble + wgt_best * MODELS[k_best]
        current_best_test_preds = (1-wgt_best) * current_best_test_preds + wgt_best * test_preds[k_best]
        MODELS.drop(k_best, axis=1, inplace=True)
        if MODELS.shape[1]==0:
            STOP = True
        print(f'Iteration: {i}, Model added: {k_best}, Best weight: {wgt_best:.2f}, Best AUC: {potential_new_best_cv_score:.5f}')
        history.append(potential_new_best_cv_score)
    else:
        STOP = True


plt.figure(figsize=(10,4))
plt.plot(np.arange(len(history))+1, history, marker="x")
plt.title("CV AUC vs. Number of Models with Hill Climbing")
plt.xlabel("Number of models")
plt.ylabel("AUC")
plt.show()


plt.figure(figsize=(10,4))
sns.histplot(current_best_test_preds)
plt.title("Distribution of final predictions")
plt.show()


# Submit predictions
submission = sub.copy()
submission["Attrition"] = current_best_test_preds.values
submission.to_csv("submission.csv", index=True)

