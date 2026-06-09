import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv('/kaggle/input/grand-xray-slam-division-a/train1.csv')
df.head()


len(df)


###### Configuration setup
BASE_PATH = '/kaggle/input/grand-xray-slam-division-a/'
TRAIN_IMG_PATH = os.path.join(BASE_PATH, 'train1/')
TEST_IMG_PATH = os.path.join(BASE_PATH, 'test1/')
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
    'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = int(os.cpu_count() // 2)
LR = 0.001
WEIGHT_DECAY = 0.0001
EPOCHS = 4
BATCH_SIZE = 16


class XRayDataset(Dataset):
    def __init__(self, df, dr = TRAIN_IMG_PATH, is_Test = False, transforms = None):
        self.df = df
        self.is_Test = is_Test
        self.labels = LABELS
        self.dr = dr
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def load_image(self, idx):
        img_path = self.df.iloc[idx]['Image_name']
        img = Image.open(img_path).convert('RGB')
        return img
        
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['Image_name']
        img_path = os.path.join(self.dr, img_path)
        img = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx][LABELS].values.astype(np.float32)
        if self.transforms:
            img =  self.transforms(img)
        if not self.is_Test:
            return img, torch.tensor(label, dtype = torch.float32)
        else:
            return img
    


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
obj = XRayDataset(df[:5], is_Test=False, transforms=transform)
img, label = obj[2]
classes = np.where(np.array(label) == 1)[0]
label_names = [LABELS[i] for i in classes]
print(classes)
print(label_names)
plt.imshow(img.permute(1,2, 0))
plt.axis('off')
plt.show()


weights_effnet = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
effnet_model = torchvision.models.efficientnet_v2_s(weights = weights_effnet)
# effnet_model


effnet_transforms = weights_effnet.transforms()
effnet_transforms


for param in effnet_model.features.parameters():
    param.requires_grad = False


## get model summary
try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.")
    !pip install -q torchinfo
    from torchinfo import summary
summary(model=effnet_model, 
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=10,
        row_settings=["var_names"]
)


effnet_model.classifier = nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features = 1280,
                   out_features = len(LABELS),
                   bias = True)
).to(DEVICE)


loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(effnet_model.parameters(), lr = LR, weight_decay = WEIGHT_DECAY)


optimizer


def train_step(model: torch.nn.Module, 
              dataloader : torch.utils.data.DataLoader,
              loss_fn : torch.nn.Module, 
              optimizer : torch.optim.Optimizer, 
              device : torch.device):
    
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y, = X.to(device), y.to(device)

        ## fwd pass
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## convert logits to probs
        y_pred_probs = torch.sigmoid(y_pred)
        y_pred_class = (y_pred_probs > 0.5).float()

        correct = (y_pred_class == y).float().mean()
        train_acc += correct.item()
        # print(f"Train: {batch}")
        
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc
    
def test_step(model: torch.nn.Module, 
             dataloader: torch.utils.data.DataLoader, 
             loss_fn: torch.nn.Module, 
             device : torch.device):

    model.eval()

    test_loss, test_acc = 0, 0
    
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = torch.sigmoid(test_pred_logits)
            test_pred_class = (test_pred_labels > 0.5).float()

            test_acc += (test_pred_class == y).float().mean().item()
            # print(f"Test: {batch}")

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    
    return test_loss, test_acc
        


def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device):
    results = {
        'train_loss' : [],
        'train_acc' : [],
        'test_loss' : [],
        'test_acc': []
    }

    ## to save model 
    best_loss = float('inf')
    model_name = model.__class__.__name__
    best_model_path = f"{model_name}_best.pth"
    checkpoint_path = f"{model_name}_checkpoint.pth"

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model, 
                                          dataloader=train_dataloader,
                                          loss_fn = loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model, 
                                       dataloader = test_dataloader,
                                       loss_fn = loss_fn,
                                       device=device)

        ## printing results
        print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Save best model: {best_model_path} (epoch {epoch + 1})")
        
        ## checkpoint to resume
        checkpoint = {
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'best_loss' : best_loss
        }
        torch.save(checkpoint, checkpoint_path)

    print(f"Training Complete.... Best Model saved as: {best_model_path}")
    return results


train_df, test_df = train_test_split(df, test_size=0.1, shuffle=True)


train_data = XRayDataset(train_df, transforms=effnet_transforms)
test_data = XRayDataset(test_df, transforms=effnet_transforms)


train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)


len(train_dataloader), len(test_dataloader)


results = train(effnet_model, train_dataloader, test_dataloader, optimizer=optimizer, loss_fn=loss_fn, epochs = EPOCHS, device=DEVICE)


fig, axes = plt.subplots(1, 2, figsize=(12, 8))

axes[0].plot(results['train_loss'], color='blue')
axes[0].set_title('Train Loss')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Train Loss')

axes[1].plot(results['test_loss'], color='green')
axes[1].set_title('Test Loss')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Test Loss')

plt.tight_layout()
plt.show()


sample_submission = pd.read_csv('/kaggle/input/grand-xray-slam-division-a/sample_submission_1.csv')
final_dataset = XRayDataset(sample_submission, dr=TEST_IMG_PATH, is_Test = True, transforms = effnet_transforms)
final_dataloader = DataLoader(final_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
effnet_model.eval()
predictions=[]

with torch.inference_mode():
    for batch, X in enumerate(final_dataloader):
        X = X.to(DEVICE)
        pred_probs = effnet_model(X)
        preds = torch.sigmoid(pred_probs)
        predictions.append(preds.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)


final_submission = sample_submission.copy()
final_submission[LABELS] = predictions
final_submission.head()


final_submission.to_csv('submission.csv', index=False)
print('Submission csv created... THE END !!')




