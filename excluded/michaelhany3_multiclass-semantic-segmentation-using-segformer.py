!pip install evaluate


import os
import gc
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torchmetrics import AUROC
from PIL import Image
from tqdm import tqdm
import evaluate
from torchvision.transforms.functional import to_pil_image
import random
# Albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, Normalize, RandomCrop, HorizontalFlip, ShiftScaleRotate, HueSaturationValue

# Transformers / HuggingFace
from transformers import (
    SegformerConfig,
    SegformerForSemanticSegmentation,
    SegformerPreTrainedModel,
)

# Metrics
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, roc_curve



def crop(imgfile, maskfile, size=224):
    mask = maskfile.astype(np.float32)
    if np.sum(mask) / mask.size > 0.9:
        # if the object mask has more than 50% of the whole mask, return the full image and mask
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
        cropped_img = cv2.resize(imgfile, (size, size), interpolation=cv2.INTER_NEAREST)
        return cropped_img, mask
    else:
        # otherwise, crop the image and mask to the bounding box of the object
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        mask = mask[y1:y2+1, x1:x2+1]
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
        cropped_img = cv2.resize(imgfile[y1:y2+1, x1:x2+1], (size, size), interpolation=cv2.INTER_NEAREST)
        return cropped_img, mask

def decode_rle_to_mask(rle, height, width, viz=False):
    '''
    rle : run-length as string formated (start value, count)
    height : height of the mask 
    width : width of the mask
    returns binary mask
    '''
    rle = np.array(rle.split(' ')).reshape(-1, 2)
    mask = np.zeros((height*width, 1, 3))
    if viz:
        color = np.random.rand(3)
    else:
        color = [1,1,1]
    for i in rle:
        mask[int(i[0]):int(i[0])+int(i[1]), :, :] = color

    return mask.reshape(height, width, 3)


df_train = pd.read_csv('/kaggle/input/ai-dl-multiclass-segmentation/train.csv')
df_test = pd.read_csv('/kaggle/input/ai-dl-multiclass-segmentation/test_class.csv')
trainpath = '/kaggle/input/ai-dl-multiclass-segmentation/TrainImages'
testpath = '/kaggle/input/ai-dl-multiclass-segmentation/TestImages'

dfcp=df_train.copy()
dfcp = dfcp.drop_duplicates(subset=['ImageName'], keep='first')
print('length of the dataset:', len(dfcp))

traindata = {"imgs": [], "masks": []}
counter = 0

for i in dfcp['ImageName']:
    imgpath=os.path.join(trainpath,i+'.jpg')
    masks=[]
    num=0
    for j in df_train[df_train['ImageName'] == i]['Encoding']:#ImageHeight
        wi = df_train[df_train['ImageName'] == i]['ImageWidth'].values[0]
        hi = df_train[df_train['ImageName'] == i]['ImageHeight'].values[0]
        classes = int(df_train[df_train['ImageName'] == i]['ClassNumber'].values[num])
        masks.append(decode_rle_to_mask(j, int(hi), int(wi))*classes)
        num+=1
    mask = np.sum(masks, axis=0)[:,:,0]
    try:
        i,m=crop(cv2.imread(imgpath),mask,size=224)
        traindata['imgs'].append(np.array(i))
        traindata['masks'].append(np.array(m))
        counter += 1
    except:
        print("Failed to add the image and mask of index",counter)
        continue

print("Images added: ",len(traindata['imgs'])," ,Masks added: ",len(traindata['masks']))


data_transforms = Compose([
                        ToTensorV2()
                    ])
train_transform = []
for i in range(len(traindata['imgs'])):
    train_transform.append((data_transforms(image=traindata['imgs'][i]),traindata['masks'][i]))



#PyTorch
ALPHA = 0.5
BETA = 0.5
GAMMA = 1
class CustomSegFormer2(SegformerPreTrainedModel):
    def __init__(self, config, in_channel, n_classes=5):
        super().__init__(config)
        self.segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640", ignore_mismatched_sizes=True)
        self.logits_outputs = nn.Conv2d(in_channel, out_channels=n_classes, kernel_size=1, stride=1, padding=0)
        self.distance_outputs = nn.Conv2d(in_channel, out_channels=1, kernel_size=1, stride=1, padding=0)
        

    def forward(self, pixel_values):
        # forward to get raw logits
        outputs = self.segformer(pixel_values) # what's the number of output channels?
        logits_outputs = self.logits_outputs(outputs[0])
        distancemap_outputs = self.distance_outputs(outputs[0])        

        upsampled_logits = nn.functional.interpolate(logits_outputs, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)
        upsampled_distancemaps = nn.functional.interpolate(distancemap_outputs , size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)

        
        return  upsampled_logits, upsampled_distancemaps
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.val = val
        self.sum += val * count
        self.count += count
        self.avg = self.sum / self.count

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        print(inputs.shape)
        print(targets.shape)
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        print(inputs.shape)
        print(targets.shape)
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky

def dice_loss(output, target, num_classes = 5):
    # Compute softmax over the channel dimension
    output = F.softmax(output, dim=1)
    
    # Create a one-hot encoding of the target
    target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    # Compute intersection and union between output and target
    intersection = torch.sum(output * target_onehot, dim=[2, 3])
    union = torch.sum(output, dim=[2, 3]) + torch.sum(target_onehot, dim=[2, 3])
    
    # Compute the Dice coefficient for each class
    dice = 2 * intersection / union.clamp(min=1e-7)
    
    # Compute the average Dice loss across all classes
    dice_loss = 1 - torch.mean(dice)
    
    return dice_loss


def train_model(model,dataloaders, criterion, optimizer, scheduler, name, num_epochs=25, feature_extractor = None, metric = None):

    #Creating a folder to save the model performance.
    try: os.mkdir('/kaggle/working/modelPerformance')
    except: print('directory exists')
    try: os.mkdir(f'/kaggle/working/modelPerformance/{name}')
    except: print(f"{name} alreadt exists")
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0  
    acc = 0.0
    loss = 0.0
    loss_avg = AverageMeter()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            print(phase)
            if phase == 'train':
                loss_avg = AverageMeter()
                model = model.train()  # Set model to training mode
            else:
                loss_avg = AverageMeter()
                model = model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0

            #epochs
            epoch=int(len(dataloaders[phase]))
            for _ in tqdm(range(epoch)):
                #Loading Data
                optimizer.zero_grad()
                inputs, labels = next(iter(dataloaders[phase]))
                inputs = inputs['image']
                labels = labels.type(torch.LongTensor)
                inputs = inputs.to(device)
                print()
                labels = labels.to(device)
                # forward
                # track history if only in train
                if phase == 'train':
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, ssn = model(inputs.float())
                        pred = outputs.softmax(dim=1)
                        preds=pred.argmax(dim=1)
                        loss = criterion(outputs, labels)
                        print(loss.item())

                        loss.backward()
                        optimizer.step()
                    loss_avg.update(loss.item())

                if phase == 'test':
                    outputs, ssn = model(inputs.float())
                    pred = outputs.softmax(dim=1)
                    preds=pred.argmax(dim=1)

                    loss = criterion(outputs, labels)
                    loss_avg.update(loss.item())           

   
                acc = metric.compute(predictions = preds, references = labels, num_labels = 5, ignore_index = 5, reduce_labels=False)
                print("Acc:",acc)
                print("Loss:",loss.item())
                
            if phase == 'train':
                scheduler.step()
                
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, loss_avg.val, acc['overall_accuracy']))
    
            
            # deep copy the model
            if phase == 'test' and acc['overall_accuracy'] > best_acc:
                best_acc = acc['overall_accuracy']
                torch.save(model.state_dict(),'/kaggle/working/modelPerformance/{}/best_model_{:.4f}acc_{}epochs.pth'.format(name,acc['overall_accuracy'],num_epochs))

                train_losses = []
                valid_losses = []

        print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataloaders = {
    'train': torch.utils.data.DataLoader(train_transform[:1600], batch_size=8, shuffle=True, num_workers=2),
    'test': torch.utils.data.DataLoader(train_transform[1600:], batch_size=8, shuffle=True, num_workers=2)
                                            }
configuration = SegformerConfig(hidden_dropout_prob = 0.3, attention_probs_dropout_prob = 0.3, classifier_dropout_prob = 0.5)
model=CustomSegFormer2(configuration,in_channel=150)
logits,DM = model(torch.rand(8,3,224,224))
logits.shape


criterion = nn.CrossEntropyLoss(ignore_index = 5, weight = torch.tensor([1.0, 5.0,5.0,8.0,10.0]).to(device) )
#criterion = dice_loss
#criterion = FocalTverskyLoss()
mean_iou = evaluate.load("abidlabs/mean_iou")


#model.load_state_dict(torch.load("/kaggle/working/segformers.pth"))
model_ft = model.to(device)
optimizer_ft = optim.AdamW(model_ft.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft,dataloaders, criterion, optimizer_ft, exp_lr_scheduler, name="Segformer", num_epochs=2, metric = mean_iou)


torch.save(model_ft.state_dict(),'/kaggle/working/segformers.pth')



model.load_state_dict(torch.load("/kaggle/working/segformers.pth"))
model.to(device)


carry, labels = next(iter(dataloaders['test']))
output = model(carry['image'].to(device).float())


# Extract logits from the first element of the output tuple
logits = output[0]

# Apply softmax to get class probabilities
pred = logits.softmax(dim=1)

# Get the predicted classes (argmax over the class dimension)
preds = pred.argmax(dim=1)

print(preds.shape)


def plot_images(image, masks, predicted_masks, num_images=1, color_map = None):
    # Select num_images random indices to plot
    indices = np.random.choice(range(len(image)), num_images, replace=False)

    # Plot each image and corresponding masks
    for i, index in enumerate(indices):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        #np.transpose(carry['image'][0].numpy(),(1, 2, 0)), cmap='gray')
        print(image[0].shape)
        axs[0].imshow(np.transpose(image[index].numpy(), (1,2,0)), cmap='gray')
        axs[0].set_title("Image")
        
        print(masks.shape)
        axs[1].imshow(masks[index], cmap=color_map)
        axs[1].set_title("Ground Truth")
        
        axs[2].imshow(predicted_masks[index].cpu(), cmap=color_map)
        axs[2].set_title("Prediction")
        
        plt.show()


# Example usage
# images = torch.randn(4, 3, 224, 224)  # 4 images with shape (3, 224, 224)
# masks = torch.randint(0, 2, (4, 1, 224, 224))  # 4 ground truth masks with shape (1, 224, 224)
# predicted_masks = torch.randint(0, 2, (4, 1, 224, 224))  # 4 predicted masks with shape (1, 224, 224)
color_map = plt.cm.get_cmap('tab20', 5)

plot_images(carry['image'], labels, preds, 8, color_map)



# Load the trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CustomSegFormer2(configuration, in_channel=150)
model.load_state_dict(torch.load("/kaggle/working/segformers.pth"))
model.to(device)
model.eval()

# Test data
test_csv = '/kaggle/input/ai-dl-multiclass-segmentation/test_class.csv'

test_images_path = '/kaggle/input/ai-dl-multiclass-segmentation/TestImages'
test_data = pd.read_csv(test_csv)




submission_data = []
# Prepare test loader
def prepare_test_data(test_df, images_path):
    data = []
    for _, row in test_df.iterrows():
        img_path = os.path.join(images_path, row['ImageName'] + ".jpg")
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
            data.append((image, row['ClassNumber']))
    return data

test_data_prepared = prepare_test_data(test_data, test_images_path)

# Transform function (modify as per your train transforms)
transform = Compose([Normalize(), ToTensorV2()])

# Generate predictions
def generate_mask(model, image, class_number, transform):
    input_tensor = transform(image=np.array(image))['image']
    input_tensor = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(input_tensor)
    probs = torch.softmax(logits, dim=1)
    mask = (probs.argmax(dim=1) == class_number).squeeze().cpu().numpy()
    return mask

def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# Visualize first 5 predictions and generate submission data
def visualize_first_5_predictions(test_data, test_df, model, transform):
    # Select the first 5 samples
    samples = test_data[0:50]
    for image, class_number in samples:
        
        # Extract the image name from the test_df DataFrame using the index
        # Assuming the test_df has 'ImageName' and 'ClassNumber' columns, we get the first match
        image_name = test_df.iloc[samples.index((image, class_number))]['ImageName']
        
        # Generate the mask for the image
        mask = generate_mask(model, image, class_number, transform)
        
        # Append to submission_data
        submission_data.append({'ImageName': image_name, 'Encoding': encode_mask_to_rle(mask)})
        
        # Print the RLE encoding with the correct image name
        # print(f"Image: {image_name}, Encoding: {encode_mask_to_rle(mask)}")
        # c+=1
        # print(c)
        # Visualize the image and its predicted mask
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image)
        ax[0].set_title(f"Original Image - Class {class_number}")
        ax[1].imshow(mask, cmap="gray")
        ax[1].set_title(f"Predicted Mask - Class {class_number}")
        plt.show()

# Run visualization for the first 5 images




visualize_first_5_predictions(test_data_prepared, test_data, model, transform)





