!pip install -U -q lightning timm SimpleITK


# PyTorch Imports
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms, models


# Helper Imports
import os
import pandas as pd
import numpy as np
from PIL import Image
from datetime import timedelta
# import pydicom
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Lightning Imports
import lightning as L
from lightning.pytorch import seed_everything
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

seed = 0
seed_everything(seed, workers=True)






class VinBigDataCXR(Dataset):
    def __init__(self, img_dir, annotations, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.annotations = annotations
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_id = row['image_id']
        class_name = row['class_name']
        label = row['class_id']
        rad_id = row['rad_id']
        x_min, y_min, x_max, y_max = row[['x_min', 'y_min', 'x_max', 'y_max']]

        dicom_path = os.path.join(self.img_dir, f"{img_id}.dicom")
        
        # Use SimpleITK to read the DICOM image
        image = sitk.ReadImage(dicom_path)
        image_array = sitk.GetArrayFromImage(image)

        # Ensure the image is a 2D array (if it's 3D, we take the first slice)
        if len(image_array.shape) > 2:
            image_array = image_array[0, :, :]  # Take the first slice if 3D

        # Convert to 3-channel (RGB) if it's grayscale
        image_rgb = np.stack([image_array] * 3, axis=-1)  # Replicate the grayscale to create 3 channels
        
        # Convert numpy array to PIL image
        image = Image.fromarray(image_rgb.astype(np.uint8))  # Ensure uint8 format for PIL

        # Get image dimensions and normalize bounding box
        width, height = image.size
        bbox = torch.tensor([x_min/width, y_min/height, (x_max-x_min)/width, (y_max-y_min)/height])

        if self.transform:
            image = self.transform(image)
        
        return image, label, bbox


class VinBigDataCXRDatamodule(L.LightningDataModule):
    def __init__(self, img_dir, csv_file, batch_size=32, num_workers=16, val_split=0.2):
        super().__init__()
        self.img_dir = img_dir
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.train_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,), std=(0.229,))
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,), std=(0.229,))
        ])
    
    def setup(self, stage=None):
        annotations = pd.read_csv(self.csv_file)

        train_annotations, val_annotations = train_test_split(
            annotations, test_size=self.val_split, random_state=42
        )
        
        self.train_dataset = VinBigDataCXR(img_dir=self.img_dir, annotations=train_annotations, transform=self.train_transform)
        self.val_dataset = VinBigDataCXR(img_dir=self.img_dir, annotations=val_annotations, transform=self.val_transform)

        if stage == "fit":
            print(f"Training Set Size: {len(self.train_dataset)}")
            print(f"Validation Set Size: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)



print('Verification#####################################')
print('Seed			 : ', seed)
print('CUDA available   : ', torch.cuda.is_available())
print('CUDA devices	 : ', torch.cuda.device_count())
print('CUDA version	 : ', torch.__version__)
print('#################################################')

# class SwinV2ObjectDetector(nn.Module):
# 	def __init__(self, num_classes=16):
# 		super().__init__()
		
# 		self.swin_backbone = models.swin_v2_b()
# 		self.swin_backbone.head = nn.Identity()
		
# 		self.classifier = nn.Linear(1024, num_classes)
		
# 		self.regressor = nn.Linear(1024, 4)
		
# 	def forward(self, x):
# 		features = self.swin_backbone(x)
# 		# print(f"Features shape: {features.shape}")
		
# 		class_scores = self.classifier(features)
		
# 		bbox_preds = self.regressor(features)
		
# 		return class_scores, bbox_preds

class NeuralNet(L.LightningModule):
	def __init__(self, num_classes, learning_rate):
		super().__init__()
		self.automatic_optimization=False
		self.num_classes = num_classes
		self.learning_rate = learning_rate

		self.swin_backbone = models.swin_v2_b()
		self.swin_backbone.head = nn.Identity()
		
		self.classifier = nn.Linear(1024, num_classes)
		
		self.regressor = nn.Linear(1024, 4)

		self.classifier_loss = F.cross_entropy
		self.regressor_loss = F.smooth_l1_loss

		self.classifier_train_loss = []
		self.regressor_train_loss = []
		self.classifier_val_loss = []
		self.regressor_val_loss = []
		self.classifier_test_loss = []
		self.regressor_test_loss = []
		
		# print(model)
		# self.save_hyperparameters(ignore=['model', 'ipython_dir'])

		print("INIT SwinV2ObjectDetector#############################")
		print("Learning Rate 		:", learning_rate)
		print("Classes 				:", num_classes)
		print("Classifier Loss 		:", self.classifier_loss)
		print("Regressor Loss	   :", self.regressor_loss)
		print("######################################################")

	def forward(self, images):
		pass

	def forward_classifier(self, images):
		features = self.swin_backbone(images)
		preds = self.classifier(features)
		return preds
	def forward_regressor(self, images):
		features = self.swin_backbone(images)
		bbox_preds = self.regressor(features)
		return bbox_preds

	def training_step(self, batch, batch_idx):
		opt_classifier, opt_regressor = self.optimizers()

		images, labels, bboxs = batch
		
		label_preds = self.forward_classifier(images)
		classifier_loss = self.classifier_loss(label_preds, labels)
		self.classifier_train_loss.append(classifier_loss.item())

		opt_classifier.zero_grad()
		classifier_loss.backward(retain_graph=True)
		opt_classifier.step()

		bbox_preds = self.forward_regressor(images)
		regressor_loss = self.regressor_loss(bbox_preds, bboxs)
		self.regressor_train_loss.append(regressor_loss.item())

		opt_regressor.zero_grad()
		regressor_loss.backward()
		opt_regressor.step()

		self.log('classifier_loss', classifier_loss.item(), on_step=True, prog_bar=True)
		self.log('regressor_loss', regressor_loss.item(), on_step=True, prog_bar=True)

	def on_train_epoch_end(self):
		classifier_train_mean_loss = torch.mean(torch.tensor(self.classifier_train_loss))
		regressor_train_mean_loss = torch.mean(torch.tensor(self.regressor_train_loss))


		self.print('Epoch : 							', self.current_epoch)
		self.print('Classifier Train Mean loss : 		', classifier_train_mean_loss)
		self.print('Regressor Train Mean loss : 		', regressor_train_mean_loss)

		self.classifier_train_loss = []
		self.regressor_train_loss = []

	def validation_step(self, batch, batch_idx):
		images, labels, bboxs = batch

		label_preds = self.forward_classifier(images)
		bbox_preds = self.forward_regressor(images)

		classifier_loss = self.classifier_loss(label_preds, labels)
		self.classifier_val_loss.append(classifier_loss.item())
		regressor_loss = self.regressor_loss(bbox_preds, bboxs)
		self.regressor_val_loss.append(regressor_loss.item())


		return classifier_loss + regressor_loss

	def on_validation_epoch_end(self):
		classifier_val_mean_loss = torch.mean(torch.tensor(self.classifier_val_loss))
		regressor_val_mean_loss = torch.mean(torch.tensor(self.regressor_val_loss))

		self.log('classifier_val_loss', classifier_val_mean_loss.item(), on_epoch=True, sync_dist=True)
		self.log('regressor_val_loss', regressor_val_mean_loss.item(), on_epoch=True, sync_dist=True)

		self.print('Epoch								:', self.current_epoch)
		self.print('Classifier Train Mean loss			:', classifier_val_mean_loss)
		self.print('Regressor Train Mean loss			:', regressor_val_mean_loss)

		# self.save_hyperparameters()

		self.classifier_val_loss = []
		self.regressor_val_loss = []

	# def test_step(self, batch, batch_idx):
	# 	images = batch
	# 	label_preds, bbox_preds = self.forward(images)
		
	# 	test_loss = self.loss(logits, labels)
	# 	self.test_loss.append(test_loss)

	# 	test_acc = self.test_acc(logits, labels)
		
	# 	return(test_loss)

	# def on_test_epoch_end(self):
	# 	test_loss = torch.mean(torch.tensor(self.test_loss))		
	# 	test_acc = self.test_acc.compute()

	# 	self.log('test_loss', test_loss.item(), on_epoch=True, sync_dist=True)
	# 	self.log('test_acc', test_acc.item(), on_epoch=True, sync_dist=True)

	# 	self.print('Epoch : 		', self.current_epoch)
	# 	self.print('Test accuracy : 	', test_acc.item())
	# 	self.print('Test mean loss : 	', test_loss.item())	

	# 	self.test_acc.reset()
	# 	self.test_loss = []
	
	def lr_scheduler_step(self, scheduler, metric):
		if metric:
			print('metric', metric)
			scheduler.step(metric)
		else:
			scheduler.step()
		
	def configure_optimizers(self):
		optimizer_classifier = optim.Adam(list(self.swin_backbone.parameters()) + list(self.classifier.parameters()), lr=self.learning_rate)
		optimizer_regressor = optim.Adam(list(self.swin_backbone.parameters()) + list(self.regressor.parameters()), lr=self.learning_rate)

		scheduler_classifier = optim.lr_scheduler.MultiStepLR(
			optimizer_classifier,
			milestones=[10, 15, 20],
			gamma=0.1,
			verbose=True
		)
		scheduler_regressor = optim.lr_scheduler.MultiStepLR(
			optimizer_regressor,
			milestones=[10, 15, 20],
			gamma=0.1,
			verbose=True
		)

		return (
			{
				"optimizer": optimizer_classifier,
				"lr_scheduler": {
					"scheduler": scheduler_classifier,
					"monitor": "classifier_val_loss",
					"interval": "epoch",
					"frequency": 1
				}
			}, 
			{
				"optimizer": optimizer_regressor,
				"lr_scheduler": {
					"scheduler": scheduler_regressor,
					"monitor": "regressor_val_loss",
					"interval": "epoch",
					"frequency": 1
				}
			}
		)

img_dir = '/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train'
csv_file = '/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv'

data_module = VinBigDataCXRDatamodule(img_dir=img_dir, csv_file=csv_file, batch_size=16, num_workers=4)

data_module.setup(stage="fit")

train_dataloader = data_module.train_dataloader()
val_dataloader = data_module.val_dataloader()

num_classes = 16
learning_rate = 1e-4

# swinv2_model = SwinV2ObjectDetector(num_classes=num_classes)

lightning_model = NeuralNet(num_classes=num_classes, learning_rate=learning_rate)

# logger = TensorBoardLogger("logs", name="swinv2_object_detection")

# checkpoint_callback = ModelCheckpoint(
# 	monitor="regressor_val_loss",
# 	dirpath="./model_history/checkpoints/",
# 	filename="checkpoint-{epoch:02d}-{val_loss:.2f}",
# 	save_top_k=1,
# 	mode="min",
# 	every_n_epochs=1,
# )

# lr_monitor = LearningRateMonitor(logging_interval="step")

trainer = Trainer(
	max_epochs=5,
	devices=1,
	accelerator="gpu",
	max_time=timedelta(hours=1)
	# logger=logger,
	# callbacks=[checkpoint_callback, lr_monitor]
)

trainer.fit(lightning_model, train_dataloader, val_dataloader)



# m = models.swin_v2_b()
# print(m)




