!pip install py7zr


# Required imports
import numpy as np
import pandas as pd
from py7zr import unpack_7zarchive
import shutil
from fastai.vision.all import *
from functools import partial


# Define CutOut transformation manually
class CutOut(RandTransform):
    def __init__(self, n_holes=1, length=20, **kwargs):
        super().__init__(**kwargs)
        self.n_holes, self.length = n_holes, length
        
    def encodes(self, img:TensorImage):
        h, w = img.shape[-2:]
        mask = torch.ones((h, w), device=img.device)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        return img * mask.unsqueeze(0)


# Unpack the dataset
shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)
shutil.unpack_archive('../input/cifar-10/train.7z', '/kaggle/temp/')
train_labels = pd.read_csv("../input/cifar-10/trainLabels.csv", header="infer")
classes = train_labels['label'].unique()
print(classes)

# Create validation folder
if not os.path.exists("/kaggle/temp/valid"):
    os.mkdir("/kaggle/temp/valid")

# Set paths
parent_path_train = "/kaggle/temp/train"
parent_path_valid = "/kaggle/temp/valid"

# Create class folders
for class1 in classes:
    path_train = os.path.join(parent_path_train, class1)
    if not os.path.exists(path_train):
        os.mkdir(path_train)
    path_valid = os.path.join(parent_path_valid, class1)
    if not os.path.exists(path_valid):
        os.mkdir(path_valid)

# Split training data into train and valid sets
for (int_ind, row) in train_labels.iterrows():
    id = str(row["id"])+".png"
    source_path = os.path.join(parent_path_train, id)
    
    p = np.random.random()
    if p <= 0.8:
        target_path = os.path.join(parent_path_train, row["label"], id)
        os.replace(source_path, target_path)
    else:
        target_path = os.path.join(parent_path_valid, row["label"], id)
        os.replace(source_path, target_path)

# Define advanced data augmentation methods including CutOut
batch_tfms = [*aug_transforms(), Normalize.from_stats(*imagenet_stats), CutOut(n_holes=1, length=20)]


# Fixed hyperparameters
best_params = {'weight_decay': 3.1595250466032746e-05,
               'learning_rate': 0.0007394699205144519,
               'batch_size': 16, 
               'dropout_rate': 0.10845782679503657}


num_epochs = 100  # Set the number of epochs variable

# Create DataLoaders with fixed batch size
dls = ImageDataLoaders.from_folder(path='/kaggle/temp', train='train', valid='valid',
                                   item_tfms=Resize(224), batch_tfms=batch_tfms, bs=best_params['batch_size'])

# Define AdamW using partial
adamw_opt = partial(Adam, wd=best_params['weight_decay'])

# Create learner with resnet50 with randomly initialized weights and advanced optimizer AdamW
learn = vision_learner(dls, resnet50, pretrained=False, metrics=accuracy, wd=best_params['weight_decay'], opt_func=adamw_opt)

# Apply the fixed dropout rate
if hasattr(learn.model[1], 'p'):
    learn.model[1].p = best_params['dropout_rate']

# Manually implement warmup + cosine annealing
def cosine_warmup_schedule(learn, warmup_epochs=5, max_lr=best_params['learning_rate'], num_epochs=num_epochs):
    # Warmup phase
    for epoch in range(warmup_epochs):
        warmup_lr = max_lr * (epoch + 1) / warmup_epochs
        print(f"Epoch {epoch + 1}/{num_epochs}: Warmup with LR = {warmup_lr}")
        learn.fit_one_cycle(1, lr_max=warmup_lr)
    
    # Cosine annealing phase
    learn.fit_flat_cos(num_epochs - warmup_epochs, lr=max_lr)

# Train the model using cosine annealing with manual warmup
cosine_warmup_schedule(learn)


# Test set prediction and submission (same as before)
shutil.unpack_archive('/kaggle/input/cifar-10/test.7z', '/kaggle/temp/test')
shutil.unregister_unpack_format('7zip')
path = '/kaggle/temp/test/test'
f = os.listdir(path)
new = [str(path)+'/'+s for s in f]
test_dl = learn.dls.test_dl(new)
class_score, y = learn.get_preds(dl=test_dl)
class_score = np.argmax(class_score, axis=1)
classScore = class_score.tolist()

learn.dls.vocab
classes = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
predicted_classes = np.empty(shape=300000, dtype=np.dtype('U20'))

ind = 0
for i in classScore:
    predicted_classes[ind] = classes[i]
    ind += 1

directory = '/kaggle/temp/test/test'
ImageId = [''.join(filter(str.isdigit, name)) for name in os.listdir(directory)]
submission = pd.DataFrame({
    "id": ImageId,
    "label": predicted_classes
})

submission.to_csv("submission.csv", index=False)
learn.save('/kaggle/working/vision')

