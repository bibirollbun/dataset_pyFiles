!pip -q install -U fastai

from fastai.vision.all import *
import fastai
print("fastai:", fastai.__version__)
print("ImageDataLoaders exists?", "ImageDataLoaders" in dir())


from pathlib import Path
path = Path("/kaggle/input/state-farm-distracted-driver-detection/imgs/train")

dls = ImageDataLoaders.from_folder(
    path,
    valid_pct=0.2,
    seed=42,
    item_tfms=Resize(224),
    batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)],
    bs=32
)

print(dls.vocab)
dls.show_batch(max_n=9, figsize=(7,8))


from fastai.vision.all import *
from fastai.metrics import error_rate # 1 - accuracy
learn = cnn_learner(
    dls,
    resnet34,
    metrics=error_rate
)
learn.path = Path("/kaggle/working")
learn.model_dir = "models"


learn


from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback

cbs = [
    EarlyStoppingCallback(monitor='valid_loss', patience=5),
    SaveModelCallback(monitor='valid_loss', fname='best_resnet34')
]

learn.fine_tune(10, cbs=cbs)
learn.recorder.plot_loss()


!ls {learn.path/'models'}


learn.load('best_resnet34')


def find_appropriate_lr(model:Learner, lr_diff:int = 15, loss_threshold:float = .05, adjust_value:float = 1, plot:bool = False) -> float:
    #Run the Learning Rate Finder
    model.lr_find()
    
    #Get loss values and their corresponding gradients, and get lr values
    losses = np.array(model.recorder.losses)
    min_loss_index = np.argmin(losses)
    
    
    #loss_grad = np.gradient(losses)
    lrs = model.recorder.lrs
    
    #return the learning rate that produces the minimum loss divide by 10   
    return lrs[min_loss_index] / 10


from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Validation set tahminleri
preds, targs = learn.get_preds(dl=dls.valid)   # data = ImageDataLoaders
y_pred = preds.argmax(dim=1).cpu().numpy()
y_true = targs.cpu().numpy()

# Sınıf isimleri
class_names = dls.vocab  # örn: ['Cam Atık', 'Kağıt Atık', ...]

# Classification report
report = classification_report(
    y_true, y_pred,
    target_names=class_names,
    digits=4
)

print(report)


learn.recorder.plot_metrics()


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(8,8))


interp.print_classification_report()


interp.plot_top_losses(9, figsize=(10,10))

