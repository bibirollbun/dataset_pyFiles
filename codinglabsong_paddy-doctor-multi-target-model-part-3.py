# install fastkaggle if not available
try: import fastkaggle
except ModuleNotFoundError:
    !pip install -q fastkaggle

from fastkaggle import *

comp = 'paddy-disease-classification'
path = setup_comp(comp, install='"fastcore>=1.4.5" "fastai>=2.7.1" "timm>=0.6.2.dev0"')

from fastai.vision.all import *
set_seed(42)

# some additional imports and define trn_path
from fastcore.parallel import *
trn_path = path/'train_images'


df = pd.read_csv(path/'train.csv', index_col='image_id')
df.head()


df.loc['100365.jpg', 'variety']


def get_variety(p): return df.loc[p.name, 'variety']


dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock, CategoryBlock),
    n_inp=1,
    get_items=get_image_files,
    get_y = [parent_label, get_variety],
    splitter=RandomSplitter(0.2, seed=42),
    item_tfms=Resize(192, method='squish'),
    batch_tfms=aug_transforms(size=128, min_scale=0.75)
).dataloaders(trn_path)


dls.show_batch(max_n=6)


arch = 'convnext_base_in22k'


learn = vision_learner(dls, arch, n_out=20).to_fp16()


def disease_loss(inp,disease,variety): return F.cross_entropy(inp[:,:10],disease)
def variety_loss(inp,disease,variety): return F.cross_entropy(inp[:,10:],variety)


def combine_loss(inp,disease,variety): return disease_loss(inp,disease,variety)+variety_loss(inp,disease,variety)


def disease_err(inp,disease,variety): return error_rate(inp[:,:10],disease)
def variety_err(inp,disease,variety): return error_rate(inp[:,10:],variety)


all_metrics = (disease_err,variety_err,disease_loss,variety_loss)


learn = vision_learner(dls, arch, loss_func=combine_loss, 
                       metrics=all_metrics, n_out=20).to_fp16()


# check the number of GPUs available and print it
num_gpus = torch.cuda.device_count()
print(f"Using {num_gpus} GPUs")

if num_gpus > 1:
    learn.model = torch.nn.DataParallel(learn.model)


learn.lr_find(suggest_funcs=(valley, slide))


learn.fine_tune(5, 0.005)


import gc

class ModelTrainer:
    def __init__(self, arch, item, batch, bs=64):
        # create dataloaders using a DataBlock
        self.dls = DataBlock(
            blocks=(ImageBlock, CategoryBlock, CategoryBlock),
            n_inp=1,
            get_items=get_image_files,
            get_y=[parent_label, get_variety],
            splitter=RandomSplitter(0.2, seed=42),
            item_tfms=item,
            batch_tfms=batch,
        ).dataloaders(trn_path, bs=bs,)
        
        # initialize the learner with the specified architecture and loss/metrics
        self.learner = vision_learner(self.dls, arch,
            loss_func=combine_loss, metrics=all_metrics,
            n_out=20
        ).to_fp16()
    
    def find_lr(self):
        # this will run the learning rate finder using the valley and slide suggestion functions
        self.learner.lr_find(suggest_funcs=(valley, slide))
    
    def train(self, epochs=5, lr=0.01):
        # check the number of GPUs available and use DataParallel if more than one is detected
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs")
        if num_gpus > 1:
            self.learner.model = torch.nn.DataParallel(self.learner.model)
        
        # fine-tune the learner
        self.learner.fine_tune(epochs, lr)

        # manually clear up GPU memory
        gc.collect()
        torch.cuda.empty_cache()
        return self.learner



trainer = ModelTrainer(arch=arch, item=Resize(192, method='squish'),
                      batch=aug_transforms(size=128, min_scale=0.75),
                      bs=256)


trainer.find_lr()


learn = trainer.train(lr=0.02)


trainer_crop = ModelTrainer(arch=arch, 
                            item=Resize(192),
                            batch=aug_transforms(size=128, min_scale=0.75),
                            bs=256)


trainer_crop.train(lr=0.02)


trainer_pad = ModelTrainer(arch=arch, 
                            item=Resize((256,192), method=ResizeMethod.Pad, pad_mode=PadMode.Zeros),
                            batch=aug_transforms(size=(171,128), min_scale=0.75),
                            bs=256)


trainer_pad.train(lr=0.02)


trainer_scaled = ModelTrainer(arch=arch, 
                            item=Resize((480,360), method=ResizeMethod.Pad, pad_mode=PadMode.Zeros),
                            batch=aug_transforms(size=(256,192), min_scale=0.75),
                            bs=128)


trainer_scaled.find_lr()


learner = trainer_scaled.train(epochs=12, lr=0.01)


# do inference to get predictions
valid = learner.dls.valid
preds,targs = learner.get_preds(dl=valid)

# find the appropriate error rate
error_rate(preds[:,:10],targs[0])


tta_preds,targs = learner.tta(dl=valid)
error_rate(tta_preds[:,:10],targs[0])


tst_files = get_image_files(path/"test_images").sorted()
test_dl = learner.dls.test_dl(tst_files)

# Then get the predictions
preds,_ = learner.tta(dl=test_dl) # We do not need the ground truth target labels this time


idxs = preds[:, :10].argmax(dim=1)


vocab = np.array(learner.dls.vocab)
vocab


vocab = vocab[0]
results = pd.Series(vocab[idxs], name="idxs")


ss = pd.read_csv(path/'sample_submission.csv')
ss['label'] = results
ss.to_csv('submission.csv', index=False)
!head submission.csv


# submit if not on kaggle website
if not iskaggle:
    from kaggle import api
    api.competition_submit_cli('submission.csv', 'convnext_base_in22k 256x192 12 epochs tta', comp)

