from pathlib import Path
input_dir = Path("/kaggle/input")
file_paths = [f for f in input_dir.rglob("*") if f.is_file()]


import torch
import numpy as np
from pathlib import Path
from fastai.vision.all import *

# Set seed for reproducibility
set_seed(42)

# Define the dataset path
path = Path("/kaggle/input/paddy-disease-classification")

# Use fastai's convenience method
print(path.ls())   # lists files/folders inside /kaggle/input


trn_path = path/'train_images'

# Recursively get all images inside class subfolders
files = get_image_files(trn_path)

print(len(files))   # should print 10407
print(files[:5])    # peek at first 5



img = PILImage.create(files[0])
print(img.size)
img.to_thumb(400)



from fastcore.parallel import *

def f(o): return PILImage.create(o).size

sizes = parallel(f, files, n_workers=8)
pd.Series(sizes).value_counts()


dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, seed=42,
    item_tfms=Resize(480, method='squish'),
    batch_tfms=aug_transforms(size=128, min_scale=0.75))

dls.show_batch(max_n=6)


!git clone https://github.com/tinygrad/tinygrad.git
%cd tinygrad
!python3 -m pip install -e .


import time, json
from pathlib import Path
import numpy as np
import pandas as pd
from fastai.vision.all import *
from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_parameters
from tinygrad.nn import optim
from extra.models.resnet import ResNet18   # use ResNet9 for faster debug
from tinygrad import Device


BENCH_FILE = Path("benchmarks.json")

def log_benchmark(entry: dict, file: Path = BENCH_FILE):
    """Append benchmark entry to JSON file."""
    if file.exists():
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(entry)
    with open(file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"ðŸ“Š Benchmark appended to {file}")

def show_benchmarks(file: Path = BENCH_FILE):
    """Display stored benchmark history."""
    if not file.exists():
        print("No benchmarks logged yet.")
        return
    with open(file, "r") as f:
        data = json.load(f)
    print("=== Benchmark History ===")
    for i, run in enumerate(data, 1):
        print(f"Run {i}:")
        print(f"  Model:   {run['model']}")
        print(f"  Device:  {run['device']}")
        print(f"  Epochs:  {run['epochs']}")
        print(f"  Time/epoch: {run['time_per_epoch']:.2f}s")
        print(f"  Last loss:  {run['last_loss']:.4f}")
        print(f"  Eval time:  {run['eval_time']:.2f}s")
        print("")


path = untar_data(URLs.PETS)  # example dataset
dls = ImageDataLoaders.from_name_func(
    path,
    get_image_files(path/"images"),
    valid_pct=0.2,
    seed=42,
    label_func=lambda f: f[0].isupper(),
    item_tfms=Resize(128),
    bs=64,
    device='cpu'    # important: keep fastai dataloader on CPU
)



train_batches = []
for xb, yb in dls.train:
    xb_tiny = Tensor(xb.numpy(), requires_grad=False).to(Device.DEFAULT)
    yb_tiny = Tensor(yb.numpy(), requires_grad=False).to(Device.DEFAULT)
    train_batches.append((xb_tiny, yb_tiny))


print(f"Using tinygrad device: {Device.DEFAULT}")
num_classes = len(dls.vocab)
model = ResNet18(num_classes=num_classes)
optimizer = optim.Adam(get_parameters(model), lr=1e-3)


Tensor.training = True

def train_step(xb, yb):
    out = model(xb)
    loss = out.sparse_categorical_crossentropy(yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss




Tensor.training = True
epochs = 1
epoch_times = []
for epoch in range(epochs):
    start_time = time.time()
    for i, (xb, yb) in enumerate(train_batches):
        loss = train_step(xb, yb)
        if i % 5 == 0:
            elapsed = time.time() - start_time
            print(f"[Epoch {epoch} Batch {i}] "
                  f"loss: {loss.numpy():.4f} | elapsed {elapsed:.2f}s")
    epoch_time = time.time() - start_time
    epoch_times.append(epoch_time)
    print(f"âœ… Epoch {epoch} finished in {epoch_time:.2f}s "
          f"(last loss {loss.numpy():.4f})")


Tensor.training = False
def predict(xb):
    return model(xb).argmax(axis=1)


# -----------------------------
# Evaluation
# -----------------------------
Tensor.training = False
all_preds = []
eval_start = time.time()

tst_files = get_image_files(path/"images").sorted()[:200]  # subset for demo
tst_dl = dls.test_dl(tst_files)

for batch in tst_dl:
    xb = batch[0].numpy().astype(np.float32)
    xb = Tensor(xb).to(Device.DEFAULT)
    preds = predict(xb).numpy().astype(np.int64)
    all_preds.append(preds)

eval_time = time.time() - eval_start
print(f"ðŸ”Ž Evaluation done in {eval_time:.2f}s")
print(f"All preds {all_preds}")



# -----------------------------
# Save predictions
# -----------------------------
idxs = np.concatenate(all_preds)
mapping = dict(enumerate(dls.vocab))
results = pd.Series(idxs, name="idxs").map(mapping)
results


results.to_csv("/kaggle/working/submission.csv", index=False)
print("ðŸ’¾ Submission file written: submission.csv")

# -----------------------------
# Log benchmark
# -----------------------------
log_benchmark({
    "model": "ResNet18",
    "device": str(Device.DEFAULT),
    "epochs": epochs,
    "time_per_epoch": sum(epoch_times)/len(epoch_times),
    "last_loss": float(loss.numpy()),
    "eval_time": eval_time,
})


pd.read_csv("/kaggle/working/submission.csv")




