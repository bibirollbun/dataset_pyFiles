import numpy as np
import pandas as pd
import fastai
import torch
from fastai.vision.all import *
from ipywidgets import widgets


train_path = Path("/kaggle/input/pnevmoniya/train")
normal_path = Path(f"{train_path}/NORMAL")
pneumonia_path = Path(f"{train_path}/PNEUMONIA")


# Datablock create
transports = DataBlock(
    blocks = (ImageBlock, CategoryBlock),
    get_items = get_image_files,
    splitter = RandomSplitter(valid_pct=0.2, seed=12),
    get_y = parent_label,
    item_tfms = Resize(224)
)

# Dataloader create
dls = transports.dataloaders(train_path)

# Datasetni tekshirish
dls.train.show_batch(max_n=32, nrows=4)





learn = cnn_learner(dls, resnet34, metrics=accuracy)
learn.fine_tune(epochs=4, base_lr=0.001)


# checking
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


interp.plot_top_losses(5)


test_path = Path('/kaggle/input/pnevmoniya/test')
test_files = get_image_files(test_path)

test_dl = learn.dls.test_dl(test_files, with_labels=False)

preds, _ = learn.get_preds(dl=test_dl)
predicted_classes = preds.argmax(dim=1).numpy()  # Convert softmax output to class indices

# Extract image file names
image_ids = [os.path.basename(file) for file in test_files]

# Create a DataFrame for submission
submission_df = pd.DataFrame({
    'id': image_ids,
    'labels': predicted_classes
})

# Save the DataFrame to a CSV file suitable for Kaggle submission
submission_df.to_csv('submission.csv', index=False)

# Print out to confirm completion
print("Submission CSV is ready!")


submission_df.head()




