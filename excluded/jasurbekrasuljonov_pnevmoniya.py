from fastai.vision.all import *

#path
path = Path('/kaggle/input/pnevmoniya/train')

#datablock yaratamiz
pnev = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128)
)

#dataloaders yaratamiz
dls = pnev.dataloaders(path)

#O'qitish train
learn = vision_learner(dls, resnet34, metrics=accuracy)
learn.fine_tune(4)


# Test papkani ko'rsatamiz
test_path = Path('/kaggle/input/pnevmoniya/test')
test_images = get_image_files(test_path)

# DataFrame yaratish
results = []

for img_path in test_images:
    img = PILImage.create(img_path)  # Rasmni ochamiz
    pred, pred_idx, probs = learn.predict(img)  # Bashorat qilamiz
    results.append({"id": img_path.name, "labels": pred})


# DataFrame hosil qilish
df = pd.DataFrame(results)
df["labels"] = df["labels"].map({"NORMAL": 0, "PNEUMONIA": 1})
df.to_csv("/kaggle/working/solution.csv", index=False)

