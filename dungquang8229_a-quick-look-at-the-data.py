import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import requests
import PIL
from io import BytesIO


data_dir = Path("/kaggle/input/plantclef-2025")


df_trn = pd.read_csv(data_dir/"PlantCLEF2024_single_plant_training_metadata.csv", sep=";", low_memory=False)
print (f"Shape: {df_trn.shape}")
df_trn.head()


learn_tag_counts = df_trn['learn_tag'].value_counts()
learn_tag_counts


for tag, count in learn_tag_counts.items():
    percent = (count / len(df_trn)) * 100
    print(f"{tag}: {percent:.2f}%")


df_trn.groupby("learn_tag")["species_id"].nunique()


sampled_data = {}
for tag in ["train", "val", "test"]:
    sampled_data[tag] = df_trn[df_trn["learn_tag"] == tag].sample(5)

fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 9))
fig.subplots_adjust(hspace=0.5)

for row, (tag, data) in enumerate(sampled_data.items()):
    for col, (url, species_id) in enumerate(zip(data["url"], data["species_id"])):
        ax = axes[row, col]  # Lấy ô tương ứng
        try:
            response = requests.get(url, timeout=5)
            img = PIL.Image.open(BytesIO(response.content))
            ax.imshow(img)
            ax.set_title(f"{tag}\nSpecies: {species_id}", fontsize=10)
            ax.axis("off")
        except Exception as e:
            ax.set_title(f"{tag}\nError", fontsize=10)
            ax.axis("off")

plt.show()


organ_counts = df_trn["organ"].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(organ_counts, labels=organ_counts.index, autopct="%1.1f%%", startangle=140, colors=plt.cm.Paired.colors)

plt.title("Organ Distribution")
plt.show()


species_counts = df_trn['species'].value_counts()
genus_counts = df_trn['genus'].value_counts()
family_counts = df_trn['family'].value_counts()

print(f"There are {len(species_counts)} species, {len(genus_counts)} genuses and {len(family_counts)} families")


top_species = species_counts.nlargest(10)
top_genuses = genus_counts.nlargest(10)
top_families = family_counts.nlargest(10)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].pie(top_species, labels=top_species.index, autopct='%1.1f%%', startangle=90)
axes[0].set_title("Top 10 popular species")

axes[1].pie(top_genuses, labels=top_genuses.index, autopct='%1.1f%%', startangle=90)
axes[1].set_title("Top 10 popular genuses")

axes[2].pie(top_genuses, labels=top_genuses.index, autopct='%1.1f%%', startangle=90)
axes[2].set_title("Top 10 popular families")

plt.tight_layout()
plt.show()


df_tst = pd.read_csv(data_dir/"PlantCLEF2025_test.csv", sep=';')
df_tst.tail()


img_dir = Path(data_dir/"PlantCLEF2025_test_images/PlantCLEF2025_test_images")
img_files = [f for f in os.listdir(img_dir)]
sample_images = random.sample(img_files, 5)

fig, axes = plt.subplots(1, len(sample_images), figsize=(15, 5))
for ax, img_name in zip(axes, sample_images):
    img_path = os.path.join(img_dir, img_name)
    img = PIL.Image.open(img_path)
    ax.imshow(img)
    ax.set_title(img_name[:20])
    ax.axis("off")

plt.tight_layout()
plt.show()


author_counts = df_tst['author'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(author_counts, labels=author_counts.index, autopct="%1.1f%%", startangle=140, colors=plt.cm.Paired.colors)

plt.title("Authors")
plt.show()


train_authors = set(df_trn["author"])
test_authors = set(df_tst["author"])

common_authors = test_authors.intersection(train_authors)
print(f"Authors appeared in both train and test: {len(common_authors)} / {len(test_authors)}")
print(list(common_authors))


train_common_authors = df_trn[df_trn["author"].isin(common_authors)]
species_per_author_train = train_common_authors.groupby("author")["species_id"].nunique()

print("The amount of species each authot in common authors took pictures:")
print(species_per_author_train.sort_values(ascending=False))


df_com_trn = pd.read_csv(data_dir/"pseudoquadrats_without_labels_complementary_training_set_urls.csv")
print(df_com_trn.shape)
df_com_trn.tail()


sample_data = df_com_trn.sample(5)

fig, axes = plt.subplots(1, len(sample_data), figsize=(15, 5))

for ax, url in zip(axes, sample_data.iloc[:, 0]):
    try:
        response = requests.get(url, timeout=5)
        img = PIL.Image.open(BytesIO(response.content))
        ax.imshow(img)
        ax.axis("off")
    except Exception as e:
        ax.set_title(f"Error: {e}")
        ax.axis("off")

plt.tight_layout()
plt.show()

