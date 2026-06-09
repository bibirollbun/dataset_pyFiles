import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_data = pd.read_csv("/kaggle/input/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Train.csv")
train_data.head()


validation_data = pd.read_csv("/kaggle/input/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Val.csv")
test_data = pd.read_csv("/kaggle/input/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Test.csv")

test_data.head()


print("Length of train data before merger:", len(train_data))
train_data = pd.concat([train_data, validation_data], axis=0)
print("Length of train data after merger:", len(train_data))


numeric_cols = []
object_cols = []
for i in train_data.columns:
    if train_data[i].dtypes == "int64" or train_data[i].dtypes == "float64":
        numeric_cols.append(i)
    else:
        object_cols.append(i)
numeric_cols, object_cols


df_numeric = train_data.loc[:, numeric_cols]
df_numeric.head()


object_df = train_data.loc[:, object_cols]
object_df.head()


less_uniques_in_object_cols = []
for i in object_cols:
    print(i)
    u = len(train_data[i].unique())
    if u <= 20:
        print("Uniques: ", train_data[i].unique())
        less_uniques_in_object_cols.append(i)
    else:
        print("Number of uniques: ", u)
    print("++++++++++++++++\n")


less_uniques_in_object_cols.remove("hasCoordinate")
less_uniques_in_object_cols


import os
images_dic_wrt_category_id = {}
iteration_train_data = train_data.loc[:, ["category_id", "filename"]]
train_image_folder_path = "/kaggle/input/fungi-clef-2025/images/FungiTastic-FewShot/train/300p"
val_image_folder_path = "/kaggle/input/fungi-clef-2025/images/FungiTastic-FewShot/val/300p"
train_image_paths = os.listdir(train_image_folder_path)
for idx, row in iteration_train_data.iterrows():
    if row["filename"] not in train_image_paths:
        filename = os.path.join(val_image_folder_path, row["filename"])
    else:
        filename = os.path.join(train_image_folder_path, row["filename"])

    if row["category_id"] not in images_dic_wrt_category_id.keys():
        images_dic_wrt_category_id[row["category_id"]] = [filename]
    else:
        images_dic_wrt_category_id[row["category_id"]].append(filename)


category_val_counts = train_data.loc[:, ["category_id"]].value_counts().reset_index()


class_to_augment = category_val_counts[category_val_counts["count"]<5]["category_id"].tolist()
class_to_augment[:5]


import os
import numpy as np
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


num_augmented_image = 5
datagen = ImageDataGenerator(
    rotation_range=30, 
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

output_dir = "/kaggle/working/dataset"
os.makedirs(output_dir, exist_ok=True)


import cv2
for label, image_list in images_dic_wrt_category_id.items():
    output_path = os.path.join(output_dir, str(label))
    os.makedirs(output_path, exist_ok=True)
    if int(label) not in class_to_augment:
        a=0
        for image_path in image_list:
            img = cv2.imread(image_path)
            cv2.imwrite(os.path.join(output_path, os.path.basename(image_path)), img)
            a+=1
            if a==5:
                break

    else:
        number_of_images = len(image_list)
        for image_path in image_list:
            img = cv2.imread(image_path)
            img_array = np.expand_dims(img, axis=0)
            cv2.imwrite(os.path.join(output_path, os.path.basename(image_path)), img)
        i=number_of_images
        for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_path, save_prefix="aug", save_format="jpg"):
            i+=1
            if i>=5:
                break


!pip install ftfy regex tqdm git+https://github.com/openai/CLIP.git


import os
import clip
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


DATASET_PATH = "/kaggle/working/dataset"

# Dict to hold class prototypes
class_prototypes = {}

def get_embedding(image_path):
    try:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image)
        return embedding.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


for class_name in tqdm(os.listdir(DATASET_PATH), desc="Building prototypes"):
    class_dir = os.path.join(DATASET_PATH, class_name)
    if not os.path.isdir(class_dir): continue

    embeddings = []
    for image_file in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_file)
        emb = get_embedding(image_path)
        if emb is not None:
            embeddings.append(emb)

    if len(embeddings) >= 1:
        prototype = np.mean(embeddings, axis=0)
        class_prototypes[class_name] = prototype
    else:
        print(f"Skipping {class_name}: no valid embeddings")

print(f"Built prototypes for {len(class_prototypes)} classes.")


embeddings[0]


a = 0
for i in os.listdir(DATASET_PATH):
    print(i)
    a+=1
    if a%5==0:
        break


def classify_image(test_image_path, top_k=10):
    test_emb = get_embedding(test_image_path).reshape(1, -1)
    all_classes = list(class_prototypes.keys())
    all_prototypes = np.vstack([class_prototypes[c] for c in all_classes])

    similarities = cosine_similarity(test_emb, all_prototypes)[0]
    top_indices = similarities.argsort()[::-1][:top_k]

    results = [(all_classes[i], similarities[i]) for i in top_indices]
    return results


test_path = os.path.join(DATASET_PATH, "130")
test_path = os.path.join(test_path, os.listdir(test_path)[0])


results = classify_image(test_path)





test_data.head()


test_data_path = "/kaggle/input/fungi-clef-2025/images/FungiTastic-FewShot/test/300p"
result_dic = {}
for oID, filename in test_data.loc[:, ["observationID", "filename"]].values:
    test_image_path = os.path.join(test_data_path, filename)
    results = classify_image(test_image_path)
    r = []
    for res in results:
        r.append(res[0])
    top_10_class = " ".join(r)
    result_dic[oID] = top_10_class


result_dic


sub_df = pd.DataFrame(list(result_dic.items()), columns=["observationId", "predictions"])
sub_df.head()


sub_df.to_csv("submission.csv", index=False)




