# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
dataset_dir = '/kaggle/input/object-detection-of-football-matches-objects/dataset'

os.listdir(dataset_dir)


training_dir = f"{dataset_dir}/train"
files = os.listdir(training_dir)


is_json = lambda x: x.endswith("json")
is_img = lambda x: x.endswith("jpg")


import json
coco_json = next(filter(is_json, files))


with open(f"{training_dir}/{coco_json}") as coco_file:
    coco_json = json.load(coco_file)
    # print(coco_json.keys())
    coco_annotations = coco_json.get("annotations")


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


categories = coco_json['categories']
category_map = {cat['id']: cat['name'] for cat in categories}


category_counts = {cat['id']: 0 for cat in categories}

for ann in coco_json['annotations']:
    category_counts[ann['category_id']] += 1


categories


class_distribution = pd.DataFrame([
    {'category_id': cat_id, 'category_name': category_map[cat_id], 'count': count} 
    for cat_id, count in category_counts.items()
])

# Sort by count
class_distribution = class_distribution.sort_values('count', ascending=False)


plt.figure(figsize=(15, 8))
sns.barplot(data=class_distribution, x='category_name', y='count')
plt.title('Class Distribution in COCO Dataset')
plt.xlabel('Class Name')
plt.ylabel('Number of Instances')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Print basic statistics
print("Class Distribution Statistics:")
print(f"Total number of classes: {len(class_distribution)}")
print(f"Total number of instances: {class_distribution['count'].sum()}")
print(f"Average instances per class: {class_distribution['count'].mean():.1f}")
print(f"Class with most instances: {class_distribution.iloc[0]['category_name']} ({class_distribution.iloc[0]['count']})")
print(f"Class with least instances: {class_distribution.iloc[-1]['category_name']} ({class_distribution.iloc[-1]['count']})")

# Calculate imbalance ratio
max_count = class_distribution['count'].max()


import random
import enum


class FootballObject(enum.Enum):
    ball = 1
    coach = 2
    goalkeeper = 3
    player = 4
    referee = 5


num_images = 100
IMG_WIDTH = 640
IMG_HEIGHT = 640

with open(f"submission.csv", "w") as f:
    f.write("img_id,prediction_string\n")

    for n in range(num_images):
        img_id = n

        num_boxes = random.randint(3, 18)
        prediction_strings_arr = []

        for i in range(num_boxes):
            class_id = random.choice(list(FootballObject.__members__.values()))

            conf = round(
                random.uniform(0.5, 0.99), 2
            )  # confidence score for each predicted object

            x_min = random.randint(20, IMG_WIDTH - 50)
            y_min = random.randint(20, IMG_HEIGHT - 50)
            width = random.randint(20, 200)
            height = random.randint(30, 200)
            prediction_string = (
                f"{class_id.value} {x_min} {y_min} {width} {height} {conf}"
            )
            prediction_strings_arr.append(prediction_string)

        prediction_strings = " ".join(prediction_strings_arr)
        f.write(f"{img_id},{prediction_strings}\n")


!head submission.csv


