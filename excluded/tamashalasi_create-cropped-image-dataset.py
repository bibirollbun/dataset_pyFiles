import os
import random
from shutil import copy, make_archive
import pandas as pd
from PIL import Image
import xml.etree.ElementTree as xml

data_root = '/kaggle/input/imagenet-object-localization-challenge'
k = 100 # randomly select 100 images from both train and test data folder
os.makedirs('./dataset', exist_ok=True) # create a dataset folder to hold all the files that I wanted to download

objs = ['banana', 'pineapple', 'electric guitar', 'envelope', 'football helmet', 'frypan', 'hammer', 'handkerchief', 'joystick', 'lipstick', 'loudspeaker', 'microwave', 'paper towel', 'pillow', 'ping-pong ball', 'pot', 'refrigerator', 'saxophone', 'soccer ball', 'tripod', 'plate', 'pretzel', 'hotdog', 'pizza']

with open(os.path.join(data_root, 'LOC_synset_mapping.txt')) as f: mapping_file = f.read()
mappings = [
    mapping
    for mapping in (
        (
            # 2. first word is the ID
            words[0],
            # 3. every other word can be joined back to get our label, which is a comma-separated list of different names of the category
            [version.strip() for version in " ".join(words[1:]).split(",")],
        )
        for words in
        # 1. split file into lines, then the each line into words
        (line.lower().split() for line in mapping_file.split("\n") if line)
    )
    # 4. filter out the mappings that are not in our list of objects
    if any((category in objs) for category in mapping[1])
]

for mapping in mappings:
    id, names = mapping
    files = os.listdir(os.path.join(data_root, 'ILSVRC/Annotations/CLS-LOC/train', id))
    target_dir = os.path.join('./dataset', names[0])
    os.makedirs(target_dir, exist_ok=True)
    print(names[0])
    
    for file in files:
        image_filename = file.replace('.xml', '.JPEG')
        src_image_path = os.path.join(data_root, 'ILSVRC/Data/CLS-LOC/train', id, image_filename)
        target_image_path = os.path.join(target_dir, file.replace('.xml', '.JPEG'))

        if not os.path.exists(src_image_path):
            print(f"Image not found: {src_image_path}")
            continue
        if os.path.exists(target_image_path):
            print(f"Target image already exists: {target_image_path}")
            continue
        
        # get bounding box
        annotation_path = os.path.join(data_root, 'ILSVRC/Annotations/CLS-LOC/train', id, file)
        with open(annotation_path) as f:
            annotation = f.read()
        tree = xml.parse(annotation_path)
        root = tree.getroot()
        bndbox = root.find('object/bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # crop and save the image
        image = Image.open(src_image_path)
        image = image.crop((xmin, ymin, xmax, ymax))
        
        image.save(target_image_path)


from shutil import copy, make_archive
make_archive(base_name='download_dataset', format='zip', root_dir='dataset')

