import os
import cv2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches


dataset_root = '../input/dl-4-cv-aircraft'

examples = ['airplane_11', 'helicopter_45', 'glider_73', 'paraglider_155']
images = [cv2.imread(f'{dataset_root}/train/images/{e}.jpg') for e in examples]
labels = [np.loadtxt(f'{dataset_root}/train/labels/{e}.txt').reshape(-1, 5) for e in examples]

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for img, label, ex, ax in zip(images, labels, examples, axs):
    h, w, _ = img.shape
    x1s, y1s = label[:, 1]*w - label[:, 3]*w / 2, label[:, 2]*h - label[:, 4]*h / 2
    x2s, y2s = label[:, 1]*w + label[:, 3]*w / 2, label[:, 2]*h + label[:, 4]*h / 2

    for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s):
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor='none', edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    ax.set_title(ex.split('_')[0])
    ax.imshow(img)

plt.tight_layout()
plt.show()


# there should be a prediction for each image in the test set
images = [img.split('.')[0] for img in os.listdir(f'{dataset_root}/test/images')]

# randomly generated yolo format bounding boxes,
# with shape (n_img, n_box, 5)
predictions = np.random.uniform(size=(len(images), 2, 5))
predictions[:, :, 0] = np.floor(predictions[:, :, 0] * 3)

# format the boxes to fit the competition submission
formatted = [
    ';'.join(' '.join(box) for box in pred.astype(str))
    for pred in predictions
]

# dataframe it and send it!
random_submission = pd.DataFrame({'image': images, 'boxes': formatted})
random_submission.to_csv('random_submission.csv', index=False)
random_submission.head()

