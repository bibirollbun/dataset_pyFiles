import numpy as np
import pandas as pd

df = pd.read_csv('/kaggle/input/h690/h690/jd_sherds_info.csv')
df.head()


df.head(50)


df.nunique()


import cv2
import matplotlib.pyplot as plt

IMG_DIR = '/kaggle/input/h690/h690/sherd_images/'
for i, row in df[:5].iterrows():
    img = cv2.imread(IMG_DIR + row['image_id'] + '.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.title(f'type: {row.type}; part: {row.part}; side: {row.image_side} ')
    plt.axis('off')
    plt.show()


sherd_groups = {}
for _, row in df.iterrows():
    sherd_id = row['sherd_id']
    if sherd_id not in sherd_groups:
        sherd_groups[sherd_id] = {'exterior': [], 'interior': []}
    
    if row['image_side'] == 'exterior':
        sherd_groups[sherd_id]['exterior'].append(row['image_id'])
    else:
        sherd_groups[sherd_id]['interior'].append(row['image_id'])

all_sherds = list(sherd_groups.keys())
np.random.shuffle(all_sherds)

groups = []
used_sherds = set()

for group_id in range(1, 21):
    group_size = np.random.randint(1, 2)
    available_sherds = [s for s in all_sherds if s not in used_sherds]
    
    if len(available_sherds) < group_size:
        available_sherds = all_sherds.copy()
        used_sherds.clear()
    
    group_sherds = available_sherds[:group_size]
    used_sherds.update(group_sherds)

    exterior_list = []
    interior_list = []
    
    for sherd in group_sherds:
        exterior_list.extend(sherd_groups[sherd]['exterior'])
        interior_list.extend(sherd_groups[sherd]['interior'])

    groups.append({
        'group_id': group_id,
        'exterior_ids': ';'.join(exterior_list),
        'interior_ids': ';'.join(interior_list),
        'image_id': ';'.join(group_sherds)
    })

submission_df = pd.DataFrame(groups)
submission_df.to_csv('MySubmission.csv', index=False)
submission_df.head()


submission = []
all_image_ids = df['image_id'].unique().tolist()

for i, image_id in enumerate(all_image_ids[:20], 1):
    sherd_id = df[df['image_id'] == image_id]['sherd_id'].iloc[0]
    submission.append({
        'image_id': image_id,
        'sherd_ids': sherd_id
    })

submission_df = pd.DataFrame(submission)
submission_df.to_csv('submission.csv', index=False)
submission_df.head()




