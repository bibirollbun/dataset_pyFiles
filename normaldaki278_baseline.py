from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Путь к папке с изображениями
train_dir = Path('/kaggle/input/neoai-2025/train')

image_paths = sorted(train_dir.glob('*.jpg'))

for path in image_paths[:6]:
    img = Image.open(path)
    plt.imshow(img)
    plt.title(path.name)
    plt.axis('off')
    plt.show()


import pandas as pd
import numpy as np

# Читаем шаблон сабмита
sub = pd.read_csv('/kaggle/input/neoai-2025/sample_submission.csv')
print('✔️ sample_submission.csv считан. Размер:', sub.shape)

sub.head()


random_values = np.random.randint(0, 6, size=sub.iloc[:, 1:].shape)
sub.iloc[:, 1:] = random_values

sub.to_csv('/kaggle/working/submission.csv', index=False)
print('submission.csv сохранён — можно сабмитить или скачать')

