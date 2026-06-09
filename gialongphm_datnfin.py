import torch

torch.cuda.is_available()


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import itertools
import seaborn as sns
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.data import Dataset
from skimage.io import imread
from sklearn.metrics import *
from sklearn.model_selection import *
from skimage.io import *
from glob import glob
import warnings


warnings.filterwarnings('ignore')
print("Necessary modules have been imported")


def parse_image(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32) / 255.0  # cháº¯c cháº¯n dtype float32
    return image, label


def load_dataset(file_paths, labels, batch_size=32):
    dataset = Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(file_paths)).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


!unzip -o ../input/diabetic-retinopathy-detection/trainLabels.csv.zip
trainLabels = pd.read_csv("./trainLabels.csv")


!apt install p7zip-full -y
!7z x ../input/diabetic-retinopathy-detection/train.zip.001 "-i!train/11*.jpeg" -y 
# restrict extracted file to about 100 for the disk restriction
!mkdir data
!mv train data/train_11


file_paths = glob("./data/train_11/*.jpeg")

file_basenames = [os.path.basename(f).replace(".jpeg", "") for f in file_paths]


filtered_labels = trainLabels[trainLabels['image'].isin(file_basenames)]['level'].values

print(f"Number of image files: {len(file_paths)}")
print(f"Number of filtered labels: {len(filtered_labels)}")

dataset = load_dataset(file_paths, filtered_labels)


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(5):  
        ax = plt.subplot(1, 5, n + 1)
        plt.imshow(image_batch[n])
        plt.title(int(label_batch[n]))
        plt.axis("off")
    plt.show()


def get_images_by_label(dataset, num_classes=5):
    images = [None] * num_classes  
    labels = [None] * num_classes
    label_counts = {i: 0 for i in range(num_classes)}  

    for image_batch, label_batch in dataset:
        for img, lbl in zip(image_batch, label_batch):
            label = int(lbl)
            if label_counts[label] == 0: 
                images[label] = img
                labels[label] = lbl
                label_counts[label] += 1
            if sum(label_counts.values()) == num_classes: 
                return np.array(images), np.array(labels)
    return np.array(images), np.array(labels)


image_batch, label_batch = get_images_by_label(dataset)


sorted_indices = np.argsort(label_batch)
image_batch = image_batch[sorted_indices]
label_batch = label_batch[sorted_indices]


show_batch(image_batch, label_batch)


import os

base_image_dir = os.path.join('.', 'data/train_11')

trainLabels['path'] = trainLabels['image'].map(lambda x: os.path.join(base_image_dir, '{}.jpeg'.format(x)))
trainLabels['exists'] = trainLabels['path'].map(os.path.exists)
df = trainLabels[trainLabels['exists']]
df = df.drop(columns=['image', 'exists'])
df = df.sample(frac=1).reset_index(drop=True)
df['level'] = df['level'].astype(str)


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ThÆ° má»¥c áº£nh gá»‘c
base_image_dir = './data/train_11/'

# Táº¡o thÆ° má»¥c lÆ°u áº£nh tÄƒng cÆ°á»�ng
aug_dir = './data/augmented'
os.makedirs(aug_dir, exist_ok=True)

# Ä�áº¿m sá»‘ áº£nh má»—i nhÃ£n
levels = df['level'].value_counts()
max_count = levels.max()

# Thiáº¿t láº­p ImageDataGenerator vá»›i augmentation cÆ¡ báº£n
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Duyá»‡t tá»«ng nhÃ£n thiáº¿u áº£nh
for level, count in levels.items():
    if count >= max_count:
        continue  # Bá»� qua nhÃ£n Ä‘Ã£ Ä‘á»§ dá»¯ liá»‡u

    n_to_augment = max_count - count  # sá»‘ áº£nh cáº§n táº¡o thÃªm

    # Láº¥y danh sÃ¡ch áº£nh cá»§a nhÃ£n Ä‘Ã³
    img_paths = df[df['level'] == level]['path'].tolist()

    # ThÆ° má»¥c con lÆ°u áº£nh tÄƒng cÆ°á»�ng theo nhÃ£n
    save_dir = os.path.join(aug_dir, f'level_{level}')
    os.makedirs(save_dir, exist_ok=True)

    augmented_count = 0
    i = 0

    # Láº·p sinh áº£nh tÄƒng cÆ°á»�ng cho tá»›i Ä‘á»§ sá»‘ lÆ°á»£ng cáº§n thiáº¿t
    while augmented_count < n_to_augment:
        img_path = img_paths[i % len(img_paths)]
        img = tf.keras.preprocessing.image.load_img(img_path)
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = x.reshape((1,) + x.shape)  # reshape batch size 1

        # Táº¡o 1 áº£nh tÄƒng cÆ°á»�ng/láº§n, lÆ°u vÃ o thÆ° má»¥c tÆ°Æ¡ng á»©ng
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=save_dir,
                                  save_prefix='aug',
                                  save_format='jpeg'):
            augmented_count += 1
            if augmented_count >= n_to_augment:
                break
        i += 1

print("Augmentation completed.")


# ThÆ° má»¥c áº£nh gá»‘c vÃ  áº£nh tÄƒng cÆ°á»�ng
base_dir_original = './data/train_11'
base_dir_aug = './data/augmented'

# 1. Láº¥y dá»¯ liá»‡u áº£nh gá»‘c tá»« DataFrame df Ä‘Ã£ cÃ³ (Ä‘Ã£ lá»�c áº£nh tá»“n táº¡i vÃ  nhÃ£n)
df_original = df.copy()  # df báº¡n Ä‘Ã£ táº¡o trÆ°á»›c Ä‘Ã³ vá»›i cá»™t 'path' vÃ  'level'

# 2. Láº¥y dá»¯ liá»‡u áº£nh tÄƒng cÆ°á»�ng tá»« thÆ° má»¥c augmented
augmented_data = []

# Láº·p qua cÃ¡c thÆ° má»¥c con tÆ°Æ¡ng á»©ng vá»›i tá»«ng level
for level_dir in os.listdir(base_dir_aug):
    level_path = os.path.join(base_dir_aug, level_dir)
    if not os.path.isdir(level_path):
        continue
    level_num = int(level_dir.split('_')[-1])  # Láº¥y sá»‘ level tá»« tÃªn thÆ° má»¥c nhÆ° 'level_2' -> 2

    # Láº¥y táº¥t cáº£ file áº£nh trong thÆ° má»¥c con
    for file_name in os.listdir(level_path):
        if file_name.lower().endswith(('.jpeg', '.jpg', '.png')):
            img_path = os.path.join(level_path, file_name)
            augmented_data.append({'path': img_path, 'level': str(level_num)})

# Táº¡o DataFrame cho dá»¯ liá»‡u áº£nh tÄƒng cÆ°á»�ng
df_augmented = pd.DataFrame(augmented_data)

# 3. Káº¿t há»£p 2 DataFrame láº¡i
df_full = pd.concat([df_original, df_augmented], ignore_index=True)

# Kiá»ƒm tra sá»‘ lÆ°á»£ng áº£nh má»—i level trong bá»™ dá»¯ liá»‡u káº¿t há»£p
print(df_full['level'].value_counts())

# Hiá»ƒn thá»‹ 5 dÃ²ng Ä‘áº§u tiÃªn Ä‘á»ƒ kiá»ƒm tra
print(df_full.head())


import matplotlib.pyplot as plt

# In sá»‘ lÆ°á»£ng sample trong dataset káº¿t há»£p
print(f"Total number of samples in combined dataset: {len(df_full)}")

# Ä�áº¿m sá»‘ lÆ°á»£ng sample cá»§a tá»«ng lá»›p
level_counts = df_full['level'].value_counts().sort_index()

# Váº½ biá»ƒu Ä‘á»“ cá»™t
plt.figure(figsize=(8,5))
level_counts.plot(kind='bar')
plt.title('Number of samples per class after combining datasets')
plt.xlabel('Class Level')
plt.ylabel('Number of samples')
plt.xticks(rotation=0)
plt.show()



import tensorflow as tf
import os
from tqdm import tqdm  # Thanh tiáº¿n trÃ¬nh

def gaussian_kernel_3ch(size: int, sigma: float):
    # Táº¡o kernel Gaussian 2D chuáº©n
    x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
    y = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
    X, Y = tf.meshgrid(x, y)
    kernel_2d = tf.exp(-(X**2 + Y**2) / (2 * sigma ** 2))
    kernel_2d /= tf.reduce_sum(kernel_2d)  # chuáº©n hÃ³a

    # reshape kernel thÃ nh [size, size, 1, 1]
    kernel_2d = tf.reshape(kernel_2d, [size, size, 1, 1])

    # táº¡o ma tráº­n Ä‘Æ¡n vá»‹ 3x3
    identity = tf.eye(3, dtype=tf.float32)  # shape [3, 3]

    # nhÃ¢n kernel gaussian vá»›i ma tráº­n Ä‘Æ¡n vá»‹ Ä‘á»ƒ cÃ³ kernel 4D [size, size, 3, 3]
    kernel = kernel_2d * identity

    return kernel

def denoise_gaussian_filter(image, kernel_size=5, sigma=1.0):
    # image: tensor [H, W, 3], float32 trong [0,1]
    kernel = gaussian_kernel_3ch(kernel_size, sigma)  # shape [size, size, 3, 3]
    image = tf.expand_dims(image, axis=0)  # ThÃªm batch dim: [1, H, W, 3]
    filtered = tf.nn.conv2d(image, kernel, strides=1, padding='SAME')
    filtered = tf.squeeze(filtered, axis=0)  # Bá»� batch dim -> [H, W, 3]
    filtered = tf.clip_by_value(filtered, 0.0, 1.0)  # Giá»›i háº¡n giÃ¡ trá»‹
    return filtered

def load_and_preprocess_image(image_path):
    # Ä�á»�c file áº£nh, decode, resize, chuáº©n hÃ³a vá»� [0,1]
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32) / 255.0
    return image

def save_image(image_tensor, save_path):
    # Chuyá»ƒn áº£nh float32 [0,1] thÃ nh uint8 [0,255], encode jpeg rá»“i lÆ°u file
    image_uint8 = tf.image.convert_image_dtype(image_tensor, dtype=tf.uint8)
    encoded = tf.io.encode_jpeg(image_uint8)
    tf.io.write_file(save_path, encoded)

# ThÆ° má»¥c lÆ°u áº£nh Ä‘Ã£ khá»­ nhiá»…u
denoised_base_dir = './data/denoised'
os.makedirs(denoised_base_dir, exist_ok=True)

# Duyá»‡t toÃ n bá»™ áº£nh trong df_full (DataFrame báº¡n Ä‘Ã£ táº¡o, gá»“m cá»™t 'path' vÃ  'level')
for idx, row in tqdm(df_full.iterrows(), total=len(df_full)):
    img_path = row['path']
    level = row['level']

    # Táº¡o thÆ° má»¥c con theo nhÃ£n level náº¿u chÆ°a cÃ³
    save_dir = os.path.join(denoised_base_dir, f'level_{level}')
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.basename(img_path)
    save_path = os.path.join(save_dir, filename)

    # Load áº£nh, khá»­ nhiá»…u vÃ  lÆ°u áº£nh
    image = load_and_preprocess_image(img_path)
    denoised_image = denoise_gaussian_filter(image, kernel_size=5, sigma=1.0)
    save_image(denoised_image, save_path)

print("HoÃ n thÃ nh khá»­ nhiá»…u toÃ n bá»™ dataset.")



import os
import pandas as pd

denoised_base_dir = './data/denoised'

denoised_paths = []
denoised_labels = []

for level in os.listdir(denoised_base_dir):
    label = level.split('_')[-1]  # 'level_2' â†’ '2'
    level_dir = os.path.join(denoised_base_dir, level)
    for fname in os.listdir(level_dir):
        if fname.endswith(('.jpeg', '.jpg', '.png')):
            denoised_paths.append(os.path.join(level_dir, fname))
            denoised_labels.append(label)

df_denoised = pd.DataFrame({'path': denoised_paths, 'level': denoised_labels})
df_denoised = df_denoised.sample(frac=1).reset_index(drop=True)




from sklearn.model_selection import train_test_split

df = df_denoised.copy()

df_train, df_temp = train_test_split(
    df, 
    test_size=0.3, 
    stratify=df['level'], 
    random_state=42
)

df_val, df_test = train_test_split(
    df_temp, 
    test_size=0.5, 
    stratify=df_temp['level'], 
    random_state=42
)



print("Train:", df_train['level'].value_counts(normalize=True))
print("Val:  ", df_val['level'].value_counts(normalize=True))
print("Test: ", df_test['level'].value_counts(normalize=True))



print("ğŸ“¦ Tá»•ng sá»‘ áº£nh:")
print(f"Train: {len(df_train)} áº£nh")
print(f"Val:   {len(df_val)} áº£nh")
print(f"Test:  {len(df_test)} áº£nh")

print("\nğŸ”� PhÃ¢n phá»‘i nhÃ£n trong tá»«ng táº­p (sá»‘ lÆ°á»£ng):")
print("Train:")
print(df_train['level'].value_counts().sort_index())
print("\nVal:")
print(df_val['level'].value_counts().sort_index())
print("\nTest:")
print(df_test['level'].value_counts().sort_index())



df_train.to_csv("train_metadata.csv", index=False)
df_val.to_csv("val_metadata.csv", index=False)
df_test.to_csv("test_metadata.csv", index=False)



import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import random
from collections import deque

# --- Cáº¥u hÃ¬nh ---
num_classes = 5
input_shape = (256, 256, 3)
learning_rate = 1e-4
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
batch_size = 8
memory_capacity = 10000

# --- Máº¡ng Q-network ---
def create_q_network():
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes)
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate), loss='mse')
    return model

# --- Bá»™ nhá»› replay buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- MÃ´i trÆ°á»�ng giáº£ láº­p ---
class RetinopathyEnv:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.index = 0

    def reset(self):
        self.index = 0
        return self.get_state()

    def get_state(self):
        img_path = self.df.loc[self.index, 'path']
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=input_shape[:2])
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        return img

    def step(self, action):
        true_label = int(self.df.loc[self.index, 'level'])
        reward = 1 if action == true_label else -1
        done = (self.index == len(self.df) - 1)
        self.index += 1
        next_state = self.get_state() if not done else None
        return next_state, reward, done

# --- DQN Agent ---
class DQNAgent:
    def __init__(self):
        self.q_network = create_q_network()
        self.target_network = create_q_network()
        self.update_target_network()

        self.memory = ReplayBuffer(memory_capacity)
        self.epsilon = epsilon

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(num_classes)
        q_values = self.q_network.predict(state[np.newaxis], verbose=0)[0]
        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < batch_size:
            return
        minibatch = self.memory.sample(batch_size)

        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] if m[3] is not None else np.zeros(input_shape) for m in minibatch])
        dones = np.array([m[4] for m in minibatch])

        q_next = self.target_network.predict(next_states, verbose=0)
        q_target = self.q_network.predict(states, verbose=0)

        for i in range(batch_size):
            if dones[i]:
                q_target[i][actions[i]] = rewards[i]
            else:
                q_target[i][actions[i]] = rewards[i] + gamma * np.amax(q_next[i])

        self.q_network.train_on_batch(states, q_target)

        if self.epsilon > epsilon_min:
            self.epsilon *= epsilon_decay

# --- Huáº¥n luyá»‡n DQN vá»›i lÆ°u mÃ´ hÃ¬nh tá»‘t nháº¥t ---
def train_dqn(agent, env, episodes=10, save_path='best_dqn_model.h5'):
    best_reward = -np.inf  # Khá»Ÿi táº¡o giÃ¡ trá»‹ reward tá»‘t nháº¥t ráº¥t tháº¥p
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
            step += 1
        agent.update_target_network()
        print(f"Episode {e+1}/{episodes} - Total reward: {total_reward} - Steps: {step} - Epsilon: {agent.epsilon:.3f}")

        # LÆ°u mÃ´ hÃ¬nh náº¿u total_reward cao hÆ¡n trÆ°á»›c Ä‘Ã³
        if total_reward > best_reward:
            best_reward = total_reward
            agent.q_network.save(save_path)
            print(f"Best model saved with total reward: {best_reward}")

# --- Sá»­ dá»¥ng ---
env = RetinopathyEnv(df_train)  # df_full chá»©a dá»¯ liá»‡u áº£nh + nhÃ£n
agent = DQNAgent()
train_dqn(agent, env, episodes=10, save_path='best_dqn_model.h5')


import os
import shutil

# CÃ¡c biáº¿n chá»©a dataframe chia split
splits = {
    'train': df_train,
    'val': df_val,
    'test': df_test
}

# ThÆ° má»¥c gá»‘c Ä‘á»ƒ táº¡o
base_output_dir = './data/split_dataset'

for split_name, df_split in splits.items():
    print(f"Processing {split_name} set with {len(df_split)} images...")

    for idx, row in df_split.iterrows():
        src_path = row['path']
        label = row['level']

        # Táº¡o thÆ° má»¥c Ä‘Ã­ch theo split vÃ  label
        dest_dir = os.path.join(base_output_dir, split_name, label)
        os.makedirs(dest_dir, exist_ok=True)

        # TÃªn file giá»¯ nguyÃªn
        filename = os.path.basename(src_path)
        dest_path = os.path.join(dest_dir, filename)

        # Copy file áº£nh sang thÆ° má»¥c Ä‘Ã­ch
        shutil.copy(src_path, dest_path)

print("Copy hoÃ n táº¥t, cáº¥u trÃºc thÆ° má»¥c dataset Ä‘Ã£ sáºµn sÃ ng.")



import shutil

# Ä�Æ°á»�ng dáº«n thÆ° má»¥c gá»‘c dataset Ä‘Ã£ chia
base_dir = './data/split_dataset'

# Danh sÃ¡ch cÃ¡c táº­p
splits = ['train', 'val', 'test']

for split in splits:
    folder_path = f'{base_dir}/{split}'
    zip_path = f'/mnt/data/{split}.zip'  # NÃ©n vÃ o thÆ° má»¥c /mnt/data Ä‘á»ƒ dá»… táº£i vá»�
    shutil.make_archive(base_name=zip_path.replace('.zip', ''), format='zip', root_dir=folder_path)
    print(f"Ä�Ã£ nÃ©n {split} thÃ nh {zip_path}")



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import tensorflow as tf

def evaluate_dqn(agent, df_eval):
    y_true = []
    y_pred = []

    for idx, row in df_eval.iterrows():
        img_path = row['path']
        true_label = int(row['level'])

        # Load áº£nh, resize vÃ  chuáº©n hÃ³a
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0

        # Dá»± Ä‘oÃ¡n nhÃ£n
        pred = agent.act(img_array)

        y_true.append(true_label)
        y_pred.append(pred)

    # TÃ­nh cÃ¡c thÆ°á»›c Ä‘o
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Váº½ confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[str(i) for i in range(agent.q_network.output_shape[-1])],
                yticklabels=[str(i) for i in range(agent.q_network.output_shape[-1])])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    return accuracy, precision, recall, f1



# VÃ­ dá»¥ Ä‘Ã¡nh giÃ¡ trÃªn táº­p test
accuracy, precision, recall, f1 = evaluate_dqn(agent, df_test)





