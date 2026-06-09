
from fastai.vision.all import *
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



import fastai
print(fastai.__version__)


import torch
print(torch.cuda.get_device_name(0))


MODEL_NAME = 'Resnet50'
TRAIN = Path('../input/humpback-whale-identification/train/')
TEST = Path('../input/humpback-whale-identification/test/')
LABELS = Path('../input/humpback-whale-identification/train.csv')
SAMPLE_SUB = Path('../input/humpback-whale-identification/sample_submission.csv')
BBOX = Path('../input/generating-whale-bounding-boxes/bounding_boxes.csv')
# Backbone architecture
arch = resnet50
# Number of workers for data preprocessing
num_workers = 4


df = pd.read_csv(LABELS).set_index('Image')
new_whale_df = df[df.Id == "new_whale"] # only new_whale dataset
train_df = df[df.Id != "new_whale"].copy()

unique_labels = np.unique(train_df.Id.values)
labels_list = unique_labels.tolist()
labels_dict = {label: i for i, label in enumerate(unique_labels)}
# labels_dict = dict()
# labels_list = []
# for i in range(len(unique_labels)):
#     labels_dict[unique_labels[i]] = i
#     labels_list.append(unique_labels[i])
# print("Number of classes: {}".format(len(unique_labels)))
# train_df.Id = train_df.Id.apply(lambda x: labels_dict[x])
# train_labels = np.asarray(train_df.Id.values)
# test_names = [f for f in os.listdir(TEST)]
# train_df['image_name'] = train_df.index


# labels_count = train_df.Id.value_counts()
labels_count = df[df.Id != "new_whale"].Id.value_counts()

plt.figure(figsize=(18, 4))
plt.subplot(121)
_, _,_ = plt.hist(labels_count.values)
plt.ylabel("frequency")
plt.xlabel("class size")

plt.title('class distribution; log scale')
labels_count.head()

plt.subplot(122)
_ = plt.plot(labels_count[1:].values)
plt.title('w/o class new_whale; log scale')
plt.xlabel("class")
plt.ylabel("log(size)")
plt.gca().set_yscale('log')


train_df['image_name'] = train_df.index
bbox_df = pd.read_csv(BBOX).set_index('Image')

rs = np.random.RandomState(42) # set random seed to be equal to the sense of life
perm = rs.permutation(len(train_df))

tr_n = train_df['image_name'].values

val_n = train_df['image_name'].values[perm][:1000]

train_labels = set(train_df.loc[tr_n, 'Id'])
valid_df = train_df.loc[val_n]
valid_df = valid_df[valid_df['Id'].isin(train_labels)]
val_n = valid_df['image_name'].values
# train_ids_for_vocab = train_df.loc[train_n, 'Id']
# label_vocab = train_ids_for_vocab.unique().tolist()

# valid_df = train_df.loc[val_n]
# valid_df = valid_df[valid_df['Id'].isin(set(label_vocab))]
# val_n = valid_df['image_name'].values


print('Train/val:', len(tr_n), len(val_n))
print('Train classes', len(train_df.loc[tr_n].Id.unique()))
print('Val classes', len(train_df.loc[val_n].Id.unique()))


from fastai.vision.augment import *
import random
import cv2

# 自定义模糊变换类
class RandomBlur(Transform):
    def __init__(self, blur_strengths=3, p=0.5):
        super().__init__()
        self.blur_strengths = blur_strengths
        self.p = p
        
    def encodes(self, img):
        if random.random() < self.p:
            # 将PIL图像转换为opencv格式
            img_array = np.array(img)
            
            # 随机高斯模糊
            blur_amount = random.randint(1, self.blur_strengths) * 2 + 1  # 必须是奇数
            img_array = cv2.GaussianBlur(img_array, (blur_amount, blur_amount), 0)
            
            # 转回PIL图像
            return PILImage.create(img_array)
        return img

# 基本数据增强
base_aug = aug_transforms(
    max_rotate=20,    # 20度旋转
    max_zoom=2,       # 2倍缩放
    max_warp=0,       # 不使用warp变换
    max_lighting=0.2, # 亮度变化
    do_flip=True,     # 启用翻转
    p_affine=0.75,    # 仿射变换概率
    p_lighting=0.75   # 亮度变换概率
)

blur_transform = RandomBlur(blur_strengths=3, p=0.5)

# 组合所有变换
final_transforms = base_aug + [blur_transform]



# def open_cropped_image(fname):
#     # 拼接完整路径并读取图像
#     img_path = TRAIN / fname
#     img = PILImage.create(img_path)

#     # 查找裁剪框
#     bbox = bbox_df.loc[fname]  # fname 是纯图片名，如 '0000e88ab.jpg'
#     x0, y0, x1, y1 = bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1']

#     # 如果 bbox 合法，就裁剪
#     if x0 < x1 and y0 < y1:
#         img = img.crop((x0, y0, x1, y1))  # PIL 图像支持 crop
#     return img

def open_cropped_image(fname):
    img_path = TRAIN / fname
    # 使用PIL加载图像
    img = PILImage.create(img_path)
    
    # 获取边界框
    bbox = bbox_df.loc[fname]
    x0, y0, x1, y1 = bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1']
    
    # 裁剪(使用与参考代码相同的条件检查)
    if not (x0 >= x1 or y0 >= y1):
        img = img.crop((x0, y0, x1, y1))
    
    # 转换为numpy并调整大小(模拟参考代码行为)
    img_array = np.array(img)
    img_array = cv2.resize(img_array, (384, 384))
    
    # 转回PIL图像格式
    return PILImage.create(img_array)

whale_df = train_df.copy()

vocab = CategoryMap(labels_list, sort=False)

whale_block = DataBlock(
    blocks=(ImageBlock, CategoryBlock(vocab=vocab)),
    get_items=lambda df: df.index.tolist(),  # 从 df 拿图片名
    get_x=open_cropped_image,
    # 关键修改：确保返回的是字符串标签，而不是数字ID
    # get_y=lambda o: df.loc[o, 'Id'] if o in df.index else None,
    get_y=lambda o: train_df.loc[o, 'Id'] if o in train_df.index else None,
    splitter=FuncSplitter(lambda o: o in val_n),  # 用你已有的验证集划分
    item_tfms=Resize(384),
    batch_tfms=aug_transforms(do_flip=True, max_rotate=20, max_zoom=2,
                              max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)
    # batch_tfms=final_transforms
)

# dls = whale_block.dataloaders(df.loc[df.Id != "new_whale"], bs=32, num_workers=4)
dls = whale_block.dataloaders(train_df, bs=32, num_workers=4)



learn = vision_learner(
    dls,
    resnet50,
    metrics=accuracy,
    pretrained=True,
    opt_func=Adam,
    lin_ftrs=[]
)


lrs = slice(1e-4,1e-3)
learn.freeze()
learn.fit_one_cycle(2, lr_max=1e-3)

learn.unfreeze()
learn.fit_one_cycle(16, lr_max=lrs)



test_files = get_image_files(TEST)
test_dl = dls.test_dl(test_files, with_labels=False)
preds_t, _ = learn.tta(dl=test_dl, n=8)
probs = preds_t.softmax(dim=1).numpy()

# 插入 new_whale
best_th = 0.38
probs = np.concatenate([np.full((probs.shape[0], 1), best_th), probs], axis=1)

labels_list_full = ["new_whale"] + labels_list

# 拼 top5
top5_preds = [[labels_list_full[i] for i in p.argsort()[-5:][::-1]] for p in probs]
test_fnames = [f.name for f in test_files]
pred_dic = dict(zip(test_fnames, top5_preds))

sample_df = pd.read_csv(SAMPLE_SUB)
sample_list = list(sample_df.Image)
pred_list_cor = [' '.join(pred_dic[img]) for img in sample_list]

df_sub = pd.DataFrame({'Image': sample_list, 'Id': pred_list_cor})
df_sub.to_csv(f'submission_{MODEL_NAME}.csv', index=False)
df_sub.head()

