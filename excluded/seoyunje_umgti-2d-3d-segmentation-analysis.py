import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from colorama import Fore

import warnings
warnings.filterwarnings('ignore')

print(f"{Fore.BLUE} Complete importing necessary library!!" )


!pip -q install palettable
import palettable.colorbrewer.qualitative as pbq


# Folder Format: <Case>/<Case-day>/<Scans>

# Scans Format: <Slice>_<Slice_Order>_<Slice Width>_<Slice Height>_<Width Pixel Spacing>_<Height Pixel Spacing>


case_list = []
case_day_list = []
case_day_slice_list = []


for path in glob("../input/uw-madison-gi-tract-image-segmentation/train/*"):
    case_list.append(path.split('/')[-1])

for _, day in tqdm(enumerate(case_list), total=len(case_list), desc='Getting MetaData'):
    case_day = [x.split('/')[-1] for x in glob(f'../input/uw-madison-gi-tract-image-segmentation/train/{day}/*')]
    case_day_list.extend(case_day)
    
    for cd in case_day:
        case_day_slice = [cd + '_' + x.split('/')[-1] for x in glob(f'../input/uw-madison-gi-tract-image-segmentation/train/{day}/{cd}/scans/*')]
        case_day_slice_list.extend(case_day_slice)

print(f'{Fore.BLUE}#'*25)
print(f"{Fore.BLUE}Total Case Count is", len(case_list))
print(f"{Fore.BLUE}Total Case Day Count is", len(case_day_list))
print(f"{Fore.BLUE}Total Slice Count is", len(case_day_slice_list))
print('#'*25)

print('\n')
print('#'*25)
print(f"{Fore.BLUE}The Average of Day per Case", len(case_day_list)/len(case_list))
print(f"{Fore.BLUE}The Average of Slice per Scans", len(case_day_slice_list)/len(case_day_list))
print('#'*25)


### Data Processing for Distribution of case & day 

tmp_list = []

for case_day in case_day_list:
    case, day = case_day.split('_')
    case_number = int(case[4:]) 
    day_number = int(day[3:])
    
    tmp = {
        'case_number': case_number,
        'day_number': day_number,
    }
    tmp_list.append(tmp)

tmp_df = pd.DataFrame(tmp_list, columns=['case_number', 'day_number'])
tmp_df = tmp_df.sort_values('day_number')


tmp = tmp_df.groupby(['case_number']).count()['day_number'].value_counts().reset_index()
tmp = tmp.rename(columns={'day_number': 'day_number_count'})
tmp['percentage'] = tmp['count'] / np.sum(tmp['count']) * 100 # Count -> Percentage

sns.set(context='notebook')
plt.figure(figsize=(12,6))
plt.title("Distribution of Day Number Count for each case", size=12, fontweight='bold')
ax = sns.barplot(x=tmp['day_number_count'], y=tmp['percentage'], edgecolor='black', palette=pbq.Pastel1_7.hex_colors)
for rect in ax.patches:
    x = rect.get_x() + rect.get_width() / 2.0
    y = rect.get_height()
    plt.text(x,y,f'{y:.2f}%',ha='center', va='bottom', size=12)
plt.ylabel("percentage")

plt.grid(True)
plt.show()


tmp2 = tmp_df.groupby('case_number')['day_number'].nunique().reset_index()
tmp2 = tmp2.rename(columns={'day_number': 'day_number_count'})

tmp_df = tmp_df.merge(tmp2, on='case_number', how='left')

sns.set(context='notebook', palette='muted')

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
fig.suptitle("Day Number Distributions by Count of Day Occurrence per Case", fontsize=14, fontweight='bold', y=1.02)

# 컬러 팔레트
palette = sns.color_palette("viridis", 6)

# 반복으로 서브플롯 생성
for i in range(1, 7):
    ax = axes[(i-1)//3, (i-1)%3]
    sns.histplot(
        tmp_df[tmp_df['day_number_count'] == i]['day_number'],
        bins='auto',
        kde=True,
        ax=ax,
        color=palette[i-1],
        edgecolor='black'
    )
    ax.set_title(f"Day Count = {i}", fontsize=11)
    ax.set_xlabel("Day Number")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.subplots_adjust(top=0.9)  # 제목 공간 확보
plt.show()



results = []

for slice_ in tqdm(case_day_slice_list, total=len(case_day_slice_list)):

    result = {
        'case': slice_.split('_')[0],
        'day': slice_.split('_')[1],
        'slice_order': int(slice_.split('_')[3]),
        'slice_width': int(slice_.split('_')[4]),
        'slice_height': int(slice_.split('_')[5]),
        'pixel_width_spacing': float(slice_.split('_')[6]),
        'pixel_height_spacing': float(slice_.split('_')[7][:-4]),
    }
    results.append(result)

df_result = pd.DataFrame(results)


## Every Case_Day: 274 cases
## Case_Day which has max slice order is 80: 15 cases 
## Case_Day which has max slice order is 144: 259 cases
## 274 cases = 15 cases + 259 cases

df_result.groupby(['case','day'])['slice_order'].max().value_counts()


from scipy.stats import gaussian_kde

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.title("Distribution of Slice Width & Height", size=12)
sns.kdeplot(x=df_result['slice_width'], shade=True, color='skyblue', label='slice_width', alpha=0.7)
sns.kdeplot(x=df_result['slice_height'], shade=True, color='red', label='slice_height', alpha=0.7)
plt.legend()

plt.subplot(1,2,2)
plt.title("Distribution of Pixel Width & Height Spacing", size=12)
sns.kdeplot(x=df_result['pixel_width_spacing'], shade=True, color='yellow', label='pixel_width', alpha=0.7)
sns.kdeplot(x=df_result['pixel_height_spacing'], shade=True, color='green', label = 'pixel_height', alpha=0.7)
plt.legend()


plt.tight_layout()
plt.show()


tmp_df = df_result.groupby(['case','day']).first().reset_index()

day_counts = tmp_df.groupby('case')['day'].nunique().reset_index(name='day_number_count')
tmp_df = tmp_df.merge(day_counts, on='case')
tmp_df['day_number'] = tmp_df['day'].apply(lambda x: int(x[3:]))


from sklearn.manifold import TSNE

tsne = TSNE(n_components = 2, perplexity = 40, random_state=42, n_iter=5000)
data_X = tmp_df[['slice_height','slice_width','pixel_width_spacing','pixel_height_spacing']]
embs = tsne.fit_transform(data_X)
# Add to dataframe for convenience
plot_x = embs[:, 0]
plot_y = embs[:, 1]


sns.set(style='whitegrid')  # 깔끔한 스타일 적용

plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)
scatter = plt.scatter(
    plot_x, plot_y,
    marker='o',
    s=30,                 # 점 크기 조금 키움
    c=tmp_df['day_number_count'],
    alpha=0.7,            # 약간 더 진한 투명도
    cmap='coolwarm',
    edgecolor='k',        # 점 테두리 검정으로 선명하게
    linewidth=0.3
)

cbar = plt.colorbar(scatter, pad=0.02, fraction=0.05)  # 컬러바 간격 및 크기 조절
cbar.set_label('Number of Days (day_number_count)', size=12, fontweight=12)

plt.title('t-SNE Embedding Colored by day_number_count', fontsize=12, weight='bold')


plt.subplot(1,2,2)

scatter = plt.scatter(
    plot_x, plot_y,
    marker='o',
    s=30,                 # 점 크기 조금 키움
    c=tmp_df['day_number'],
    alpha=0.7,            # 약간 더 진한 투명도
    cmap='coolwarm',
    edgecolor='k',        # 점 테두리 검정으로 선명하게
    linewidth=0.3
)

cbar = plt.colorbar(scatter, pad=0.02, fraction=0.05)  # 컬러바 간격 및 크기 조절
cbar.set_label('Day Number', size=12, fontweight=12)

plt.title('t-SNE Embedding Colored by day_number', fontsize=12, weight='bold')

plt.tight_layout()
plt.show()


from PIL import Image
import cv2

def display_image(path_list, max_images_per_row, apply_clahe=False):
    num_images = len(path_list)
    num_rows = (num_images + max_images_per_row - 1) // max_images_per_row

    image_width = 2.0  # 각 이미지의 가로 크기 (inches)
    image_height = 2.0 # 각 이미지의 세로 크기 (inches)

    fig_width = max_images_per_row * image_width
    fig_height = num_rows * image_height


    fig, axes = plt.subplots(num_rows, max_images_per_row, figsize=(fig_width, fig_height))
    axes = axes.flatten()
    for i, path in enumerate(tqdm(path_list, total=len(path_list))):
    
        img = Image.open(path)
        img = np.array(img) # Image 객체 -> np.array
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
        
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Slice {i+1}")

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()


path_list = glob('/kaggle/input/uw-madison-gi-tract-image-segmentation/train/case107/case107_day0/scans/slice_*')
path_list.sort()


display_image(path_list[:24], 12, apply_clahe=True)


from matplotlib import animation, rc
rc('animation', html='jshtml')

def create_animation(path_list):
    ims = []

    for path in tqdm(path_list, desc="Loading images"):
        img = Image.open(path)
        img = img.resize((128,128))
        img = np.array(img)

        # 16비트 이미지를 8비트로 변환
        img = (img / 65535 * 255).astype(np.uint8)

        # CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        ims.append(img)

    # 애니메이션 생성
    fig = plt.figure(figsize=(6, 6))
    plt.axis('off')
    im = plt.imshow(ims[0], cmap="gray")

    def animate_func(i):
        im.set_array(ims[i])
        return [im]

    plt.close(fig) 

    return animation.FuncAnimation(fig, animate_func, frames=len(ims), interval=1000 // 24)

# 사용 예시
anim = create_animation(path_list)
anim


import plotly.graph_objects as go
import math

def load_image_stack(path_list):
    volume = []

    for path in tqdm(path_list, desc="Loading images"):
        img = Image.open(path)
        img = img.resize((64,64))
        img = np.array(img)
        

        # 16비트 이미지를 8비트로 변환
        img = (img / 65535 * 255).astype(np.uint8)

        # CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        img = img/255

        volume.append(img)

    # (depth, height, width) 형태로 변환
    volume = np.stack(volume, axis=0)
    return volume

def show_3d_volume(volume):
    z, y, x = np.mgrid[0:volume.shape[0], 0:volume.shape[1], 0:volume.shape[2]]

    # 3. Plotly Volume 시각화
    fig = go.Figure(data=go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=volume.flatten(),
        isomin = 0.05, # graph 내 표시 최소 pixel 값
        isomax = 1.0, # graph 내 표시 최대 pixel 값
        opacity=0.3,           # 투명도 증가
        opacityscale="extremes",
        surface_count=10,      # isosurface 수 조절
        colorscale='gray',
    ))

    fig.update_layout(scene_xaxis_showticklabels=False,
                  scene_yaxis_showticklabels=False,
                  scene_zaxis_showticklabels=False)

    angle = math.pi / 4
    
    fig.update_layout(scene_camera = dict(
        up=dict(x=math.cos(angle), y=math.sin(angle), z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0.0, y=0.0, z=2) # XY Plane
    ))
    
    fig.show(renderer="iframe")

# 사용 예시
volume = load_image_stack(path_list[30:60])
show_3d_volume(volume)



# Stomach(위장)
# Small Bowel/Small Intestine(소장)
# Large Bowel/Large Intestine(대장)

train_df = pd.read_csv('/kaggle/input/uw-madison-gi-tract-image-segmentation/train.csv')
train_df['Null'] = train_df['segmentation'].apply(lambda x: 1 if pd.isna(x) else 0)
train_df['slice'] = train_df['id'].apply(lambda x: int(x.split('_')[-1]))
print('Shape of DataFrame: ', train_df.shape)
print(display(train_df))


def check(row):
    if pd.notnull(row['segmentation']):
        class_ = row['class']
        return class_
    else:
        return np.nan

train_df['seg_class'] = train_df.apply(lambda x: check(x), axis=1)


tmp = train_df['Null'].value_counts()

plt.style.use("ggplot")
plt.figure(figsize=(20,5))

plt.subplot(1,2,1)
plt.title("< Distribution of Null>", size=12, fontweight='bold')
plt.pie(tmp, labels=['Null', 'Stomach or Bowel'], autopct='%.1f%%', colors=pbq.Pastel1_7.hex_colors, wedgeprops={'linewidth': 5, 'edgecolor':'white'})
my_circle = plt.Circle((0,0), 0.7, color='white')
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.legend()

tmp = train_df.groupby('slice')['Null'].value_counts().reset_index()
plt.subplot(1,2,2)
sns.barplot(data=tmp[tmp['Null'] == 0], x='slice', y='count', label='Stomach or Bowel', color='blue', alpha=0.3) ## small bowel, large bowel, stomach
sns.barplot(data=tmp[tmp['Null'] == 1], x='slice', y='count', label='Null', color='red', alpha=0.3) 

# 꾸미기
plt.title('Null Value Distribution per Slice', fontsize=18, weight='bold')
plt.xlabel('< Slice >', fontsize=14, fontweight='bold')
plt.ylabel('< Count >', fontsize=14, fontweight='bold')
plt.xticks(rotation=90, size=8)
plt.legend(title='Is Null', title_fontsize=12, fontsize=11)

plt.tight_layout()
plt.show()


tmp = train_df['seg_class'].value_counts()

# 전체 Figure & Subplot 생성
fig, axes = plt.subplots(1, 2, figsize=(20, 5))

axes[0].set_title("< Distribution of Intestine >", fontsize=12, fontweight='bold')
wedges, texts, autotexts = axes[0].pie(
    tmp,
    labels=tmp.index,
    autopct='%.1f%%',
    colors=pbq.Pastel1_7.hex_colors,
    wedgeprops={'linewidth': 5, 'edgecolor': 'white'}
)
my_circle = plt.Circle((0, 0), 0.7, color='white')
axes[0].add_artist(my_circle)
axes[0].legend()

tmp2 = train_df.groupby(['slice'])['seg_class'].value_counts().unstack(fill_value=0)
tmp2.plot(
    kind='bar',
    stacked=True,
    ax=axes[1], 
    edgecolor='black',
    width=1.0,
    colormap='Pastel1'
)

axes[1].set_title("Distribution of Intestine for each slice")
axes[1].tick_params(axis='x', rotation=90, labelsize=8)

plt.tight_layout()
plt.show()


path_list = []

for idx, row in tqdm(df_result.iterrows(), total=len(df_result)):

    case = row['case']
    day = row['day']
    slice_order = row['slice_order']
    
    path = glob(f'/kaggle/*/*/train/{case}/{case}_{day}/scans/slice_{str(slice_order).zfill(4)}_*')
    path_list.extend(path)

df_result['path'] = pd.Series(path_list)

df_result['id'] = df_result['case'] + '_' + df_result['day'] + '_' + 'slice' + '_' + df_result['slice_order'].astype(str).str.zfill(4)
df_result = df_result.drop(columns=['slice_order'])

train_df = train_df.merge(df_result, on='id', how='left')


## background: 0 , mask: 1

def rle_mask(mask_rle, shape):

    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape)


mask_list = []

for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
    rle = row['segmentation']
    shape = (row['slice_height'], row['slice_width'])
    if not pd.isna(rle):
        mask = rle_mask(rle, shape)
        mask_list.append(mask)
    else:
        mask_list.append(np.zeros((shape[0],shape[1]), dtype=np.uint8))

train_df['mask'] = pd.Series(mask_list)


train_df = train_df.sort_values(['id', 'class'])

tmp1 = train_df.groupby('id').agg('first').reset_index().drop(columns=['class', 'seg_class','mask','Null','segmentation'])

tmp2 = train_df.groupby('id')['mask'].agg(list).reset_index()

train_df = tmp1.merge(tmp2, on='id', how='left')


def is_empty(mask_list):
    mask_stack = np.stack(mask_list, axis=-1)  # shape: (H, W, 3)
    return not np.any(mask_stack)  # True if all zeros

train_df['is_empty'] = train_df['mask'].apply(lambda m: 'empty' if is_empty(m) else 'non-empty')


def get_center(mask):
    coord = np.argwhere(mask)
    center = coord.mean(axis=0)
    return center

class_name = ['large_bowel', 'small_bowel', 'stomache']
large_bowel_x_mid = []; small_bowel_x_mid = []; stomache_x_mid = []
large_bowel_y_mid = []; small_bowel_y_mid = []; stomache_y_mid = []

for i, intenstine in enumerate(class_name):
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        center = get_center(row['mask'][i])
        center[0] = center[0] / row['slice_height'] # normalization
        center[1] = center[1] / row['slice_width'] # normalization

        if intenstine == 'large_bowel':
            large_bowel_y_mid.append(center[0])
            large_bowel_x_mid.append(center[1])
        elif intenstine == 'small_bowel':
            small_bowel_y_mid.append(center[0])
            small_bowel_x_mid.append(center[1])
        else:
            stomache_y_mid.append(center[0])
            stomache_x_mid.append(center[1])

train_df['large_bowel_x_mid'] = pd.Series(large_bowel_x_mid)
train_df['large_bowel_y_mid'] = pd.Series(large_bowel_y_mid)
train_df['small_bowel_x_mid'] = pd.Series(small_bowel_x_mid)
train_df['small_bowel_y_mid'] = pd.Series(small_bowel_y_mid)
train_df['stomache_x_mid'] = pd.Series(stomache_x_mid)
train_df['stomache_y_mid'] = pd.Series(stomache_y_mid)


plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
plt.title("Distribution of mid coord for large bowel", size=10, fontweight='bold')
xy = np.vstack([train_df['large_bowel_x_mid'].dropna(), train_df['large_bowel_y_mid'].dropna()])
z = gaussian_kde(xy)(xy)
sns.scatterplot(x=train_df['large_bowel_x_mid'].dropna(), y=train_df['large_bowel_y_mid'].dropna(), c=z, cmap='coolwarm')

plt.subplot(1,3,2)
plt.title("Distribution of mid coord for small bowel", size=10, fontweight='bold')
xy = np.vstack([train_df['small_bowel_x_mid'].dropna(), train_df['small_bowel_y_mid'].dropna()])
z = gaussian_kde(xy)(xy)
sns.scatterplot(x=train_df['small_bowel_x_mid'].dropna(), y=train_df['small_bowel_y_mid'].dropna(), c=z, cmap='coolwarm')

plt.subplot(1,3,3)
plt.title("Distribution of mid coord for stomache", size=10, fontweight='bold')
xy = np.vstack([train_df['stomache_x_mid'].dropna(), train_df['stomache_y_mid'].dropna()])
z = gaussian_kde(xy)(xy)
sns.scatterplot(x=train_df['stomache_x_mid'].dropna(), y=train_df['stomache_y_mid'].dropna(), c=z, cmap='coolwarm')

plt.tight_layout()
plt.show()


tmp_large = train_df[~train_df['large_bowel_x_mid'].isna()]
tmp_small = train_df[~train_df['small_bowel_x_mid'].isna()]
tmp_stomache = train_df[~train_df['stomache_x_mid'].isna()]


import plotly.express as px

xyz = np.vstack([tmp_large['large_bowel_x_mid'], tmp_large['large_bowel_y_mid'], tmp_large['slice']])
z = gaussian_kde(xyz)(xyz)

fig = px.scatter_3d(
    tmp_large,
    x='large_bowel_x_mid',
    y='large_bowel_y_mid',
    z='slice',
    color=z, 
    opacity=0.7,
)

# 카메라 시점 설정
fig.update_layout(
    scene=dict(
        xaxis_title='X (중앙)',
        yaxis_title='Y (중앙)',
        zaxis_title='Slice',
    ),
    title=dict(text='3D Scatter of Large Bowel Midpoints'),
    template="plotly_white"
)

fig.update_traces(marker=dict(
    size=3,
    opacity=0.9,
    colorscale='Plasma',  # 예시: 'Viridis', 'Cividis', 'Inferno', 'Plasma', 'Jet', 'Turbo', 'Bluered'
))

fig.update_layout(
        scene_camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25) # XY Plane
        ))

fig.show(renderer="iframe")


import plotly.express as px

xyz = np.vstack([tmp_small['small_bowel_x_mid'], tmp_small['small_bowel_y_mid'], tmp_small['slice']])
z = gaussian_kde(xyz)(xyz)

fig = px.scatter_3d(
    tmp_small,
    x='small_bowel_x_mid',
    y='small_bowel_y_mid',
    z='slice',
    color=z, 
    opacity=0.7,
)

# 카메라 시점 설정
fig.update_layout(
    scene=dict(
        xaxis_title='X (중앙)',
        yaxis_title='Y (중앙)',
        zaxis_title='Slice',
    ),
    title=dict(text='3D Scatter of Small Bowel Midpoints'),
    template="plotly_white"
)

fig.update_traces(marker=dict(
    size=3,
    opacity=0.9,
    colorscale='Plasma',  # 예시: 'Viridis', 'Cividis', 'Inferno', 'Plasma', 'Jet', 'Turbo', 'Bluered'
))

fig.update_layout(
        scene_camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25) # XY Plane
        ))

fig.show(renderer="iframe")


import plotly.express as px

xyz = np.vstack([tmp_stomache['stomache_x_mid'], tmp_stomache['stomache_y_mid'], tmp_stomache['slice']])
z = gaussian_kde(xyz)(xyz)

fig = px.scatter_3d(
    tmp_stomache,
    x='stomache_x_mid',
    y='stomache_y_mid',
    z='slice',
    color=z, 
    opacity=0.7,
)

# 카메라 시점 설정
fig.update_layout(
    scene=dict(
        xaxis_title='X (중앙)',
        yaxis_title='Y (중앙)',
        zaxis_title='Slice',
    ),
    title=dict(text='3D Scatter of Stomache Midpoints'),
    template="plotly_white"
)

fig.update_traces(marker=dict(
    size=3,
    opacity=0.9,
    colorscale='Plasma',  # 예시: 'Viridis', 'Cividis', 'Inferno', 'Plasma', 'Jet', 'Turbo', 'Bluered'
))

fig.update_layout(
        scene_camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25) # XY Plane
        ))

fig.show(renderer="iframe")


train_df['large_bowel_area'] = train_df['mask'].apply(lambda x: np.sum(x[0]))
train_df['small_bowel_area'] = train_df['mask'].apply(lambda x: np.sum(x[1]))
train_df['stomache_area'] = train_df['mask'].apply(lambda x: np.sum(x[2]))

plt.figure(figsize=(20, 5))
plt.title('Distribution of Area for each Intensite')
sns.kdeplot(train_df['large_bowel_area'], label='Large Bowel', shade=True)
sns.kdeplot(train_df['small_bowel_area'], label='Small Bowel', shade=True)
sns.kdeplot(train_df['stomache_area'], label='Stomache', shade=True)

plt.legend()
plt.show()


tmp_large = train_df.groupby(['slice'])['large_bowel_area'].agg('mean').reset_index(); max_large = np.max(tmp_large['large_bowel_area'])
tmp_small = train_df.groupby(['slice'])['small_bowel_area'].agg('mean').reset_index(); max_small = np.max(tmp_small['small_bowel_area'])
tmp_stomache =  train_df.groupby(['slice'])['stomache_area'].agg('mean').reset_index(); max_stomache = np.max(tmp_stomache['stomache_area'])

plt.figure(figsize=(12,6))
plt.title("Mean of Intenstine Area for each Slice", size=12, fontweight='bold')

sns.scatterplot(x=tmp_large['slice'], y=tmp_large['large_bowel_area'], color='red', alpha=0.3, label='large_bowel')
plt.plot([tmp_large[tmp_large['large_bowel_area'] == max_large]['slice'],tmp_large[tmp_large['large_bowel_area'] == max_large]['slice']],
         [0, max_large], '--', color='red')

sns.scatterplot(x=tmp_small['slice'], y=tmp_small['small_bowel_area'], color='blue', alpha=0.3, label='small_bowel')
plt.plot([tmp_small[tmp_small['small_bowel_area'] == max_small]['slice'],tmp_small[tmp_small['small_bowel_area'] == max_small]['slice']],
         [0, max_small], '--', color='blue')

sns.scatterplot(x=tmp_stomache['slice'], y=tmp_stomache['stomache_area'], color='green', alpha=0.3, label='stomache')
plt.plot([tmp_stomache[tmp_stomache['stomache_area'] == max_stomache]['slice'],tmp_stomache[tmp_stomache['stomache_area'] == max_stomache]['slice']],
         [0, max_stomache], '--', color='green')

plt.ylabel('Mean of Area')
plt.legend()
plt.show()


tmp = train_df.groupby(['case','day'])['large_bowel_area'].agg('sum').reset_index() 
tmp2 = train_df.groupby(['case','day'])['small_bowel_area'].agg('sum').reset_index()
tmp3 =train_df.groupby(['case','day'])['stomache_area'].agg('sum').reset_index()


plt.figure(figsize=(12,6))

plt.title('Distribution of 3D Volume for each Intestine')
sns.kdeplot(tmp['large_bowel_area'], shade=True, label='large_bowel', alpha=0.3, color='skyblue')
sns.kdeplot(tmp2['small_bowel_area'], shade=True, label='small_bowel', alpha=0.3, color='yellow')
sns.kdeplot(tmp3['stomache_area'], shade=True, label='stomache', alpha=0.1, color='green')

plt.legend()
plt.show()

## The small intestine shows the greatest variability in size among individuals, followed by the large intestine, while the stomach exhibits the least variation.
## 3D Volume size: Large Bowel ~= Small Bowel > Stomache 


### Helper Function ###

def mask2contour(mask, width=5):
    h = mask.shape[0]
    w = mask.shape[1]

    mask2 = np.concatenate([mask[:,width:], np.zeros((h,width))], axis=1)
    mask2 = np.logical_xor(mask, mask2)
    mask3 = np.concatenate([mask[width:,:], np.zeros((width,w))], axis=0)
    mask3 = np.logical_xor(mask,mask3)

    return np.logical_or(mask2, mask3)


from matplotlib.patches import Patch
legend_elements = [
        Patch(facecolor='red', edgecolor='r', label='Large Bowel'),
        Patch(facecolor='green', edgecolor='g', label='Small Bowel'),
        Patch(facecolor='blue', edgecolor='b', label='Stomach')
    ]

## Diplay Method1: mask 
## Display Method2: mask2contour

def display_mask(df, num_images, max_images_per_row, method='mask'):
    num_rows = (num_images + max_images_per_row - 1) // max_images_per_row

    image_width = 2.0  # 각 이미지의 가로 크기 (inches)
    image_height = 2.0 # 각 이미지의 세로 크기 (inches)

    fig_width = max_images_per_row * image_width
    fig_height = num_rows * image_height


    fig, axes = plt.subplots(num_rows, max_images_per_row, figsize=(fig_width, fig_height))
    axes = axes.flatten()
    
    for i, idx in enumerate(range(num_images)):

        temp = df.loc[idx]
        path = temp['path']
        img = Image.open(path)
        img = np.array(img) # Image 객체 -> np.array
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)
        img = np.stack([img,img,img], axis=-1)

        large_bowel = temp['mask'][0]
        small_bowel = temp['mask'][1]
        stomache = temp['mask'][2]

        if method == 'mask':
            img[large_bowel != 0,0] = 255 # Red Channel
            img[small_bowel != 0,1] = 255 # Green Channel
            img[stomache != 0,2] = 255 # Blue Channel
        elif method == 'mask2contour':
            img[mask2contour(large_bowel) != 0,0] = 255 # Red Channel
            img[mask2contour(small_bowel) != 0,1] = 255 # Green Channel
            img[mask2contour(stomache) != 0,2] = 255 # Blue Channel
        
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Slice {i+1}")
        axes[i].legend(handles=legend_elements, loc='upper right', 
                       fontsize=6,
                       labelspacing=0.1,     
                       borderpad=0.1,         
                       handlelength=0.5,    
                       handletextpad=0.2,   
                       borderaxespad=0.2,    
                       markerscale=0.5 )
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()


tmp = train_df[train_df['is_empty'] != 'empty'].reset_index(drop=True)

display_mask(tmp, 36, 12, method='mask') # Default


display_mask(tmp, 36, 12, method='mask2contour') # Custom


def mask_animation(df, case, day, num_images=None, include_empty=True):
    ims = []

    if include_empty:
        temp = df[df['id'].str.contains(f'case{case}_day{day}')]
        if num_images: temp = temp[:num_images]
    else:
        df = df[df['is_empty'] == 'non_empty']
        temp = df[df['id'].str.contains(f'case{case}_day{day}')]
        if num_images: temp = temp[:num_images]
    
    for _, row in tqdm(temp.iterrows(), total=len(temp), desc="Loading images"):
        
        path = row['path']
        img = Image.open(path)
        img = img.resize((128,128))
        img = np.array(img)

        # Nomalization & 16bit -> 8bit
        img = (img - img.min())/(img.max() - img.min() + 1e-9)
        img = (img*255).astype(np.uint8)
        img = np.stack([img]*3, axis=-1) # (Height, Width, Channel)

        ## Applying Intesnite Mask
        mask = row['mask']
        mask[0] = cv2.resize(mask[0], (128, 128), interpolation=cv2.INTER_NEAREST)
        mask[1] = cv2.resize(mask[1], (128, 128), interpolation=cv2.INTER_NEAREST)
        mask[2] = cv2.resize(mask[2], (128, 128), interpolation=cv2.INTER_NEAREST)
        
        img[mask[0] != 0, 0] = 255 # Channel 0: Large Bowel 
        img[mask[1] != 0, 1] = 255 # Channel 1: Small Bowel
        img[mask[2] != 0, 2] = 255 # Channel 2: Stomach
        
        ims.append(img)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    im = ax.imshow(ims[0])

    ax.legend(handles=legend_elements, 
                       loc='upper right', 
                       fontsize=6,
                       labelspacing=0.1,     
                       borderpad=0.1,         
                       handlelength=0.5,    
                       handletextpad=0.2,   
                       borderaxespad=0.2,    
                       markerscale=0.5 )
    
    def animate_func(i):
        im.set_array(ims[i])
        return [im]

    plt.close(fig) 

    return animation.FuncAnimation(fig, animate_func, frames=len(ims), interval=1000 // 24)


anim = mask_animation(tmp, case = 101, day = 20)
anim


anim = mask_animation(tmp, case = 102, day = 0)
anim


# intestine 0: large bowel
# intestine 1: small bowel
# intestine 2: stomache


def load_image_stack(df, case, day, start=0, num_images=None, include_empty=True):
    volume = []

    if include_empty:
        temp = df[df['id'].str.contains(f'case{case}_day{day}')]
        if num_images: temp = temp[start:start+num_images]
    else:
        df = df[df['is_empty'] == 'non-empty']
        temp = df[df['id'].str.contains(f'case{case}_day{day}')]
        if num_images: temp = temp[start:start+num_images]
    
    for _, row in tqdm(temp.iterrows(), total=len(temp), desc="Loading images"):
    

        ## Applying Intesnite Mask
        mask_0 = row['mask'][0]
        mask_0 = cv2.resize(mask_0, (64, 64), interpolation=cv2.INTER_NEAREST)
        
        mask_1 = row['mask'][1]
        mask_1 = cv2.resize(mask_1, (64, 64), interpolation=cv2.INTER_NEAREST)

        mask_2 = row['mask'][2]
        mask_2 = cv2.resize(mask_2, (64, 64), interpolation=cv2.INTER_NEAREST)

        mask_0[mask_0 != 0] = 1  # large bowel
        mask_1[mask_1 != 0] = 2  # small bowel
        mask_2[mask_2 != 0] = 3  # stomach

        # 겹치는 부분이 있으면 우선순위 높은 것으로
        mask = np.maximum.reduce([mask_0, mask_1, mask_2])
        
        volume.append(mask)

    # (depth, height, width) 형태로 변환
    volume = np.stack(volume, axis=0)
    return volume

def show_3d_volume(volume):
    z, y, x = np.mgrid[0:volume.shape[0], 0:volume.shape[1], 0:volume.shape[2]]

    # 3. Plotly Volume 시각화
    fig = go.Figure(data=go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=volume.flatten(),
        isomin = 1, # graph 내 표시 최소 pixel 값
        isomax = 3, # graph 내 표시 최대 pixel 값
        opacity=0.3,           # 투명도 증가
        opacityscale="extremes",
        surface_count=10,      # isosurface 수 조절
        colorscale=[
        [0.0, 'red'],     
        [0.5, 'green'], 
        [1.0, 'blue']
    ],
    ))

    fig.update_layout(
                  scene=dict(
                        xaxis_title='X축',
                        yaxis_title='Y축',
                        zaxis_title='Slice',
                    ),     )
        
    angle = math.pi / 4
    
    fig.update_layout(scene_camera = dict(
        up=dict(x=math.cos(angle), y=math.sin(angle), z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0.0, y=2.0, z=0.0) # XY Plane
    ))
    
    fig.show(renderer="iframe")


volume = load_image_stack(train_df, case=9, day=22,  include_empty=False)
show_3d_volume(volume)


volume = load_image_stack(train_df, case=101, day=20,  include_empty=False)
show_3d_volume(volume)


volume = load_image_stack(train_df, case=107, day=0,  include_empty=False)
show_3d_volume(volume)

