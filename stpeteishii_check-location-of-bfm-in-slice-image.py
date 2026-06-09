from PIL import Image
import glob
import pandas as pd
import random
import os
import cv2
import matplotlib.pyplot as plt
!mkdir train


df=pd.read_csv('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv')
print(df.columns.tolist())
df=df[df['Number of motors']!=0]
display(df)
unique_names = df['tomo_id'].unique().tolist()
os.makedirs("frames", exist_ok=True)
names=random.sample(unique_names,1)
print(names)


for name in names:
    print(name)
    folder_path ="/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/"+name+"/*.jpg"
    image_files = sorted(glob.glob(folder_path))
    images = [
        Image.open(img).resize((Image.open(img).width // 2, Image.open(img).height // 2), Image.LANCZOS)
        for img in image_files
    ]
    images2 = []
    for i in range(0, len(images), 5):
        blended = Image.blend(images[i], images[i+1], alpha=0.20)
        for j in range(i+2, i+5):
            blended = Image.blend(blended, images[j], alpha=0.20)
        images2.append(blended)
    images2[0].save('./train/'+name+".gif", save_all=True, append_images=images2[1:], duration=200, loop=0)


df=df[df['tomo_id'] == name]
display(df)


def show_2d_name(name):
    fig, ax = plt.subplots(figsize=(6,6))  # 2D plot
    dfi = df[df['tomo_id'] == name]
    display(dfi)
    #path = paths[z]
    #image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = dfi.iloc[0,6],dfi.iloc[0,7] 
    #ax.imshow(image, cmap='gray', extent=[0, w, 0, h], origin='upper')
    ax.scatter(dfi['Motor axis 2'], dfi['Motor axis 1'], color='red', alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(0,w)
    ax.set_ylim(h,0)
    #filename = f"frames/{name}_2d.png"
    #plt.savefig(filename)
    plt.show()


def show_2d_name_i(name,i):
    fig, ax = plt.subplots(figsize=(6,6))  # 2D plot
    dfi = df[df['tomo_id'] == name]
    display(dfi)
    path =f"/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/{name}/slice_{i:04d}.jpg"
    print(path)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = dfi.iloc[0,6],dfi.iloc[0,7] 
    ax.imshow(image, cmap='gray', extent=[0, w, 0, h], origin='upper')
    ax.scatter(dfi['Motor axis 2'], h-dfi['Motor axis 1'], color='red', alpha=0.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(0,w)
    ax.set_ylim(0,h)
    #filename = f"frames/{name}_2d.png"
    #plt.savefig(filename)
    plt.show()


#name=unique_names[32]
name=names[0]
print(name)
jpaths=[]
for dirname, _, filenames in os.walk(f'/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/{name}'):
    for filename in filenames:
        jpaths+=[(os.path.join(dirname, filename))]
jpaths.sort()
#display(df)
#zs=sorted(df['Motor axis 0'].astype(int).unique().tolist())


from IPython.display import Image
i=int(df.iloc[0,2]) #Motor axis 0
path=jpaths[i]
print(path)
Image(open(path,'rb').read())


show_2d_name(name)


print(name,i)
show_2d_name_i(name,i)


from IPython.display import Image
path=f'./train/{name}.gif'
Image(open(path,'rb').read())


#!rm -rf frames

