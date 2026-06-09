from PIL import Image
import glob
import pandas as pd
import random
import os
!mkdir train


names0=os.listdir('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/')
print(len(names0))
names=random.sample(names0,5)
print(names)


for name in names:
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


paths=[]
for dirname, _, filenames in os.walk('./train'):
    for filename in filenames:
        paths+=[(os.path.join(dirname, filename))]


from IPython.display import Image

path=paths[0]
print(path.split('/')[-1])
Image(open(path,'rb').read())


path=paths[1]
print(path.split('/')[-1])
Image(open(path,'rb').read())


path=paths[2]
print(path.split('/')[-1])
Image(open(path,'rb').read())


path=paths[3]
print(path.split('/')[-1])
Image(open(path,'rb').read())


path=paths[4]
print(path.split('/')[-1])
Image(open(path,'rb').read())




