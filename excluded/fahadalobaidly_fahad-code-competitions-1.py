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


from fastai.vision.all import *



!pip install py7zr



from py7zr import unpack_7zarchive
import shutil
shutil.register_unpack_format('7zip',['.7z'],unpack_7zarchive)


shutil.unpack_archive('../input/cifar-10/train.7z', '/kaggle/temp/')



train_labels = pd.read_csv("../input/cifar-10/trainLabels.csv", header="infer")

classes = train_labels['label'].unique()
print(classes)


if not os.path.exists("/kaggle/temp/valid"):
    os.mkdir("/kaggle/temp/valid")
    
parent_path_train = "/kaggle/temp/train"
parent_path_valid = "/kaggle/temp/valid"
# parent_path_test = "/kaggle/temp/test"

for class1 in classes:
    path_train = os.path.join(parent_path_train,class1)
    if not os.path.exists(path_train):
        os.mkdir(path_train)
    path_valid = os.path.join(parent_path_valid,class1)
    if not os.path.exists(path_valid):
        os.mkdir(path_valid)
for (int_ind,row) in train_labels.iterrows():
    id = str(row["id"])+".png"
    source_path = os.path.join(parent_path_train,id)
    
    p=np.random.random()
    if p<=0.8:
        target_path = os.path.join(parent_path_train,row["label"],id)
        os.replace(source_path, target_path)
    else:
        target_path = os.path.join(parent_path_valid,row["label"],id)
        os.replace(source_path, target_path)


dls=ImageDataLoaders.from_folder(path='/kaggle/temp',
                            train='train', valid='valid',item_tfms=Resize(224),bs=64)


dls=ImageDataLoaders.from_folder(path='/kaggle/temp',
                            train='train', valid='valid',item_tfms=Resize(224),bs=64)


dls.show_batch()


learn = vision_learner(dls, resnet50, metrics=accuracy)



learn.lr_find()



learn.fit_one_cycle(10, 3e-3)



learn.unfreeze()



learn.lr_find()



learn.fit_one_cycle(5, lr_max=slice(1e-6,3e-5))



learn.recorder.plot_loss()



shutil.unpack_archive('/kaggle/input/cifar-10/test.7z','/kaggle/temp/test')
shutil.unregister_unpack_format('7zip')


learn.save('/kaggle/working/vision')


len(os.listdir('/kaggle/temp/test/test'))


path='/kaggle/temp/test/test'
f=os.listdir(path)
new=[str(path)+'/'+s for s in f]


test_dl=learn.dls.test_dl(new)



class_score,y=learn.get_preds(dl=test_dl)



class_score


class_score = np.argmax(class_score, axis=1)



class_score[1].item()



class_score


classScore=class_score.tolist()


len(classScore)


learn.dls.vocab


classes={0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}


predicted_classes=np.empty(shape=300000,dtype=np.dtype('U20'))


ind=0
for i in (classScore):
    predicted_classes[ind]=classes[i]
    ind=ind+1


predicted_classes


directory = '/kaggle/temp/test/test'
ImageId = [ (''.join(filter(str.isdigit, name ))) for name in os.listdir(directory)]


submission  = pd.DataFrame({
    "id": ImageId,
    "label": predicted_classes
})
# submission.sort_values(by=["ImageId"], inplace = True)
submission.to_csv("submission.csv", index=False)
display(submission.head(3))
display(submission.tail(3))


submission['id']=submission['id'].astype(int)



submission.to_csv('submission.csv', index=False)


im = Image.open('/kaggle/temp/test/test/134190.png')
im.to_thumb(254,254)


