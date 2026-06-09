!git clone https://github.com/lanl/OpenFWI

%cd /kaggle/working/OpenFWI


!ls /kaggle/input/waveform-inversion/train_samples/FlatVel_A/data


import numpy as np
velocity = np.load('/kaggle/input/waveform-inversion/train_samples/FlatVel_A/model/model2.npy')
data = np.load('/kaggle/input/waveform-inversion/train_samples/FlatVel_A/data/data2.npy')


print('Velocity map size:', velocity.shape)
print('Seismic data size:', data.shape)



import matplotlib.pyplot as plt
## Select a sample in the data
sample=14


from matplotlib.colors import ListedColormap
fig, ax = plt.subplots(1, 1, figsize=(11, 5))
img=ax.imshow(velocity[sample,0,:,:],cmap='jet')
ax.set_xticks(range(0, 70, 10))
ax.set_xticklabels(range(0, 700, 100))
ax.set_yticks(range(0, 70, 10))
ax.set_yticklabels(range(0, 700, 100))
ax.set_ylabel('Depth (m)', fontsize=12)
ax.set_xlabel('Offset (m)', fontsize=12)
clb=plt.colorbar(img, ax=ax)
clb.ax.set_title('km/s',fontsize=8)
plt.show()


! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/tutorial_wave.mp4?download=1'  -O /kaggle/working/wave_propagation.mp4

from IPython.display import HTML
from base64 import b64encode
mp4 = open('/kaggle/working/wave_propagation.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=1200 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)


print('Seismic data size:', data.shape)

fig,ax=plt.subplots(1,5,figsize=(20,5))
ax[0].imshow(data[sample,0,:,:],extent=[0,70,1000,0],aspect='auto',cmap='gray',vmin=-0.5,vmax=0.5)
ax[1].imshow(data[sample,1,:,:],extent=[0,70,1000,0],aspect='auto',cmap='gray',vmin=-0.5,vmax=0.5)
ax[2].imshow(data[sample,2,:,:],extent=[0,70,1000,0],aspect='auto',cmap='gray',vmin=-0.5,vmax=0.5)
ax[3].imshow(data[sample,3,:,:],extent=[0,70,1000,0],aspect='auto',cmap='gray',vmin=-0.5,vmax=0.5)
ax[4].imshow(data[sample,4,:,:],extent=[0,70,1000,0],aspect='auto',cmap='gray',vmin=-0.5,vmax=0.5)
for axis in ax:
   axis.set_xticks(range(0, 70, 10))
   axis.set_xticklabels(range(0, 700, 100))
   axis.set_yticks(range(0, 2000, 1000))
   axis.set_yticklabels(range(0, 2,1))
   axis.set_ylabel('Time (s)', fontsize=12)
   axis.set_xlabel('Offset (m)', fontsize=12)
plt.show()



!python train.py -ds flatvel-tutorial -n tutorial -m InversionNet -g1v 1 -g2v 0  --tensorboard -t kaggle_tutorial_train.txt -v kaggle_tutorial_val.txt  --lr 0.0001 -b 120 -eb 10 -nb 5


! ls /kaggle/working/OpenFWI/Invnet_models/tutorial


!python test.py -ds flatvel-tutorial -n tutorial -m InversionNet -v kaggle_tutorial_train.txt -r checkpoint.pth --vis -vb 2 -vsa 3 -o /kaggle/working/OpenFWI/Invnet_models/

from IPython.display import Image
from IPython.display import display
a=Image('./Invnet_models/tutorial/visualization/V_0_0.png', width = 600, height = 300)
b=Image('./Invnet_models/tutorial/visualization/V_0_1.png', width = 600, height = 300)
c=Image('./Invnet_models/tutorial/visualization/V_0_2.png', width = 600, height = 300)
display(a,b,c)


!python test.py -ds flatvel-tutorial -n tutorial -m InversionNet -v kaggle_tutorial_val.txt -r checkpoint.pth --vis -vb 2 -vsa 3 -o /kaggle/working/OpenFWI/Invnet_models/

a=Image('./Invnet_models/tutorial/visualization/V_0_0.png', width = 600, height = 300)
b=Image('./Invnet_models/tutorial/visualization/V_0_1.png', width = 600, height = 300)
c=Image('./Invnet_models/tutorial/visualization/V_0_2.png', width = 600, height = 300)
display(a,b,c)



! mkdir -p /kaggle/working/Invnet_models/pretrained_model/
! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/fva_l1.pth?download=1'   -O /kaggle/working/Invnet_models/pretrained_model/fva_l1.pth


!python test.py -ds flatvel-tutorial -n pretrained_model -m InversionNet -v kaggle_tutorial_val.txt -r fva_l1.pth --vis -vb 2 -vsa 3 -o /kaggle/working/Invnet_models/
a=Image('/kaggle/working/Invnet_models/pretrained_model/visualization/V_0_0.png', width = 600, height = 300)
b=Image('/kaggle/working/Invnet_models/pretrained_model/visualization/V_0_1.png', width = 600, height = 300)
c=Image('/kaggle/working/Invnet_models/pretrained_model/visualization/V_0_2.png', width = 600, height = 300)
display(a,b,c)


!mkdir -p /kaggle/working/Invnet_models/transfer_learning/
! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/cva_l1.pth?download=1'   -O /kaggle/working/Invnet_models/transfer_learning/cva_l1.pth


!python test.py -ds flatvel-tutorial -n transfer_learning -m InversionNet -v kaggle_tutorial_val.txt -r cva_l1.pth --vis -vb 2 -vsa 3 -o /kaggle/working/Invnet_models/

from IPython.display import Image
from IPython.display import display
a=Image('/kaggle/working/Invnet_models/transfer_learning/visualization/V_0_0.png', width = 600, height = 300)
b=Image('/kaggle/working/Invnet_models/transfer_learning/visualization/V_0_1.png', width = 600, height = 300)
c=Image('/kaggle/working/Invnet_models/transfer_learning/visualization/V_0_2.png', width = 600, height = 300)
display(a,b,c)


!python train.py -ds flatvel-tutorial -n transfer_learning -m InversionNet -g1v 1 -g2v 0  --tensorboard -t kaggle_tutorial_train.txt -v kaggle_tutorial_val.txt\
  --lr 0.0001 -b 120 -eb 25 -nb 5 -r cva_l1.pth -o /kaggle/working/Invnet_models/


!python test.py -ds flatvel-tutorial -n transfer_learning -m InversionNet -v kaggle_tutorial_val.txt -r checkpoint.pth --vis -vb 2 -vsa 3 -o /kaggle/working/Invnet_models/
a=Image('/kaggle/working/Invnet_models/transfer_learning/visualization/V_0_0.png', width = 600, height = 300)
b=Image('/kaggle/working/Invnet_models/transfer_learning/visualization/V_0_1.png', width = 600, height = 300)
c=Image('/kaggle/working/Invnet_models/transfer_learning/visualization/V_0_2.png', width = 600, height = 300)
display(a,b,c)




