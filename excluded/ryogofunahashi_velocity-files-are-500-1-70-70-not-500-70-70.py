import numpy as np

path = '/kaggle/input/waveform-inversion/train_samples/FlatVel_A/'
d1 = np.load(path + 'data/data1.npy', mmap_mode='r')
m1 = np.load(path + 'model/model1.npy', mmap_mode='r')
print(d1.dtype)
print(d1.shape)

print(m1.dtype)
print(m1.shape) # !!!
# It's important to note that the actual shape is not (500, 70, 70), 
# contrary to what's indicated in the comment.

