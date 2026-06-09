import numpy as np
import matplotlib.pyplot as plt
import cv2


img = cv2.imread('/kaggle/input/petfinder-pawpularity-score/train/0007de18844b0dbbb5e1f607da0606e0.jpg')


print(type(img))


img.shape


plt.imshow(img)
plt.show()


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()


img_resize = cv2.resize(img,(256,256))
plt.imshow(img_resize)
plt.show()


h = int(img.shape[1]/2) 
w = int(img.shape[0]/2)
print(h,w)
img_resize = cv2.resize(img,(h,w))
plt.imshow(img_resize)
plt.show()


img_flip = cv2.flip(img,0) #vertical
plt.imshow(img_flip)
plt.show()


img_flip = cv2.flip(img,1) #horizontal
plt.imshow(img_flip)
plt.show()


plt.imshow(img)
plt.show()
img_crop = img[100:500,100:300]
plt.imshow(img_crop)
plt.show()


cv2.imwrite('cropped_image.jpg',img_crop)




