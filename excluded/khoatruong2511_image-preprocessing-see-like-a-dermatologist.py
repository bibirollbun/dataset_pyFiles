from scipy import ndimage
import operator
import cv2
import numpy as np 
import os 
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt 


image_paths=os.listdir('../input/siim-isic-melanoma-classification/jpeg/train')
image_paths= ["../input/siim-isic-melanoma-classification/jpeg/train/" + str(x) for x in image_paths]


def resize_and_crop_to_square(img, target_size):
    """
    Resize ảnh theo cạnh nhỏ hơn và cắt thành hình vuông từ phần trung tâm.
    
    Args:
        img : Ảnh đầu vào (numpy array).
        target_size : Kích thước mong muốn của hình vuông đầu ra.
    
    Returns:
        result : Ảnh sau khi resize và cắt thành hình vuông.
    """
    # Lấy kích thước ảnh gốc
    height, width = img.shape[:2]

    # Tính tỷ lệ resize dựa trên cạnh nhỏ hơn
    if width < height:
        resize_ratio = target_size / width
        new_width = target_size
        new_height = int(height * resize_ratio)
    else:
        resize_ratio = target_size / height
        new_height = target_size
        new_width = int(width * resize_ratio)
    
    # Resize ảnh theo tỷ lệ
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Tính tọa độ để cắt hình vuông từ trung tâm
    x1 = (new_width - target_size) // 2
    y1 = (new_height - target_size) // 2
    x2 = x1 + target_size
    y2 = y1 + target_size

    # Cắt hình vuông từ phần trung tâm của ảnh đã resize
    cropped_img = resized_img[y1:y2, x1:x2]

    return cropped_img


def crop_and_zoom(img):
    return resize_and_crop_to_square(img,256)


example_image='../input/siim-isic-melanoma-classification/jpeg/train/ISIC_0368894.jpg'
z=plt.imread(example_image)
plt.imshow(z)


plt.imshow(crop_and_zoom(z))
plt.imsave("example1.png",z)


example_2="../input/siim-isic-melanoma-classification/jpeg/train/ISIC_0094775.jpg"
z2=plt.imread(example_2)
plt.imshow(z2)


plt.imshow(crop_and_zoom(z2))
plt.imsave("example2.png",z2)


example_3="../input/siim-isic-melanoma-classification/jpeg/train/ISIC_0166988.jpg"
z3=plt.imread(example_3)
plt.imshow(z3)


plt.imshow(crop_and_zoom(z3))
plt.imsave("example3.png",z3)


def generate_images(imagelist):
    for x in imagelist:
        img_name=x.split(sep='/')[-1]
        img= plt.imread(x)
        img=crop_and_zoom(img)
        plt.imsave(img_name,img)


generate_images(image_paths)

