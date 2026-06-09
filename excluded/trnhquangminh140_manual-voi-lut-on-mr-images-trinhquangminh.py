# Tham khảo từ https://www.kaggle.com/code/davidbroberts/manual-voi-lut-on-mr-images
# Module OS trong Python cung cấp nhiều phương thức hữu ích để quản lý file và folder, 
# giúp tương tác với hệ điều hành dễ dàng hơn. Dưới đây là một số phương thức quan trọng:
# tham khảo import os o trang https://viettuts.vn/python/mo-dun-os-trong-python
# tham khảo import os o trang https://pythonve.ikitai.net/entry/2024/01/08/013800#google_vignette
import os
#-----------------------------------------------------------------------------------
# Numpy là một thư viện lõi phục vụ cho khoa học máy tính của Python, hỗ trợ cho việc 
# tính toán các mảng nhiều chiều, có kích thước lớn với các hàm đã được tối ưu áp dụng
# lên các mảng nhiều chiều đó. Numpy đặc biệt hữu ích khi thực hiện các hàm liên quan 
# tới Đại Số Tuyến Tính.
# tham khảo trang https://viblo.asia/p/
#   gioi-thieu-ve-numpy-mot-thu-vien-chu-yeu-phuc-vu-cho-khoa-hoc-may-tinh-cua-python-maGK7kz9Kj2
# https://codelearn.io/sharing/tim-hieu-thu-vien-numpy-trong-python
import numpy as np
#-----------------------------------------------------------------------------------
import pandas as pd
#-----------------------------------------------------------------------------------
import pydicom
#-----------------------------------------------------------------------------------
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------
import sys
import cv2
import numpy as np


# 8-neighboars - định nghĩa về lân cận của một điểm ảnh
x=0
y=0
arr2 = np.array(([(x-1,y-1,x,y-1,x+1,y-1), (x-1,y-1,x-1,y,x-1,y+1)],
                 [(x-1,y,x,y,x+1,y), (x,y-1,x,y,x,y+1)],
                 [(x-1,y+1,x,y+1,x+1,y+1), (x+1,y-1,x+1,y,x+1,y+1)]), dtype = int)

print(arr2)
print("1- Kiểu dữ liệu của phần tử trong mảng:", arr2.dtype)
print("2- Kích thước của mảng:", arr2.shape)
print("3- Số phần tử trong mảng:", arr2.size)
print("4- Số chiều của mảng:", arr2.ndim)
print("5- Giá trị lớn nhất của mảng arr là:", np.max(arr2))
print("6- Giá trị nhỏ nhất của mảng arr là:", np.min(arr2))
print("7- Tổng tất cả các phần tử của mảng arr là:", np.sum(arr2))
print("8- Trung bình cộng tất cả các phần tử của mảng arr là:", np.mean(arr2))
print("9- Giá trị trung vị của mảng arr là:", np.median(arr2))


# Hàm này cung cấp tên của Module os được import.
# Hiện tại, nó đăng ký 'posix', 'nt', 'os2', 'ce', 'java' và 'riscos'. 
# Ví dụ: posix của kaggle.com
print(os.name)
# Lấy folder làm việc hiện tại (output là /kaggle/working )
current_dir = os.getcwd()
print(f'folder làm việc hiện tại: {current_dir}')
# Liệt kê các file trong folder
files = os.listdir('.')
print('Danh sách file trong folder hiện tại:', files)


# Make a simple linear VOI LUT from the raw (stored) pixel data
def make_lut(pixels, width, center, p_i):
    
    # Slope and Intercept set to 1 and 0 for MR. Get these from DICOM tags instead if using 
    # on a modality that requires them (CT, PT etc)
    slope = 1.0
    intercept = 0.0
    min_pixel = int(np.amin(pixels))
    max_pixel = int(np.amax(pixels))

    # Make an empty array for the LUT the size of the pixel 'width' in the raw pixel data
    lut = [0] * (max_pixel + 1)
    
    # Invert pixels and cent for MONOCHROME1. We invert the specified center so that 
    # increasing the center value makes the images brighter regardless of photometric intrepretation
    invert = False
    if p_i == "MONOCHROME1":
        invert = True
    else:
        center = (max_pixel - min_pixel) - center
        
    # Loop through the pixels and calculate each LUT value
    for pix_value in range(min_pixel, max_pixel):
        lut_value = pix_value * slope + intercept
        voi_value = (((lut_value - center) /  width + 0.5) * 255.0)
        clamped_value = min(max(voi_value, 0), 255)
        if invert:
            lut[pix_value] = round(255 - clamped_value)
        else:
            lut[pix_value] = round(clamped_value)
        
    return lut


# Apply the LUT to a pixel array
def apply_lut(pixels_in, lut):
    
    pixels_in = pixels_in.flatten()
    pixels_out = [0] * len(pixels_in)
    
    for i in range(0, len(pixels_in)):
        pixel = pixels_in[i]
        pixels_out[i] = int(lut[pixel])
        
    return pixels_out


from pydicom.pixel_data_handlers.util import apply_voi_lut
# Load an image
image = pydicom.dcmread('../input/rsna-miccai-brain-tumor-radiogenomic-classification/train/00148/T1wCE/Image-95.dcm')

pixels = image.pixel_array

# Print out the pixel 'width'
print("Min pixel value: " + str(np.min(pixels)))
print("Max pixel value: " + str(np.max(pixels)))

print(image.PhotometricInterpretation)


plt.figure(figsize= (6,6))
plt.imshow(pixels, cmap='gray');
#--------------------------------------------------------------
def change_brightness(img, alpha, beta):
    img_new = np.asarray(alpha*img + beta, dtype=int)   # cast pixel values to int
    img_new[img_new>255] = 255
    img_new[img_new<0] = 0
    return img_new


#if __name__ == "__main__":
#    alpha = 1.0
#    beta = 35.0
#    if len(sys.argv) == 3: 
#        alpha = float(sys.argv[1])
#        beta = float(sys.argv[2])
#    img = cv2.imread('/kaggle/input/mri-images/image3.JPG')  
#    #img = cv2.imread(print(image.PhotometricInterpretation))  
#    # [height, width, channel]
#    # change image brightness g(x,y) = alpha*f(x,y) + beta
#    img_new = change_brightness(img, alpha, beta)
#    cv2.imwrite('/kaggle/input/mri-images/image3_new_tao_moi.jpg', img_new)
#--------------------------------------------------------------


# Plot a histogram of the raw pixel data
fig, axes = plt.subplots(nrows=1, ncols=1,sharex=False, sharey=False, figsize=(10,4))
plt.title('Pixel Range: ' + str(np.min(pixels)) + '-' + str(np.max(pixels)))
plt.hist(pixels.ravel(), np.max(pixels), (1, np.max(pixels)))
plt.tight_layout()
plt.show()


# Apply three different WW/WL settings via LUT. We'll set the center slightly less than half to adjust for brightness.
window_width_1 = np.max(image.pixel_array)
window_center_1 = window_width_1 / 2

lut = make_lut(image.pixel_array, window_width_1, window_center_1, image.PhotometricInterpretation)
image1 = np.reshape(apply_lut(pixels, lut), (pixels.shape[0],pixels.shape[1]))

window_width_2 = 450
window_center_2 = 450

lut = make_lut(image.pixel_array, window_width_2, window_center_2, image.PhotometricInterpretation)
image2 = np.reshape(apply_lut(pixels, lut), (pixels.shape[0],pixels.shape[1]))

window_width_3 = 900
window_center_3 = 90

lut = make_lut(image.pixel_array, window_width_3, window_center_3, image.PhotometricInterpretation)
image3 = np.reshape(apply_lut(pixels, lut), (pixels.shape[0],pixels.shape[1]))


fig, axes = plt.subplots(nrows=2, ncols=2,sharex=True, sharey=True, figsize=(12, 12))
ax = axes.ravel()
ax[0].set_title('Default Image')
ax[0].imshow(image.pixel_array, cmap='gray')
ax[1].set_title(f'Width: {window_width_1} / Center: {window_center_1}')
ax[1].imshow(image1, cmap='gray')
ax[2].set_title(f'Width: {window_width_2} / Center: {window_center_2}')
ax[2].imshow(image2, cmap='gray')
ax[3].set_title(f'Width: {window_width_3} / Center: {window_center_3}')
ax[3].imshow(image3, cmap='gray')
plt.tight_layout()
plt.show()


# Load an image
image = pydicom.dcmread('../input/rsna-miccai-brain-tumor-radiogenomic-classification/train/00014/FLAIR/Image-126.dcm')
pixels = image.pixel_array

# Print out the pixel 'width'
print("Min pixel value: " + str(np.min(pixels)))
print("Max pixel value: " + str(np.max(pixels)))

plt.figure(figsize= (6,6))
plt.imshow(pixels, cmap='gray');


# Apply three different WW/WL settings via LUT. We'll set the level slightly less than half to adjust for brightness.
window_width_1 = 1000
window_center_1 = 900

lut = make_lut(image.pixel_array, window_width_1, window_center_1, image.PhotometricInterpretation)
image1 = np.reshape(apply_lut(pixels, lut), (pixels.shape[0],pixels.shape[1]))

window_width_2 = 600
window_width_2 = 900

lut = make_lut(image.pixel_array, window_width_2, window_center_2, image.PhotometricInterpretation)
image2 = np.reshape(apply_lut(pixels, lut), (pixels.shape[0],pixels.shape[1]))

window_width_3 = 300
window_center_3 = 900

lut = make_lut(image.pixel_array, window_width_3, window_center_3, image.PhotometricInterpretation)
image3 = np.reshape(apply_lut(pixels, lut), (pixels.shape[0],pixels.shape[1]))


fig, axes = plt.subplots(nrows=2, ncols=2,sharex=True, sharey=True, figsize=(12, 12))
ax = axes.ravel()
ax[0].set_title('Default Image')
ax[0].imshow(image.pixel_array, cmap='gray')
ax[1].set_title(f'Width: {window_width_1} / Center: {window_center_1}')
ax[1].imshow(image1, cmap='gray')
ax[2].set_title(f'Width: {window_width_2} / Center: {window_center_2}')
ax[2].imshow(image2, cmap='gray')
ax[3].set_title(f'Width: {window_width_3} / Center: {window_center_3}')
ax[3].imshow(image3, cmap='gray')
plt.tight_layout()
plt.show()


# > pip install aspose-imaging-python-net 
# Tham khảo từ https://blog.aspose.com/vi/imaging/adjust-image-contrast-brightness-gamma-python/
# Điều chỉnh độ tương phản của hình ảnh trong Python
# Đầu tiên, tải hình ảnh bằng phương thức Image.load().
# Sau đó, truyền đối tượng sang loại RasterImage.
# Sau đó, cache hình ảnh nếu nó không sử dụng phương thức RasterImage.cachedata().
# Điều chỉnh độ tương phản trong phạm vi [-100, 100] bằng phương pháp RasterImage.adjustcontrast().
# Cuối cùng, lưu hình ảnh thu được bằng phương thức RasterImage.save().
import aspose.pycore as aspycore
from aspose.imaging import RasterImage, Image
from aspose.imaging.fileformats.tiff.enums import TiffExpectedFormat, TiffPhotometrics
from aspose.imaging.imageoptions import TiffOptions
import os


if 'TEMPLATE_DIR' in os.environ:
	templates_folder = os.environ['TEMPLATE_DIR']
else:
	templates_folder = r"/kaggle/input/dataset-mri/"

delete_output = 'SAVE_OUTPUT' not in os.environ
data_dir = templates_folder
# Tải một hình ảnh trong một phiên bản của Hình ảnh
with Image.load(os.path.join(data_dir, "Template.JPG")) as image:
	# Truyền đối tượng của Image sang RasterImage
	raster_image = aspycore.as_of(image, RasterImage)
	# Kiểm tra xem RasterImage có được lưu vào bộ nhớ cache hay không và Cache RasterImage để có hiệu suất tốt hơn
	if not raster_image.is_cached:
		raster_image.cache_data()

	# Điều chỉnh độ tương phản
	raster_image.adjust_contrast(10)
	# Tạo một phiên bản TiffOptions cho hình ảnh kết quả, Đặt các thuộc tính khác nhau cho đối tượng của TiffOptions và Lưu hình ảnh kết quả thành định dạng TIFF
	tiff_options = TiffOptions(TiffExpectedFormat.DEFAULT)
	tiff_options.bits_per_sample = [8, 8, 8]
	tiff_options.photometric = TiffPhotometrics.RGB
	raster_image.save(os.path.join(data_dir, "result.tiff"), tiff_options)

if delete_output:
	os.remove(os.path.join(data_dir, "result.tiff"))


