!pip install pydicom Pillow


import pydicom
from PIL import Image
import os
import numpy as np



def convert_dicom_to_jpg(input_dirs, output_dir):
    # ایجاد دایرکتوری خروجی در صورت عدم وجود
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for input_dir in input_dirs:
        for root, _, files in os.walk(input_dir):
            for filename in files:
                if filename.endswith('.dcm'):
                    dicom_path = os.path.join(root, filename)
                    dicom_image = pydicom.dcmread(dicom_path)

                    # تبدیل تصویر DICOM به فرمت مناسب
                    image_data = dicom_image.pixel_array

                    # بررسی نوع داده تصویر و تبدیل به حالت مناسب
                    if image_data.dtype == np.uint16:
                        # نرمال‌سازی 16 بیتی به 8 بیتی (حفظ مقیاس مناسب)
                        image_data = (image_data - np.min(image_data)) * (255.0 / (np.max(image_data) - np.min(image_data)))
                        image_data = np.clip(image_data, 0, 255).astype(np.uint8)
                    elif image_data.dtype == np.int16:
                        # نرمال‌سازی به 8 بیتی (حفظ مقیاس مناسب)
                        image_data = (image_data - np.min(image_data)) * (255.0 / (np.max(image_data) - np.min(image_data)))
                        image_data = np.clip(image_data, 0, 255).astype(np.uint8)

                    # افزایش کنتراست با استفاده از Histogram Equalization (اختیاری)
                    # در اینجا، از OpenCV برای انجام Equalization استفاده می‌کنیم
                    # img = cv2.equalizeHist(image_data)  # اگر بخواهید از این روش استفاده کنید

                    img = Image.fromarray(image_data)
                    img = img.convert('L')  # تبدیل به سیاه و سفید

                    # تبدیل مجدد به RGB برای ذخیره‌سازی
                    img = img.convert('RGB')

                    # تغییر اندازه تصویر به 224 در 224
                    img = img.resize((224, 224))

                    # ساخت دایرکتوری مقصد با حفظ ساختار دایرکتوری اصلی
                    relative_path = os.path.relpath(root, input_dir)
                    output_subdir = os.path.join(output_dir, relative_path)

                    if not os.path.exists(output_subdir):
                        os.makedirs(output_subdir)

                    # ذخیره‌سازی تصویر به فرمت JPG
                    jpg_filename = filename.replace('.dcm', '.jpg')
                    jpg_path = os.path.join(output_subdir, jpg_filename)
                    img.save(jpg_path)

# دایرکتوری‌های ورودی و خروجی
input_dirs = ['/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images']
output_dir = '/kaggle/working/test_images'

convert_dicom_to_jpg(input_dirs, output_dir)
print("تبدیل به اتمام رسید!")



def convert_dicom_to_jpg(input_dirs, output_dir):
    # ایجاد دایرکتوری خروجی در صورت عدم وجود
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for input_dir in input_dirs:
        for root, _, files in os.walk(input_dir):
            for filename in files:
                if filename.endswith('.dcm'):
                    dicom_path = os.path.join(root, filename)
                    dicom_image = pydicom.dcmread(dicom_path)

                    # تبدیل تصویر DICOM به فرمت مناسب
                    image_data = dicom_image.pixel_array

                    # بررسی نوع داده تصویر و تبدیل به حالت مناسب
                    if image_data.dtype == np.uint16:
                        # نرمال‌سازی 16 بیتی به 8 بیتی (حفظ مقیاس مناسب)
                        image_data = (image_data - np.min(image_data)) * (255.0 / (np.max(image_data) - np.min(image_data)))
                        image_data = np.clip(image_data, 0, 255).astype(np.uint8)
                    elif image_data.dtype == np.int16:
                        # نرمال‌سازی به 8 بیتی (حفظ مقیاس مناسب)
                        image_data = (image_data - np.min(image_data)) * (255.0 / (np.max(image_data) - np.min(image_data)))
                        image_data = np.clip(image_data, 0, 255).astype(np.uint8)

                    # افزایش کنتراست با استفاده از Histogram Equalization (اختیاری)
                    # در اینجا، از OpenCV برای انجام Equalization استفاده می‌کنیم
                    # img = cv2.equalizeHist(image_data)  # اگر بخواهید از این روش استفاده کنید

                    img = Image.fromarray(image_data)
                    img = img.convert('L')  # تبدیل به سیاه و سفید

                    # تبدیل مجدد به RGB برای ذخیره‌سازی
                    img = img.convert('RGB')

                    # تغییر اندازه تصویر به 224 در 224
                    img = img.resize((224, 224))

                    # ساخت دایرکتوری مقصد با حفظ ساختار دایرکتوری اصلی
                    relative_path = os.path.relpath(root, input_dir)
                    output_subdir = os.path.join(output_dir, relative_path)

                    if not os.path.exists(output_subdir):
                        os.makedirs(output_subdir)

                    # ذخیره‌سازی تصویر به فرمت JPG
                    jpg_filename = filename.replace('.dcm', '.jpg')
                    jpg_path = os.path.join(output_subdir, jpg_filename)
                    img.save(jpg_path)
                    
# دایرکتوری‌های ورودی و خروجی
input_dirs = ['/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images']
output_dir = '/kaggle/working/train_images'

convert_dicom_to_jpg(input_dirs, output_dir)
print("تبدیل به اتمام رسید!")


import zipfile
import os

def zip_directory(folder_path, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w') as zip_file:
        for root, _, files in os.walk(folder_path):
            for file in files:
                zip_file.write(os.path.join(root, file),
                               os.path.relpath(os.path.join(root, file), 
                               os.path.join(folder_path, '..')))

# دایرکتوری خروجی
output_dir = '/kaggle/working/'
# مسیر فایل ZIP نهایی
output_zip_path = '/kaggle/working/dataset.zip'

zip_directory(output_dir, output_zip_path)
print("فایل ZIP با موفقیت ایجاد شد!")

