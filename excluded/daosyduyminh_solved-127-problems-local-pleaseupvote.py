!pip install -U gdown



import gdown

# URL folder Google Drive
url = "https://drive.google.com/drive/folders/1HLlj6YUHH4dCJ9Hu8pQDQ4fTCfsL86Yt"
gdown.download_folder(url, quiet=False, use_cookies=False)



import shutil
import zipfile
import os

# Đường dẫn hiện tại của file zip
src_path = "/kaggle/working/Golf Code/submission.zip"

# Đường dẫn đích (đưa ra ngoài folder)
dst_path = "/kaggle/working/submission.zip"

# Di chuyển file zip ra ngoài
shutil.move(src_path, dst_path)

# Giải nén file zip
with zipfile.ZipFile(dst_path, 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working/")

# Kiểm tra kết quả
print("Giải nén xong. Các file gồm:")
os.listdir("/kaggle/working/")





