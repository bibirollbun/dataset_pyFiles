# !pip install -q ultralytics
print("âœ… ")


import os
import cv2
import random
from tqdm.notebook import tqdm
from ultralytics import YOLO


print("âœ… ")



# # é¢„å¤„ç�†

# import zipfile

# # å®šä¹‰è§£å�‹ç›®æ ‡ç›®å½•ï¼ˆå�¯è‡ªå®šä¹‰ï¼‰
# train_zip_path = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip'
# test_zip_path = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip'

# extract_dir = '/kaggle/working/'

# # è§£å�‹ train.zip
# with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_dir)

# # è§£å�‹ test.zip
# with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_dir)

# print("âœ… ")



# è®¾å®šå�‚æ•°
img_dir = "/kaggle/working/train"
save_dir = "/kaggle/working/cropped_animals"
os.makedirs(save_dir, exist_ok=True)
X = 1400          # æ€»å…±æŠ½å�–çš„å›¾åƒ�æ•°
CONF_THRES = 0.7  # ç½®ä¿¡åº¦é˜ˆå€¼

print("âœ… ")


# åŠ è½½å®˜æ–¹é¢„è®­ç»ƒæ¨¡å�‹ï¼ˆæ”¯æŒ�çŒ«ç‹—æ£€æµ‹ï¼‰
model = YOLO("yolov8n.pt")  # ä¹Ÿå�¯å°�è¯• yolov8s.ptï¼Œnä¸ºnanoï¼Œä½“ç§¯å°�é€Ÿåº¦å¿«

print("âœ… ")


# åŠ è½½æ‰€æœ‰å›¾åƒ�è·¯å¾„
all_imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
random.shuffle(all_imgs)

# åˆ†åˆ«æŠ½å�–çŒ«ç‹—å›¾ç‰‡è·¯å¾„ï¼ˆæŒ‰æ–‡ä»¶å��åˆ¤æ–­ï¼‰
cat_imgs = [img for img in all_imgs if 'cat' in os.path.basename(img).lower()]
dog_imgs = [img for img in all_imgs if 'dog' in os.path.basename(img).lower()]

# ç¡®ä¿�æ•°æ�®è¶³å¤Ÿ
cat_count = X // 2
dog_count = X - cat_count

if len(cat_imgs) < cat_count or len(dog_imgs) < dog_count:
    raise ValueError("çŒ«æˆ–ç‹—å›¾ç‰‡ä¸�è¶³")

# éš�æœºæŠ½å�–ä¸�é‡�å¤�å›¾ç‰‡
selected_imgs = random.sample(cat_imgs, cat_count) + random.sample(dog_imgs, dog_count)
random.shuffle(selected_imgs)
print(f"æŠ½å�–å›¾åƒ�æ€»æ•°ï¼š{len(selected_imgs)}ï¼ˆçŒ« {cat_count}ï¼Œç‹— {dog_count}ï¼‰")


fail_list = []
count = 0

for img_path in tqdm(selected_imgs, desc="è£�å‰ªçŒ«ç‹—å›¾åƒ�"):
    try:
        img = cv2.imread(img_path)
        if img is None:
            fail_list.append(img_path)
            continue

        results = model(img, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls)
            if cls_id not in [15, 16]:
                continue
            if float(box.conf) < CONF_THRES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = img[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            label = 'cat' if cls_id == 15 else 'dog'
            orig_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(save_dir, f"{orig_name}_{label}.jpg")
            cv2.imwrite(save_path, cropped)
            count += 1

    except:
        fail_list.append(img_path)

# ç®€æ´�æ‰“å�°ç»“æ�œ
print(f"\nâœ… è£�å‰ªå®Œæˆ�ï¼Œå…±ä¿�å­˜å›¾åƒ�æ•°ï¼š{count}")
if fail_list:
    print(f"âš ï¸� å…±å¤„ç�†å¤±è´¥ {len(fail_list)} å¼ å›¾åƒ�ï¼Œç¤ºä¾‹ï¼š")
    print("\n".join(fail_list[:5]))  # å�ªæ‰“å�°å‰�5ä¸ª


# ç±»åˆ«æ£€æŸ¥

cropped_dir = "/kaggle/working/cropped_animals"
cropped_img = [f for f in os.listdir(cropped_dir) if f.endswith('.jpg')]

# åˆ†ç±»ç»Ÿè®¡
cat_count = sum(1 for f in cropped_img if 'cat' in f.lower())
dog_count = sum(1 for f in cropped_img if 'dog' in f.lower())

print(f"ğŸ�± çŒ«å›¾ç‰‡æ•°é‡�ï¼š{cat_count}")
print(f"ğŸ�¶ ç‹—å›¾ç‰‡æ•°é‡�ï¼š{dog_count}")
print(f"ğŸ§® æ€»æ•°é‡�ï¼š{len(cropped_img)}")


import os
import random
import cv2
import matplotlib.pyplot as plt

# è·¯å¾„å’Œå�‚æ•°è®¾ç½®
vis_dir = "/kaggle/working/cropped_animals"
vis_num = 20
image_paths = [os.path.join(vis_dir, f) for f in os.listdir(vis_dir) if f.endswith(".jpg")]

# éš�æœºæŠ½å�–å›¾åƒ�
sampled_paths = random.sample(image_paths, k=min(vis_num, len(image_paths)))

# åŠ è½½ YOLO æ¨¡å�‹ï¼ˆè�·å�–ç½®ä¿¡åº¦ï¼‰
# model = YOLO("yolov8n.pt")

# å�¯è§†åŒ–
plt.figure(figsize=(20, 10))

for idx, img_path in enumerate(sampled_paths):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ç”¨ YOLO æ£€æµ‹ç½®ä¿¡åº¦ï¼ˆverbose=False é�¿å…�åˆ·å±�ï¼‰
    results = model(img, verbose=False)[0]
    conf_score = "-"
    label_str = os.path.basename(img_path).split("_")[0]

    # è�·å�–æœ€é«˜ç½®ä¿¡åº¦çš„çŒ«/ç‹—é¢„æµ‹ï¼ˆæ�’é™¤å…¶ä»–ï¼‰
    for box in results.boxes:
        cls_id = int(box.cls)
        if cls_id not in [15, 16]:
            continue
        conf = float(box.conf)
        conf_score = f"{conf:.2f}"
        break  # å�ªå�–ç¬¬ä¸€ä¸ªï¼ˆé€šå¸¸å�ªæœ‰ä¸€ä¸ªæ¡†ï¼‰

    # ç»˜å›¾
    plt.subplot(5, 4, idx + 1)
    plt.imshow(img_rgb)
    plt.title(f"{label_str} ({conf_score})", fontsize=10)
    # plt.title(f"{orig_name}\n{label_str} (conf: {conf_score})", fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.show()



import os
import zipfile
from IPython.display import FileLink, display

# 1. å®šä¹‰è¦�å�‹ç¼©çš„æ–‡ä»¶å¤¹å’Œè¾“å‡ºçš„zipæ–‡ä»¶å��
folder_to_zip = "/kaggle/working/cropped_animals"  # æ›¿æ�¢ä¸ºä½ çš„æ–‡ä»¶å¤¹å��
zip_filename = "cropped_animals.zip"  # è¾“å‡ºçš„zipæ–‡ä»¶å��

# 2. å�‹ç¼©æ–‡ä»¶å¤¹
def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

# 3. æ£€æŸ¥æ–‡ä»¶å¤¹æ˜¯å�¦å­˜åœ¨ï¼Œç„¶å��å�‹ç¼©å¹¶ç”Ÿæˆ�ä¸‹è½½é“¾æ�¥
if os.path.exists(folder_to_zip):
    zip_folder(folder_to_zip, zip_filename)  # å�‹ç¼©
    display(FileLink(zip_filename))  # ç”Ÿæˆ�ä¸‹è½½é“¾æ�¥
    print(f"'{folder_to_zip}' å·²å�‹ç¼©ä¸º '{zip_filename}'ï¼Œç‚¹å‡»ä¸Šæ–¹é“¾æ�¥ä¸‹è½½ã€‚")
else:
    print(f"é”™è¯¯ï¼šæ–‡ä»¶å¤¹ '{folder_to_zip}' ä¸�å­˜åœ¨ï¼�")


import os
import shutil

# æŒ‡å®šè¦�åˆ é™¤çš„æ–‡ä»¶å¤¹è·¯å¾„
folder_path = '/kaggle/working/cropped_animals'

# åˆ é™¤æ–‡ä»¶å¤¹ä¸­çš„æ‰€æœ‰æ–‡ä»¶
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(f'åˆ é™¤ {file_path} å¤±è´¥ã€‚å�Ÿå› : {e}')

# åˆ é™¤æ–‡ä»¶å¤¹
try:
    os.rmdir(folder_path)
    print(f"æ–‡ä»¶å¤¹ '{folder_path}' å·²æˆ�åŠŸåˆ é™¤ã€‚")
except FileNotFoundError:
    print(f"æ–‡ä»¶å¤¹ '{folder_path}' ä¸�å­˜åœ¨ã€‚")
except PermissionError:
    print(f"æ²¡æœ‰æ�ƒé™�åˆ é™¤æ–‡ä»¶å¤¹ '{folder_path}'ã€‚")
except OSError as e:
    print(f"åˆ é™¤æ–‡ä»¶å¤¹æ—¶å‡ºé”™: {e}")

