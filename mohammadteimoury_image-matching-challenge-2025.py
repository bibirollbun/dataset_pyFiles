!pip install opencv-python-headless



# Ø¢Ù…Ø§Ø¯Ù‡â€ŒØ³Ø§Ø²ÛŒ Ú©ØªØ§Ø¨Ø®Ø§Ù†Ù‡â€ŒÙ‡Ø§
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.notebook import tqdm
import torch

# Ù…Ø³ÛŒØ± Ø§ØµÙ„ÛŒ Ø¯ÛŒØªØ§Ø³Øª Ø±Ù‚Ø§Ø¨Øª
BASE_DIR = "/kaggle/input/image-matching-challenge-2025"

# Ø¨Ø±Ø±Ø³ÛŒ Ø§ÛŒÙ†Ú©Ù‡ CUDA Ø¯Ø± Ø¯Ø³ØªØ±Ø³ Ù‡Ø³Øª ÛŒØ§ Ù†Ù‡
device = "cuda" if torch.cuda.is_available() else "cpu"
print("âœ… Torch device:", device)



from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "/kaggle/input/clip-vit-b32-savedmodel-offline"
model = SentenceTransformer(model_path, device=device)

print("âœ… CLIP model loaded successfully!")




from transformers import CLIPProcessor, CLIPModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Ù…Ø³ÛŒØ± Ø¯ÛŒØªØ§Ø³ØªØª Ø±Ùˆ ØªÙ†Ø¸ÛŒÙ… Ú©Ø±Ø¯Ù…
model_path = "/kaggle/input/clip-vit-b32-savedmodel-offline"

# Ù„ÙˆØ¯ Ù…Ø¯Ù„ Ùˆ Ù¾Ø±Ø¯Ø§Ø²Ø´Ú¯Ø±
model = CLIPModel.from_pretrained(model_path).to(device)
processor = CLIPProcessor.from_pretrained(model_path)

print("âœ… CLIP model & processor loaded successfully!")



from PIL import Image
import torch
from torchvision import transforms

def extract_clip_features(image_path, model, processor, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.cpu().numpy()




import os
for root, dirs, files in os.walk("/kaggle/input/test-image"):
    for file in files:
        print(file)



sample_path = "/kaggle/input/test-image/H2604-L374320503_original.jpg"
features = extract_clip_features(sample_path, model, processor, device)
print("âœ… Embedding shape:", features.shape)




import os

test_image_dir = "/kaggle/input/image-matching-challenge-2025/test"
image_list = []

# Ù¾ÛŒÙ…Ø§ÛŒØ´ Ù‡Ù…Ù‡ Ø²ÛŒØ±Ù¾ÙˆØ´Ù‡â€ŒÙ‡Ø§ Ùˆ Ø¬Ù…Ø¹â€ŒÚ©Ø±Ø¯Ù† ØªØµØ§ÙˆÛŒØ± .png
for root, dirs, files in os.walk(test_image_dir):
    for file in files:
        if file.lower().endswith(".png"):
            image_list.append(file)

image_list = sorted(image_list)

print(f"âœ… ØªØ¹Ø¯Ø§Ø¯ ØªØµØ§ÙˆÛŒØ± ØªØ³Øª: {len(image_list)}")
print(f"ğŸ”� Ù†Ù…ÙˆÙ†Ù‡â€ŒÙ‡Ø§: {image_list[:3]}")



import os
from PIL import Image
import matplotlib.pyplot as plt

# Ù…Ø³ÛŒØ± ØªØµØ§ÙˆÛŒØ± ØªØ³Øª
test_dir = "/kaggle/input/test-image"

# Ù„ÛŒØ³Øª ØªØµØ§ÙˆÛŒØ±
test_images = [f for f in os.listdir(test_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

print(f"âœ… ØªØ¹Ø¯Ø§Ø¯ ØªØµØ§ÙˆÛŒØ± Ù…ÙˆØ¬ÙˆØ¯ Ø¯Ø± test-image: {len(test_images)}")
print("ğŸ–¼ï¸� Ú†Ù†Ø¯ Ù†Ù…ÙˆÙ†Ù‡ ØªØµÙˆÛŒØ±:")
for i, name in enumerate(test_images[:5]):  # Ù†Ù…Ø§ÛŒØ´ Ù�Ù‚Ø· Ûµ ØªØµÙˆÛŒØ± Ø§ÙˆÙ„
    print(f"{i+1}. {name}")



from PIL import Image
import matplotlib.pyplot as plt
import os

test_image_dir = "/kaggle/input/image-matching-challenge-2025/test"
test_image_name = "H2604-L374320503_original.jpg"
image_path = os.path.join(test_dir, test_image_name)

image = Image.open(image_path)

plt.imshow(image)
plt.title("Test Image")
plt.axis("off")
plt.show()



# Ù…Ø³ÛŒØ± ØªØµÙˆÛŒØ± Ø¯ÛŒØªØ§Ø¨ÛŒØ³

db_image_path = "/kaggle/input/solasali/sol.jpg"

# Ø§Ø³ØªØ®Ø±Ø§Ø¬ ÙˆÛŒÚ˜Ú¯ÛŒ
db_feature = extract_clip_features(db_image_path, model, processor, device)
print("âœ… Database feature shape:", db_feature.shape)




import torch
import numpy as np

# Ù…Ø­Ø§Ø³Ø¨Ù‡ Ø´Ø¨Ø§Ù‡Øª Ú©Ø³ÛŒÙ†ÙˆØ³ÛŒ Ø¨ÛŒÙ† ØªØµÙˆÛŒØ± ØªØ³Øª Ùˆ Ø¯ÛŒØªØ§Ø¨ÛŒØ³
similarity = torch.nn.functional.cosine_similarity(
    torch.tensor(features),
    torch.tensor(db_feature)
).item()

print(f"ğŸ�¯ Cosine Similarity: {similarity:.4f}")



import os

test_image_dir = "/kaggle/input/image-matching-challenge-2025/test"  
image_list = []

for root, dirs, files in os.walk(test_image_dir):
    for file in files:
        if file.lower().endswith(".png"):  
            image_list.append(file)

image_list = sorted(image_list)

print(f"âœ… ØªØ¹Ø¯Ø§Ø¯ ØªØµØ§ÙˆÛŒØ± ØªØ³Øª: {len(image_list)}")
print(f"ğŸ–¼ï¸� Ù†Ù…ÙˆÙ†Ù‡â€ŒØ§ÛŒ Ø§Ø² Ù„ÛŒØ³Øª ØªØµØ§ÙˆÛŒØ±: {image_list[:5]}")



import pandas as pd
import numpy as np

# Ù�Ø±Ø¶ Ú©Ù†ÛŒÙ… Ù‚Ø¨Ù„Ø§Ù‹ Ù„ÛŒØ³Øª Ø§Ø³Ø§Ù…ÛŒ Ø¹Ú©Ø³ Ù‡Ø§ Ø¢Ù…Ø§Ø¯Ù‡ Ø´Ø¯Ù‡
# Ù…Ø«Ù„Ø§: image_list = ['another_et_another_et001.png', 'another_et_another_et002.png', ...]

num_images = len(image_list)

# Ø³Ø§Ø®Øª Ù…Ù‚Ø¯Ø§Ø±Ù‡Ø§ÛŒ Ø¯Ø±Ø³Øª Ø¨Ø±Ø§ÛŒ Ø³ØªÙˆÙ† Ù‡Ø§
rotation_matrix_list = ["1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0"] * num_images
translation_vector_list = ["0.0 0.0 0.0"] * num_images

# Ø³Ø§Ø®Øª Ø¯ÛŒØªØ§Ù�Ø±ÛŒÙ…
submission = pd.DataFrame({
    "dataset": ["test"] * num_images,
    "scene": [f"scene{i+1}" for i in range(num_images)],
    "image": image_list,
    "rotation_matrix": rotation_matrix_list,
    "translation_vector": translation_vector_list,
})

# Ø°Ø®ÛŒØ±Ù‡ Ù�Ø§ÛŒÙ„ CSV
submission.to_csv("/kaggle/working/submission.csv", index=False)

print("âœ… Ù�Ø§ÛŒÙ„ Ù†Ù‡Ø§ÛŒÛŒ submission.csv Ø³Ø§Ø®ØªÙ‡ Ø´Ø¯!")
submission.head()



# Import libraries
import pandas as pd
import numpy as np
import os

# Ù…Ø³ÛŒØ± Ø¹Ú©Ø³â€ŒÙ‡Ø§ (Ø¬Ø§ÛŒÛŒ Ú©Ù‡ Ù�Ø§ÛŒÙ„â€ŒÙ‡Ø§ÛŒ ØªØ³ØªØª Ù‡Ø³ØªÙ†)
test_image_dir = "/kaggle/input/image-matching-challenge-2025/test"

# Ø³Ø§Ø®Øª Ù„ÛŒØ³Øª Ø§Ø³Ù… Ù�Ø§ÛŒÙ„â€ŒÙ‡Ø§ÛŒ Ø¹Ú©Ø³
image_list = []
for root, dirs, files in os.walk(test_image_dir):
    for file in files:
        if file.lower().endswith('.png'):
            image_list.append(file)

image_list = sorted(image_list)

# ØªØ¹Ø¯Ø§Ø¯ Ø¹Ú©Ø³ Ù‡Ø§
num_images = len(image_list)

# Ø³Ø§Ø®Øª Ø³ØªÙˆÙ†â€ŒÙ‡Ø§
rotation_matrix_list = ["1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0"] * num_images
translation_vector_list = ["0.0 0.0 0.0"] * num_images

# Ø³Ø§Ø®Øª Ø¯ÛŒØªØ§Ù�Ø±ÛŒÙ… Ù†Ù‡Ø§ÛŒÛŒ
submission = pd.DataFrame({
    "dataset": ["test"] * num_images,
    "scene": [f"scene{i+1}" for i in range(num_images)],
    "image": image_list,
    "rotation_matrix": rotation_matrix_list,
    "translation_vector": translation_vector_list,
})

# Ø°Ø®ÛŒØ±Ù‡ Ù�Ø§ÛŒÙ„ Ù†Ù‡Ø§ÛŒÛŒ
submission.to_csv("/kaggle/working/submission.csv", index=False)

print("âœ… Ù�Ø§ÛŒÙ„ Ø³Ø§Ù„Ù… submission.csv Ø³Ø§Ø®ØªÙ‡ Ø´Ø¯!")
submission.head()



import pandas as pd

df = pd.read_csv("/kaggle/working/submission.csv")
print(df["rotation_matrix"].head(3))
print(df["translation_vector"].head(3))



!ls /kaggle/working/



import pandas as pd

df = pd.read_csv("/kaggle/working/submission.csv")
print(df.head())
print(df.columns)



import os
print(os.listdir("/kaggle/working"))



!head /kaggle/working/submission.csv


