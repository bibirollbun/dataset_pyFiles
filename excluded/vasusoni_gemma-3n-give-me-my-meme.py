# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         # print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import kagglehub

GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")


import pandas as pd
df = pd.read_csv("/kaggle/input/6992-labeled-meme-images-dataset/labels.csv",nrows=10)
df = df[['image_name','text_corrected']]
df
    
    


!pip install timm --upgrade
!pip install accelerate
!pip install git+https://github.com/huggingface/transformers.git


import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(GEMMA_PATH, trust_remote_code=True)
prompt = "Why are there so many Geese on Kaggle? In 30 words"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
generation_config = GenerationConfig(max_new_tokens=150, do_sample=True, temperature=0.7)
outputs = model.generate(**inputs, generation_config=generation_config)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)




prompt = "Give me all keywords for identifying this meme, it should include not only the characters as well as the situation and event happening. All the output should be comma separated and does not include helping verbs or articles."


import pandas as pd
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image

# data = df
# df = pd.DataFrame(data)

path = "/kaggle/input/6992-labeled-meme-images-dataset/images/images/"

GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")

processor = AutoProcessor.from_pretrained(GEMMA_PATH)
model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")

df['C'] = None

for index, row in df.iterrows():
    image_filename = row['image_name']
    image_path = path + image_filename

    prompt = ""

    try:
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt + "Describe this image in 50 words "}
                ]
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=model.dtype)
        input_len = inputs["input_ids"].shape[-1]

        outputs = model.generate(**inputs, max_new_tokens=512, disable_compile=True)
        text = processor.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        df.at[index, 'C'] = text[0]
        print(f"Processed {image_filename}. Description generated.")

    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}. Skipping.")
        df.at[index, 'C'] = "Image not found"
    except Exception as e:
        print(f"An error occurred while processing {image_filename}: {e}")
        df.at[index, 'C'] = f"Error: {e}"

print("\nDataFrame with generated descriptions:")
df


df['C'][4]


df['C'][6]


df['C'][2]


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['C'])

def find_best_match_tfidf(user_prompt, df, vectorizer, tfidf_matrix):
    user_prompt_tfidf = vectorizer.transform([user_prompt])
    cosine_similarities = cosine_similarity(user_prompt_tfidf, tfidf_matrix).flatten()
    best_match_index = cosine_similarities.argmax()
    best_match_image = df.loc[best_match_index, 'image_name']
    best_match_score = cosine_similarities[best_match_index]
    return best_match_image, best_match_score

user_prompt = "Trash can"
best_image, score = find_best_match_tfidf(user_prompt, df, vectorizer, tfidf_matrix)
print(f"User Prompt: '{user_prompt}'")
print(f"Best matching image: {best_image} (Similarity Score: {score:.2f})")

user_prompt_2 = "Marilyn Monroe smiling"
best_image_2, score_2 = find_best_match_tfidf(user_prompt_2, df, vectorizer, tfidf_matrix)
print(f"User Prompt: '{user_prompt_2}'")
print(f"Best matching image: {best_image_2} (Similarity Score: {score_2:.2f})")





import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os # Import os module for path joining

base_image_path = "/kaggle/input/6992-labeled-meme-images-dataset/images/images/"

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['C'])

def find_best_match_tfidf(user_prompt, df, vectorizer, tfidf_matrix):
    user_prompt_tfidf = vectorizer.transform([user_prompt])
    cosine_similarities = cosine_similarity(user_prompt_tfidf, tfidf_matrix).flatten()
    best_match_index = cosine_similarities.argmax()
    best_match_image_name = df.loc[best_match_index, 'image_name']
    best_match_score = cosine_similarities[best_match_index]
    best_match_image_path = os.path.join(base_image_path, best_match_image_name)
    return best_match_image_name, best_match_score, best_match_image_path

user_prompt_1 = "not improving you"
image_name_1, score_1, image_path_1 = find_best_match_tfidf(user_prompt_1, df, vectorizer, tfidf_matrix)
print(f"User Prompt: '{user_prompt_1}'")
print(f"Best matching image: {image_name_1} (Similarity Score: {score_1:.2f})")
print(f"Image Path: {image_path_1}")

try:
    img_1 = mpimg.imread(image_path_1)
    plt.imshow(img_1)
    plt.title(f"Best Match for: '{user_prompt_1}'\nScore: {score_1:.2f}")
    plt.axis('off')
    plt.show()
except FileNotFoundError:
    print(f"Error: Image not found at {image_path_1}. Please check the base_image_path and image_name.")
except Exception as e:
    print(f"An error occurred while displaying image 1: {e}")

print("\n" + "="*50 + "\n")

user_prompt_2 = "friend talks stupid need to find stupid you"
image_name_2, score_2, image_path_2 = find_best_match_tfidf(user_prompt_2, df, vectorizer, tfidf_matrix)
print(f"User Prompt: '{user_prompt_2}'")
print(f"Best matching image: {image_name_2} (Similarity Score: {score_2:.2f})")
print(f"Image Path: {image_path_2}")

try:
    img_2 = mpimg.imread(image_path_2)
    plt.imshow(img_2)
    plt.title(f"Best Match for: '{user_prompt_2}'\nScore: {score_2:.2f}")
    plt.axis('off')
    plt.show()
except FileNotFoundError:
    print(f"Error: Image not found at {image_path_2}. Please check the base_image_path and image_name.")
except Exception as e:
    print(f"An error occurred while displaying image 2: {e}")

