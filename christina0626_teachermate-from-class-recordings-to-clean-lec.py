# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# !pip install sqlalchemy==2.0.4
# !pip install "torch==2.6.0+cu124" --force-reinstall --index-url https://download.pytorch.org/whl/cu124
# !pip install "rich<14"
# !pip install numpy==1.25



# !pip install datasets


!pip install yt-dlp



ted_urls=['https://www.youtube.com/watch?v=eIho2S0ZahI&list=PLlT0ph_Ig5Rc8xVBw47aZfLsOGw-lxRWb&index=3','https://www.youtube.com/watch?v=Hu4Yvq-g7_Y&list=PLlT0ph_Ig5Rc8xVBw47aZfLsOGw-lxRWb&index=4',
   'https://www.youtube.com/watch?v=95ovIJ3dsNk&list=PLlT0ph_Ig5Rc8xVBw47aZfLsOGw-lxRWb&index=9' ,'https://www.youtube.com/watch?v=xp0O2vi8DX4&list=PLlT0ph_Ig5Rc8xVBw47aZfLsOGw-lxRWb&index=10',
     'https://www.youtube.com/watch?v=3VTsIju1dLI&list=PLlT0ph_Ig5Rc8xVBw47aZfLsOGw-lxRWb&index=18'
    ]


# from datasets import load_dataset

# tedlium = load_dataset("LIUM/tedlium", "release1") # for Release 1

# # see structure
# print(tedlium)

# # load audio sample on the fly
# audio_input = tedlium["train"][0]["audio"]  # first decoded audio sample
# transcription = tedlium["train"][0]["text"] 


## Trascribe ted to mp3: 

import yt_dlp

def download_mp3_from_youtube(url, output_dir="./ted_mp3s"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_dir}/%(title).30s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# å»ºç«‹è³‡æ–™å¤¾
import os
os.makedirs("ted_mp3s", exist_ok=True)

# ä¸‹è¼‰æ‰€æœ‰ TED æ¼”è¬›
for url in ted_urls:
    download_mp3_from_youtube(url)






# # See one ted  audio example

# from IPython.display import Audio
# sample = tedlium["train"][1]
# Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"])


!pip install -q git+https://github.com/openai/whisper.git
!sudo apt update && sudo apt install -y ffmpeg  # è‹¥æœªå…§å»ºè½‰æª”å·¥å…·



import whisper

#  'tiny', 'base', 'small', 'medium', 'large'
model = whisper.load_model("base")


import whisper
from pydub import AudioSegment
import os

# è¼‰å…¥ Whisper æ¨¡å�‹
model = whisper.load_model("base")

# è¼¸å…¥èˆ‡è¼¸å‡ºè¨­å®š
audio_folder = "./ted_mp3s"
audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".mp3")]

# å„²å­˜è¾¨è­˜çµ�æ�œçš„å­—å…¸
result_texts = {}

for filename in audio_files:
    mp3_path = os.path.join(audio_folder, filename)

    # è½‰æ�›ç‚º WAVï¼ˆWhisper æ�¨è–¦ä½¿ç”¨ 16kHz monoï¼‰
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    wav_path = mp3_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")

    # èª�éŸ³è¾¨è­˜
    result = model.transcribe(wav_path)

    # å­˜å…¥ dictï¼ˆå�¯é�¸ï¼šå�»é™¤å‰¯æª”å��ï¼‰
    result_texts[filename] = result["text"]

    print(f" loadï¼š{filename}")

# å�¯æŸ¥çœ‹çµ�æ�œå­—å…¸ï¼š
print("\nğŸ“„ part of transcribe resultï¼š\n")
for fname, text in list(result_texts.items())[:3]:
    print(f"â–¶ï¸� {fname}:\n{text[:300]}...\n")



yt_result_texts=result_texts


import os
import random
from sklearn.model_selection import train_test_split
import pandas as pd

DATASET_ROOT = "/kaggle/input/spoken-wikipedia-corpus-dutch/dutch/dutch"

# å»ºç«‹æ¨£æœ¬è·¯å¾‘æ¸…å–® (audio.ogg + wiki.txt)
samples = []
for folder in os.listdir(DATASET_ROOT):
    folder_path = os.path.join(DATASET_ROOT, folder)
    audio_path = os.path.join(folder_path, "audio.ogg")
    text_path = os.path.join(folder_path, "wiki.txt")
    if os.path.exists(audio_path) and os.path.exists(text_path):
        samples.append((audio_path, text_path))

# åˆ‡åˆ†è¨“ç·´ã€�é©—è­‰èˆ‡æ¸¬è©¦é›†ï¼ˆ70/10/20ï¼‰
random.seed(42)
random.shuffle(samples)
train_val, wiki_test_path = train_test_split(samples, test_size=0.2, random_state=42)
wiki_train_path, wiki_val_path = train_test_split(train_val, test_size=0.125, random_state=42)

# é¡¯ç¤ºè·¯å¾‘æ•¸é‡�åˆ†ä½ˆ
df = pd.DataFrame({
    "split": ["train", "validation", "test"],
    "count": [len(wiki_train_path), len(wiki_val_path), len(wiki_test_path)]
})





# import os
# os.environ["WHISPER_FORCE_FP32"] = "1"  # å¼·åˆ¶ FP32 (åœ¨éƒ¨åˆ† whisper fork æ”¯æ�´)









# from tqdm import tqdm

# def transcribe_ted_dataset(dataset, name="test"):
#     results = []
#     dataset_split=dataset['test']
#     for sample in tqdm(dataset_split, desc=f"Transcribing {name}"):
#         audio_array = sample["audio"]["array"]  # numpy array
#         result = model.transcribe(audio_array)  # Whisper å�¯ä»¥å�ƒ raw array

#         results.append({
#             "audio_path": sample["file"],               # æˆ–ç”¨ sample["id"] è­˜åˆ¥ä¹Ÿå�¯ä»¥
#             "prediction": result["text"],
#             "ground_truth": sample["text"]
#         })
    
#     return results
# ted_pred_results= transcribe_ted_dataset(tedlium , name="tedlium")





wiki_test_path[0]


# need to use GPU to train
from tqdm import tqdm

def transcribe_dataset(dataset_split, name="test"):
    results = []
    for audio_path, text_path in tqdm(dataset_split):
        result = model.transcribe(audio_path)
        with open(text_path, "r", encoding="utf-8") as f:
            ground_truth = f.read().strip()
        results.append({
            "audio_path": audio_path,
            "prediction": result["text"],
            "ground_truth": ground_truth,
        })
    return results

# ç¯„ä¾‹ï¼šè·‘ test set
#wiki_test_results = transcribe_dataset(wiki_test_path, name="wiki_test_path")



import json
with open("/kaggle/input/wiki-transcribe-file/wiki_test_results.json", "r", encoding="utf-8") as f:
    wiki_test_results = json.load(f)


wiki_test_results[0]['prediction'][:20]


from IPython.display import Audio

# è·¯å¾‘å°�æ‡‰ä½ æˆªåœ–ä¸­çš„ç¬¬ä¸€å€‹é …ç›®
ogg_path =wiki_test_path[0][0]

# æ’­æ”¾éŸ³è¨Š
Audio(ogg_path)



# import json

# with open("wiki_test_results.json", "w", encoding="utf-8") as f:
#     json.dump(wiki_test_results, f, ensure_ascii=False, indent=2)








!pip show gradio


import gradio as gr
import whisper

model = whisper.load_model("base")

def transcribe(audio):
    result = model.transcribe(audio)
    return result["text"]


# gr.Interface(
#     fn=transcribe,
#     inputs=gr.Audio(source="microphone", type="filepath"),
#     outputs="text",
#     title="ğŸ�™ï¸� å�³æ™‚èª�éŸ³è½‰æ–‡å­— with Whisper"
# ).launch()
gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="ğŸ�™ å�³æ™‚èª�éŸ³è½‰æ–‡å­— with Whisper"
).launch()











!pip install git+https://github.com/huggingface/transformers.git


#!pip install --upgrade transformers


!pip list --format=freeze > current_env.txt














# import transformers
# help(transformers.GenerationMixin)





!pip install transformers==4.53.2


# from transformers import pipeline
# import kagglehub

# # ä¸‹è¼‰ Gemma æ¨¡å�‹
# GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")

# # ä½¿ç”¨ pipelineï¼Œè‡ªå‹•è¼‰å…¥ tokenizer + model
# summarizer = pipeline("text-generation", model=GEMMA_PATH, tokenizer=GEMMA_PATH, trust_remote_code=True)

# # æ¸¬è©¦ä¸€æ®µè¼¸å…¥
# prompt = "ä½ æ˜¯ä¸€ä½�å°ˆæ¥­ç­†è¨˜æ•´ç�†åŠ©ç�†ï¼Œè«‹ç¸½çµ�ä»¥ä¸‹æ®µè�½ï¼š Bleiswijk is een plaats in..."
# out = summarizer(prompt, max_new_tokens=200, do_sample=False)

# print(out[0]['generated_text'])




# è¼‰å…¥ Gemma-3n æ¨¡å�‹
import kagglehub

GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(GEMMA_PATH, trust_remote_code=True)
#summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer)



transcript = yt_result_texts.values[1] # å�‡è¨­å·²æœ‰ Whisper è¼¸å‡º

prompt = f"""
ä½ æ˜¯ä¸€ä½�å°ˆæ¥­ç­†è¨˜æ•´ç�†åŠ©ç�†ï¼Œè«‹å¹«æˆ‘å°‡ä¸‹åˆ—è¬›èª²å…§å®¹è½‰æ�›ç‚ºæ¢�åˆ—å¼�é‡�é»�ï¼š

---
{transcript}
---

è«‹ä»¥ Markdown æ ¼å¼�è¼¸å‡ºï¼ŒåŒ…å�«æ¨™é¡Œèˆ‡æ¢�åˆ—é‡�é»�ã€‚
"""

result = summarizer(prompt, max_new_tokens=512, do_sample=False)
note = result[0]["generated_text"]
print(note)




