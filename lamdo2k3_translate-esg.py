import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
from accelerate import Accelerator
import accelerate
accelerator = Accelerator(kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)])
accelerator.print(f"ACCELERATOR DEVICE:{accelerator.distributed_type}---- NUM OF PROCESSES: {accelerator.num_processes }")



df = pd.read_csv('/kaggle/input/esg-data-19k/esg_articles_async.csv')
df


model_name = "VietAI/envit5-translation"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map ='auto', torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("/kaggle/input/esg-data-19k/esg_articles_async.csv") 
titles = df['Tiêu đề'].fillna("").tolist()
contents = df['Nội dung'].fillna("").tolist()

def translate_batch(text_list, prefix="en:", batch_size=16):
    results = []
    for i in tqdm(range(0, len(text_list), batch_size)):
        batch_texts = [prefix + " " + text.strip() for text in text_list[i:i+batch_size]]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        outputs = model.generate(**inputs, max_length=512)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)
    return results

# translated_titles = translate_batch(titles, prefix="en:", batch_size=128)
translated_contents = translate_batch(contents, prefix="en:", batch_size=128)

df_translated = pd.DataFrame({
    "Tiêu đề gốc": titles,
    "Nội dung gốc": contents,
    # "Tiêu đề dịch": translated_titles,
    "Nội dung dịch": translated_contents
})

df_translated.to_csv("translated_output.csv", index=False)
print("✅ Dịch hoàn tất. File đã lưu vào 'translated_output.csv'")



model_name = "VietAI/envit5-translation"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16).to(accelerator.device)
device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("/kaggle/input/esg-data-19k/esg_articles_async.csv") 
titles = df['Tiêu đề'].fillna("").tolist()
contents = df['Nội dung'].fillna("").tolist()

def translate_batch(text_list, prefix="en:", batch_size=16):
    results = []
    for i in tqdm(range(0, len(text_list), batch_size)):
        batch_texts = [prefix + " " + text.strip() for text in text_list[i:i+batch_size]]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(accelerator.device)
        outputs = model.generate(**inputs, max_length=512)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)
    return results

# translated_titles = translate_batch(titles, prefix="en:", batch_size=128)
translated_contents = translate_batch(contents, prefix="en:", batch_size=128)

df_translated = pd.DataFrame({
    "Tiêu đề gốc": titles,
    "Nội dung gốc": contents,
    # "Tiêu đề dịch": translated_titles,
    "Nội dung dịch": translated_contents
})

df_translated.to_csv("translated_output.csv", index=False)
print("✅ Dịch hoàn tất. File đã lưu vào 'translated_output.csv'")





