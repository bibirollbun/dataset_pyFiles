# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


if False: 
    import requests
    from bs4 import BeautifulSoup
    import re
    import csv
    from urllib.parse import urljoin
    
    # Base URL of the website
    base_url = "https://ir.voanews.com/"
    
    # Output file path in Kaggle working directory
    output_file = "/kaggle/working/voa_articles.csv"
    
    # Function to extract text and title from a given page
    def extract_article_content(url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
    
            title = soup.find("title").get_text(strip=True) if soup.find("title") else "Untitled"
            content_div = soup.find("div", id="article-content")
            content = content_div.get_text(strip=True) if content_div else None
    
            if content:
                return {"title": title, "text": content}
            return None
        except Exception as e:
            print(f"Error while fetching {url}: {e}")
            return None
    
    # Function to find all article links on a page
    def find_article_links(soup):
        links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            # Check for relevant article links based on your needs
            if re.match(r"^/.*", href):  # Adjust this pattern for specific links if necessary
                full_url = urljoin(base_url, href)
                links.append(full_url)
        return set(links)  # Remove duplicates
    
    # Main crawling function
    def crawl_voa_website():
        visited_urls = set()
        urls_to_visit = [base_url]
        all_articles = []
    
        while urls_to_visit and len(all_articles) < 50:
            current_url = urls_to_visit.pop(0)
            if current_url in visited_urls:
                continue
            print(f"Visiting: {current_url}")
            try:
                response = requests.get(current_url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
    
                # Find and save article content
                article = extract_article_content(current_url)
                if article:
                    all_articles.append(article)
    
                # Discover new links
                new_links = find_article_links(soup)
                for link in new_links:
                    if link not in visited_urls:
                        urls_to_visit.append(link)
    
                visited_urls.add(current_url)
            except Exception as e:
                print(f"Failed to process {current_url}: {e}")
    
        # Save all collected articles to a CSV file
        with open(output_file, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["title", "text"])
            writer.writeheader()
            writer.writerows(all_articles)
    
        print(f"Saved {len(all_articles)} articles to {output_file}")
    
    # Execute the crawler
    if __name__ == "__main__":
        crawl_voa_website()


!pip install pip3-autoremove
!pip-autoremove torch torchvision torchaudio -y
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
!pip install unsloth
!pip -q uninstall transformers -y
!pip -q install transformers==4.47.1 


from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2-9b",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)


alpaca_prompt = """Below is a question in persian, you need to answer it based on the provided instruction.

### Prompt:
{}

### Instruction:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"] 
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass
# Load the dataset
from datasets import load_dataset

data = load_dataset("Bruce-Azar-Wayne/alpacca_far", split = "train")
split_data = data.train_test_split(test_size=0.2, seed=42)
train_dataset = split_data["train"]
val_dataset = split_data["test"]

train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)
val_dataset = val_dataset.map(formatting_prompts_func, batched = True,)

train_dataset
val_dataset


FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

inputs = tokenizer(
    [
        alpaca_prompt.format(
            "تفاوت  خودرو و هواپیما در چیست؟",  # instruction
            """               :برای پاسخ به این سوال از ساختار نمونه زیر الحام بگیر                     
                                               تفاوت  میز و صندلی در چیست؟ 
                   .میز: میز وسیله ای هست که از ان برای نگهداشتم اجسام استفاده میشود
                     .صندلی: صندلی وسیله ای هست که از ان برای نشستن استفاده میشود
.تفاوت میز و صندلی: میز برای نگهداری اجسام استفاده میشود ولی صندلی برای نشستن استفاده میشود""",  # input
            ""   # output - leave this blank for generation!
        )
    ],
    return_tensors="pt"
).to("cuda")
outputs = model.generate(**inputs, max_new_tokens=800, use_cache=True, repetition_penalty=1.1)
answer = tokenizer.batch_decode(outputs)
answer = answer[0].replace('\u200c', '')
print(answer)


!pip install -qU \
  transformers \
  sentence-transformers\
  pinecone-client\
  datasets\
  accelerate\
  einops \
  langchain \
  xformers \
  bitsandbytes \
   "torch<2.5" \
  -U langchain-community


from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

embed_model_id = "HooshvareLab/bert-fa-base-uncased"

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)


import os
from pinecone import Pinecone
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
secret_value= user_secrets.get_secret("PineconeKey")

# configure client
pc = Pinecone(secret_value)


from pinecone import ServerlessSpec

 #Fill this data based on what you used to create your index.
cloud = os.environ.get('aws')   
region = os.environ.get('us-east-1') 

spec = ServerlessSpec(cloud=cloud, region=region)


index_name = 'radiofarda'


import time

# check if index already exists 
if index_name not in pc.list_indexes().names():
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=len(embeddings[0]),
        metric='cosine', 
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
# view index stats
index.describe_index_stats()


if False:
    from datasets import load_dataset
    
    data = load_dataset(
        'community-datasets/farsi_news',
        split='radiofarda'
    )
    data


if False:
    import hashlib
    data = data.to_pandas()
    
    batch_size = 32
    
    for i in range(0, len(data), batch_size):
        i_end = min(len(data), i+batch_size)
        batch = data.iloc[i:i_end]
        ids = [x['link'] for i, x in batch.iterrows()]   #Note: Your title should be in ASCII, meaning latin words, numbers and special characters.
        texts = [x['summary'] for i, x in batch.iterrows()] #This is the actual column that contains your content.
        embeds = embed_model.embed_documents(texts)
        # get metadata to store in Pinecone
        metadata = [
            {'text': x['summary'],
             'title': x['title']} for i, x in batch.iterrows()  #Its a good idea to have quality titles since metadata is used for the search.
        ]
        # add to Pinecone
        index.upsert(vectors=zip(ids, embeds, metadata))


index.describe_index_stats()


from langchain.vectorstores import Pinecone

text_field = 'text'  # field in metadata that contains text content

vectorstore = Pinecone(
    index, embed_model.embed_query, text_field
)
     


def fetch_query_results(vectorstore, prompt, k=1): #K refers to the rank of the content in terms of similarity. K=1 means that we only want the most similar answer.
    """Fetch the top `k` most relevant chunks of text."""
    return vectorstore.similarity_search(prompt, k=k)


import re

def format_alpaca_prompt(query_results, prompt):
    """Format the Alpaca prompt with query results."""
    # Assuming each query result is a Document object with a 'text' attribute.
    context = "\n\n".join([doc.page_content for doc in query_results if hasattr(doc, 'page_content')])
    return alpaca_prompt.format(prompt, context, "")  #Here we are feeding our alpaca_prompt with the prompt and context which is the result of the query.

def generate_response(tokenizer, model, prompt, max_new_tokens=256):
    """Generate a response using the Alpaca model."""
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True, repetition_penalty=1.1) #Add a repetition penalty or the model will keep repeating the generation.
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def run_pipeline(vectorstore, tokenizer, model, prompt, k=1):
    """
    Run the complete RAG pipeline using a single prompt for query and Alpaca.
    
    Args:
        vectorstore: The vector store instance.
        tokenizer: The tokenizer instance.
        model: The Alpaca model instance.
        prompt: The input prompt string (used for both query and Alpaca).
        k: Number of results to fetch from the query.
        
    Returns:
        The generated response from the Alpaca model.
    """
    # Step 1: Fetch query results using the same prompt
    query_results = fetch_query_results(vectorstore, prompt, k)

    # Step 2: Format the Alpaca prompt using the query results
    alpaca_prompt_formatted = format_alpaca_prompt(query_results, prompt)

    # Step 3: Generate the response
    response = generate_response(tokenizer, model, alpaca_prompt_formatted)
    
    response = response[0].replace('\u200c', '') 
    return response



prompt = "ایا دستگاه ویروس یاب سپاه پاسداران مجوز وزارت بهداشت را دارد؟"
response = run_pipeline(vectorstore, tokenizer, model, prompt)

print(response)


prompt = "استفاده از داروی  درمان لوزالمعده در ژاپن و کره برای درمان کرونا را توضیح بده؟"
response = run_pipeline(vectorstore, tokenizer, model, prompt)

print(response)


from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset= val_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        #num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
        evaluation_strategy="epoch",  # Enable evaluation during training
    ),
   
)


trainer_stats = trainer.train()


import math
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

inputs = tokenizer(
    [
        alpaca_prompt.format(
            "موارد زیر را به دسته غذا ، لوازم الکتریکی و شهر طبقه بندی کنید",  # instruction
            "تلفن همرا , پیتزا, پاریس",  # input
            ""   # output - leave this blank for generation!
        )
    ],
    return_tensors="pt"
).to("cuda")
outputs = model.generate(**inputs, max_new_tokens=250, use_cache=True, repetition_penalty=1.1)
answer = tokenizer.batch_decode(outputs)
answer = answer[0].replace('\u200c', '')
print(answer)


trainer.save_model('/kaggle/working/fine_tuned_gemma-2-9b_model')
tokenizer.save_pretrained('/kaggle/working/fine_tuned_gemma-2-9b_model')

