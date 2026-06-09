!pip install anthropic==0.39.0
# need these for scorer to not fail
!pip install --upgrade widgetsnbextension
!pip install --upgrade ipywidgets
!jupyter nbextension enable --py widgetsnbextension


import anthropic
from tqdm import tqdm
from kaggle_secrets import UserSecretsClient
import csv
import json
import re
import os
user_secrets = UserSecretsClient()
ANTHROPIC_API_KEY = user_secrets.get_secret("ANTHROPIC_API_KEY")


client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def generate_sentence(model="claude-3-5-sonnet-20241022"):
    SYSTEM_PROMPT = """write a chrismas sentence from a song or a movie, using chrismas words like: "advent chimney elf family fireplace gingerbread mistletoe ornament reindeer scrooge" but use different words than the ones I gave you.
After that remove all stopwords and connectors and mix the words left.
place original sentence in between <sentence></sentence> tags and the mixed version in between <mixed><mixed/> tags.
    """
    messages = [{"role": "user", "content": "Your sentence:"}]
    response = client.messages.create(
        model=model,
        system=SYSTEM_PROMPT,
        max_tokens=1000,
        messages=messages,
        temperature=0.7
    )
    return response.content[0].text
    
def get_reasoning(mixed, expected, model="claude-3-5-sonnet-20241022"):
    SYSTEM_PROMPT = """Given a set of random Christmas words in between <mixed><mixed/> tags.Provide the reasoning neccesary to group the mixed words so that they become the sentence given in between <expected></expected> tags.
    Think step by step about what is the most natural way to group the mixed words and give clear reasoning statement in between <think></think> tags.
    """
    messages = [{"role": "user", "content": f"<mixed>{mixed}</mixed>\n<expected>{expected}</expected>\nYour thinking:"}]
    response = client.messages.create(
        model=model,
        system=SYSTEM_PROMPT,
        max_tokens=1000,
        messages=messages,
        temperature=0
    )
    return response.content[0].text


def extract_sentence_text(text):
   sentence_pattern = r'<sentence>(.*?)</sentence>'
   match = re.search(sentence_pattern, text, re.DOTALL)
   return match.group(1).strip() if match else None

def extract_mixed_text(text):
   sentence_pattern = r'<mixed>(.*?)</mixed>'
   match = re.search(sentence_pattern, text, re.DOTALL)
   return match.group(1).strip() if match else None

def extract_reasoning_text(text):
   sentence_pattern = r'<think>(.*?)</think>'
   match = re.search(sentence_pattern, text, re.DOTALL)
   return match.group(1).strip() if match else None






RAW_GENERATIONS_DATASET_PATH = "/kaggle/input/sft-santa2024/raw_generations.csv"
raw_generations_saved = os.path.exists(RAW_GENERATIONS_DATASET_PATH)


if not raw_generations_saved:
    dataset = []
    print("generating sentences")
    for i in tqdm(range(500), desc="Generating sentences"):
        try:
            generation = generate_sentence()
            dataset.append(generation)
        except Exception as e:
            time.sleep(59)
            print("retrying...")
            generation = generate_sentence()
            dataset.append(generation)
    with open("/kaggle/working/raw_generations.csv", 'w', newline='') as file:
       writer = csv.writer(file)
       writer.writerow(['sentence'])  # Header
       for sentence in dataset:
           writer.writerow([sentence])
else:
    print("using cached sentences")
    with open(RAW_GENERATIONS_DATASET_PATH, 'r') as file:
       reader = csv.reader(file)
       next(reader)  # Skip header row
       dataset = [row[0] for row in reader]


CLEAN_GENERATIONS_DATASET_PATH = "/kaggle/input/sft-santa2024/clean_dataset.json"
clean_generations_saved = os.path.exists(CLEAN_GENERATIONS_DATASET_PATH)


if not clean_generations_saved:
    print("generating clean sentences")
    clean_dataset = []
    for d in tqdm(dataset, desc="Processing data", unit="samples"):
        expected = extract_sentence_text(d)
        mixed = extract_mixed_text(d)
        try:
            reasoning_raw = get_reasoning(mixed, expected)
        except Exception as e:
            time.sleep(59)
            print("retrying...")
            reasoning_raw = get_reasoning(mixed, expected)
        reasoning = extract_reasoning_text(reasoning_raw)
        clean_dataset.append({"mixed": mixed, "expected":expected, "reasoning": reasoning, "raw_reasoning": reasoning_raw})
    
    with open("/kaggle/working/clean_dataset.json", 'w') as file:
        json.dump(clean_dataset, file)
else:
    print("Loading from cache...")
    with open(CLEAN_GENERATIONS_DATASET_PATH, 'r') as file:
        clean_dataset = json.load(file)


def get_root(word):
    """Get root form of word by removing common suffixes"""
    word = word.lower()
    # Strip common suffixes in order
    suffixes = ["'s", "'", "s", "ing", "ed", "er", "ers"]
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

def sort_words(words_str, sentence):
    # Get original word list
    words_orig = words_str.split()
    
    # Create lookup dictionary with root forms
    words_dict = {}
    for word in words_orig:
        cleaned = ''.join(c for c in word if c.isalnum())
        root = get_root(cleaned)
        words_dict[root] = word
    
    result = []
    seen = set()  # Track roots we've already added
    
    for sentence_word in sentence.split():
        # Clean and get root of sentence word
        cleaned = ''.join(c for c in sentence_word if c.isalnum())
        root = get_root(cleaned)
        
        # If root matches and we haven't seen it, add original version
        if root in words_dict and root not in seen:
            result.append(words_dict[root])
            seen.add(root)
    
    return ' '.join(result)
    
    
for cd in clean_dataset:
    cd["answer"] = sort_words(cd["mixed"], cd["expected"])


with open('/kaggle/working/clean_dataset_full.json', 'w') as file:
   json.dump(clean_dataset, file)


#https://www.kaggle.com/code/pablomarino/llama-vs-deepseek-distill
SYSTEM_PROMPT ="""Rearrange the words to create the most coherent order. Think step by step and provide your answer between <answer></answer> tags.
Example:
Input: hung stockings canes bright sleigh bells echoed night candy
<think>
1. Group natural word pairs:
   - "stockings hung"
   - "candy canes"
   - "bright sleigh bells"
   - "echoed night"
2. Arrange for flow and meaning
</think>
<answer>stockings hung candy canes bright sleigh bells echoed night</answer>
Input:
"""
alpaca_dataset = []

for example in clean_dataset:
    output = f"<think>{example['reasoning']}. This leaves me with: \"{example['expected']}\"</think><answer>{example['answer']}</answer>"
    e = {"instruction": SYSTEM_PROMPT, "input":example["mixed"], "output": output}
    alpaca_dataset.append(e)
    
with open('/kaggle/working/alpaca_dataset_santa2024.json', 'w') as file:
   json.dump(alpaca_dataset, file)


len(alpaca_dataset)


clean_dataset[0]


clean_dataset[1]


alpaca_dataset[0]


alpaca_dataset[1]


alpaca_dataset[5]




