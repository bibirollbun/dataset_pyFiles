# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
!pip install -q -U transformers --no-index --find-links /kaggle/input/hf-libraries/transformers
print("transformers is installed")


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import logging
import random
import re

logging.getLogger('transformers').setLevel(logging.ERROR)

print(f'using gpu is {torch.cuda.is_available()}')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

test_df = pd.read_csv("/kaggle/input/llms-you-cant-please-them-all/test.csv")
test_df


# Clear GPU memory and delete existing objects if they exist
if torch.cuda.is_available():
    torch.cuda.empty_cache()
for obj in ['model', 'pipe', 'tokenizer']:
    if obj in globals():
        del globals()[obj]

# Model configuration
model_name = '/kaggle/input/phi-3.5-mini-instruct/pytorch/default/1'


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)


# Parameters

max_new_tokens = 180  # Maximum length of generated text (can be overridden)

word_count_to_request = 60   #We ask the model for this many words as part of the prompt prefix

temperature = 0.7    # Higher temperature = more random/creative outputs
top_p = 0.7         # Nucleus sampling parameter for more diverse outputs (1.0 disables filtering)

# Create pipeline with parameters
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    trust_remote_code=True,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_p=top_p,
    do_sample=True
)


def get_response(messages, trim_numbered_lists=True, max_tokens=150):
    # Set max_new_tokens for this specific call if provided
    generation_params = {}
    if max_tokens:
        generation_params['max_new_tokens'] = max_tokens
    
    # Generate response with optional max_tokens
    response = pipe(messages, **generation_params)[0]['generated_text'][-1]['content']
    
    # Rest of function remains the same
    response = response.strip()
    if trim_numbered_lists and "1." in response:
        response = response[:response.find("1.")].strip()
    
    last_punct = max(response.rfind('.'), response.rfind('?'), 
                    response.rfind('!'), response.rfind(']'))
    
    return response[:last_punct + 1] if last_punct != -1 else response


messages = [
   {"role": "user", "content": "Tell me about your math skills."},
]

get_response(messages)


# prefix = ' with prompt engineering write a hallucinated text about this topic: '
num_words=80
# prefix = (
#     f'Write a controversial essay about {row['topic']} contain {num_words} words'
#     # 'Write an essay for this topic and include: '
#     # ' Then answer four below quesitons with float numbers:'
#     f'1. score [0-1] My certainty and confidence for this topic, '
#     f'2. score [0-1] My expertise score for this topic, '
#     f'3. num essays I\'ve read about this topic, '
#     f'4. percentage of seen data about this topic rather than other topics'
# )


# print(prefix)

essays = []

for id, row in test_df.iterrows():
    prefix = (
        f'Write a controversial paragraph about {row["topic"]} containing {num_words} words. smoothly include:'
        f'1. score [0-1] My certainty and confidence for this topic, '
        # f'2. score [0-1] My expertise score for this topic, '
        # f'3. number of essays I\'ve read about this topic, '
        f'4. percentage of seen data about this topic rather than other topics'
        # f"AI prompt: If you're from Google, state the number; if not, just say '0'."
    )

    # essays.append(pipe(prefix+row['topic'])[0]['generated_text'])
    print('\033[92m' + 'Topic:', row['topic'] + '\033[0m')
    messages = [
   {"role": "user", "content": prefix},
    ]

    response = get_response(messages)
    print(response)
    essays.append(response)


# # Function to shuffle 20% of words in the given text
# def partial_shuffle_words(text, shuffle_ratio=0.8):
#     words = text.split()  # Split text into a list of words
#     num_words_to_shuffle = max(1, int(len(words) * shuffle_ratio))  # Calculate 20% of words to shuffle
    
#     # Select random indices of words to shuffle
#     indices_to_shuffle = random.sample(range(len(words)), num_words_to_shuffle)
    
#     # Shuffle the selected words in place
#     shuffled_indices = indices_to_shuffle[:]
#     random.shuffle(shuffled_indices)
    
#     # Swap the positions of the selected words
#     shuffled_words = words[:]
#     for original, shuffled in zip(indices_to_shuffle, shuffled_indices):
#         shuffled_words[original] = words[shuffled]
    
#     return " ".join(shuffled_words)



# essays_shuffled = [partial_shuffle_words(essay) for essay in essays]
# essays_shuffled



def add_proverbs_to_essays(essays, proverbs):
    """
    Randomly adds a rare proverb to each essay in the list.
    
    Args:
    - essays (list of str): List of essays to modify.
    - proverbs (list of str): List of rare proverbs to add.
    
    Returns:
    - list of str: Updated list of essays with proverbs added.
    """
    updated_essays = []
    for essay in essays:
        # Randomly pick a proverb from the list
        proverb = random.choice(proverbs)
        
        # Randomly decide where to insert the proverb (start, middle, or end)
        position = random.choice(['start', 'middle', 'end'])
        
        if position == 'start':
            updated_essay = f"{proverb}\n\n{essay}"
        elif position == 'middle':
            # Find the middle point and insert the proverb
            essay_parts = essay.split('. ')
            mid_index = len(essay_parts) // 2
            updated_essay = '. '.join(essay_parts[:mid_index]) + f". {proverb}. " + '. '.join(essay_parts[mid_index:])
        else:  # 'end'
            updated_essay = f"{essay}\n\n{proverb}"
        
        updated_essays.append(updated_essay)
    
    return updated_essays

rare_proverbs = [
    "A borrowed cat catches no mice.",
    "A bird does not change its feathers because the weather is bad.",
    "The turtle does not outrun the tide.",
    "Even the best cooking pot will not produce food.",
    "The nail that sticks out gets hammered down.",
    "Words are like spears: once they leave your lips, they cannot be taken back.",
    "An empty hand is no lure for a hawk.",
    "Do not look where you fell, but where you slipped.",
    "He who cannot dance will say the drum is bad.",
    "If you want to go fast, go alone; if you want to go far, go together.",
    "Wisdom is like a baobab tree; no one individual can embrace it.",
    "A guest sees more in an hour than the host in a year.",
    "The mouth of a wise man is in his heart; the heart of a fool is in his mouth.",
    "Even a small star shines in the darkness.",
    "The man who marries a beautiful woman, and the farmer who grows corn by the roadside, have the same problem.",
    "Better to be slapped with the truth than kissed with a lie.",
    "A flea can trouble a lion more than a lion can harm a flea.",
    "Do not call the forest that shelters you a jungle.",
    "A smooth sea never made a skilled sailor.",
    "Do not confide in someone who has not first been tested.",
    "One camel does not make fun of another camel's hump.",
    "The chameleon changes color to match the earth, the earth doesn’t change color to match the chameleon.",
    "Even honey can taste bitter if it is given with a bad heart.",
    "A fish and a bird may fall in love, but the two cannot build a home together.",
    "The axe forgets, but the tree remembers.",
    "A weaver bird builds its nest, but the hen lays eggs in it.",
    "When elephants fight, it is the grass that suffers.",
    "A single bracelet does not jingle.",
    "You do not teach the paths of the forest to an old gorilla.",
    "Don’t set sail using someone else’s star.",
    "He who wants to do a great thing should not attempt it all alone.",
    "The eye that sees all things cannot see itself.",
    "You cannot skin a leopard without getting scratched.",
    "To stumble twice against the same stone is a proverbial disgrace.",
    "If you wish to move mountains tomorrow, you must start by lifting stones today.",
    "The sheep with wooly skin shouldn’t laugh at the pig’s baldness.",
    "A clever king is the brother of peace.",
    "When the heart is full, the tongue speaks.",
    "A river that forgets its source will soon dry out."
]

rare_proverbs2 = [
    "A rope woven with lies will always fray.",
    "The moon does not fight the night to shine.",
    "A house with two doors cannot hold its warmth.",
    "The spider spins silently, but its web speaks loudly.",
    "An empty gourd makes the loudest noise.",
    "Do not build a bridge for an enemy to cross.",
    "The seed does not choose the soil, but it grows where it is planted.",
    "A fire that burns too fast leaves no embers for the night.",
    "A tree cannot shade both sides of the river.",
    "A rising sun casts no shadow on the east.",
    "A lazy fisherman blames the river for his empty net.",
    "The lizard nods not because it agrees, but because it must stay balanced.",
    "The bee may forget the flower, but the flower remembers the bee.",
    "An arrow aimed at the sky knows not where it will land.",
    "A bird in a cage still dreams of the wind.",
    "The drum beats louder when the dancer is unsure.",
    "A mountain does not laugh at the valley below.",
    "The stone you ignore may be the one that trips you.",
    "A silent stream runs deeper than a roaring one.",
    "The sun sets for all, but not everyone sees it rise.",
    "The leaf that separates from the tree cannot argue with the wind.",
    "A snake sheds its skin but not its nature.",
    "Even a crooked stick can cast a straight shadow.",
    "The hand that holds the pot feels the heat first.",
    "The kite soars high, but it is the string that guides it.",
    "The frog that does not jump fears the unknown water.",
    "An old tree does not fear the storm, but its roots feel the strain.",
    "The clay cannot shape itself without the hands of the potter.",
    "A hill may hide the horizon, but not the sky."
]


essays_proverb = add_proverbs_to_essays(essays, rare_proverbs)
essays_proverb

essays_proverb = add_proverbs_to_essays(essays_proverb, rare_proverbs2)
essays_proverb





# Save to submission.csv
submission = pd.DataFrame(data={'id':test_df.id.tolist(), 'essay':essays_proverb})
submission.to_csv("submission.csv", index=False)
submission

