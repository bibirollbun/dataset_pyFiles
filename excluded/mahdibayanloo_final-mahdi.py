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
    "A clock with no hands measures the patience of time."
    "The mirror reflects all but hides its own cracks."
    "A storm’s roar is softer than the silence before it."
    "The path with no footprints still leads somewhere."
    "A lantern’s light cannot reveal the stars."
    "The river that overflows drowns its own banks."
    "A blade sharpens faster in a rival’s hand."
    "The mountain listens longer than the valley speaks."
    "A whisper in the void echoes louder than a scream in a crowd."
    "The shadow of a tree dances only when the wind commands."
    "A lock with no key fears no thief."
    "The tide’s retreat reveals treasures lost to the sand."
    "A kite soars highest when tethered tightly."
    "A journey of no destination still wears the soles of shoes."
    "The stars guide those who dare to look up."
    "A fallen leaf is not forgotten by the tree."
    "The fire’s warmth hides the pain of its burn."
    "The bridge you refuse to cross may be the safest path."
    "A mirror shows faces but conceals thoughts."
    "The drum that beats alone forgets its rhythm."
    "The bird that flies into the storm knows its wings."
    "A whisper to the sea is louder than a scream to the mountain."
    "The horizon belongs to those who walk towards it."
    "A silent book speaks louder than an empty room."
    "The rain falls for all, yet no two drops touch the same ground."
    "A firefly lights its world but cannot warm it."
    "The ship without an anchor knows only the wind’s will."
    "A stone thrown into a pond leaves ripples that outlive the thrower."
    "The tallest tree casts the longest shadow at dusk."
    "The mountain’s peak hides in the clouds but rests on the earth."
]

rare_proverbs2 = [
    "The river knows no walls, but its flow carves its own boundaries."
    "A ship with no sail drifts faster in the wind’s absence."
    "The echo argues not with the voice that created it."
    "A candle’s flame dances to the song of its own demise."
    "The rain that floods the field forgets the seeds it waters."
    "The owl’s silence speaks louder than the sparrow’s chatter."
    "A bridge built with haste sways with every step."
    "The stone that resists the river polishes its own surface."
    "A star forgotten by the night still burns in the void."
    "The wind’s song is heard only by those who stand still."
    "The sky bows to no horizon, yet embraces them all."
    "The wave that crashes learns the patience of the shore."
    "The sand remembers every footstep, though none linger."
    "A flame without fuel consumes itself."
    "The forest hides its secrets beneath the roots."
    "A storm carries the whispers of distant clouds."
    "The ocean’s roar silences the whispers of a thousand streams."
    "A seed buried deep finds strength in the darkness."
    "The hill sees the sunrise first but feels the shadow last."
    "The flame teaches the moth the cost of desire."
    "The cloud that weeps waters the roots it cannot see."
    "The lighthouse sees the storm yet never leaves its place."
    "The silent wind bends the tallest grass."
    "The river’s song fades, but the rocks remember the tune."
    "The frost does not ask before painting the leaves white."
    "The night hides the sun but cannot steal its warmth."
    "The sky’s colors argue with the dusk before resting at night."
    "A pebble thrown starts a ripple but forgets its origin."
    "The moon does not envy the sun; its light is borrowed but unique."
    "The forest burns, yet the seeds remain in the ash."
]


essays_proverb = add_proverbs_to_essays(essays, rare_proverbs)
essays_proverb

essays_proverb = add_proverbs_to_essays(essays_proverb, rare_proverbs2)
essays_proverb





# Save to submission.csv
submission = pd.DataFrame(data={'id':test_df.id.tolist(), 'essay':essays_proverb})
submission.to_csv("submission.csv", index=False)
submission

