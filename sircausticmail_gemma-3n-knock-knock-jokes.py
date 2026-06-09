


%%time

!pip install --quiet timm --upgrade 2> /dev/null
!pip install --quiet accelerate     2> /dev/null
!pip install --quiet git+https://github.com/huggingface/transformers.git 2> /dev/null





%%time

# Source: https://www.kaggle.com/models/google/gemma-3n

import kagglehub
from transformers import AutoProcessor, AutoModelForImageTextToText

GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e4b-it")  # = 4 billion parameters - instruction tuned
# GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e4b")  # = 4 billion parameters
# GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b")  # = 2 billion parameters
processor  = AutoProcessor.from_pretrained(GEMMA_PATH)
model      = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")

global jokehistory
jokehistory = []

"""
# VERSION 1

def knockknock():
    if jokehistory:
        knockknockprompt = f"Tell me A knock knock joke. And make sure to not repeat these jokes ---: {'; '.join(jokehistory[-3:])} ---"  # Last 3 jokes
    else:
        knockknockprompt = "Tell me A knock knock joke."

    # print(knockknockprompt)        
    input_ids = processor(text=str(knockknockprompt), return_tensors="pt").to(model.device, dtype=model.dtype)
    outputs   = model.generate(**input_ids, max_new_tokens=512, disable_compile=True
                              , do_sample = True, temperature = 1.5, top_p = 0.9 )
    currjoke = processor.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces = True
    )

    jokehistory.append(currjoke[0])

    return currjoke[0]  # text[0] injects <bos>prompt
"""

# VERSION 2
def knockknock():
    if jokehistory:
        knockknockprompt = f"Tell me only ONE knock knock joke. And make sure to not repeat these jokes ---: {'; '.join(jokehistory)} ---"
    else:
        knockknockprompt = "Tell me only ONE knock knock joke."
    
    # Tokenize the input
    input_ids = processor(text=str(knockknockprompt), return_tensors="pt").to(model.device, dtype=model.dtype)
    input_length = input_ids['input_ids'].shape[1]  # Get length of input tokens
    
    # Generate response
    outputs = model.generate(**input_ids, max_new_tokens = 128, disable_compile=True,
                            do_sample=True, temperature=0.75, top_p=0.9,
                            eos_token_id=processor.tokenizer.eos_token_id)
    
    # Extract only the newly generated tokens (after the input)
    new_tokens = outputs[:, input_length:]
    
    # Decode only the new tokens
    currjoke = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    
    jokehistory.append(currjoke[0])
    return currjoke[0]




def print_joke_history():
    for i, joke in enumerate(jokehistory, 1):
        print(f"{i}.")
        print(joke)
        print("-" * 30)
        





%%time

# print(jokehistory)

joke = knockknock()
print(joke)
print()



print_joke_history()






%%time

# print(jokehistory)

joke = knockknock()
print(joke)
print()




print_joke_history()



%%time

# print(jokehistory)

joke = knockknock()
print(joke)
print()




print_joke_history()



%%time

# print(jokehistory)

joke = knockknock()
print(joke)
print()



print_joke_history()



%%time

# print(jokehistory)

joke = knockknock()
print(joke)
print()



print_joke_history()








