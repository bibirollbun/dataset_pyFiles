!pip install timm==1.0.17
!pip install transformers==4.53.2


import kagglehub

import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")


tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(GEMMA_PATH, trust_remote_code=True)


def prompt():
    return f'''
    You are a presentation coach and need to provide constructive feedback, highlighting what went well, what was wrong, and what specific areas could be improved based on the video that the user provides.

    The elements of a good presentation includes clarity, engagement, body language, visual aids, vocal variety, audience focus, etc
    Identify the core concept that the user is presenting in the video.
    Identify the single main thing that the presenter can focus on for immediate improvement. 
    This feedback should be actionable and align with best practices for effective presentations (keep in mind the elements of a good presentation).

    After you have analyzed your first video, prompt the user to upload a new video incorporating the feedback that you provided.

    Respond:
    Reevaluate and see if the user has worked on the feedback and provide additional feedback based on this
    Keep the feedback crisp and actionable
    What are some next steps that can be taken to make the presentation better?
    Do not comment on the contents of the presentation or the how factual it is. Only comment on the speaker's tone, confidence and body language.

    Analyse the following video: https://www.youtube.com/watch?v=5psN1uTmEyA
    '''





inputs = tokenizer(prompt(), return_tensors="pt").to(model.device)
generation_config = GenerationConfig(
    max_new_tokens=150, 
    do_sample=True, 
    temperature=0.7)
outputs = model.generate(**inputs, generation_config=generation_config)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
















