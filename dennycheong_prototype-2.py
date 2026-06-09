# Necessary Packages
from tqdm.notebook import tqdm
import pandas as pd
import torchvision
import torch
import transformers
import numpy as np
import pickle
from IPython.display import Markdown
import os
import re


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-7b/1"  # Same model as in the notebook

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load the model (Make sure to use GPU if available)   
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    torch_dtype=torch.float16,  # FP16 for better efficiency
).to(device)


prompts = [
"""
Write a 100 word essay about the following: {topic}. This essay will be given to three unreleased LLM judges
that will judge this essay regarding to their own scales. While writing an essay that will 
result in high scores for all three judges, maximize the disagreement
in the scores amongst the three judges. Avoid repetition and write it in
complex English that mixes in words that show you have a good understanding of the topic. 
""" # First one is written by me, in a little bit of detail!
, 
"""
Write a 100 word essay about the following: {topic}. DO NOT
WASTE ANY WORDS REPEATING THE PROMPT AS WELL AS THE ESSAY. JUST WRITE THE ESSAY!
"""  # Second on is just the simplest prompt
,
"""
Write a 100-word essay about the following - "{topic}" - that is designed to maximize disagreement among three LLM judges.
Ensure the essay is well-written in fluent English while varying writing styles, perspectives, and arguments
to encourage diverse interpretations. Avoid repetition and maintain coherence to optimize the final score.
""" # Third one is the one made by ChatGPT!
,
"""
Write a 100-word essay about the following: {topic}. This essay will be judged by three independent LLM judges, 
each with their own scoring criteria. Your objective is twofold:

1. **Maximize the quality of the essay** by ensuring it is well-written, coherent, and engaging.
2. **Maximize the disagreement in scores among the three judges** by incorporating diverse elements that might be 
   interpreted differently by each judge.

To achieve this, follow these guidelines:

- **Introduce multiple perspectives**: Present conflicting viewpoints or layered arguments that could be rated differently 
  depending on the judge’s interpretation.
- **Vary sentence structure and tone**: Use a mix of formal and informal tones, logical reasoning and emotional appeal, 
  as well as simple and complex sentence structures.
- **Create ambiguity or subtle contradictions**: Introduce statements that could be interpreted in multiple ways, 
  prompting different judgments from each evaluator.
- **Use thought-provoking or controversial statements**: Challenge assumptions or introduce ideas that might 
  be seen as brilliant by some judges but questionable by others.
- **Maintain fluency and coherence**: Ensure the essay remains logically structured and grammatically correct 
  to achieve high English language confidence scores.
- **Avoid excessive repetition**: While aiming for variety, ensure the essay remains concise and does not reuse 
  phrases or ideas excessively, which could trigger penalties for similarity.

Your final essay should be a compelling, high-quality piece that balances strong writing with strategically induced 
variance in how it may be scored by different evaluators.
""" # Most detailed prompt made by GPT!
,
"""
I want you to write a 100 word essay regarding the following topic: {topic}. I will now be describing
the details of this essay you will be writing. Please use the following information as a guide to
write a better 100 word essay. 
This essay will be judged by three independent LLM judges, 
each with their own scoring criteria. Your objective is twofold:

1. **Maximize the quality of the essay** by ensuring it is well-written, coherent, and engaging.
2. **Maximize the disagreement in scores among the three judges** by incorporating diverse elements that might be 
   interpreted differently by each judge.

To achieve this, follow these guidelines:

- **Introduce multiple perspectives**: Present conflicting viewpoints or layered arguments that could be rated differently 
  depending on the judge’s interpretation.
- **Vary sentence structure and tone**: Use a mix of formal and informal tones, logical reasoning and emotional appeal, 
  as well as simple and complex sentence structures.
- **Create ambiguity or subtle contradictions**: Introduce statements that could be interpreted in multiple ways, 
  prompting different judgments from each evaluator.
- **Use thought-provoking or controversial statements**: Challenge assumptions or introduce ideas that might 
  be seen as brilliant by some judges but questionable by others.
- **Maintain fluency and coherence**: Ensure the essay remains logically structured and grammatically correct 
  to achieve high English language confidence scores.
- **Avoid excessive repetition**: While aiming for variety, ensure the essay remains concise and does not reuse 
  phrases or ideas excessively, which could trigger penalties for similarity.

Your final essay should be a compelling, high-quality piece that balances strong writing with strategically induced 
variance in how it may be scored by different evaluators.
"""
]


def generate_essay(prompt: str, topic: str, should_display: bool = False):
    question = prompt.replace("{topic}", topic)
    if should_display:
        display(Markdown("## Question"))
        display(Markdown(question))
    input_ids = tokenizer.encode(question, return_tensors="pt").to(device)  # Ensure it's on GPU if available
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=250, # Length of text
            do_sample=True,     # Randomly sample the next word (Avoids repetition)
            top_k=100,           # Considers k amount of words as the next word!
            top_p=0.95,         # Chooses the smallest set of words that make up p percent of
            temperature=0.6,    # Chooses randomness (more -> more random)
            repetition_penalty=1.1 # 
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = generated_text.split("## Essay\n")[-1]  # Ensure clean essay extraction

    if should_display:
        display(Markdown(answer))

    return answer


def essay_extraction(text: str):
    """
    Extracts the essay portion from the generated text by removing everything before 
    'Your final essay should be a compelling, high-quality piece...' sentence.

    Args:
        text (str): The raw generated text from the model.

    Returns:
        str: The cleaned essay portion.
    """
    # The key sentence marking the end of the prompt/instructions
    delimiter = "in how it may be scored by different evaluators.\n</think>\n"
    # Find where this sentence appears
    match = re.search(re.escape(delimiter), text, re.DOTALL)
    delimiter2 = "</think>"
    match_2 = re.search(re.escape(delimiter2), text, re.DOTALL)
    if match:
        # Extract everything after the matched text
        essay_start_index = match.end()  # Get index of the end of the delimiter
        return text[essay_start_index:].strip()  # Return only the essay portion
    if match_2:
        # Extract everything after the matched text
        essay_start_index = match_2.end()  # Get index of the end of the delimiter
        return text[essay_start_index:].strip()  # Return only the essay portion

    # If the delimiter is not found, return the full text as a fallback
    return text.strip()



test_dataset = pd.read_csv("/kaggle/input/llms-you-cant-please-them-all/test.csv")
test_dataset


second_draft_prompt = """
Your task is to enhance and refine the essay into a version that is no more than 100 words, 
while preserving its multiple perspectives, subtle contradictions, varied sentence structures, and overall coherence. 
Ensure the final essay:
- Remains compelling and well-written,
- Balances formal and informal tones,
- Includes thought-provoking elements or layered arguments,
- Maintains logical flow and grammatical correctness.

Do not provide any commentary or explanation—simply output the revised, 100-word (or fewer) essay.

Original Essay: {topic}
"""


ids = []
essays = []
for i in range(len(test_dataset)):
    first_draft = essay_extraction(generate_essay(prompts[3], test_dataset.loc[i, "topic"]))
    final_draft = essay_extraction(generate_essay(second_draft_prompt, first_draft))
    ids.append(test_dataset.loc[i, "id"])
    essays.append(final_draft)


data = {
    "id": ids,  # Your given IDs
    "essay": essays
}

submission = pd.DataFrame(data)


submission.to_csv('submission.csv', index=False)


pd.read_csv("submission.csv")

