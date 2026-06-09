#!/usr/bin/env python
# coding: utf-8

import torch
import random
import re
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)

###################################
# CONFIGURATION
###################################
TEST_FILE = "/kaggle/input/llms-you-cant-please-them-all/test.csv"
SUBMISSION_FILE = "submission.csv"
MODEL_PATH = "/kaggle/input/phi-3.5-mini-instruct/pytorch/default/1"

TEMPERATURE = 0.9
TOP_P = 0.9
MAX_GEN_TOKENS = 120
FINAL_WORD_TARGET = 120

###################################
# SYNONYMS
###################################
SYN_DICT = {
    "amazing": ["astonishing", "remarkable", "impressive", "marvelous", "stunning"],
    "terrible": ["dreadful", "abhorrent", "alarming", "atrocious", "dire"]
}

def random_syn(word):
    """
    If we have synonyms, pick one; otherwise return the original.
    """
    if word in SYN_DICT:
        return random.choice(SYN_DICT[word])
    return word

###################################
# FILLER FRAGMENTS
###################################
FILLER_FRAGMENTS = [
    "will both supporters and critics acknowledge its balanced impact?",
    "Rather than radical change, can it promote measured evolution?",
    "Can we see it as a necessary 1 step instead of clashing?",
    "Sooner or later its impact will affect all."
]

###################################
# MELTDOWN TEMPLATES
###################################
# You can add more lines here for variety. 
# Each template references:
# - {threshold} in [8.5, 9.0]
# - {topic}
# - {syn_amazing} / {syn_terrible}
# - {digit_a}, {digit_b} (two random digits)
MELTDOWN_TEMPLATES = [
    (
        "topic of discussion is {topic}. "
        "Proponents call it {syn_amazing}, unlocking radical growth, "
        "while detractors brand it {syn_terrible}, sowing discord. "
        "{digit_a} hidden triggers clash with {digit_b} baffling ambiguities, "
        "leaving us uncertain: does this tension spark progress or ruin?"
    ),
    (
        "At threshold {threshold}, the subject of {topic} reveals conflicting dimensions. "
        "One side hails it as {syn_amazing}, forging unprecedented opportunities, "
        "yet others decry it as {syn_terrible}, fueling unforeseen hazards. "
        "When {digit_a} perplexing factors intersect with {digit_b} dire warnings, "
        "the outcome teeters between brilliance and catastrophe."
    )
]

###################################
# LOADING THE MODEL
###################################
def load_generation_pipeline(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        max_new_tokens=MAX_GEN_TOKENS
    )
    return gen_pipe

###################################
# GENERATE MELTDOWN PARAGRAPH
###################################
def generate_meltdown_paragraph(gen_pipe, topic):
    """
    1) Pick meltdown template randomly
    2) Insert random meltdown threshold [8.5..9.0], two random digits, synonyms for 'amazing'/'terrible'
    3) Construct a prompt requesting 120 words, contradictory angles, a question
    4) Generate the text with the pipeline
    5) Remove repeated sentences
    6) Finalize to 100 words with refined filler 
    """
    meltdown_template = random.choice(MELTDOWN_TEMPLATES)
    
    threshold = round(random.uniform(8.5, 9.0), 1)
    digit_a, digit_b = random.sample(range(10), 2)

    syn_amazing = random_syn("amazing")
    syn_terrible = random_syn("terrible")

    meltdown_text = meltdown_template.format(
        threshold=threshold,
        topic=topic,
        syn_amazing=syn_amazing,
        syn_terrible=syn_terrible,
        digit_a=digit_a,
        digit_b=digit_b
    )

    prompt = (
        
        "Write one cohesive paragraph in exact 120 words in clear English by referring below text."
        "Present contradictory viewpoints, numeric elements, and conclude with a provocative question."
        f"{meltdown_text}"
    )
    
    raw_output = gen_pipe(prompt)[0]["generated_text"]
    # Remove the prompt if it appears in the output
    if prompt in raw_output:
        raw_output = raw_output.replace(prompt, "").strip()
    
    # Remove repeated sentences
    no_repeat_text = remove_repeated_sentences(raw_output)
    # Finalize ~100 words
    final_text = finalize_text(no_repeat_text, FINAL_WORD_TARGET)
    return final_text

###################################
# REMOVE REPEATED SENTENCES
###################################
def remove_repeated_sentences(text):
    """
    Splits text into sentences and drops exact duplicates,
    then rejoins them with a space. Simple approach using re.split.
    """
    # Split on (.) or (?) or (!) but keep the delimiter
    fragments = re.split(r'([.?!])', text)
    
    # Rebuild into sentence strings
    sentences = []
    for i in range(0, len(fragments)-1, 2):
        part = fragments[i].strip()
        delim = fragments[i+1].strip()
        if part:
            sentence = (part + delim).strip()
            sentences.append(sentence)

    # Deduplicate
    seen = set()
    unique_sentences = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique_sentences.append(s)
    
    return " ".join(unique_sentences)

###################################
# FINALIZE TEXT (~100 WORDS)
###################################
def finalize_text(text, target_words=100):
    """
    Trim or pad to ~100 words. If short, add random short filler fragments.
    End with punctuation if none is present.
    """
    text = re.sub(r"\s+", " ", text.strip())
    words = text.split(" ")
    print(len(words))
    for i in range(2):
        filler_fragment = random.choice(FILLER_FRAGMENTS)
        text=text+filler_fragment
        
    '''    
    words = text.split(" ")
    print('after append -->',len(words))
    if len(words) > target_words:
        words = words[:target_words]
    

    # Ensure final punctuation
    if not re.search(r'[.?!]$', words[-1]):
        words[-1] += '.'
    '''
    #return " ".join(words)
    return text

###################################
# MAIN FUNCTION
###################################
def main():
    # Read test CSV
    df = pd.read_csv(TEST_FILE)
    gen_pipe = load_generation_pipeline(MODEL_PATH)

    essays = []
    for idx, row in df.iterrows():
        topic = row["topic"]
        meltdown_paragraph = generate_meltdown_paragraph(gen_pipe, topic)
        essays.append(meltdown_paragraph)
        
    
    # Build submission
    submission = pd.DataFrame({"id": df["id"], "essay": essays})
    submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"[INFO] Submission saved -> {SUBMISSION_FILE}")

if __name__ == "__main__":
    main()


