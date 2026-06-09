# need these for scorer to not fail
!pip install --upgrade widgetsnbextension
!pip install --upgrade ipywidgets
!jupyter nbextension enable --py widgetsnbextension


!pip install "trl<0.15.0"
!pip install -qqq "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --progress-bar off
!pip install -qqq --no-deps xformers=="0.0.27.post2" peft accelerate bitsandbytes triton==3.1.0 --progress-bar off



from torch import __version__; from packaging.version import Version as V



import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported
import wandb
import torch 
from kaggle_secrets import UserSecretsClient
import csv
import os
import gc

user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret("HF_TOKEN")

WANDB_TOKEN = user_secrets.get_secret("WANDB_TOKEN") 
FINETUNED_MODEL_ID = "Pablonm/FineLlama-3.1-8B_v13"
MAX_TOKENS = 5000

wandb.login(key=WANDB_TOKEN)



model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=FINETUNED_MODEL_ID,
    max_seq_length=MAX_TOKENS,
    load_in_4bit=True,
    dtype=None,
)
model = FastLanguageModel.for_inference(model)


PROMPT1 = """Rearrange the words to create the most coherent order. Think step by step and provide your answer between <answer></answer> tags.\nInput:\n"""
messages = [
    {"role": "system", "content": PROMPT1},
    {"role": "user", "content": "advent chimney elf family fireplace gingerbread mistletoe ornament reindeer scrooge"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")
text_streamer = TextStreamer(tokenizer)
r = model.generate(input_ids=inputs, streamer=text_streamer, use_cache=True)


PROMPT1 = """Rearrange the words to create the most coherent order. Think step by step and provide your answer between <answer></answer> tags.\nInput:\n"""
messages = [
    {"role": "system", "content": PROMPT1},
    {"role": "user", "content": "red advent chimney elf family bless gingerbread mistletoe ornament reindeer scrooge"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")
text_streamer = TextStreamer(tokenizer)
r = model.generate(input_ids=inputs, streamer=text_streamer, use_cache=True)


import pandas as pd

data = pd.read_csv('/kaggle/input/santa-2024/sample_submission.csv')
print(data.head())


def get_prediction(model, tokenizer, sentence, prompt):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": sentence},
    ]
    inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
    output = model.generate(input_ids=inputs, temperature=0.0001, max_new_tokens=2000, use_cache=True)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output


sentence = "red advent chimney elf family bless gingerbread mistletoe ornament reindeer scrooge"
PROMPT = """Rearrange the words to create the most coherent order. Think step by step and provide your answer between <answer></answer> tags.\nInput:\n"""

get_prediction(model, tokenizer, sentence, PROMPT)


%%time
"""Evaluation metric for Santa 2024."""

import gc
import os
from math import exp
from collections import Counter
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import transformers
import torch

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ParticipantVisibleError(Exception):
    pass


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    model_path: str = '/kaggle/input/gemma-2/transformers/gemma-2-9b/2',
    load_in_8bit: bool = False,
    clear_mem: bool = False,
) -> float:
    """
    Calculates the mean perplexity of submitted text permutations compared to an original text.

    Parameters
    ----------
    solution : DataFrame
        DataFrame containing the original text in a column named 'text'.
        Includes a row ID column specified by `row_id_column_name`.

    submission : DataFrame
        DataFrame containing the permuted text in a column named 'text'.
        Must have the same row IDs as the solution.
        Includes a row ID column specified by `row_id_column_name`.

    row_id_column_name : str
        Name of the column containing row IDs.
        Ensures aligned comparison between solution and submission.

    model_path : str, default='/kaggle/input/gemma-2/transformers/gemma-2-9b/2'
        Path to the serialized LLM.

    load_in_8bit : bool, default=False
        Use 8-bit quantization for the model. Requires CUDA.

    clear_mem : bool, default=False
        Clear GPU memory after scoring by clearing the CUDA cache.
        Useful for testing.

    Returns
    -------
    float
        The mean perplexity score. Lower is better.

    Raises
    ------
    ParticipantVisibleError
        If the submission format is invalid or submitted strings are not valid permutations.

    Examples
    --------
    >>> import pandas as pd
    >>> model_path = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"
    >>> solution = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'text': ["this is a normal english sentence", "the quick brown fox jumps over the lazy dog"]
    ... })
    >>> submission = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'text': ["sentence english normal a is this", "lazy the over jumps fox brown quick the dog"]
    ... })
    >>> score(solution, submission, 'id', model_path=model_path, clear_mem=True) > 0
    True
    """
    # Check that each submitted string is a permutation of the solution string
    sol_counts = solution.loc[:, 'text'].str.split().apply(Counter)
    sub_counts = submission.loc[:, 'text'].str.split().apply(Counter)
    invalid_mask = sol_counts != sub_counts
    if invalid_mask.any():
        raise ParticipantVisibleError(
            'At least one submitted string is not a valid permutation of the solution string.'
        )

    # Calculate perplexity for the submitted strings
    sub_strings = [
        ' '.join(s.split()) for s in submission['text'].tolist()
    ]  # Split and rejoin to normalize whitespace
    scorer = PerplexityCalculator(
        model_path=model_path,
        load_in_8bit=load_in_8bit,
    )  # Initialize the perplexity calculator with a pre-trained model
    perplexities = scorer.get_perplexity(
        sub_strings
    )  # Calculate perplexity for each submitted string

    if clear_mem:
        # Just move on if it fails. Not essential if we have the score.
        try:
            scorer.clear_gpu_memory()
        except:
            print('GPU memory clearing failed.')

    return float(np.mean(perplexities))


class PerplexityCalculator:
    """
    Calculates perplexity of text using a pre-trained language model.

    Adapted from https://github.com/asahi417/lmppl/blob/main/lmppl/ppl_recurrent_lm.py

    Parameters
    ----------
    model_path : str
        Path to the pre-trained language model

    load_in_8bit : bool, default=False
        Use 8-bit quantization for the model. Requires CUDA.

    device_map : str, default="auto"
        Device mapping for the model.
    """

    def __init__(
        self,
        model_path: str,
        load_in_8bit: bool = False,
        device_map: str = 'auto',
    ):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        # Configure model loading based on quantization setting and device availability
        if load_in_8bit:
            if DEVICE.type != 'cuda':
                raise ValueError('8-bit quantization requires CUDA device')
            quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map=device_map,
            )
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
                device_map=device_map,
            )

        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        self.cache = {}
        self.model.eval()

    def get_perplexity(
        self, input_texts: Union[str, List[str]], debug=False
    ) -> Union[float, List[float]]:
        """
        Calculates the perplexity of given texts.

        Parameters
        ----------
        input_texts : str or list of str
            A single string or a list of strings.

        batch_size : int, default=None
            Batch size for processing. Defaults to the number of input texts.

        debug : bool, default=False
            Print debugging information.

        Returns
        -------
        float or list of float
            A single perplexity value if input is a single string,
            or a list of perplexity values if input is a list of strings.

        Examples
        --------
        >>> import pandas as pd
        >>> model_path = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"
        >>> scorer = PerplexityCalculator(model_path=model_path)

        >>> submission = pd.DataFrame({
        ...     'id': [0, 1, 2],
        ...     'text': ["this is a normal english sentence", "thsi is a slihgtly misspelled zr4g sentense", "the quick brown fox jumps over the lazy dog"]
        ... })
        >>> perplexities = scorer.get_perplexity(submission["text"].tolist())
        >>> perplexities[0] < perplexities[1]
        True
        >>> perplexities[2] < perplexities[0]
        True

        >>> perplexities = scorer.get_perplexity(["this is a sentence", "another sentence"])
        >>> all(p > 0 for p in perplexities)
        True

        >>> scorer.clear_gpu_memory()
        """
        single_input = isinstance(input_texts, str)
        input_texts = [input_texts] if single_input else input_texts

        loss_list = []
        with torch.no_grad():
            # Process each sequence independently
            for text in input_texts:
                if text in self.cache:
                    print("score cache hit")
                    loss_list.append(self.cache[text])
                    print(f"score: {self.cache[text]} for text: {text}")
                    continue
                    
                # Explicitly add sequence boundary tokens to the text
                text_with_special = f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}"

                # Tokenize
                model_inputs = self.tokenizer(
                    text_with_special,
                    return_tensors='pt',
                    add_special_tokens=False,
                )

                if 'token_type_ids' in model_inputs:
                    model_inputs.pop('token_type_ids')

                model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}
                self.loss_fct = self.loss_fct.to(DEVICE)  # Add this after initializing loss_fct
                
                # Get model output
                output = self.model(**model_inputs, use_cache=False)
                logits = output['logits']

                # Shift logits and labels for calculating loss
                shift_logits = logits[..., :-1, :].contiguous()  # Drop last prediction
                shift_labels = model_inputs['input_ids'][..., 1:].contiguous()  # Drop first input

                # Calculate token-wise loss
                loss = self.loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                # Calculate average loss
                sequence_loss = loss.sum() / len(loss)
                loss_item = sequence_loss.cpu().item()
                loss_list.append(loss_item)
                self.cache[text] = loss_item


                # Debug output
                if debug:
                    print(f"\nProcessing: '{text}'")
                    print(f"With special tokens: '{text_with_special}'")
                    print(f"Input tokens: {model_inputs['input_ids'][0].tolist()}")
                    print(f"Target tokens: {shift_labels[0].tolist()}")
                    print(f"Input decoded: {self.tokenizer.decode(model_inputs['input_ids'][0])}")
                    print(f"Target decoded: {self.tokenizer.decode(shift_labels[0])}")
                    print(f"Individual losses: {loss.tolist()}")
                    print(f"Average loss: {sequence_loss.item():.4f}")

        ppl = [exp(i) for i in loss_list]

        if debug:
            print("\nFinal perplexities:")
            for text, perp in zip(input_texts, ppl):
                print(f"Text: '{text}'")
                print(f"Perplexity: {perp:.2f}")
        return ppl[0] if single_input else ppl

    def clear_gpu_memory(self) -> None:
        """Clears GPU memory by deleting references and emptying caches."""
        if not torch.cuda.is_available():
            return

        # Delete model and tokenizer if they exist
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer

        # Run garbage collection
        gc.collect()

        # Clear CUDA cache and reset memory stats
        with DEVICE:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect
            torch.cuda.reset_peak_memory_stats()


import ast

def fix_llm_output(llm_output, original_sentence):
    start_tag = "<answer>"
    end_tag = "</answer>"
    
    try:
        # Find the last occurrence of start tag
        last_start_index = llm_output.rfind(start_tag)
        
        if last_start_index == -1:
            print("Answer tags not found in LLM output(defaulting to original sentence): ", llm_output)
            return original_sentence
            
        # Find the end tag that follows the last start tag
        start_index = last_start_index + len(start_tag)
        end_index = llm_output.find(end_tag, start_index)
        
        if end_index == -1:
            print("Closing answer tag not found after last opening tag(defaulting to original sentence): ", llm_output)
            return original_sentence
            
        answer = llm_output[start_index:end_index].strip()
        if not answer:
            print("empty answer, returning original sentence")
            return original_sentence
        # Create sets of words from input and answer (case-insensitive)
        input_words = set(word.lower() for word in original_sentence.split())
        answer_words = answer.split()
        
        # Filter answer words to keep only those present in input (case-insensitive)
        filtered_answer = []
        for word in answer_words:
            if word.lower() in input_words:
                filtered_answer.append(word)
        
        # Join the filtered words back into a sentence
        final_answer = ' '.join(filtered_answer)
        return final_answer
        
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return llm_output  # Return full output as fallback

def score_outputs(llm_outputs):
    preds = []
    i = 0
    original_sentences = data.text.tolist()
    
    for llm_output in llm_outputs:
        try:
            original_sentence = original_sentences[i]
            clean_sentence = fix_llm_output(llm_output, original_sentence)
            preds.append(clean_sentence)
        except (ValueError, SyntaxError, IndexError) as e:
            # If parsing fails, use the original sentence and log the error
            print(f"Error parsing output at index {i}: {str(e)}")
            print(f"Original LLM output: {llm_output}")
            print("Will fallback to original sentence...")
            llm_outputs.append(original_sentences[i])
            preds.append(original_sentences[i])
        
        i += 1
    
    scores = scorer.get_perplexity(preds)
    return llm_outputs, preds, np.mean(scores)
    


%%time

sentences_model1 = []
for sentence in data.text.tolist():
    print("getting prediction for sentence:", sentence)
    prediction = get_prediction(model, tokenizer, sentence, PROMPT)
    sentences_model1.append(prediction)


model_path = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"
scorer = PerplexityCalculator(model_path)


outputs1, answers1, score1 = score_outputs(sentences_model1)
score1




