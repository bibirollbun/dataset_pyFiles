# need these for scorer to not fail
!pip install --upgrade widgetsnbextension
!pip install --upgrade ipywidgets
!jupyter nbextension enable --py widgetsnbextension
!pip install transformers==4.47.1
!pip install bitsandbytes


# Use a pipeline as a high-level helper
from transformers import pipeline
import torch 
from kaggle_secrets import UserSecretsClient
import csv
import os
import gc

user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret("HF_TOKEN")

#https://www.kaggle.com/code/pablomarino/prompt-optimization-on-claude-haiku?scriptVersionId=218473070&cellId=15
PROMPT1 = """Rearrange the words in the given sentence as to minimize perplexity, only reorder the words and output them as given but in a more coherent order, let's think step by step and output final answer final answer in between <answer></answer> tags.
Input: hung stockings canes bright sleigh bells echoed night candy
Output: One possible way to build a sentence with these words is: "The stockings were hung with candy canes bright, as Santa's sleigh bells echoed through the snowy night". I will keep only words provided in the input to end with: <answer>stockings hung candy canes bright sleigh bells echoed night</answer>
Input: evergreen carolers gathered evergreen tree ribbons glee tinsel sparkled
Output: One possible way to build a sentence with these words is: "Carolers gathered around the evergreen tree, while ribbons and tinsel sparkled with holiday glee.". I will keep only words provided in the input to end with: <answer>evergreen carolers gathered evergreen tree ribbons glee tinsel sparkled</answer>
Input:
"""
#https://www.kaggle.com/code/pablomarino/prompt-optimization-on-claude-haiku?scriptVersionId=218473070&cellId=15
PROMPT2 = """Rearrange the words in the given sentence as to minimize perplexity, only reorder the words and output them as given but in a more coherent order, let's think step by step and output final answer final answer in between <answer></answer> tags.
Input:
"""

PROMPT3 = """Rearrange the words in the given sentence as to minimize perplexity, only reorder the words and output them as given but in a more coherent order, let's think step by step and output final answer final answer in between <answer></answer> tags, be concise.
Input:
"""



def format_cuda_memory(free_bytes, total_bytes):
   gb = lambda x: round(x / 1024**3, 2)
   used_bytes = total_bytes - free_bytes
   return {
       'free': f"{gb(free_bytes)} GB",
       'total': f"{gb(total_bytes)} GB", 
       'used': f"{gb(used_bytes)} GB"
   }

# Usage
free, total = torch.cuda.mem_get_info()
mem = format_cuda_memory(free, total)
print(f"Free: {mem['free']}")
print(f"Total: {mem['total']}")
print(f"Used: {mem['used']}")


import pandas as pd

data = pd.read_csv('/kaggle/input/santa-2024/sample_submission.csv')
print(data.head())


def get_using_pipe(pipe, sentence, prompt):
    messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": sentence},
    ]
    response = pipe(messages,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        truncation = True,
        max_length=2000,)
    return response


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
                print(f"score: {self.cache[text]} for text: {text}")


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
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()


import ast

def fix_llm_output(llm_output, original_sentence):
    start_tag = "<answer>"
    end_tag = "</answer>"
    
    try:
        start_index = llm_output.find(start_tag) + len(start_tag)
        end_index = llm_output.find(end_tag)
        
        if start_index == -1 or end_index == -1:
            print("Answer tags not found in LLM output(defaulting to original sentence): ", llm_output)
            return original_sentence
            
        answer = llm_output[start_index:end_index].strip()
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
    clean_llm_outputs = []
    
    for llm_output in llm_outputs:
        try:
            parsed_llm_output = ast.literal_eval(llm_output)
            content = parsed_llm_output[0]["generated_text"][-1]["content"]
            clean_llm_outputs.append(content)
            original_sentence = original_sentences[i]
            clean_sentence = fix_llm_output(content, original_sentence)
            preds.append(clean_sentence)
        except (ValueError, SyntaxError, IndexError) as e:
            # If parsing fails, use the original sentence and log the error
            print(f"Error parsing output at index {i}: {str(e)}")
            print(f"Original LLM output: {llm_output}")
            print("Will fallback to original sentence...")
            clean_llm_outputs.append(original_sentences[i])
            preds.append(original_sentences[i])
        
        i += 1
    
    scores = scorer.get_perplexity(preds)
    return clean_llm_outputs, preds, np.mean(scores)
    


%%time

MODEL1_OUTPUTS_PATH = "/kaggle/input/model-outputs2/output_model1.1.csv"
model1_outputs = os.path.exists(MODEL1_OUTPUTS_PATH)

if not model1_outputs:
    print("model outputs not cached, will generate them")
    sentences_model1 = []
    pipe1 = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    for sentence in data.text.tolist():
        print("getting prediction for sentence:", sentence)
        prediction = get_using_pipe(pipe1, sentence, PROMPT1)
        sentences_model1.append(prediction)
    del pipe1
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("loading from cache")
    with open(MODEL1_OUTPUTS_PATH, 'r') as file:
       reader = csv.reader(file)
       next(reader)  # Skip header row
       sentences_model1 = [row[0] for row in reader]
with open('/kaggle/working/output_model1.1.csv', 'w', newline='') as file:
   writer = csv.writer(file)
   writer.writerow(['sentence'])  # Header
   for sentence in sentences_model1:
       writer.writerow([sentence])


%%time

MODEL2_OUTPUTS_PATH = "/kaggle/input/model-outputs2/output_model2.csv"
model2_outputs = os.path.exists(MODEL2_OUTPUTS_PATH)

if not model2_outputs:
    print("model outputs not cached, will generate them")
    sentences_model2 = []
    pipe2 = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct", token=HF_TOKEN, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    for sentence in data.text.tolist():
        print("getting prediction for sentence:", sentence)
        prediction = get_using_pipe(pipe2, sentence, PROMPT1)
        sentences_model2.append(prediction)
    del pipe2
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("loading from cache")
    with open(MODEL2_OUTPUTS_PATH, 'r') as file:
       reader = csv.reader(file)
       next(reader)  # Skip header row
       sentences_model2 = [row[0] for row in reader]
with open('/kaggle/working/output_model2.csv', 'w', newline='') as file:
   writer = csv.writer(file)
   writer.writerow(['sentence'])  # Header
   for sentence in sentences_model2:
       writer.writerow([sentence])


model_path = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"
scorer = PerplexityCalculator(model_path)


outputs1, answers1, score1 = score_outputs(sentences_model1)
score1


outputs2, answers2, score2 = score_outputs(sentences_model2)
score2


del scorer
gc.collect()
torch.cuda.empty_cache()


%%time
MODEL3_OUTPUTS_PATH = "/kaggle/input/model-outputs2/output_model3.2.csv"
model3_outputs = os.path.exists(MODEL3_OUTPUTS_PATH)

if not model3_outputs:
    print("model outputs not cached, will generate them")
    sentences_model3 = []
    pipe3 = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    for sentence in data.text.tolist():
        print("getting prediction for sentence:", sentence)
        prediction = get_using_pipe(pipe3, sentence, PROMPT2)
        sentences_model3.append(prediction)
    del pipe3
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("loading from cache")
    with open(MODEL3_OUTPUTS_PATH, 'r') as file:
       reader = csv.reader(file)
       next(reader)  # Skip header row
       sentences_model3 = [row[0] for row in reader]
with open('/kaggle/working/output_model3.2.csv', 'w', newline='') as file:
   writer = csv.writer(file)
   writer.writerow(['sentence'])  # Header
   for sentence in sentences_model3:
       writer.writerow([sentence])


%%time
MODEL4_OUTPUTS_PATH = "/kaggle/input/model-outputs2/output_model4.1.csv"
model4_outputs = os.path.exists(MODEL4_OUTPUTS_PATH)

if not model4_outputs:
    print("model outputs not cached, will generate them")
    sentences_model4 = []
    pipe4 = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct", token=HF_TOKEN, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    for sentence in data.text.tolist():
        print("getting prediction for sentence:", sentence)
        prediction = get_using_pipe(pipe4, sentence, PROMPT2)
        sentences_model4.append(prediction)
    del pipe4
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("loading from cache")
    with open(MODEL4_OUTPUTS_PATH, 'r') as file:
       reader = csv.reader(file)
       next(reader)  # Skip header row
       sentences_model4 = [row[0] for row in reader]
with open('/kaggle/working/output_model4.1.csv', 'w', newline='') as file:
   writer = csv.writer(file)
   writer.writerow(['sentence'])  # Header
   for sentence in sentences_model4:
       writer.writerow([sentence])


model_path = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"
scorer = PerplexityCalculator(model_path)


outputs3, answers3, score3 = score_outputs(sentences_model3)
score3


outputs4, answers4, score4 = score_outputs(sentences_model4)
score4


del scorer
gc.collect()
torch.cuda.empty_cache()


%%time
MODEL5_OUTPUTS_PATH = "/kaggle/input/model-outputs2/output_model5.csv"
model5_outputs = os.path.exists(MODEL5_OUTPUTS_PATH)

if not model5_outputs:
    print("model outputs not cached, will generate them")
    sentences_model5 = []
    pipe5 = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    for sentence in data.text.tolist():
        print("getting prediction for sentence:", sentence)
        prediction = get_using_pipe(pipe5, sentence, PROMPT3)
        sentences_model5.append(prediction)
    del pipe5
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("loading from cache")
    with open(MODEL5_OUTPUTS_PATH, 'r') as file:
       reader = csv.reader(file)
       next(reader)  # Skip header row
       sentences_model5 = [row[0] for row in reader]
with open('/kaggle/working/output_model5.csv', 'w', newline='') as file:
   writer = csv.writer(file)
   writer.writerow(['sentence'])  # Header
   for sentence in sentences_model5:
       writer.writerow([sentence])


%%time
MODEL6_OUTPUTS_PATH = "/kaggle/input/model-outputs2/output_model6.csv"
model6_outputs = os.path.exists(MODEL6_OUTPUTS_PATH)

if not model6_outputs:
    print("model outputs not cached, will generate them")
    sentences_model6 = []
    pipe6 = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct", token=HF_TOKEN, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    for sentence in data.text.tolist():
        print("getting prediction for sentence:", sentence)
        prediction = get_using_pipe(pipe6, sentence, PROMPT3)
        sentences_model6.append(prediction)
    del pipe6
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("loading from cache")
    with open(MODEL6_OUTPUTS_PATH, 'r') as file:
       reader = csv.reader(file)
       next(reader)  # Skip header row
       sentences_model6 = [row[0] for row in reader]
with open('/kaggle/working/output_model6.csv', 'w', newline='') as file:
   writer = csv.writer(file)
   writer.writerow(['sentence'])  # Header
   for sentence in sentences_model6:
       writer.writerow([sentence])


model_path = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"
scorer = PerplexityCalculator(model_path)


outputs5, answers5, score5 = score_outputs(sentences_model5)
score5


outputs6, answers6, score6 = score_outputs(sentences_model6)
score6


del scorer
gc.collect()
torch.cuda.empty_cache()




