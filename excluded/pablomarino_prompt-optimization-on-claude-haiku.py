import warnings
warnings.filterwarnings('ignore')
#Supress default INFO logging

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


!pip install anthropic==0.40.0
# need these for scorer to not fail
!pip install --upgrade widgetsnbextension
!pip install --upgrade ipywidgets
!jupyter nbextension enable --py widgetsnbextension
!pip install dspy


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
                loss_list.append(sequence_loss.cpu().item())

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


%%time
model_path = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"
scorer = PerplexityCalculator(model_path)


import anthropic
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
ANTHROPIC_API_KEY = user_secrets.get_secret("ANTHROPIC_API_KEY")


client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def call_LLM(sentence, system_prompt, model):
    messages = [{"role": "user", "content": sentence}]
    response = client.messages.create(
        model=model,
        system=system_prompt,
        max_tokens=2000,
        messages=messages,
        temperature=0
    )
    return response.content[0].text


def fix_sentence(sentence, prompt, model="claude-3-5-haiku-20241022"):
    print("sentence: ", sentence)
    llm_output = call_LLM(sentence, prompt, model)
    start_tag = "<answer>"
    end_tag = "</answer>"
    
    try:
        start_index = llm_output.find(start_tag) + len(start_tag)
        end_index = llm_output.find(end_tag)
        
        if start_index == -1 or end_index == -1:
            raise ValueError("Answer tags not found in LLM output")
            
        answer = llm_output[start_index:end_index].strip()
        print(f"Answer: {answer}")
        # Create sets of words from input and answer (case-insensitive)
        input_words = set(word.lower() for word in sentence.split())
        answer_words = answer.split()
        
        # Filter answer words to keep only those present in input (case-insensitive)
        filtered_answer = []
        for word in answer_words:
            if word.lower() in input_words:
                filtered_answer.append(word)
            else:
                print(f"word: {word} not in input_words")
        
        # Join the filtered words back into a sentence
        final_answer = ' '.join(filtered_answer)
        print("final_answer: ", final_answer)
        return final_answer
        
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return llm_output  # Return full output as fallback


def score_prompt(sentences, prompt):
    preds = []
    for sentence in sentences:
        sentence_prediction = fix_sentence(sentence, prompt)
        preds.append(sentence_prediction) 
    scores = scorer.get_perplexity(preds)
    return preds, np.mean(scores)

def clean_sentence(answer, original):
    input_words = set(word.lower() for word in original.split())
    answer_words = answer.split()
    
    # Filter answer words to keep only those present in input (case-insensitive)
    filtered_answer = []
    for word in answer_words:
        if word.lower() in input_words:
            filtered_answer.append(word)
        else:
            print(f"word: {word} not in input_words")
    
    # Join the filtered words back into a sentence
    final_answer = ' '.join(filtered_answer)
    return final_answer


import pandas as pd

data = pd.read_csv('/kaggle/input/santa-2024/sample_submission.csv')
print(data.head())


sentence_list = data.text.tolist()
sentence_list[0]


PROMPT1 = "Rearrange the words in the given sentence as to minimize perplexity, only reorder the words and output them as given but in a more coherent order, let's think step by step and output final answer final answer in between <answer></answer> tags"
preds1, score1 = score_prompt(sentence_list, PROMPT1)
score1


PROMPT2 = """Rearrange the words in the given sentence as to minimize perplexity, only reorder the words and output them as given but in a more coherent order, let's think step by step and output final answer final answer in between <answer></answer> tags.
Input: hung stockings canes bright sleigh bells echoed night candy
Output: One possible way to build a sentence with these words is: "The stockings were hung with candy canes bright, as Santa's sleigh bells echoed through the snowy night". I will keep only words provided in the input to end with: <answer>stockings hung candy canes bright sleigh bells echoed night</answer>
Input:
"""
preds2, score2 = score_prompt(sentence_list, PROMPT2)
score2



PROMPT3 = """Rearrange the words in the given sentence as to minimize perplexity, only reorder the words and output them as given but in a more coherent order, let's think step by step and output final answer final answer in between <answer></answer> tags.
Input: hung stockings canes bright sleigh bells echoed night candy
Output: One possible way to build a sentence with these words is: "The stockings were hung with candy canes bright, as Santa's sleigh bells echoed through the snowy night". I will keep only words provided in the input to end with: <answer>stockings hung candy canes bright sleigh bells echoed night</answer>
Input: evergreen carolers gathered evergreen tree ribbons glee tinsel sparkled
Output: One possible way to build a sentence with these words is: "Carolers gathered around the evergreen tree, while ribbons and tinsel sparkled with holiday glee.". I will keep only words provided in the input to end with: <answer>evergreen carolers gathered evergreen tree ribbons glee tinsel sparkled</answer>
Input:
"""
preds3, score3 = score_prompt(sentence_list, PROMPT3)
score3



PROMPT4 = """Rearrange the words in the given sentence as to minimize perplexity, only reorder the words and output them as given but in a more coherent order, let's think step by step and output final answer final answer in between <answer></answer> tags.
Input: hung stockings canes bright sleigh bells echoed night candy
Output: One possible way to build a sentence with these words is: "The stockings were hung with candy canes bright, as Santa's sleigh bells echoed through the snowy night". I will keep only words provided in the input to end with: <answer>stockings hung candy canes bright sleigh bells echoed night</answer>
Input: evergreen carolers gathered evergreen tree ribbons glee tinsel sparkled
Output: One possible way to build a sentence with these words is: "Carolers gathered around the evergreen tree, while ribbons and tinsel sparkled with holiday glee.". I will keep only words provided in the input to end with: <answer>evergreen carolers gathered evergreen tree ribbons glee tinsel sparkled</answer>
Input: fruitcake wrapping scattered paper twinkling eggnog warmed winter
Output: A natural way to build a sentence with these words is: "Wrapping paper scattered beneath twinkling lights, as fruitcake and eggnog warmed the winter nights". I will keep only words provided in the input to end with: <answer>Wrapping paper scattered twinkling fruitcake eggnog warmed winter</answer>
"""
preds4, score4 = score_prompt(sentence_list, PROMPT4)
score4



import dspy
lm = dspy.LM("claude-3-5-haiku-20241022", api_key=ANTHROPIC_API_KEY, temperature=0)
dspy.configure(lm=lm)


class FixSentenceDspy(dspy.Signature):
    """Rearrange the words in the given sentence as to minimize perplexity, only reorder the words and output them as given but in a more coherent order"""

    mixed_words: str = dspy.InputField()
    sorted_words: str = dspy.OutputField()

module = dspy.Predict(FixSentenceDspy)

mixed_words = sentence_list[0]
response = module(mixed_words=mixed_words)
response


lm.history[-1].keys()  # access the last call to the LM, with all metadata



lm.history[-1]["cost"]


test_set = [dspy.Example(mixed_words=sentence, sorted_words="").with_inputs("mixed_words") for sentence in sentence_list]


import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



scores = []
preds = []
for x in test_set:
    pred = module(**x.inputs())
    pred = clean_sentence(pred.sorted_words, x.mixed_words)
    preds.append(pred)
scores = scorer.get_perplexity(preds)
scores


np.mean(scores)


train_sentences = [
    "echo stockings hung candy canes bright santa snowy sleigh bells night",
    "tinsel sparkle glee caroler around evergreen holiday ribbons tree",
    "wrap paper scatter twinkle night light fruitcake eggnog warm beneath winter",
    "garlands windowsill holly danced jack along fresh bake cookie chill tempt Frost winter",
    "sugar adorn plums nutcracker december parlor care air chestnuts crackle",
    "hymns glow yuletide below giftwrap bows float lamplight snow"
]

trainset = [dspy.Example(mixed_words=sentence, sorted_words="").with_inputs("mixed_words") for sentence in train_sentences]



%%time
model_path = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"
scorer.clear_gpu_memory()
scorer = PerplexityCalculator(model_path)

def dspy_metric(example, pred, trace=None):
    pred = clean_sentence(pred.sorted_words, example.mixed_words)
    perplexity = scorer.get_perplexity([pred])[0]
    normalized_score = 1 / (1 + perplexity/1000)  # Divide by 1000 to adjust the scale
    return normalized_score # multiply by -1 since we want to minimize metric, not maximize


prompt_model_lm =   dspy.LM("claude-3-5-haiku-20241022", api_key=ANTHROPIC_API_KEY, temperature=0)
task_model_lm =  dspy.LM("claude-3-5-haiku-20241022", api_key=ANTHROPIC_API_KEY, temperature=0)

tp = dspy.MIPROv2(metric=dspy_metric, auto="medium", num_threads=1,prompt_model=prompt_model_lm, task_model=task_model_lm) # use 1 thread, metric function fails when running in // bc it runs an LLM in GPU and can't be paralelized
optimized_module = tp.compile(module, trainset=trainset)


scores = []
preds = []
for x in test_set:
    pred = optimized_module(**x.inputs())
    pred = clean_sentence(pred.sorted_words, x.mixed_words)
    preds.append(pred)
scores = scorer.get_perplexity(preds)
scores


np.mean(scores)


preds


optimized_module.signature


optimized_module.demos




