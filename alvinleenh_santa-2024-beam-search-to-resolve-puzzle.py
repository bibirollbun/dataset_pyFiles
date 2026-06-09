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


# import torch
# from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


google_gemma_2_transformers_gemma_2_9b_2_path='/kaggle/input/gemma-2/transformers/gemma-2-9b/2'

scorer = PerplexityCalculator(google_gemma_2_transformers_gemma_2_9b_2_path)


# Function to perform beam search
def beam_search(input_words, beam_width=1, max_length=20,print_output=0):
    # Initialize the beam with the input words and an empty sequence
    beam = [([], 0, input_words)]  # (sequence, score, remaining words)
    for _ in range(max_length):
        new_beam = []
        for seq, score, remaining_words in beam:
            if not remaining_words:
                break
            else:
                # Calculate the perplexity of new sequence for each remaining word
                for idx , word in enumerate(remaining_words):
                    new_seq = seq + [word]
                    new_score=scorer.get_perplexity(" ".join(new_seq))
                    if print_output:
                        print(f"new_seq {new_seq} score {new_score}")
                    new_remaining_words = remaining_words[:idx] + remaining_words[idx + 1:]
                    new_beam.append((new_seq, new_score, new_remaining_words))
                    
        if not remaining_words:
            break
        # Sort the new beam by score and keep the top beam_width sequences
        new_beam = sorted(new_beam, key=lambda x: x[1], reverse=False)[:beam_width]
        if print_output:
            print("\n")
        beam = new_beam

    if print_output:
        print(beam)
        
    # Return the best sequence
    best_seq = beam[0][0]
    return ' '.join(best_seq)


# Input words
input_text = "am i here"
score = scorer.get_perplexity(input_text)
print(f"Input sentence: {input_text}. score = {score}")
input_seq = input_text.split()
# Perform beam search
output_text = beam_search(input_seq, beam_width=2, max_length=len(input_seq),print_output=1)
score = scorer.get_perplexity(output_text)
print(f"Generated sentence: {output_text}. score = {score}")


def load_data(file_path):
    data = pd.read_csv(file_path)
    data['text'] = data['text'].str.strip()
    return data

input_file = "/kaggle/input/santa-2024/sample_submission.csv"
data = load_data(input_file)


decoded_passages = []
i = 0
for passage in data["text"]:
    score = scorer.get_perplexity(passage)
    print(f"passage: {passage} {score}\n")
    input_seq = passage.split()
    reordered_text = beam_search(input_seq, beam_width=1, max_length=len(input_seq), print_output=0)
    score = scorer.get_perplexity(reordered_text)
    print(f"reordered_text: {reordered_text} {score}\n\n")
    decoded_passages.append((reordered_text, score))


# decoded_passages = []
# # "reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament", 469
# # "reindeer mistletoe scrooge elf gingerbread chimney fireplace ornament family advent", 514, beam=5
# decoded_passages.append(("scrooge mistletoe ornament family advent fireplace chimney elf reindeer gingerbread",1328)) #beam=1
# decoded_passages.append(("scrooge mistletoe ornament and reindeer family advent fireplace chimney elf night sleep the gingerbread bake walk drive give laugh jump",1466))
# decoded_passages.append(("yuletide gifts grinch ornament nutcracker decorations holiday stocking holly jingle sleigh carol cheer chimney naughty nice beard workshop polar magi",967))
# decoded_passages.append(("yuletide gifts unwrap holiday cheer the nutcracker and grinch decorations ornament stocking holly jingle sleigh carol sing cheer chimney visit naughty nice beard workshop polar eat relax of is magi",755))
# decoded_passages.append(("eggnog fruitcake poinsettia snowglobe wreath candle cookie star candy peppermint chocolate milk hohoho merry and joy peace hope angel wish dream believe wonder night season greeting card paper wrapping bow toy doll game puzzle fireplace to the from of you we with in it that as not have workshop kaggle",283))
# decoded_passages.append(("advent chimney elf family fireplace gingerbread mistletoe ornament reindeer scrooge walk give jump drive bake the sleep night laugh and yuletide decorations gifts cheer holiday carol magi nutcracker polar grinch sleigh chimney workshop stocking ornament holly jingle beard naughty nice sing cheer and of the is eat visit relax unwrap hohoho candle poinsettia snowglobe peppermint eggnog fruitcake chocolate candy puzzle game doll toy workshop wonder believe dream hope peace joy merry season greeting card wrapping paper bow fireplace night cookie milk star wish wreath angel the to of and in that have it not with as you from we kaggle",354))


# replace 1st passage with brute-force golden result
decoded_passages[0] = ("reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament", 469)


output_file = "reordered_passages.txt"
total_perplexity = 0
with open(output_file, "w") as f:
    for idx, (text, perplexity) in enumerate(decoded_passages):
        total_perplexity += perplexity
        f.write(f"Passage {idx + 1}:\n{text}\nPerplexity: {perplexity}\n\n")

average_perplexity = total_perplexity / len(decoded_passages)
print(f"Average Perplexity: {average_perplexity}")



output_csv = "submission.csv"
import csv

with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "text"])  # Header

    for idx, (text, perplexity) in enumerate(decoded_passages):
        writer.writerow([idx, text])

print(f"Predictions saved to {output_csv}")


with open("submission.csv") as fp:
    print(fp.read())


import pandas as pd

# Prepare the data for display
predictions = [{"Passage_ID": idx + 1, "Reordered_Text": text, "Perplexity": perplexity}
               for idx, (text, perplexity) in enumerate(decoded_passages)]

# Convert to DataFrame
df = pd.DataFrame(predictions)

# Display the DataFrame in a tabular format
from IPython.display import display
display(df)

# Show the first few rows for preview
print("Preview of Results:")
print(df.head(30))





