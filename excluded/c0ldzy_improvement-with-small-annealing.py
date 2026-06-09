import yaml
import datetime
import time
import gc
import os
import transformers
import torch
import numpy as np
import pandas as pd
import random
import math
from collections import Counter
from pprint import pprint
from typing import List, Union


os.environ["TOKENIZERS_PARALLELISM"] = "false"
PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ParticipantVisibleError(Exception):
    pass


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    model_path: str = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2",
    load_in_8bit: bool = True,
    clear_mem: bool = False,
) -> float:

    # Check that each submitted string is a permutation of the solution string
    sol_counts = solution.loc[:, "text"].str.split().apply(Counter)
    sub_counts = submission.loc[:, "text"].str.split().apply(Counter)
    invalid_mask = sol_counts != sub_counts
    if invalid_mask.any():
        raise ParticipantVisibleError(
            "At least one submitted string is not a valid permutation of the solution string."
        )

    # Calculate perplexity for the submitted strings
    sub_strings = [" ".join(s.split()) for s in submission["text"].tolist()]  # Split and rejoin to normalize whitespace
    scorer = PerplexityCalculator(
        model_path=model_path,
        load_in_8bit=load_in_8bit,
    )  # Initialize the perplexity calculator with a pre-trained model
    perplexities = scorer.get_perplexity(sub_strings)  # Calculate perplexity for each submitted string

    if clear_mem:
        # Just move on if it fails. Not essential if we have the score.
        try:
            scorer.clear_gpu_memory()
        except:
            print("GPU memory clearing failed.")

    return float(np.mean(perplexities))


class PerplexityCalculator:
    def __init__(
        self,
        model_path: str,
        load_in_8bit: bool = False,
        device_map: str = "auto",
    ):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, padding_side="right")
        # Configure model loading based on quantization setting and device availability
        if load_in_8bit:
            if DEVICE.type != "cuda":
                raise ValueError("8-bit quantization requires CUDA device")

            # quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
            # quantization_config = transformers.BitsAndBytesConfig(load_in_4bit=True)

            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",  # fp4 nf4
                bnb_4bit_use_double_quant=False,
                bnb_4bit_compute_dtype=torch.float16,
            )

            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map=device_map,
            )
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
                device_map=device_map,
            )

        self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        self.model.eval()

    def get_perplexity(self, input_texts: Union[str, List[str]], batch_size: 32) -> Union[float, List[float]]:
        single_input = isinstance(input_texts, str)
        input_texts = [input_texts] if single_input else input_texts

        loss_list = []

        batches = len(input_texts) // batch_size + (len(input_texts) % batch_size != 0)
        for j in range(batches):

            a = j * batch_size
            b = (j + 1) * batch_size
            input_batch = input_texts[a:b]

            with torch.no_grad():

                # Explicitly add sequence boundary tokens to the text
                text_with_special = [
                    f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}" for text in input_batch
                ]

                # Tokenize
                model_inputs = self.tokenizer(
                    text_with_special, return_tensors="pt", add_special_tokens=False, padding=True
                )

                if "token_type_ids" in model_inputs:
                    model_inputs.pop("token_type_ids")

                model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}

                # Get model output
                output = self.model(**model_inputs, use_cache=False)
                logits = output["logits"]

                label = model_inputs["input_ids"]
                label[label == self.tokenizer.pad_token_id] = PAD_TOKEN_LABEL_ID

                # Shift logits and labels for calculating loss
                shift_logits = logits[..., :-1, :].contiguous()  # Drop last prediction
                shift_labels = label[..., 1:].contiguous()  # Drop first input

                # Calculate token-wise loss
                loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                loss = loss.view(len(logits), -1)
                valid_length = (shift_labels != PAD_TOKEN_LABEL_ID).sum(dim=-1)
                loss = torch.sum(loss, -1) / valid_length

                loss_list += loss.cpu().tolist()

        ppl = [math.exp(i) for i in loss_list]

        return ppl[0] if single_input else ppl

    def clear_gpu_memory(self) -> None:
        """Clears GPU memory by deleting references and emptying caches."""
        if not torch.cuda.is_available():
            return

        # Delete model and tokenizer if they exist
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer

        # Run garbage collection
        gc.collect()

        # Clear CUDA cache and reset memory stats
        with DEVICE:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()


def format_time(elapsed):
    """Take a time in seconds and return a string hh:mm:ss."""
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))
    
class SimulatedAnnealing:
    def __init__(self, Tmax, Tmin, nsteps, nsteps_per_T, log_freq, random_state, cooling, k):
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.nsteps = nsteps
        self.nsteps_per_T = nsteps_per_T
        self.log_freq = log_freq
        self.cooling = cooling
        self.k = k
        random.seed(random_state)

    def _generate_neighbor(self, solution):
        r = random.choice(range(2))
        if r == 0:
            neighbor = solution.copy()
            i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            return neighbor
        elif r == 1:
            shift = solution.copy()
            extract, insert = random.sample(range(len(shift) - 1), 2)
            shift_words = shift[extract : extract + 1]
            shift = shift[:extract] + shift[extract + 1 :]
            shift = shift[:insert] + shift_words + shift[insert:]
            return shift

    def _acceptance_probability(self, current_energy, new_energy, temperature):
        """
        Calculate the probability of accepting a new solution.
        """
        if new_energy < current_energy:
            return 1.0
        return math.exp(self.k * (current_energy - new_energy) / temperature)

    def solve(self, text):

        t0 = time.time()  # Measure staring time
        current_solution = text.split()
        
        A = " ".join(current_solution)
        if A in USED:
            current_energy = USED[A]
        else:
            current_energy = scorer.get_perplexity(A, batch_size=config["batch_size"])
            USED[A] = current_energy

        best_solution = current_solution.copy()
        best_energy = current_energy

        temperature = self.Tmax
        Tfactor = -math.log(self.Tmax / self.Tmin)  # for exponentil cooling

        temperatures = [temperature]
        log_energies = [current_energy]

        for step in range(self.nsteps):
            cntused = 0
            accept = 0

            for step1 in range(self.nsteps_per_T):
                # generate neighbor
                new_solution = self._generate_neighbor(current_solution)
                A = " ".join(new_solution)
                if A in USED:
                    new_energy = USED[A]
                    cntused += 1
                else:
                    new_energy = scorer.get_perplexity(A, batch_size=config["batch_size"])
                    USED[A] = new_energy

                # calculation of acceptance probability
                acceptance = self._acceptance_probability(current_energy, new_energy, temperature)

                # update current solution
                if acceptance > random.random():
                    current_solution = new_solution
                    current_energy = new_energy
                    accept += 1
                # update best solution
                if new_energy < best_energy:
                    best_solution = new_solution.copy()
                    best_energy = new_energy
                    if new_energy < 10000:
                        print(f"\nNew best score: {best_energy:8.3f}")
                        print("New text: ", " ".join(best_solution), "\n", flush=True)

                # log
                log_energies.append(current_energy)
                temperatures.append(temperature)

                t1 = format_time(time.time() - t0)

                if step1 % self.log_freq == 0 or step1 == (self.nsteps_per_T - 1):
                    print(
                        f"T: {temperature:8.3f}  Step: {step1:6} CntUsed: {cntused}  Acceptance Rate: {accept/(step1+1):7.4f}  Score: {current_energy:8.3f}  Best Score: {best_energy:8.3f}  Elapsed Time: {t1}",
                        flush=True,
                    )

            # lower the temperature
            if self.cooling == "linear":
                temperature -= (self.Tmax - self.Tmin) / self.nsteps
            elif self.cooling == "exponential":
                temperature = self.Tmax * math.exp(Tfactor * (step + 1) / self.nsteps)
            elif self.cooling == "logarithmic":
                temperature = self.Tmax / math.log10(step + 10)

            if best_energy < 30.:
                print("Stop! Target value is achieved.")
                break

        return " ".join(best_solution), best_energy, log_energies, temperatures


# store already computed perplexities, helps to speed up, especially doing many attempts
USED = {}


%%writefile config.yaml

subfile: "/kaggle/input/santa-2024/sample_submission.csv"
sample: 3
batch_size: 1024
model_path: '/kaggle/input/gemma-2/transformers/gemma-2-9b/2'

params:
    Tmax: 2
    Tmin: 1
    nsteps: 30
    nsteps_per_T: 3000
    log_freq: 1000
    random_state: 77
    cooling: 'exponential'
    k: 1.


with open("config.yaml", "r") as file_obj:
    config = yaml.safe_load(file_obj)


# We won't need it for now
# scorer = PerplexityCalculator(config["model_path"])
# optimizer = SimulatedAnnealing(**config["params"])


# Solution with 205.5 score
text = "sleigh of the magi yuletide cheer is unwrap gifts and eat cheer holiday holly jingle relax sing carol visit workshop chimney stocking naughty nice grinch polar beard nutcracker ornament decorations"

# I spent all GPU quota, so can't submit with computing the answer, below is a screenshot
# solution, score, log_scores, log_ts = optimizer.solve(text)

# So for now I will just enter the solution that is found by the code above
solution = "sleigh of the magi yuletide cheer is unwrap gifts and eat cheer holiday decorations holly jingle relax sing carol visit workshop grinch naughty nice chimney stocking nutcracker polar beard ornament"
score = 203.99134454426334

print("Final Score:", score)
print("Final Solution:", solution)


# Current best public solution
t = """reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament
reindeer sleep walk the night and drive mistletoe scrooge laugh chimney jump elf bake gingerbread family give advent fireplace ornament
sleigh yuletide beard carol cheer chimney decorations gifts grinch holiday holly jingle magi naughty nice nutcracker ornament polar workshop stocking
sleigh of the magi yuletide cheer is unwrap gifts and eat cheer holiday decorations holly jingle relax carol sing chimney visit workshop grinch naughty nice polar beard nutcracker ornament stocking
with as and from in it of not that the to we you angel believe bow candle candy chocolate cookie doll dream eggnog fireplace fruitcake game greeting card have hohoho hope joy kaggle merry milk night peace peppermint poinsettia puzzle season snowglobe star toy wrapping paper wreath wish wonder workshop
from and and as we and have the in is it of not that the to with you advent card angel bake beard believe bow candy candle carol cheer cheer chocolate chimney cookie decorations doll dream drive eat eggnog family fireplace fireplace chimney fruitcake game gifts give gingerbread greeting grinch holiday holly hohoho hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night elf nutcracker ornament ornament of the wrapping paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wreath yuletide"""

df = pd.read_csv("/kaggle/input/santa-2024/sample_submission.csv")

df['text'] = t.split('\n')

df.at[config["sample"], "text"] = solution

df.to_csv("submission.csv", index=False)


df

