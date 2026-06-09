from huggingface_hub import login

login("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")


import sys
sys.path.append("/kaggle/input/neoai-2025-dftd-baseline-code/")

import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from detector import FakeTextDetector


from torch.utils.data import Dataset


class FakeTextDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        mode: str
    ) -> None:
        super().__init__()

        assert mode in ["train", "eval", "test"]

        self.dataset_path = dataset_path
        self.mode = mode

        self.df = pd.read_csv(dataset_path)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        sample = {}
        if self.mode == "test":
            sample["prompt"] = row.prompt
        else:
            sample["prompt"] = row.prompt
            sample["human"] = row.human
        return sample

    def __len__(self) -> int:
        return self.df.shape[0]


!pip install sae_lens==4.4.5 -q


from sae_lens import SAE, HookedSAETransformer


@dataclass
class Config:
    # data params
    data_path: str = "/kaggle/input/neoai-2025-evading-generated-text-detection"
    test_dataset_name: str = "test.csv"
    num_workers: int = 1
    batch_size: int = 20
    output_submission_path: str = "submission.csv"

    # gemma params
    device_llm: str = "cuda:0"
    model_name = "google/gemma-2-2b"

    # sae params
    release: str = "gemma-scope-2b-pt-res-canonical"
    device_sae: str = "cuda:0"
    layer: int = 20
    num_latents_k: int = 16

    # detector params
    device_detector: str = "cuda:1"


# Initialize Gemma model
model = HookedSAETransformer.from_pretrained(Config.model_name, local_files_only=False, device=Config.device_llm)
model.eval()

# Initialize SAE model
sae, _, _ = SAE.from_pretrained(
    release=Config.release,
    sae_id=f"layer_{Config.layer}/width_{Config.num_latents_k}k/canonical"
)
sae = sae.to(Config.device_sae)

# Initialize detector model
detector = FakeTextDetector(device=Config.device_detector)


def infer(
        config: Config,
        model: torch.nn.Module,
        max_new_tokens: int = 128,
        stop_at_eos: bool = True,
        prepend_bos: bool = True,
        verbose: bool = False,
        skip_special_tokens: bool = True
    ) -> None:
    dataset = FakeTextDataset(os.path.join(config.data_path, config.test_dataset_name), mode="test")
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    submission = {"prompt": [], "generation": []}
    for batch in tqdm(dataloader):
        prompts = batch["prompt"]

        submission["prompt"].extend(prompts)

        with torch.no_grad():
            input_ids = model.to_tokens(prompts, prepend_bos=True).to('cuda:0')
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                stop_at_eos=stop_at_eos,
                prepend_bos=prepend_bos,
                verbose=verbose
            ).cpu().numpy()
        generated_texts = model.tokenizer.batch_decode(output, skip_special_tokens=skip_special_tokens)
        submission["generation"].extend(generated_texts)

    submission = pd.DataFrame(submission)

    submission.prompt = submission.prompt.apply(lambda x: x.replace('"', "'"))
    submission.generation = submission.generation.apply(lambda x: x.replace('"', '"'))

    submission = submission.astype(pd.StringDtype())

    submission.to_csv(config.output_submission_path, index=False)



infer(config=Config, model=model, max_new_tokens=128)


from typing import List, Union

from sentence_transformers import SentenceTransformer, util

from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TextComparator:
    def __init__(
        self,
        device: str = "cuda:0",
        model_dir: str = "/kaggle/input/text-comparator"
    ) -> None:
        self.model = SentenceTransformer(model_dir, device=device)
        self.embedding_cache = {}

    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        single_text = isinstance(texts, str)
        if single_text:
            texts = [texts]

        uncached_texts = [t for t in texts if t not in self.embedding_cache]
        if uncached_texts:
            new_embeddings = self.model.encode(uncached_texts, convert_to_numpy=True)
            for text, embedding in zip(uncached_texts, new_embeddings):
                self.embedding_cache[text] = embedding

        embeddings = np.array([self.embedding_cache[t] for t in texts])

        return embeddings[0] if single_text else embeddings

    def similarity(
        self,
        text1: Union[str, List[str]],
        text2: Union[str, List[str]]
    ) -> Union[float, np.ndarray]:
        emb1 = self.get_embeddings(text1)
        emb2 = self.get_embeddings(text2)

        return np.diag(util.cos_sim(emb1, emb2).numpy())


class FakeTextDetector:
    def __init__(
        self,
        device: str = "cuda:0",
        model_dir: str = "/kaggle/input/fake-text-detector/"
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.model.to(device)

    def get_score(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to(self.device)
        with torch.no_grad():
            score = self.model(**inputs).logits.cpu().detach().numpy()
        return score


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    similarity_threshold: float = 0.65,
    score_threshold_low: float = -2.0,
    score_threshold_high: float = 2.0
) -> float:
    assert solution.shape == submission.shape
    assert (solution[row_id_column_name] == submission[row_id_column_name]).all()

    comparator = TextComparator()
    detector = FakeTextDetector()

    similarities = comparator.similarity(
        solution["generation"].tolist(),
        submission["generation"].tolist()
    )

    steered_texts = submission["generation"].tolist()

    scores = []
    for steered_text, similarity_score in tqdm(zip(steered_texts, similarities)):
        if similarity_score < similarity_threshold:
            scores.append(0)
            continue

        scores_i = detector.get_score(steered_text).item()
        scores.append(scores_i)

    similarity_mask = similarities > similarity_threshold
    deceptions = (np.array(scores) > score_threshold_low) & (np.array(scores) < score_threshold_high) & similarity_mask
    return deceptions.mean()



submission = pd.read_csv(Config.output_submission_path)

metric = score(solution=submission, submission=submission, row_id_column_name="prompt")
print(f"LB metric in best case: {metric.item()}")




