!pip install transformers -q -U
!pip install accelerate -q


from huggingface_hub import login
login("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") #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


import sys
sys.path.append("/kaggle/input/neoai-2025-dftd-baseline-code/")

import os
import re
import numpy as np
import random
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed

from dataset import FakeTextDataset
from detector import FakeTextDetector


def seed_everything(seed: int):
  set_seed(seed)
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

seed_everything(1331)


print(torch.__version__)
print(torch.cuda.is_available()) # Should be True
print(torch.cuda.device_count()) # Should be 2
print(torch.cuda.get_device_name(0)) # Should show 'Tesla T4'
print(torch.cuda.get_device_name(1)) # Should show 'Tesla T4'


@dataclass
class Config:
    # data params
    data_path: str = "/kaggle/input/neoai-2025-evading-generated-text-detection/"
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
    device_detector: str = "cuda:0"


tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    device_map=Config.device_llm,
)

# Initialize detector model
detector = FakeTextDetector(device=Config.device_detector)


#tokenizer

tokenizer = AutoTokenizer.from_pretrained(Config.model_name)


model


model.config


def collect_residual_streams(model, target_layers, input_ids):
    # ada beberapa definisi dari residual streams: (1) output dari sebuah
    # transformer block (2) antara attention layer dan MLP di dalam sebuah
    # block. Kita akan menggunakan definisi yang pertama.
    activations = {}

    def target_act_hook(layer_idx):
        def hook(module, inputs, outputs):
            # kadang outputs[0]; kadang outputs
            # kalau qwen, outputs-nya tuple
            assert type(outputs) == tuple
            activation = outputs[0].detach()
            
            # aggregasi: mean
            # tipe data tensor di gemma sudah float32, beda dengan qwen yang bfloat16
            # and buat batch dim
            activation = activation.mean(dim=1).squeeze(0).cpu().numpy()
            
            activations[layer_idx] = activation
            return outputs
        return hook

    handles = []
    for layer_idx in target_layers:
        # perlu lihat arsitektur model-nya; beda model beda cara akses/nama layer-nya
        handle = model.model.layers[layer_idx].register_forward_hook(
            target_act_hook(layer_idx)
        )
        handles.append(handle)

    _ = model(input_ids=input_ids)

    for handle in handles:
        handle.remove()

    return activations


#Load HC3 dataset for solution

!pip install wldhx.yadisk-direct
!curl -L $(yadisk-direct https://disk.yandex.ru/d/Kz3OP8eQq49ubw) -o data.zip
!unzip -qq data.zip
!mv gemma_steering_ioai/* ./
!rm -rf gemma_steering_ioai data.zip


from torch.utils.data import Dataset, random_split

class HC3(Dataset):
    def __init__(
        self,
        dataset_path: str
    ):
        super().__init__()

        self.df = pd.read_csv(dataset_path)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        row_values = row.values

        sample = {}
        sample["question"] = row_values[0]
        sample["human_answers"] = row_values[1]
        sample["chatgpt_answers"] = row_values[2]
        sample["source"] = row_values[3]

        return sample

    def __len__(self) -> int:
        return len(self.df)

######
dataset = HC3(dataset_path="data/hc3.csv")
print(len(dataset))

# semuanya ada 23867, kebanyakan; silakan kalau mau coba semua
# tapi, kali ini, kita hanya pakai 1000 instances sample saja
n_samples = 1000
lengths = [n_samples, len(dataset) - n_samples]
dataset, _ = random_split(dataset, lengths, generator=torch.Generator().manual_seed(1331))
print(len(dataset))


TARGET_LAYERS = [12, 25] # Yang ditengah dan akhir saja

neuron_datasets = {}
for sample_idx in tqdm(range(len(dataset))):
    sample = dataset[sample_idx]

    tokens_human = tokenizer(sample["human_answers"], truncation = True, max_length=512, return_tensors = 'pt').input_ids
    tokens_gpt = tokenizer(sample["chatgpt_answers"], truncation = True, max_length=512, return_tensors = 'pt').input_ids
    
    with torch.no_grad():
        hum_activations = collect_residual_streams(model, 
                                                   TARGET_LAYERS, 
                                                   tokens_human.to(Config.device_llm)
                                                  )
        
        gpt_activations = collect_residual_streams(model, 
                                                   TARGET_LAYERS, 
                                                   tokens_gpt.to(Config.device_llm)
                                                  )
        
        for layer in TARGET_LAYERS:
            dataset_names = [f"hum-{layer}", f"gpt-{layer}"]
            for dataset_name in dataset_names:
                if dataset_name not in neuron_datasets.keys():
                    neuron_datasets[dataset_name] = []
    
                if "hum" in dataset_name:
                    neuron_datasets[dataset_name].append(hum_activations[layer])
                else:
                    neuron_datasets[dataset_name].append(gpt_activations[layer])


import xgboost as xgb

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

TOP_FEATURES = {}
choose_top_n_neurons = 20

for layer in TARGET_LAYERS:
    print(f"Layer: {layer}")

    data_hum = np.array(neuron_datasets[f"hum-{layer}"])
    data_gpt = np.array(neuron_datasets[f"gpt-{layer}"])

    data = np.concatenate([data_hum, data_gpt], axis=0)

    # 0: human, 1: gpt
    labels = np.concatenate([np.zeros(data_hum.shape[0]), np.ones(data_gpt.shape[0])])

    print(data.shape, labels.shape)

    X_train, X_eval, y_train, y_eval = train_test_split(data, 
                                                        labels, 
                                                        test_size=0.3, 
                                                        random_state=1331)

    clf = XGBClassifier(objective='binary:logistic', 
                        eval_metric='logloss', 
                        random_state=1331)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_train)
    print("f1 score (macro) - train: ", f1_score(y_train, y_pred, average='macro', labels=[0, 1]))

    y_eval_pred = clf.predict(X_eval)
    print("f1 score (macro) - dev: ", f1_score(y_eval, y_eval_pred, average='macro', labels=[0, 1]))


    ################ neuron importance #####################
    xgboost_feature_names = [f"neuron_{i}" for i in range(X_train.shape[1])]
    dtrain = xgb.DMatrix(data, label=labels, feature_names=xgboost_feature_names)
    
    params_class = {'objective': 'binary:logistic','eval_metric': 'logloss',
                    'eta': 0.1,'max_depth': 3,'seed': 1331
    }

    model_class_ = xgb.train(params_class, dtrain, num_boost_round=100)
    feature_gain = model_class_.get_score(importance_type='gain')
    feature_gain = [(feature_gain[f"neuron_{i}"] if f"neuron_{i}" in feature_gain else 0.)
                     for i in range(model.config.hidden_size)]

    chosen_features = []

    print(f"\nFeature importance (gain):")
    sorted_idx = np.argsort(feature_gain)[::-1]

    for idx in sorted_idx[:choose_top_n_neurons]:  # top n neurons
        feat_name = xgboost_feature_names[idx]
        print(f"{feat_name}: {feature_gain[idx]:.4f}")
        chosen_features.append(idx)
    
    # faktor untuk steering = delta
    # D = mean(hum) - mean(gpt) ---> untuk steering ke hum
    # D = mean(gpt) - mean(hum) ---> untuk steering ke gpt
    delta_to_hum = [data[labels == 0, idx].mean() - data[labels == 1, idx].mean() 
                                                              for idx in chosen_features]
    
    TOP_FEATURES[layer] = {
        "feature_idxs": chosen_features,
        "delta": delta_to_hum
    }

    print("=" * 80)


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
            #input_ids = model.to_tokens(prompts, prepend_bos=True)
            prompts = [tokenizer.bos_token + " " + p for p in prompts]
            
            input_ids = tokenizer(prompts, 
                                  return_tensors="pt", 
                                  padding=True, 
                                  truncation=True).input_ids.to(Config.device_llm)
            
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                #stop_at_eos=stop_at_eos,
                #prepend_bos=prepend_bos,
                eos_token_id=tokenizer.eos_token_id,
                #verbose=verbose
            ).cpu().numpy()
        generated_texts = tokenizer.batch_decode(output, skip_special_tokens=skip_special_tokens)
        submission["generation"].extend(generated_texts)

    submission = pd.DataFrame(submission)

    submission.prompt = submission.prompt.apply(lambda x: x.replace('"', "'"))
    submission.generation = submission.generation.apply(lambda x: x.replace('"', '"'))

    submission = submission.astype(pd.StringDtype())

    submission.to_csv(config.output_submission_path, index=False)



def create_steering_hook(layer: int, steering_vector: torch.Tensor):
    def hook(module, inputs, outputs):
        # ada model yang outpus saja, ada yang dibungkus dalam list, jadi perlu outputs[0]
        outputs_new = outputs[0].detach()
        return [outputs_new + steering_vector.to(Config.device_llm)]
    return hook

def steer(model, alpha=0.3):
    STEERING_DICT = {}
    for layer in TARGET_LAYERS:
        steering_vector = torch.zeros(model.config.hidden_size)
        for feature, delta in zip(
            TOP_FEATURES[layer]["feature_idxs"],
            TOP_FEATURES[layer]["delta"]
        ):
            steering_vector[feature] += delta * alpha
        STEERING_DICT[layer] = steering_vector

    # clear existing hooks, if exist
    for layer_idx in TARGET_LAYERS:
        model.model.layers[layer_idx]._forward_hooks.clear()
    
    handles = []
    for layer_idx in TARGET_LAYERS:
        # perlu lihat arsitektur model-nya; beda model beda cara akses/nama layer-nya
        handle = model.model.layers[layer_idx].register_forward_hook(
            create_steering_hook(layer, STEERING_DICT[layer])
        )
        handles.append(handle)

    ##### 
    with torch.no_grad():
        infer(config=Config, model=model, max_new_tokens=128)
    
    # remove hooks
    for handle in handles:
        handle.remove()


steer(model, alpha=0.5)


from typing import List, Union

from sentence_transformers import SentenceTransformer, util

from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TextComparator:
    def __init__(
        self,
        device: str = "cuda:1",
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
        device: str = "cuda:1",
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


# baseline sekitar 0.35...

# clear existing hooks, if exist
for layer_idx in TARGET_LAYERS:
    model.model.layers[layer_idx]._forward_hooks.clear()
    
with torch.no_grad():
    infer(config=Config, model=model, max_new_tokens=128)

submission = pd.read_csv(Config.output_submission_path)

metric = score(solution=submission, submission=submission, row_id_column_name="prompt")
print(f"LB metric in baseline: {metric.item()}")

