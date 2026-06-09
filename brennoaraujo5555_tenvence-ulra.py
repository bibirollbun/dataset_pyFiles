# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Instalando as dependências
!pip install transformers torch torchvision torchaudio textstat nltk tqdm


import torch
import torch.utils.data as data
import transformers

SCORE_RANGES = {1: (2, 12), 2: (1, 6), 3: (0, 3), 4: (0, 3), 5: (0, 4), 6: (0, 4), 7: (0, 30), 8: (0, 60)}
NUM_SCORES = {key: max_score - min_score + 1 for key, (min_score, max_score) in SCORE_RANGES.items()}

def load_samples(root, prompt_idx, fold_idx, name='train'):
    samples = pd.read_csv(os.path.join(root, 'split_data', f'fold_{fold_idx}', f'{name}.tsv'), sep='\t', header=None)
    samples = samples.loc[samples[1] == prompt_idx]
    data_ids = torch.tensor(samples[0].tolist())
    essays = samples[2].tolist()
    cls_labels = torch.tensor(samples[3].tolist()) - SCORE_RANGES[prompt_idx][0]
    return data_ids, essays, cls_labels

def encode_to_bert(essays):
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer(essays, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    return input_ids, attention_mask

def load_features(root, prompt_idx):
    with open(os.path.join(root, 'features', f'prompt_{prompt_idx}.pkl'), 'rb') as f:
        data = pickle.load(f)
    data_ids = torch.tensor(data['ids'])
    features = torch.tensor(data['features']).float()
    return data_ids, features

def load_data(root, prompt_idx, fold_idx, name):
    sample_data_ids, essays, cls_labels = load_samples(root, prompt_idx, fold_idx, name)
    feature_data_ids, features = load_features(root, prompt_idx)
    mask = torch.eq(sample_data_ids[None, :], feature_data_ids[:, None])
    features = features[torch.argmax(mask.float(), dim=0), :]
    return essays, features, cls_labels

def load_datasets(root, prompt_idx, fold_idx=0):
    train_essays, train_features, train_cls_labels = load_data(root, prompt_idx, fold_idx, 'train')
    dev_essays, dev_features, dev_cls_labels = load_data(root, prompt_idx, fold_idx, 'dev')
    test_essays, test_features, test_cls_labels = load_data(root, prompt_idx, fold_idx, 'test')

    input_ids, attention_mask = encode_to_bert(train_essays + dev_essays + test_essays)
    train_input_ids = input_ids[:len(train_essays), ...]
    train_attention_mask = attention_mask[:len(train_essays), ...]
    dev_input_ids = input_ids[len(train_essays):len(train_essays) + len(dev_essays), ...]
    dev_attention_mask = attention_mask[len(train_essays):len(train_essays) + len(dev_essays), ...]
    test_input_ids = input_ids[len(train_essays) + len(dev_essays):, ...]
    test_attention_mask = attention_mask[len(train_essays) + len(dev_essays):, ...]

    train_dataset = data.TensorDataset(train_input_ids, train_attention_mask, train_features, train_cls_labels)
    dev_dataset = data.TensorDataset(dev_input_ids, dev_attention_mask, dev_features, dev_cls_labels)
    test_dataset = data.TensorDataset(test_input_ids, test_attention_mask, test_features, test_cls_labels)

    return train_dataset, dev_dataset, test_dataset


import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_features, weight_init):
        super(Model, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.score_fc = nn.Linear(self.bert.config.hidden_size, 1)
        self.weight_memory = nn.Parameter(torch.full((num_features,), fill_value=torch.logit(torch.tensor(weight_init))), requires_grad=True)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask).last_hidden_state
        x = bert_output.mean(dim=1)
        pred_score = self.score_fc(x)
        return pred_score, self.weight_memory.sigmoid()


import time
import torch.cuda.amp as amp

def train(model, optimizer, data_loader):
    st = time.time()
    losses = []
    model.train()
    scaler = amp.GradScaler()

    for input_ids, attention_mask, features, labels in data_loader:
        input_ids = input_ids.cuda(non_blocking=True)
        attention_mask = attention_mask.cuda(non_blocking=True)
        features = features.cuda(non_blocking=True)

        batch_size, num_features = features.shape
        optimizer.zero_grad()
        with amp.autocast():
            pred_score, weight_memory = model(input_ids, attention_mask)
            total_mask = torch.ones((batch_size, batch_size))
            idx_pairs = torch.nonzero(total_mask).cuda()
            features_a = features[idx_pairs[:, 0], :]
            features_b = features[idx_pairs[:, 1], :]
            ge_mask = torch.where(features_a >= features_b, weight_memory[None, :], 1. - weight_memory[None, :])
            pred_score_a = pred_score[idx_pairs[:, 0], :]
            pred_score_b = pred_score[idx_pairs[:, 1], :]
            term = pred_score_a.exp() / (pred_score_a.exp() + pred_score_b.exp())
            loss = -torch.log(ge_mask * term + (1 - ge_mask) * (1 - term)).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.detach())

    return torch.stack(losses).mean(), time.time() - st

@torch.no_grad()
def evaluate(model, data_loader, num_classes):
    st = time.time()
    model.eval()
    pred_scores_list, gt_labels_list = [], []
    for input_ids, attention_mask, _, cls_labels in data_loader:
        input_ids = input_ids.cuda(non_blocking=True)
        attention_mask = attention_mask.cuda(non_blocking=True)
        cls_labels = cls_labels.cuda(non_blocking=True)
        with amp.autocast():
            pred_scores, weight_memory = model(input_ids, attention_mask)
        pred_scores_list.append(pred_scores)
        gt_labels_list.append(cls_labels)
    pred_scores_list = torch.cat(pred_scores_list, dim=0).cpu().squeeze()
    gt_labels_list = torch.cat(gt_labels_list, dim=0).cpu()

    reg_scores = reg_scoring(pred_scores_list, num_classes)
    reg_qwk = quadratic_weighted_kappa(gt_labels_list, reg_scores)

    gt_scores = gt_scoring(pred_scores_list, gt_labels_list)
    gt_qwk = quadratic_weighted_kappa(gt_labels_list, gt_scores)

    u_scores = uniform_scoring(pred_scores_list, num_classes)
    u_qwk = quadratic_weighted_kappa(gt_labels_list, u_scores)

    t_scores = tri_scoring(pred_scores_list, num_classes)
    t_qwk = quadratic_weighted_kappa(gt_labels_list, t_scores)

    n_scores = normal_scoring(pred_scores_list, num_classes)
    n_qwk = quadratic_weighted_kappa(gt_labels_list, n_scores)

    return gt_qwk, u_qwk, reg_qwk, t_qwk, n_qwk, time.time() - st

def reg_scoring(pred_scores_list, num_classes):
    pred_scores_list -= min(pred_scores_list.clone())
    pred_scores_list /= max(pred_scores_list.clone())
    pred_scores_list *= (num_classes - 1)
    scores = torch.round(pred_scores_list.float()).long()
    return scores

def gt_scoring(pred_scores_list, gt_labels_list):
    gt_labels_sorted, gt_indices_sorted = torch.sort(gt_labels_list)
    gt_labels_set = sorted(set(gt_labels_sorted.numpy()))
    pred_scores_sorted, pred_indices_sorted = torch.sort(pred_scores_list)
    scores = torch.zeros(len(pred_scores_sorted))
    for label in gt_labels_set:
        ids = torch.nonzero(gt_labels_sorted == label).squeeze()
        scores[pred_indices_sorted[ids]] = int(label)
    return scores

def normal_scoring(pred_scores_list, num_classes):
    pred_scores_sorted, pred_indices_sorted = torch.sort(pred_scores_list)
    k = 1 / np.sqrt(2 * np.pi) * torch.exp(-(torch.arange(num_classes) - (num_classes - 1) / 2) ** 2 / 2)
    k = k / k.sum()
    num_list = [int(np.floor(len(pred_scores_list) * i)) for i in k]
    count = len(pred_scores_list) - sum(num_list)
    for i in range(num_classes):
        num_list[i] += 1
        count -= 1
        if count == 0:
            break
    scores = torch.zeros(len(pred_scores_list))
    sorted_labels = []
    for i, a in enumerate(num_list):
        sorted_labels.extend([i for _ in range(a)])
    sorted_labels = torch.tensor(sorted_labels, dtype=torch.float)
    scores[pred_indices_sorted] = sorted_labels
    return scores

def tri_scoring(pred_scores_list, num_classes):
    pred_scores_sorted, pred_indices_sorted = torch.sort(pred_scores_list)
    num_samples = len(pred_scores_list)
    k = -torch.abs(torch.arange(num_classes) - 1 - (num_classes - 1) / 2) + (num_classes + 1) / 2
    k /= torch.sum(k)
    num_list = [int(np.floor(a * num_samples)) for a in k]
    count = num_samples - sum(num_list)
    for i in range(num_classes):
        if count <= 0:
            break
        num_list[i] += 1
        count -= 1
    scores = torch.zeros(num_samples)
    sorted_labels = []
    for i, a in enumerate(num_list):
        sorted_labels.extend([i for _ in range(a)])
    sorted_labels = torch.tensor(sorted_labels, dtype=torch.float)
    scores[pred_indices_sorted] = sorted_labels
    return scores

def uniform_scoring(pred_scores_list, num_classes):
    pred_scores_sorted, pred_indices_sorted = torch.sort(pred_scores_list)
    scores = torch.zeros(len(pred_scores_sorted))
    scores[pred_indices_sorted] = torch.arange(len(pred_scores_sorted)).float()
    scores = scores / (len(pred_scores_sorted)) * num_classes
    scores = torch.round(scores).long()
    return scores


# Métricas

def pearson(pred_score, true_score):
    pred_avg = np.average(pred_score)
    true_avg = np.average(true_score)

    num, n1, n2 = 0.0, 0.0, 0.0
    for pred_t, true_t in zip(pred_score, true_score):
        num += (pred_t - pred_avg) * (true_t - true_avg)
        n1 += (pred_t - pred_avg) * (pred_t - pred_avg)
        n2 += (true_t - true_avg) * (true_t - true_avg)

    return num / np.power(n1 * n2, 0.5)


def spearman(pred_score, true_score):
    pred_score = np.asarray(pred_score)
    true_score = np.asarray(true_score)

    pred_sort = np.sort(pred_score)

    true_sort = np.sort(true_score)

    pred_index, true_index = [], []
    for pred_t, true_t in zip(pred_score, true_score):
        index_list = np.where(pred_sort == pred_t)

        index = (index_list[0] + index_list[-1]) / 2

        pred_index.append(index[0])

        index_list = np.where(true_sort == true_t)
        index = (index_list[0] + index_list[-1]) / 2

        true_index.append(index[0])

    nb = len(pred_score)
    err = 0.0
    for pred_i, true_i in zip(pred_index, true_index):
        err += np.power(pred_i - true_i, 2)

    return 1.0 - 6.0 * err / (np.power(nb, 3) - nb)


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for _ in range(num_ratings)] for _ in range(num_ratings)]
    # print(num_ratings, max(rater_a - min_rating), min(rater_a - min_rating), max(rater_b - min_rating), min(rater_b - min_rating))
    # print(num_ratings, min_rating, max_rating)
    for a, b in zip(rater_a, rater_b):
        # print(a, b, a - min_rating, b - min_rating)
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for _ in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b, min_rating, max_rating)

    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j] / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator


# Pré-processamento dos dados para criar a estrutura de pastas esperada

import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import nltk, re #, textstat
from textstat.textstat import textstat
from collections import Counter
import requests
import csv 

# Download da lista de palavras
dale_chall_url = "https://gist.githubusercontent.com/Abhishek-P/e00edcc6f508640fe24f263f5836a7dc/raw/166225e09fb8b554deff37ec344ad5ca40dab2fb/dale-chall-3000-words.txt"
response = requests.get(dale_chall_url)
if response.status_code == 200:
    dale_chall_common_words = set(response.text.splitlines())
    #print(dale_chall_common_words)
else:
    dale_chall_common_words = set()
    
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
#nltk.download('averaged_perceptron_tagger')

def compute_signals(text):
    tokens = nltk.word_tokenize(str(text))
    words = [t for t in tokens if re.match(r"\w+", t)]
    n_chars = len(text)
    n_words = len(words)
    n_sent = max(1, text.count('.') + text.count('!') + text.count('?'))
    uniq = len(set(w.lower() for w in words))
    pos = nltk.pos_tag(words, lang="eng")
    pos_counts = Counter(tag for _, tag in pos)

    # Surface signals
    ch = n_chars
    w = n_words
    co = text.count(',')
    uw = uniq

    # Preposition signals
    nnp = pos_counts.get('NNP', 0)
    dt = pos_counts.get('DT', 0)
    nn = pos_counts.get('NN', 0)
    rb = pos_counts.get('RB', 0)
    jj = pos_counts.get('JJ', 0)
    inn = pos_counts.get('IN', 0)

    # Readability signals
    fog = textstat.gunning_fog(text) if n_words > 0 else 0
    smog = textstat.smog_index(text) if n_words > 0 else 0
    rix = textstat.rix(text) if n_words > 0 else 0
    dc = textstat.dale_chall_readability_score(text) if n_words > 0 else 0
    wt = len(set(words))
    s = n_sent
    lw = sum(1 for w in words if len(w) > 6)
    cw = sum(1 for w in words if textstat.syllable_count(w) > 2)
    nbw = sum(1 for w in words if w.lower() not in dale_chall_common_words)
    dw = sum(1 for w in words if textstat.difficult_words(w))

    feats = [ch, w, co, uw, nnp, dt, nn, rb, jj, inn, fog, smog, rix, dc, wt, s, lw, cw, nbw, dw]
    return np.array(feats, dtype=float)

def preprocess_asap(tsv_path, out_root, test_size=0.1, dev_size=0.1, seed=42):
    os.makedirs(os.path.join(out_root, "split_data/fold_0"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "features"), exist_ok=True)

    # Lendo o arquivo original mantendo apenas colunas necessárias
    df = pd.read_csv(tsv_path, sep="\t", encoding="latin1", usecols=['essay_id', 'essay_set', 'essay', 'domain1_score'])
    
    # Substituindo tabs por espaços nas redações
    df['essay'] = df['essay'].str.replace('\t', ' ')
    
    for prompt_id in sorted(df['essay_set'].unique()):
        subset = df[df['essay_set'] == prompt_id].reset_index(drop=True)

        # Split train/dev/test
        train, temp = train_test_split(subset, test_size=(dev_size+test_size),
                                       random_state=seed, stratify=None)
        dev, test = train_test_split(temp, test_size=test_size/(dev_size+test_size),
                                     random_state=seed, stratify=None)

        # Escrevendo os splits com apenas 4 colunas
        for name, part in [("train", train), ("dev", dev), ("test", test)]:
            out_tsv = os.path.join(out_root, "split_data/fold_0", f"{name}.tsv")
            part[['essay_id', 'essay_set', 'essay', 'domain1_score']].to_csv(
                out_tsv, sep='\t', header=False, index=False, mode='a')
        
        feats = []
        ids = []
        for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"prompt {prompt_id}"):
            feats.append(compute_signals(row['essay']))
            ids.append(row['essay_id'])
        feats = np.vstack(feats)
        out_pkl = os.path.join(out_root, "features", f"prompt_{prompt_id}.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump({"ids": ids, "features": feats}, f)
        print(f"Saved features for prompt {prompt_id} -> {out_pkl}")


import argparse
import random
import warnings
import torch.backends.cudnn
import torch.utils.data as data

def get_args_parser():
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', default=19970423, type=int)
    parser.add_argument('--root', default='/kaggle/working/ASAP', type=str)
    parser.add_argument('--prompt_idx', default=1, type=int, help='{1, 2, 3, 4, 5, 6, 7, 8}')
    parser.add_argument('--weight_init', default=0.9, type=float)
    parser.add_argument('--training_type', default='inductive', type=str, help='{inductive, transductive}')
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.00005, type=float)
    parser.add_argument('--memory_lr', default=0.05, type=float)
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    return parser.parse_args([])

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run(args):
    set_random_seed(args.random_seed)
    num_classes = NUM_SCORES[args.prompt_idx]
    train_dataset, dev_dataset, test_dataset = load_datasets(args.root, args.prompt_idx)

    num_features = 20
    model = Model(num_features, args.weight_init).cuda()

    optim_parameters = [
        {'params': [p for n, p in model.named_parameters() if not n.endswith('weight_memory') and p.requires_grad], 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if n.endswith('weight_memory') and p.requires_grad], 'lr': args.memory_lr, 'weight_decay': args.weight_decay}
    ]
    optimizer = torch.optim.AdamW(optim_parameters)

    if args.training_type == 'transductive':
        dataset = data.ConcatDataset([train_dataset, dev_dataset, test_dataset])
        train_dataloader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
        test_dataloader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False)
    elif args.training_type == 'inductive':
        train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
        test_dataloader = data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False)
    else:
        raise NotImplementedError

    min_ep_loss = 100
    best_reg_qwk, best_u_qwk, best_gt_qwk, best_t_qwk, best_n_qwk = 0, 0, 0, 0, 0
    for ep_idx in range(args.num_epochs):
        ep_loss, train_t = train(model, optimizer, train_dataloader)
        gt_qwk, u_qwk, reg_qwk, t_qwk, n_qwk, test_t = evaluate(model, test_dataloader, num_classes)
        info = f'P{args.prompt_idx};Epoch={ep_idx + 1}/{args.num_epochs};TrainT={train_t:.1f}s;TestT={test_t:.1f}s;L={ep_loss:.4f};QWK={reg_qwk:.4f};'
        if ep_loss <= min_ep_loss:
            min_ep_loss = ep_loss
            best_reg_qwk, best_u_qwk, best_gt_qwk, best_t_qwk, best_n_qwk = reg_qwk, u_qwk, gt_qwk, t_qwk, n_qwk
            torch.save(model.state_dict(), f'{args.training_type}_P{args.prompt_idx}.pkl')
        info += f'minL={min_ep_loss:.4f};BestQWK={best_reg_qwk:.4f}.'
        print(info)

    info = f'     R,      G,      U,      T,      N | FINAL | prompt {args.prompt_idx}\n'
    info += f'{best_reg_qwk:.4f}, {best_gt_qwk:.4f}, {best_u_qwk:.4f}, {best_t_qwk:.4f}, {best_n_qwk:.4f}\n'
    print(info)

# if __name__ == '__main__':
#     args = get_args_parser()
#     run(args)


# Limpa diretórios anteriores para evitar conflitos
import shutil

if os.path.exists("/kaggle/working/ASAP"):
    shutil.rmtree("/kaggle/working/ASAP")

tsv_path = "/kaggle/input/asap-aes/training_set_rel3.tsv"
out_root = "/kaggle/working/ASAP"

preprocess_asap(tsv_path, out_root)


args = get_args_parser()
args.root = "/kaggle/working/ASAP"
run(args)


# Executando no modo transductive
args = get_args_parser()
args.root = "/kaggle/working/ASAP"
args.training_type = 'transductive'
run(args)

