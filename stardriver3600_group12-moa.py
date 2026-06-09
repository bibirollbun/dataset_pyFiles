import sys

%mkdir ./tmp
%mkdir ./tmp/preprocessed
%mkdir ./tmp/exp


import pandas as pd
import numpy as np
import scipy as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import os
import time
from umap import UMAP

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import TSNE
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import KFold, RepeatedKFold
from joblib import parallel_backend

from tqdm import tqdm
import random
import os
import math

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False

DEFAULT_SEED = 512
seed_everything(seed_value=DEFAULT_SEED)


# iterative-stratification
# https://github.com/trent-b/iterative-stratification
from sklearn.utils import check_random_state
from sklearn.utils.validation import _num_samples, check_array
from sklearn.utils.multiclass import type_of_target

from sklearn.model_selection._split import _BaseKFold, _RepeatedSplits, \
    BaseShuffleSplit, _validate_shuffle_split

def IterativeStratification(labels, r, random_state):
    n_samples = labels.shape[0]
    test_folds = np.zeros(n_samples, dtype=int)

    # Calculate the desired number of examples at each subset
    c_folds = r * n_samples

    # Calculate the desired number of examples of each label at each subset
    c_folds_labels = np.outer(r, labels.sum(axis=0))

    labels_not_processed_mask = np.ones(n_samples, dtype=bool)

    while np.any(labels_not_processed_mask):
        # Find the label with the fewest (but at least one) remaining examples,
        # breaking ties randomly
        num_labels = labels[labels_not_processed_mask].sum(axis=0)

        # Handle case where only all-zero labels are left by distributing
        # across all folds as evenly as possible (not in original algorithm but
        # mentioned in the text). (By handling this case separately, some
        # code redundancy is introduced; however, this approach allows for
        # decreased execution time when there are a relatively large number
        # of all-zero labels.)
        if num_labels.sum() == 0:
            sample_idxs = np.where(labels_not_processed_mask)[0]

            for sample_idx in sample_idxs:
                fold_idx = np.where(c_folds == c_folds.max())[0]

                if fold_idx.shape[0] > 1:
                    fold_idx = fold_idx[random_state.choice(fold_idx.shape[0])]
                else:
                    fold_idx = fold_idx[0]

                test_folds[sample_idx] = fold_idx
                c_folds[fold_idx] -= 1

            break

        label_idx = np.where(num_labels == num_labels[np.nonzero(num_labels)].min())[0]
        if label_idx.shape[0] > 1:
            label_idx = label_idx[random_state.choice(label_idx.shape[0])]

        sample_idxs = np.where(np.logical_and(labels[:, label_idx].flatten(), labels_not_processed_mask))[0]

        for sample_idx in sample_idxs:
            # Find the subset(s) with the largest number of desired examples
            # for this label, breaking ties by considering the largest number
            # of desired examples, breaking further ties randomly
            label_folds = c_folds_labels[:, label_idx]
            fold_idx = np.where(label_folds == label_folds.max())[0]
            
            if fold_idx.shape[0] > 1:
                temp_fold_idx = np.where(c_folds[fold_idx] ==
                                         c_folds[fold_idx].max())[0]
                
                if temp_fold_idx.shape[0] > 1:
                    fold_idx = fold_idx[temp_fold_idx]
                    fold_idx = fold_idx[random_state.choice(temp_fold_idx.shape[0])]
                else:
                    fold_idx = fold_idx[temp_fold_idx[0]]
            else:
                fold_idx = fold_idx[0]

            test_folds[sample_idx] = fold_idx
            labels_not_processed_mask[sample_idx] = False

            # Update desired number of examples
            c_folds_labels[fold_idx, labels[sample_idx]] -= 1
            c_folds[fold_idx] -= 1

    return test_folds

class MultilabelStratifiedKFold(_BaseKFold):

    def __init__(self, n_splits=3, *, shuffle=False, random_state=None):
        super(MultilabelStratifiedKFold, self).__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _make_test_folds(self, X, y):
        y = np.asarray(y, dtype=bool)
        type_of_target_y = type_of_target(y)

        if type_of_target_y != 'multilabel-indicator':
            raise ValueError(
                'Supported target type is: multilabel-indicator. Got {!r} instead.'.format(type_of_target_y))

        num_samples = y.shape[0]

        rng = check_random_state(self.random_state)
        indices = np.arange(num_samples)

        if self.shuffle:
            rng.shuffle(indices)
            y = y[indices]

        r = np.asarray([1 / self.n_splits] * self.n_splits)

        test_folds = IterativeStratification(labels=y, r=r, random_state=rng)

        return test_folds[np.argsort(indices)]

    def _iter_test_masks(self, X=None, y=None, groups=None):
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            yield test_folds == i

    def split(self, X, y, groups=None):
        y = check_array(y, ensure_2d=False, dtype=None)
        return super(MultilabelStratifiedKFold, self).split(X, y, groups)


class RepeatedMultilabelStratifiedKFold(_RepeatedSplits):
    def __init__(self, n_splits=5, *, n_repeats=10, random_state=None):
        super(RepeatedMultilabelStratifiedKFold, self).__init__(
            MultilabelStratifiedKFold, n_repeats=n_repeats, random_state=random_state,
            n_splits=n_splits)


class MOADataset(torch.utils.data.Dataset):
    def __init__(self, feature_file_path, target_file_path, feature_mode):
        self.features = pd.read_csv(feature_file_path)
        if target_file_path:
            self.targets = pd.read_csv(target_file_path).drop(columns=["sig_id"]).to_numpy()
        else:
            self.targets = [0] * len(self.features)

        #self.cp_type = np.expand_dims(pd.Categorical(self.features["cp_type"], categories=["ctl_vehicle", "trt_cp"]).codes, axis=1)
        #self.cp_dose = np.expand_dims(pd.Categorical(self.features["cp_dose"], categories=['D1', 'D2']).codes, axis=1)
        #self.cp_time = np.expand_dims(pd.Categorical(self.features["cp_time"], categories=['24', '48', '72']).codes, axis=1)
        self.cp_type = pd.get_dummies(pd.Categorical(self.features["cp_type"], categories=["ctl_vehicle", "trt_cp"]).codes)
        self.cp_dose = pd.get_dummies(pd.Categorical(self.features["cp_dose"], categories=['D1', 'D2']).codes)
        self.cp_time = pd.get_dummies(pd.Categorical(self.features["cp_time"], categories=['24', '48', '72']).codes)

        self.cnt_features = self.features.drop(columns=["sig_id", "cp_type", "cp_time", "cp_dose"])

        if feature_mode == "CNT":
            self.feat = self.cnt_features.to_numpy()
        elif feature_mode == "BOTH":
            self.feat = np.hstack([self.cp_type, self.cp_time, self.cp_dose, self.cnt_features.to_numpy()])
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        return {'feature': self.feat[index], 'target': self.targets[index]}
    
    def __len__(self):
        return len(self.feat)
    
class MOABatchCollate(object):
    def __call__(self, batch):
        features = torch.from_numpy(np.array([item['feature'] for item in batch])).float()
        targets = torch.from_numpy(np.array([item['target'] for item in batch])).float()
        return {'features': features, 'targets': targets}


# pytorch implementation of mixture of experts
# reference from https://github.com/lucidrains/mixture-of-experts/

# constants
from inspect import isfunction

MIN_EXPERT_CAPACITY = 4

# helper functions

def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val

def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

# tensor related helper functions

def top1(t):
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index

def cumsum_exclusive(t, dim=-1):
    num_dims = len(t.shape)
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice   = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]

# pytorch one hot throws an error if there are out of bound indices.
# tensorflow, in contrast, does not throw an error
def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]

def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)


class Experts(nn.Module):
    def __init__(self,
        dim,
        num_experts = 16,
        hidden_dim = None,
        activation = nn.GELU):
        super().__init__()

        hidden_dim = default(hidden_dim, dim * 4)
        num_experts = cast_tuple(num_experts)

        w1 = torch.zeros(*num_experts, dim, hidden_dim)
        w2 = torch.zeros(*num_experts, hidden_dim, dim)

        w1 = init_(w1)
        w2 = init_(w2)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.act = activation()

    def forward(self, x):
        hidden = torch.einsum('...nd,...dh->...nh', x, self.w1)
        hidden = self.act(hidden)
        out    = torch.einsum('...nh,...hd->...nd', hidden, self.w2)
        return out

class Top2Gating(nn.Module):
    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        outer_expert_dims = tuple(),
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))

        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

    def forward(self, x, importance = None):
        *_, b, group_size, dim = x.shape
        num_gates = self.num_gates

        if self.training:
            policy = self.second_policy_train
            threshold = self.second_threshold_train
            capacity_factor = self.capacity_factor_train
        else:
            policy = self.second_policy_eval
            threshold = self.second_threshold_eval
            capacity_factor = self.capacity_factor_eval

        raw_gates = torch.einsum('...bnd,...de->...bne', x, self.w_gating)
        raw_gates = raw_gates.softmax(dim=-1)

        # FIND TOP 2 EXPERTS PER POSITON
        # Find the top expert for each position. shape=[batch, group]

        gate_1, index_1 = top1(raw_gates)
        mask_1 = F.one_hot(index_1, num_gates).float()
        density_1_proxy = raw_gates

        if importance is not None:
            equals_one_mask = (importance == 1.).float()
            mask_1 *= equals_one_mask[..., None]
            gate_1 *= equals_one_mask
            density_1_proxy = density_1_proxy * equals_one_mask[..., None]
            del equals_one_mask

        gates_without_top_1 = raw_gates * (1. - mask_1)

        gate_2, index_2 = top1(gates_without_top_1)
        mask_2 = F.one_hot(index_2, num_gates).float()

        if importance is not None:
            greater_zero_mask = (importance > 0.).float()
            mask_2 *= greater_zero_mask[..., None]
            del greater_zero_mask

        # normalize top2 gate scores
        denom = gate_1 + gate_2 + self.eps
        gate_1 /= denom
        gate_2 /= denom

        # BALANCING LOSSES
        # shape = [batch, experts]
        # We want to equalize the fraction of the batch assigned to each expert
        density_1 = mask_1.mean(dim=-2)
        # Something continuous that is correlated with what we want to equalize.
        density_1_proxy = density_1_proxy.mean(dim=-2)
        loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)

        # Depending on the policy in the hparams, we may drop out some of the
        # second-place experts.
        if policy == "all":
            pass
        elif policy == "none":
            mask_2 = torch.zeros_like(mask_2)
        elif policy == "threshold":
            mask_2 *= (gate_2 > threshold).float()
        elif policy == "random":
            probs = torch.zeros_like(gate_2).uniform_(0., 1.)
            mask_2 *= (probs < (gate_2 / max(threshold, self.eps))).float().unsqueeze(-1)
        else:
            raise ValueError(f"Unknown policy {policy}")

        # Each sequence sends (at most?) expert_capacity positions to each expert.
        # Static expert_capacity dimension is needed for expert batch sizes
        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)

        # COMPUTE ASSIGNMENT TO EXPERTS
        # [batch, group, experts]
        # This is the position within the expert's mini-batch for this sequence
        position_in_expert_1 = cumsum_exclusive(mask_1, dim=-2) * mask_1
        # Remove the elements that don't fit. [batch, group, experts]
        mask_1 *= (position_in_expert_1 < expert_capacity_f).float()
        # [batch, experts]
        # How many examples in this sequence go to this expert
        mask_1_count = mask_1.sum(dim=-2, keepdim=True)
        # [batch, group] - mostly ones, but zeros where something didn't fit
        mask_1_flat = mask_1.sum(dim=-1)
        # [batch, group]
        position_in_expert_1 = position_in_expert_1.sum(dim=-1)
        # Weight assigned to first expert.  [batch, group]
        gate_1 *= mask_1_flat

        position_in_expert_2 = cumsum_exclusive(mask_2, dim=-2) + mask_1_count
        position_in_expert_2 *= mask_2
        mask_2 *= (position_in_expert_2 < expert_capacity_f).float()
        mask_2_flat = mask_2.sum(dim=-1)

        position_in_expert_2 = position_in_expert_2.sum(dim=-1)
        gate_2 *= mask_2_flat
        
        # [batch, group, experts, expert_capacity]
        combine_tensor = (
            gate_1[..., None, None]
            * mask_1_flat[..., None, None]
            * F.one_hot(index_1, num_gates)[..., None]
            * safe_one_hot(position_in_expert_1.long(), expert_capacity)[..., None, :] +
            gate_2[..., None, None]
            * mask_2_flat[..., None, None]
            * F.one_hot(index_2, num_gates)[..., None]
            * safe_one_hot(position_in_expert_2.long(), expert_capacity)[..., None, :]
        )

        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        return dispatch_tensor, combine_tensor, loss

class MoE(nn.Module):
    def __init__(self,
        dim,
        num_experts = 16,
        hidden_dim = None,
        activation = nn.ReLU,
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,
        experts = None):
        super().__init__()

        self.num_experts = num_experts

        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}
        self.gate = Top2Gating(dim, num_gates = num_experts, **gating_kwargs)
        self.experts = default(experts, lambda: Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
        self.loss_coef = loss_coef

    def forward(self, inputs, **kwargs):
        b, n, d, e = *inputs.shape, self.num_experts
        dispatch_tensor, combine_tensor, loss = self.gate(inputs)
        expert_inputs = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor)

        # Now feed the expert inputs through the experts.
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(e, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)
        return output, loss * self.loss_coef


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class MLP(nn.Module):
    def __init__(self, num_features = 256, num_targets = 256, dropout_r=0.1):
        super(MLP, self).__init__()
        self.num_features = num_features
        self.num_target = num_targets
        self.mlp = nn.Sequential(nn.BatchNorm1d(self.num_features),
                                 nn.Dropout(p=dropout_r),
                                 nn.utils.weight_norm(nn.Linear(self.num_features, self.num_features * 4)),
                                 nn.Mish(),
                                 nn.BatchNorm1d(self.num_features*4),
                                 nn.Dropout(p=dropout_r),
                                 nn.utils.weight_norm(nn.Linear(self.num_features * 4, self.num_target)))
        
    def forward(self, features):
        logits = self.mlp(features)

        return logits


class MOE_MLP(nn.Module):
    def __init__(self, num_features = 256, num_targets = 256, dropout_r=0.1):
        super(MOE_MLP, self).__init__()
        self.num_features = num_features
        self.num_target = num_targets
        self.norm = nn.Sequential(  nn.BatchNorm1d(self.num_features),
                                    nn.Dropout(p=dropout_r))
        
        self.moe = MoE(
            dim = self.num_features,
            num_experts = 16,               # increase the experts (# parameters) of your model without increasing computation
            hidden_dim = self.num_features * 4,  # size of hidden dimension in each expert, defaults to 4 * dimension
            activation = nn.Mish,           # use your preferred activation, will default to GELU
            second_policy_train = 'random', # in top_2 gating, policy for whether to use a second-place expert
            second_policy_eval = 'random',  # all (always) | none (never) | threshold (if gate value > the given threshold) | random (if gate value > threshold * random_uniform(0, 1))
            second_threshold_train = 0.2,
            second_threshold_eval = 0.2,
            capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
            capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
            loss_coef = 1e-2                # multiplier on the auxiliary expert balancing auxiliary loss
        )
        
        self.mlp = nn.Sequential(nn.BatchNorm1d(self.num_features),
                                 nn.Dropout(p=dropout_r),
                                 nn.utils.weight_norm(nn.Linear(self.num_features, self.num_features * 4)),
                                 nn.Mish(),
                                 nn.BatchNorm1d(self.num_features*4),
                                 nn.Dropout(p=dropout_r),
                                 nn.utils.weight_norm(nn.Linear(self.num_features * 4, self.num_target)))
        
    def forward(self, features):
        features = self.norm(features)
        if features.shape[0] % GROUP_SIZE != 0:
            pad_len = GROUP_SIZE - (features.shape[0] % GROUP_SIZE)
            pad_tensor = torch.zeros((pad_len, features.shape[-1]), dtype=features.dtype, device=features.device)
            features = torch.cat([features, pad_tensor], dim=0)
            features_group = features.reshape(-1, GROUP_SIZE, features.shape[-1])
            moe_features_group, aux_loss = self.moe(features_group)
            moe_features = moe_features_group.reshape(-1, features.shape[-1])
            moe_features = moe_features[:moe_features.shape[0]-pad_len]
        else:
            features_group = features.reshape(-1, GROUP_SIZE, features.shape[-1])
            moe_features_group, aux_loss = self.moe(features_group)
            moe_features = moe_features_group.reshape(-1, features.shape[-1])
        logits = self.mlp(moe_features)

        return logits, aux_loss



DATA_SET_DIR = "../input/lish-moa"
MODEL_DIR = "./tmp/exp"
train_features = pd.read_csv(f'{DATA_SET_DIR}/train_features.csv')
train_targets_scored = pd.read_csv(f'{DATA_SET_DIR}/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv(f'{DATA_SET_DIR}/train_targets_nonscored.csv')

test_features = pd.read_csv(f'{DATA_SET_DIR}/test_features.csv')
sample_submission = pd.read_csv(f'{DATA_SET_DIR}/sample_submission.csv')

IS_TRAIN = True
DATA_DIR = './tmp/preprocessed'
# label smoothing
PMIN = 0.0
PMAX = 1.0

# submission smoothing
SMIN = 0.0
SMAX = 1.0



GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]
for col in tqdm((GENES + CELLS)):
    vec_len = len(train_features[col].values)
    vec_len_test = len(test_features[col].values)
    raw_vec = pd.concat([train_features, test_features])[col].values.reshape(vec_len+vec_len_test, 1)
    if IS_TRAIN:
        transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
        transformer.fit(raw_vec)
        pd.to_pickle(transformer, f'{DATA_DIR}/{col}_quantile_transformer.pkl')
    else:
        transformer = pd.read_pickle(f'{DATA_DIR}/{col}_quantile_transformer.pkl')        
    train_features[col] = transformer.transform(train_features[col].values.reshape(vec_len, 1)).reshape(1, vec_len)[0]
    test_features[col] = transformer.transform(test_features[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]


# GENES
n_comp = 50
n_dim = 15

data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])

if IS_TRAIN:
    pca = PCA(n_components=n_comp, random_state=DEFAULT_SEED).fit(train_features[GENES])
    umap = UMAP(n_components=n_dim, random_state=DEFAULT_SEED).fit(train_features[GENES])
    pd.to_pickle(pca, f"{DATA_DIR}/pca_g.pkl")
    pd.to_pickle(umap, f"{DATA_DIR}/umap_g.pkl")
else:
    pca = pd.read_pickle(f"{DATA_DIR}/pca_g.pkl")
    umap = pd.read_pickle(f"{DATA_DIR}/umap_g.pkl")
    
data2 = pca.transform(data[GENES])
data3 = umap.transform(data[GENES])

train2 = data2[:train_features.shape[0]]
test2 = data2[-test_features.shape[0]:]
train3 = data3[:train_features.shape[0]]
test3 = data3[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])
train3 = pd.DataFrame(train3, columns=[f'umap_G-{i}' for i in range(n_dim)])
test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])
test3 = pd.DataFrame(test3, columns=[f'umap_G-{i}' for i in range(n_dim)])

train_features = pd.concat((train_features, train2, train3), axis=1)
test_features = pd.concat((test_features, test2, test3), axis=1)

#CELLS
n_comp = 15
n_dim = 5

data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])


if IS_TRAIN:
    pca = PCA(n_components=n_comp, random_state=DEFAULT_SEED).fit(train_features[CELLS])
    umap = UMAP(n_components=n_dim, random_state=DEFAULT_SEED).fit(train_features[CELLS])
    pd.to_pickle(pca, f"{DATA_DIR}/pca_c.pkl")
    pd.to_pickle(umap, f"{DATA_DIR}/umap_c.pkl")
else:
    pca = pd.read_pickle(f"{DATA_DIR}/pca_c.pkl")
    umap = pd.read_pickle(f"{DATA_DIR}/umap_c.pkl")   

data2 = pca.transform(data[CELLS])
data3 = umap.transform(data[CELLS])

train2 = data2[:train_features.shape[0]]
test2 = data2[-test_features.shape[0]:]
train3 = data3[:train_features.shape[0]]
test3 = data3[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])
train3 = pd.DataFrame(train3, columns=[f'umap_C-{i}' for i in range(n_dim)])
test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])
test3 = pd.DataFrame(test3, columns=[f'umap_C-{i}' for i in range(n_dim)])

train_features = pd.concat((train_features, train2, train3), axis=1)
test_features = pd.concat((test_features, test2, test3), axis=1)


from sklearn.feature_selection import VarianceThreshold

if IS_TRAIN:
    var_thresh = VarianceThreshold(threshold=0.5).fit(train_features.iloc[:, 4:])
    pd.to_pickle(var_thresh, f"{DATA_DIR}/variance_thresh0_5.pkl")
else:
    var_thresh = pd.read_pickle(f"{DATA_DIR}/variance_thresh0_5.pkl")
                                
data = pd.concat([train_features, test_features])
data_transformed = var_thresh.transform(data.iloc[:, 4:])

train_features_transformed = data_transformed[ : train_features.shape[0]]
test_features_transformed = data_transformed[-test_features.shape[0] : ]


train_features = pd.DataFrame(train_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                              columns=['sig_id','cp_type','cp_time','cp_dose'])

train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)


test_features = pd.DataFrame(test_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                             columns=['sig_id','cp_type','cp_time','cp_dose'])

test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)

print(train_features.shape)
print(test_features.shape)

#train = train_features[train_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)
#test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

#train = train.drop('cp_type', axis=1)
#test = test.drop('cp_type', axis=1)
train_features.to_csv(f"{DATA_DIR}/train_preprocessed.csv")
test_features.to_csv(f"{DATA_DIR}/test_preprocessed.csv")


# training hyper params
EPOCHS = 15
BATCH_SIZE = 2048
NFOLDS = 10 # 10
NREPEATS = 1
NSEEDS = 5 # 5
NUM_FEATURE = 949
NUM_TARGET = 206
FEATURE_MODE = "BOTH"
GROUP_SIZE = 2048
LR = 5e-4

SEED = 8591
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

PCT_START = 0.2
DIV_FACS = 1e3
MAX_LR = 1e-2


train_dataset = MOADataset(f"{DATA_DIR}/train_preprocessed.csv", "../input/lish-moa/train_targets_scored.csv", feature_mode=FEATURE_MODE)
test_dataset = MOADataset(f"{DATA_DIR}/test_preprocessed.csv", target_file_path=None, feature_mode=FEATURE_MODE)


def train_step(model:torch.nn.Module, optimizor:torch.optim.Optimizer, scheduler,
            train_loader: DataLoader, criteria: torch.nn.Module):
    
    model.train()
    total_loss = 0
    for batch in train_loader:
        features = batch['features'].to(DEVICE)
        targets = batch['targets'].to(DEVICE)
        
        
        optimizor.zero_grad()
        #predict = model(features)
        predict, aux_loss = model(features)
        #loss = criteria(predict, targets)
        loss = criteria(predict, targets) + aux_loss
        loss.backward()
        optimizor.step()
        scheduler.step()

        total_loss += loss.item()
    
    total_loss /= len(train_loader)

    return total_loss

def valid_step(model:torch.nn.Module, data_loader: DataLoader, criteria: torch.nn.Module):
    model.eval()
    total_loss = 0
    preds = []
    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)
        
            #predict = model(features)
            predict, aux_loss = model(features)
            #loss = criteria(predict, targets)
            loss = criteria(predict, targets)# + aux_loss

            total_loss += loss.item()
            preds.append(predict.sigmoid().detach().cpu().numpy())
        
        total_loss /= len(data_loader)
        preds = np.concatenate(preds)

    return total_loss, preds

def inference_step(model:torch.nn.Module, data_loader: DataLoader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(DEVICE)
        
            #predict = model(features)
            predict, _ = model(features)
            preds.append(predict.sigmoid().detach().cpu().numpy())
        preds = np.concatenate(preds)

    return preds


def run_single_fold(model:torch.nn.Module, optimizor:torch.optim.Optimizer, scheduler,
                    train_loader: DataLoader, valid_loader: DataLoader, criteria: torch.nn.Module, seed, fold):
    seed_everything(seed)
    train_loss_history = []
    valid_loss_history = []
    best_loss = np.inf
    best_valid_preds = None
    for epoch in tqdm(range(EPOCHS)):
        train_loss = train_step(model, optimizor, scheduler, train_loader, criteria)
        valid_loss, valid_preds = valid_step(model, valid_loader, criteria)

        if valid_loss < best_loss:            
            best_loss = valid_loss
            best_loss_epoch = epoch
            model.to('cpu')
            torch.save(model.state_dict(), f"{MODEL_DIR}/mlp_SEED{seed}_FOLD{fold}_best.pth")
            model.to(DEVICE)
        
        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        print(valid_loss)

    return train_loss_history, valid_loss_history

def run_k_fold(train_dataset, test_dataset, seed):
    #mskf = RepeatedKFold(n_splits=NFOLDS, n_repeats=NREPEATS, random_state=seed)
    mskf = RepeatedMultilabelStratifiedKFold(n_splits=NFOLDS, n_repeats=NREPEATS, random_state=seed)
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    predictions = np.zeros((len(test_dataset), NUM_TARGET))
    
    for fold, (t_idx, v_idx) in enumerate(mskf.split(X=train_dataset.feat, y=train_dataset.features)):
        print(f"Training fold {fold+1}/{NFOLDS * NREPEATS}")
        
        train_fold_dataset = Subset(train_dataset, t_idx)
        valid_fold_dataset = Subset(train_dataset, v_idx)
        train_fold_loader = DataLoader(train_fold_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=MOABatchCollate(), num_workers=4, drop_last = True)  # 減少 num_workers 避免記憶體問題
        valid_fold_loader = DataLoader(valid_fold_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=MOABatchCollate(), num_workers=4)
        
        model = MOE_MLP(NUM_FEATURE, NUM_TARGET, dropout_r=0.2).to(DEVICE)
        #model = MLP(NUM_FEATURE, NUM_TARGET, dropout_r=0.2).to(DEVICE)
        optimizer = torch.optim.Adam(lr=LR, params=list(model.parameters()))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, 
            pct_start=PCT_START, 
            div_factor=DIV_FACS, 
            max_lr=MAX_LR, 
            epochs=EPOCHS, 
            steps_per_epoch=len(train_fold_loader)
        )
        
        train_loss_history, valid_loss_history = run_single_fold(model, optimizer, scheduler, train_fold_loader, valid_fold_loader, loss_fn, seed, fold)

        model = MOE_MLP(NUM_FEATURE, NUM_TARGET, dropout_r=0.2).to(DEVICE)
        #model = MLP(NUM_FEATURE, NUM_TARGET, dropout_r=0.2).to(DEVICE)
        model.load_state_dict(torch.load(f"{MODEL_DIR}/mlp_SEED{seed}_FOLD{fold}_best.pth"))
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=MOABatchCollate(), num_workers=4)
        pred = inference_step(model, test_loader)
        predictions += pred / (NFOLDS * NREPEATS)

    return predictions



# Check actual feature count 
print("Check training data shape:")
print(f"train_dataset.feat.shape: {train_dataset.feat.shape}")
print(f"Actual feature count: {train_dataset.feat.shape[1]}")

print("\nCheck test data shape:")
print(f"test_dataset.feat.shape: {test_dataset.feat.shape}")

# Check preprocessed data
train_preprocessed = pd.read_csv(f"{DATA_DIR}/train_preprocessed.csv")
test_preprocessed = pd.read_csv(f"{DATA_DIR}/test_preprocessed.csv")

print(f"\nPreprocessed training data shape: {train_preprocessed.shape}")
print(f"Preprocessed test data shape: {test_preprocessed.shape}")

# Feature count after removing ID columns
actual_feature_count = train_dataset.feat.shape[1]
print(f"Actual available feature count: {actual_feature_count}")

# Correct NUM_FEATURE parameter
NUM_FEATURE = actual_feature_count
print(f"Corrected NUM_FEATURE: {NUM_FEATURE}")



predictions = run_k_fold(train_dataset, test_dataset, SEED)
print("DONE")


import pandas as pd
sample_df = pd.read_csv("../input/lish-moa/sample_submission.csv")
target_name = pd.read_csv("../input/lish-moa/train_targets_scored.csv").drop(columns=['sig_id']).columns.tolist()
test_df = pd.read_csv("../input/lish-moa/test_features.csv")
result_df = test_df.drop(columns=test_df.columns[test_df.columns!='sig_id'])
result_df = result_df.join(pd.DataFrame(index=result_df.index, columns=target_name))
#result_df.loc[test_df['cp_type'] == "ctl_vehicle", result_df.columns[1:]] = 0.0
#result_df.loc[test_df['cp_type'] != "ctl_vehicle", result_df.columns[1:]] = predictions
result_df.loc[:, result_df.columns[1:]] = predictions
result_df.to_csv('submission.csv', index=False)
#sample_df.to_csv('sample_submission.csv', index=False)
print("Submission created.")

