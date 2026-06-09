!pip install -q /kaggle/input/cibmtr-competition/qhoptim-1.1.0-py3-none-any.whl
!pip install pytorch_tabular -q --no-index --find-links=/kaggle/input/cibmtr-competition
!pip install rtdl_num_embeddings -q --no-index --find-links=/kaggle/input/cibmtr-competition/rtdl_num_embeddings


import os
import random
import numpy as np
import pandas as pd
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from qhoptim.pyt import QHAdam
from pytorch_tabular.models.common.layers import ODST
import rtdl_num_embeddings


# ======================================================================================
# Cell with TabM source code:
# https://github.com/yandex-research/tabm/blob/main/tabm_reference.py
# ======================================================================================



# License: https://github.com/yandex-research/tabm/blob/main/LICENSE

# NOTE
# The minimum required versions of the dependencies are specified in README.md.

import itertools
from typing import Any, Literal, Union

import rtdl_num_embeddings
import torch
import torch.nn as nn
from torch import Tensor


# ======================================================================================
# Initialization
# ======================================================================================
def init_rsqrt_uniform_(x: Tensor, d: int) -> Tensor:
    assert d > 0
    d_rsqrt = d**-0.5
    return nn.init.uniform_(x, -d_rsqrt, d_rsqrt)


@torch.inference_mode()
def init_random_signs_(x: Tensor) -> Tensor:
    return x.bernoulli_(0.5).mul_(2).add_(-1)


# ======================================================================================
# Modules
# ======================================================================================
class NLinear(nn.Module):
    """N linear layers applied in parallel to N disjoint parts of the input.

    **Shape**

    - Input: ``(B, N, in_features)``
    - Output: ``(B, N, out_features)``

    The i-th linear layer is applied to the i-th matrix of the shape (B, in_features).

    Technically, this is a simplified version of delu.nn.NLinear:
    https://yura52.github.io/delu/stable/api/generated/delu.nn.NLinear.html.
    The difference is that this layer supports only 3D inputs
    with exactly one batch dimension. By contrast, delu.nn.NLinear supports
    any number of batch dimensions.
    """

    def __init__(
        self, n: int, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(n, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        d = self.weight.shape[-2]
        init_rsqrt_uniform_(self.weight, d)
        if self.bias is not None:
            init_rsqrt_uniform_(self.bias, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3
        assert x.shape[-(self.weight.ndim - 1) :] == self.weight.shape[:-1]

        x = x.transpose(0, 1)
        x = x @ self.weight
        x = x.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias
        return x


class OneHotEncoding0d(nn.Module):
    # Input:  (*, n_cat_features=len(cardinalities))
    # Output: (*, sum(cardinalities))

    def __init__(self, cardinalities: list[int]) -> None:
        super().__init__()
        self._cardinalities = cardinalities

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 1
        assert x.shape[-1] == len(self._cardinalities)

        return torch.cat(
            [
                # NOTE
                # This is a quick hack to support out-of-vocabulary categories.
                #
                # Recall that lib.data.transform_cat encodes categorical features
                # as follows:
                # - In-vocabulary values receive indices from `range(cardinality)`.
                # - All out-of-vocabulary values (i.e. new categories in validation
                #   and test data that are not presented in the training data)
                #   receive the index `cardinality`.
                #
                # As such, the line below will produce the standard one-hot encoding for
                # known categories, and the all-zeros encoding for unknown categories.
                # This may not be the best approach to deal with unknown values,
                # but should be enough for our purposes.
                nn.functional.one_hot(x[..., i], cardinality + 1)[..., :-1]
                for i, cardinality in enumerate(self._cardinalities)
            ],
            -1,
        )


class ScaleEnsemble(nn.Module):
    def __init__(
        self,
        k: int,
        d: int,
        *,
        init: Literal['ones', 'normal', 'random-signs'],
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(k, d))
        self._weight_init = init
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self._weight_init == 'ones':
            nn.init.ones_(self.weight)
        elif self._weight_init == 'normal':
            nn.init.normal_(self.weight)
        elif self._weight_init == 'random-signs':
            init_random_signs_(self.weight)
        else:
            raise ValueError(f'Unknown weight_init: {self._weight_init}')

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 2
        return x * self.weight


class LinearEfficientEnsemble(nn.Module):
    """
    This layer is a more configurable version of the "BatchEnsemble" layer
    from the paper
    "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning"
    (link: https://arxiv.org/abs/2002.06715).

    First, this layer allows to select only some of the "ensembled" parts:
    - the input scaling  (r_i in the BatchEnsemble paper)
    - the output scaling (s_i in the BatchEnsemble paper)
    - the output bias    (not mentioned in the BatchEnsemble paper,
                          but is presented in public implementations)

    Second, the initialization of the scaling weights is configurable
    through the `scaling_init` argument.

    NOTE
    The term "adapter" is used in the TabM paper only to tell the story.
    The original BatchEnsemble paper does NOT use this term. So this class also
    avoids the term "adapter".
    """

    r: Union[None, Tensor]
    s: Union[None, Tensor]
    bias: Union[None, Tensor]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        k: int,
        ensemble_scaling_in: bool,
        ensemble_scaling_out: bool,
        ensemble_bias: bool,
        scaling_init: Literal['ones', 'random-signs'],
    ):
        assert k > 0
        if ensemble_bias:
            assert bias
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_parameter(
            'r',
            (
                nn.Parameter(torch.empty(k, in_features))
                if ensemble_scaling_in
                else None
            ),  # type: ignore[code]
        )
        self.register_parameter(
            's',
            (
                nn.Parameter(torch.empty(k, out_features))
                if ensemble_scaling_out
                else None
            ),  # type: ignore[code]
        )
        self.register_parameter(
            'bias',
            (
                nn.Parameter(torch.empty(out_features))  # type: ignore[code]
                if bias and not ensemble_bias
                else nn.Parameter(torch.empty(k, out_features))
                if ensemble_bias
                else None
            ),
        )

        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.scaling_init = scaling_init

        self.reset_parameters()

    def reset_parameters(self):
        init_rsqrt_uniform_(self.weight, self.in_features)
        scaling_init_fn = {'ones': nn.init.ones_, 'random-signs': init_random_signs_}[
            self.scaling_init
        ]
        if self.r is not None:
            scaling_init_fn(self.r)
        if self.s is not None:
            scaling_init_fn(self.s)
        if self.bias is not None:
            bias_init = torch.empty(
                # NOTE: the shape of bias_init is (out_features,) not (k, out_features).
                # It means that all biases have the same initialization.
                # This is similar to having one shared bias plus
                # k zero-initialized non-shared biases.
                self.out_features,
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
            bias_init = init_rsqrt_uniform_(bias_init, self.in_features)
            with torch.inference_mode():
                self.bias.copy_(bias_init)

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (B, K, D)
        assert x.ndim == 3

        # >>> The equation (5) from the BatchEnsemble paper (arXiv v2).
        if self.r is not None:
            x = x * self.r
        x = x @ self.weight.T
        if self.s is not None:
            x = x * self.s
        # <<<

        if self.bias is not None:
            x = x + self.bias
        return x


class MLP(nn.Module):
    def __init__(
        self,
        *,
        d_in: Union[None, int] = None,
        d_out: Union[None, int] = None,
        n_blocks: int,
        d_block: int,
        dropout: float,
        activation: str = 'ReLU',
    ) -> None:
        super().__init__()

        d_first = d_block if d_in is None else d_in
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_first if i == 0 else d_block, d_block),
                    getattr(nn, activation)(),
                    nn.Dropout(dropout),
                )
                for i in range(n_blocks)
            ]
        )
        self.output = None if d_out is None else nn.Linear(d_block, d_out)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        if self.output is not None:
            x = self.output(x)
        return x


def make_efficient_ensemble(module: nn.Module, **kwargs) -> None:
    """Replace torch.nn.Linear modules with LinearEfficientEnsemble.

    NOTE
    In the paper, there are no experiments with networks with normalization layers.
    Perhaps, their trainable weights (the affine transformations) also need
    "ensemblification" as in the paper about "FiLM-Ensemble".
    Additional experiments are required to make conclusions.
    """
    for name, submodule in list(module.named_children()):
        if isinstance(submodule, nn.Linear):
            module.add_module(
                name,
                LinearEfficientEnsemble(
                    in_features=submodule.in_features,
                    out_features=submodule.out_features,
                    bias=submodule.bias is not None,
                    **kwargs,
                ),
            )
        else:
            make_efficient_ensemble(submodule, **kwargs)


def _get_first_ensemble_layer(backbone: MLP) -> LinearEfficientEnsemble:
    if isinstance(backbone, MLP):
        return backbone.blocks[0][0]  # type: ignore[code]
    else:
        raise RuntimeError(f'Unsupported backbone: {backbone}')


@torch.inference_mode()
def _init_first_adapter(
    weight: Tensor,
    distribution: Literal['normal', 'random-signs'],
    init_sections: list[int],
) -> None:
    """Initialize the first adapter.

    NOTE
    The `init_sections` argument is a historical artifact that accidentally leaked
    from irrelevant experiments to the final models. Perhaps, the code related
    to `init_sections` can be simply removed, but this was not tested.
    """
    assert weight.ndim == 2
    assert weight.shape[1] == sum(init_sections)

    if distribution == 'normal':
        init_fn_ = nn.init.normal_
    elif distribution == 'random-signs':
        init_fn_ = init_random_signs_
    else:
        raise ValueError(f'Unknown distribution: {distribution}')

    section_bounds = [0, *torch.tensor(init_sections).cumsum(0).tolist()]
    for i in range(len(init_sections)):
        # NOTE
        # As noted above, this section-based initialization is an arbitrary historical
        # artifact. Consider the first adapter of one ensemble member.
        # This adapter vector is implicitly split into "sections",
        # where one section corresponds to one feature. The code below ensures that
        # the adapter weights in one section are initialized with the same random value
        # from the given distribution.
        w = torch.empty((len(weight), 1), dtype=weight.dtype, device=weight.device)
        init_fn_(w)
        weight[:, section_bounds[i] : section_bounds[i + 1]] = w


_CUSTOM_MODULES = {
    # https://docs.python.org/3/library/stdtypes.html#definition.__name__
    CustomModule.__name__: CustomModule
    for CustomModule in [
        rtdl_num_embeddings.LinearEmbeddings,
        rtdl_num_embeddings.LinearReLUEmbeddings,
        rtdl_num_embeddings.PeriodicEmbeddings,
        rtdl_num_embeddings.PiecewiseLinearEmbeddings,
        MLP,
    ]
}


def make_module(type: str, *args, **kwargs) -> nn.Module:
    Module = getattr(nn, type, None)
    if Module is None:
        Module = _CUSTOM_MODULES[type]
    return Module(*args, **kwargs)


# ======================================================================================
# Optimization
# ======================================================================================
def default_zero_weight_decay_condition(
    module_name: str, module: nn.Module, parameter_name: str, parameter: nn.Parameter
):
    from rtdl_num_embeddings import _Periodic

    del module_name, parameter
    return parameter_name.endswith('bias') or isinstance(
        module,
        nn.BatchNorm1d
        or nn.LayerNorm
        or nn.InstanceNorm1d
        or rtdl_num_embeddings.LinearEmbeddings
        or rtdl_num_embeddings.LinearReLUEmbeddings
        or _Periodic,
    )


def make_parameter_groups(
    module: nn.Module,
    zero_weight_decay_condition=default_zero_weight_decay_condition,
    custom_groups: Union[None, list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    if custom_groups is None:
        custom_groups = []
    custom_params = frozenset(
        itertools.chain.from_iterable(group['params'] for group in custom_groups)
    )
    assert len(custom_params) == sum(
        len(group['params']) for group in custom_groups
    ), 'Parameters in custom_groups must not intersect'
    zero_wd_params = frozenset(
        p
        for mn, m in module.named_modules()
        for pn, p in m.named_parameters()
        if p not in custom_params and zero_weight_decay_condition(mn, m, pn, p)
    )
    default_group = {
        'params': [
            p
            for p in module.parameters()
            if p not in custom_params and p not in zero_wd_params
        ]
    }
    return [
        default_group,
        {'params': list(zero_wd_params), 'weight_decay': 0.0},
        *custom_groups,
    ]


# ======================================================================================
# The model
# ======================================================================================
class Model(nn.Module):
    """MLP & TabM."""

    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: list[int],
        n_classes: Union[None, int],
        backbone: dict,
        bins: Union[None, list[Tensor]],  # For piecewise-linear encoding/embeddings.
        num_embeddings: Union[None, dict] = None,
        arch_type: Literal[
            # Plain feed-forward network without any kind of ensembling.
            'plain',
            #
            # TabM-mini
            'tabm-mini',
            #
            # TabM-mini. The first adapter is initialized from the normal distribution.
            # This is used in Section 5.1 of the paper.
            'tabm-mini-normal',
            #
            # TabM
            'tabm',
            #
            # TabM. The first adapter is initialized from the normal distribution.
            # This variation is not used in the paper, but there is a preliminary
            # evidence that may be a better default strategy.
            'tabm-normal',
        ],
        k: Union[None, int] = None,
    ) -> None:
        # >>> Validate arguments.
        assert n_num_features >= 0
        assert n_num_features or cat_cardinalities
        if arch_type == 'plain':
            assert k is None
        else:
            assert k is not None
            assert k > 0

        super().__init__()

        # >>> Continuous (numerical) features
        first_adapter_sections = []  # See the comment in `_init_first_adapter`.

        if n_num_features == 0:
            assert bins is None
            self.num_module = None
            d_num = 0

        elif num_embeddings is None:
            assert bins is None
            self.num_module = None
            d_num = n_num_features
            first_adapter_sections.extend(1 for _ in range(n_num_features))

        else:
            if bins is None:
                self.num_module = make_module(
                    **num_embeddings, n_features=n_num_features
                )
            else:
                assert num_embeddings['type'].startswith('PiecewiseLinearEmbeddings')
                self.num_module = make_module(**num_embeddings, bins=bins)
            d_num = n_num_features * num_embeddings['d_embedding']
            first_adapter_sections.extend(
                num_embeddings['d_embedding'] for _ in range(n_num_features)
            )

        # >>> Categorical features
        self.cat_module = (
            OneHotEncoding0d(cat_cardinalities) if cat_cardinalities else None
        )
        first_adapter_sections.extend(cat_cardinalities)
        d_cat = sum(cat_cardinalities)

        # >>> Backbone
        d_flat = d_num + d_cat
        self.minimal_ensemble_adapter = None
        # Any backbone can be here but we provide only MLP
        self.backbone = make_module(d_in=d_flat, **backbone)

        if arch_type != 'plain':
            assert k is not None
            first_adapter_init = (
                'normal'
                if arch_type in ('tabm-mini-normal', 'tabm-normal')
                # For other arch_types, the initialization depends
                # on the presense of num_embeddings.
                else 'random-signs'
                if num_embeddings is None
                else 'normal'
            )

            if arch_type in ('tabm-mini', 'tabm-mini-normal'):
                # Minimal ensemble
                self.minimal_ensemble_adapter = ScaleEnsemble(
                    k,
                    d_flat,
                    init='random-signs' if num_embeddings is None else 'normal',
                )
                _init_first_adapter(
                    self.minimal_ensemble_adapter.weight,  # type: ignore[code]
                    first_adapter_init,
                    first_adapter_sections,
                )

            elif arch_type in ('tabm', 'tabm-normal'):
                # Like BatchEnsemble, but all multiplicative adapters,
                # except for the very first one, are initialized with ones.
                make_efficient_ensemble(
                    self.backbone,
                    k=k,
                    ensemble_scaling_in=True,
                    ensemble_scaling_out=True,
                    ensemble_bias=True,
                    scaling_init='ones',
                )
                _init_first_adapter(
                    _get_first_ensemble_layer(self.backbone).r,  # type: ignore[code]
                    first_adapter_init,
                    first_adapter_sections,
                )

            else:
                raise ValueError(f'Unknown arch_type: {arch_type}')

        # >>> Output
        d_block = backbone['d_block']
        d_out = 1 if n_classes is None else n_classes
        self.output = (
            nn.Linear(d_block, d_out)
            if arch_type == 'plain'
            else NLinear(k, d_block, d_out)  # type: ignore[code]
        )

        # >>>
        self.arch_type = arch_type
        self.k = k

    def forward(
        self,
        x_num: Union[None, Tensor] = None,
        x_cat: Union[None, Tensor] = None
    ) -> Tensor:
        x = []
        if x_num is not None:
            x.append(x_num if self.num_module is None else self.num_module(x_num))
        if x_cat is None:
            assert self.cat_module is None
        else:
            assert self.cat_module is not None
            x.append(self.cat_module(x_cat).float())
        x = torch.column_stack([x_.flatten(1, -1) for x_ in x])

        if self.k is not None:
            x = x[:, None].expand(-1, self.k, -1)  # (B, D) -> (B, K, D)
            if self.minimal_ensemble_adapter is not None:
                x = self.minimal_ensemble_adapter(x)
        else:
            assert self.minimal_ensemble_adapter is None

        x = self.backbone(x)
        x = self.output(x)
        if self.k is None:
            # Adjust the output shape for plain networks to make them compatible
            # with the rest of the script (loss, metrics, predictions, ...).
            # (B, D_OUT) -> (B, 1, D_OUT)
            x = x[:, None]
        return x


# NN full training from scratch for CIBMTR Competition

class CIBMTR_Dataset(Dataset):
    def __init__(self, num_features, cat_features, efs=None, efs_time=None, y=None, predict=True):
        super().__init__()
        self.num_features = num_features
        self.cat_features = cat_features
        self.efs = efs
        self.efs_time = efs_time
        self.y = y
        self.predict = predict

    def __len__(self):
        return len(self.num_features)

    def __getitem__(self, index):
        if self.predict:
            return self.num_features[index], self.cat_features[index]
        else:
            return self.num_features[index], self.cat_features[index], self.efs[index], self.efs_time[index], self.y[index]

class Trainer():
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self.log_train = []

    def prepare_model(self, model):
        self.model = model

    def prepare_train(self, train):
        self.train_dataloader = train
        self.num_train_batches = len(self.train_dataloader)

    def fit(self, model, train):
        self.prepare_model(model)
        self.prepare_train(train)

        self.optim = model.optimizer()

        self.train_batch_idx = 0

        global SEED_FIXING

        for self.epoch in range(self.num_epochs):
            SEED_FIXING += 1
            set_seed(SEED_FIXING)

            self.fit_epoch()

    def fit_epoch(self):
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.loss(self.model(batch[0].to(DEVICE), batch[1].to(DEVICE)),
                                   batch[2].to(DEVICE), batch[3].to(DEVICE), batch[4].to(DEVICE))
            self.log_train.append(loss.item())

            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                self.optim.step()

            self.train_batch_idx += 1

    def predict(self, data, name):
        self.predict_dataloader = data
        batches_pred = []

        self.model.eval()
        with torch.no_grad():
            if name == 'Reg1':
                for batch in self.predict_dataloader:
                    pred = self.model(batch[0].to(DEVICE), batch[1].to(DEVICE)).mean(dim=1)
                    batches_pred.append(pred)
            if name == 'Class':
                for batch in self.predict_dataloader:
                    pred = self.model(batch[0].to(DEVICE), batch[1].to(DEVICE))
                    batches_pred.append(pred)
            preds = torch.cat(batches_pred, dim=0)
            preds = preds.detach().cpu().numpy()
        return preds

class NN_Reg1(nn.Module):
    def __init__(self, model, loss_type, lr):
        super().__init__()
        self.model = model
        self.loss_type = loss_type
        self.lr = lr

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_nums, x_cats):
        x = self.model(x_nums, x_cats).squeeze(-1)
        return self.sigmoid(2*x)

    def loss(self, pred, efs, efs_time, y):
        loss_function_1 = nn.BCELoss()
        loss_function_2 = nn.MSELoss()

        pred = pred.flatten(0, 1)
        y = y.repeat_interleave(K_R)

        if self.loss_type == 'BCE':
            loss = loss_function_1(pred, y)
        if self.loss_type == 'MSE':
            loss = loss_function_2(pred, y)
        return loss

    def optimizer(self):
        return QHAdam(make_parameter_groups(self), lr=self.lr)

class NN_Class(nn.Module):
    def __init__(self, hidden_dim_1, hidden_dim_2, hidden_dim_3, lr):
        super().__init__()
        self.lr = lr

        self.embs = nn.ModuleList([nn.Embedding(CAT_SIZE[i], CAT_EMB_SIZE_C[i]) for i in range(len(CATS))])
        self.n_embs = sum(e.embedding_dim for e in self.embs)

        self.n_nums = len(NUMS)

        self.trees = ODST(self.n_nums + self.n_embs, hidden_dim_1)

        self.bn_2 = nn.BatchNorm1d(hidden_dim_1)
        self.linear_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.act_2 = nn.SiLU()

        self.bn_3 = nn.BatchNorm1d(hidden_dim_2)
        self.linear_3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.act_3 = nn.SiLU()

        self.bn_out = nn.BatchNorm1d(hidden_dim_3)
        self.out = nn.Linear(hidden_dim_3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_nums, x_cats):
        x_embs = [e(x_cats[:, i]) for i, e in enumerate(self.embs)]
        x_embs = torch.cat(x_embs, dim=1)

        x = torch.cat([x_nums, x_embs], dim=1)

        x = self.trees(x)

        x = self.act_2(self.linear_2(self.bn_2(x)))
        x = self.act_3(self.linear_3(self.bn_3(x)))
        return self.sigmoid(self.out(self.bn_out(x))).flatten()

    def loss(self, pred, efs, efs_time, y):
        loss_function = nn.BCELoss()
        return loss_function(pred, efs)

    def optimizer(self):
        return QHAdam(self.parameters(), lr=self.lr)

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def make_target(df):
    df = df.with_columns(pl.Series(name='y', values=df['efs_time']))

    df_0 = df.filter(pl.col('efs')==0)
    df_1 = df.filter(pl.col('efs')==1)

    df_0 = df_0.with_columns(pl.Series(name='y', values=df_0['y'].rank()))
    df_0 = df_0.with_columns(pl.Series(name='y', values=df_0['y']/df_0['y'].max()))

    df_1 = df_1.with_columns(pl.Series(name='y', values=df_1['y'].rank()))
    df_1 = df_1.with_columns(pl.Series(name='y', values=df_1['y']/df_1['y'].max()))

    df = pl.concat([df_0, df_1])
    df = df.sort(by='ID')

    df = df.with_columns(y=pl.when(pl.col('efs')==0).then(COLLAPSE).otherwise(pl.col('y')))

    df = df.with_columns(df['y'].cast(pl.Float32))
    return df

def make_target_by_race(df):
    df_1 = df.filter(pl.col('race_group') == 0)
    df_1 = make_target(df_1)
    
    df_2 = df.filter(pl.col('race_group') == 1)
    df_2 = make_target(df_2)
    
    df_3 = df.filter(pl.col('race_group') == 2)
    df_3 = make_target(df_3)
    
    df_4 = df.filter(pl.col('race_group') == 3)
    df_4 = make_target(df_4)
    
    df_5 = df.filter(pl.col('race_group') == 4)
    df_5 = make_target(df_5)
    
    df_6 = df.filter(pl.col('race_group') == 5)
    df_6 = make_target(df_6)

    df = pl.concat([df_1, df_2, df_3, df_4, df_5, df_6])
    df = df.sort(by='ID')
    return df

def get_preds(train, test):
    preds = np.zeros(test.shape[0])

    train = make_target_by_race(train)

    train_0 = train.filter(pl.col('efs')==0)
    train_1 = train.filter(pl.col('efs')==1)

    global SEED_FIXING

    for repeat_ext in range(NREPEATS_EXTERNAL):
        pred_r_repeated = np.zeros(test.shape[0])
        pred_c_repeated = np.zeros(test.shape[0])
        for repeat_int in range(NREPEATS_INTERNAL):
            SEED_FIXING += 1
            set_seed(SEED_FIXING)

            TabM = Model(
                        n_num_features=len(NUMS),
                        cat_cardinalities=CAT_SIZE,
                        n_classes=None,
                        backbone={
                            'type': 'MLP',
                            'n_blocks': 3,
                            'd_block': 768,
                            'dropout': 0,
                            'activation': 'SiLU',
                        },
                        bins=rtdl_num_embeddings.compute_bins(train_1.select(NUMS).to_torch(), 48),
                        num_embeddings={
                            'type': 'PiecewiseLinearEmbeddings',
                            'd_embedding': 48,
                            'activation': False,
                            'version': 'A',
                        },
                        arch_type='tabm-mini',
                        k=K_R,
                        )

            ds_train_r = CIBMTR_Dataset(num_features=train_1.select(NUMS).to_torch(),
                                        cat_features=train_1.select(CATS).to_torch(),
                                        efs=train_1['efs'].to_torch(),
                                        efs_time=train_1['efs_time'].to_torch(),
                                        y=train_1['y'].to_torch(),
                                        predict=False)
            dl_train_r = DataLoader(ds_train_r, batch_size=BS_R, shuffle=True, drop_last=True)

            ds_predict_r = CIBMTR_Dataset(num_features=test.select(NUMS).to_torch(),
                                          cat_features=test.select(CATS).to_torch())
            dl_predict_r = DataLoader(ds_predict_r, batch_size=BS_R, shuffle=False, drop_last=False)

            model_r = NN_Reg1(model=TabM, loss_type='BCE', lr=LR_R).to(DEVICE)

            trainer_r = Trainer(num_epochs=E_R)
            trainer_r.fit(model_r, dl_train_r)

            pred_r = trainer_r.predict(dl_predict_r, 'Reg1')
            pred_r_repeated += pred_r

            SEED_FIXING += 1
            set_seed(SEED_FIXING)

            ds_train_c = CIBMTR_Dataset(num_features=train.select(NUMS).to_torch(),
                                        cat_features=train.select(CATS).to_torch(),
                                        efs=train['efs'].to_torch(),
                                        efs_time=train['efs_time'].to_torch(),
                                        y=train['y'].to_torch(),
                                        predict=False)
            dl_train_c = DataLoader(ds_train_c, batch_size=BS_C, shuffle=True, drop_last=True)

            ds_predict_c = CIBMTR_Dataset(num_features=test.select(NUMS).to_torch(),
                                          cat_features=test.select(CATS).to_torch())
            dl_predict_c = DataLoader(ds_predict_c, batch_size=BS_C, shuffle=False, drop_last=False)

            model_c = NN_Class(hidden_dim_1=HD_1_C, hidden_dim_2=HD_2_C, hidden_dim_3=HD_3_C, lr=LR_C).to(DEVICE)

            trainer_c = Trainer(num_epochs=E_C)
            trainer_c.fit(model_c, dl_train_c)

            pred_c = trainer_c.predict(dl_predict_c, 'Class')
            pred_c_repeated += pred_c

        pred_r_repeated /= NREPEATS_INTERNAL
        pred_c_repeated /= NREPEATS_INTERNAL
        pred_repeated = COLLAPSE * (1 - pred_c_repeated) + pred_r_repeated * pred_c_repeated
        preds += pred_repeated

    preds /= NREPEATS_EXTERNAL
    return preds

train = pl.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
train = train.with_columns(pl.Series(name='donor_age_is_null', values=train['donor_age'].is_null().cast(pl.Int64)))

test = pl.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
test = test.with_columns(pl.Series(name='donor_age_is_null', values=test['donor_age'].is_null().cast(pl.Int64)))

IDS = ['ID']
TARGETS = ['efs', 'efs_time']
FEATURES = []

NUMS = ['donor_age', 'age_at_hct']
FEATURES.extend(NUMS)

CATS = [feat for feat in train.columns if not (feat in IDS or feat in TARGETS or feat in NUMS)]
FEATURES.extend(CATS)

for i in TARGETS:
    if train[i].dtype.is_integer():
        train = train.with_columns(train[i].cast(pl.Int32))
    if train[i].dtype.is_float():
        train = train.with_columns(train[i].cast(pl.Float32))

train_targets = train[TARGETS]
train = train.drop(TARGETS)

train_test = pl.concat([train, test])

for i in NUMS:
    if train_test[i].dtype.is_integer():
        train_test = train_test.with_columns(train_test[i].cast(pl.Int32))
    if train_test[i].dtype.is_float():
        train_test = train_test.with_columns(train_test[i].cast(pl.Float32))

for i in CATS:
    if train_test[i].dtype.is_numeric():
        train_test = train_test.with_columns(train_test[i].cast(pl.String))

CAT_SIZE = []
CAT_EMB_SIZE_C = []

for i in NUMS:
    train_test = train_test.with_columns(pl.Series(name=i, values=(train_test[i]-train_test[i].mean())/train_test[i].std()))
    train_test = train_test.with_columns(pl.Series(name=i, values=train_test[i].fill_null(strategy='zero')))

for i in CATS:
    train_test = train_test.with_columns(pl.Series(name=i, values=pd.factorize(train_test[i])[0]))
    train_test = train_test.with_columns(pl.Series(name=i, values=train_test[i]-train_test[i].min()))
    train_test = train_test.with_columns(train_test[i].cast(pl.Int64))
    CAT_SIZE.append(train_test[i].n_unique())
    CAT_EMB_SIZE_C.append(8)

train = train_test[:len(train)]
train = train.with_columns(train_targets)

test = train_test[len(train):]

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

COLLAPSE = 1.35

NREPEATS_INTERNAL = 5
NREPEATS_EXTERNAL = 5

E_R = 30
BS_R = 512
LR_R = 0.005
K_R = 48

E_C = 4
BS_C = 256
LR_C = 0.05
HD_1_C = 512
HD_2_C = 256
HD_3_C = 128

SEED_FIXING = 0

preds = get_preds(train, test)
preds = -preds

submission = pl.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')
submission = submission.with_columns(pl.Series(name='prediction', values=preds))
submission.write_csv('submission.csv')
print(submission)

