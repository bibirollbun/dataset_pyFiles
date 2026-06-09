%pip install /kaggle/input/cibmtr-packages/autograd-1.7.0-py3-none-any.whl --quiet
%pip install /kaggle/input/cibmtr-packages/autograd-gamma-0.5.0.tar.gz --quiet
%pip install /kaggle/input/cibmtr-packages/interface_meta-1.3.0-py3-none-any.whl --quiet  # noqa: E501
%pip install /kaggle/input/cibmtr-packages/formulaic-1.1.1-py3-none-any.whl --quiet
%pip install /kaggle/input/cibmtr-packages/lifelines-0.30.0-py3-none-any.whl --quiet
%pip install /kaggle/input/cibmtr-packages/rtdl_num_embeddings-0.0.11-py3-none-any.whl --quiet  # noqa: E501
%pip install /kaggle/input/cibmtr-packages/delu-0.0.26-py3-none-any.whl --quiet




import argparse
import datetime
import inspect
import itertools
import json
import math
import shutil
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import Any, TypedDict, cast

import delu
import numpy as np
import pandas as pd
import pandas.api.types
import rtdl_num_embeddings
import sklearn.preprocessing
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from scipy.stats import rankdata
from torch import Tensor
from tqdm import tqdm

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired  # noqa: UP035

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[code]

KAGGLE = Path('/kaggle').exists()
sys.path.append('/kaggle/input/cibmtr-packages' if KAGGLE else str(Path.cwd()))

from tabm_reference import (  # noqa: E402
    LinearEfficientEnsemble,
    Model,
    NLinear,
    make_parameter_groups,
)



# fmt: off
class ParticipantVisibleError(Exception):
    pass


def score(
    solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str
) -> float:
    """
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> y_pred = {'prediction': {0: 1.0, 1: 0.0, 2: 1.0}}
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred.insert(0, row_id_column_name, range(len(y_pred)))
    >>> y_true = { 'efs': {0: 1.0, 1: 0.0, 2: 0.0}, 'efs_time': {0: 25.1234,1: 250.1234,2: 2500.1234}, 'race_group': {0: 'race_group_1', 1: 'race_group_1', 2: 'race_group_1'}}
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true.insert(0, row_id_column_name, range(len(y_true)))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    0.75
    """  # noqa: E501

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
            merged_df_race[interval_label],
            -merged_df_race[prediction_label],
            merged_df_race[event_label],
        )
        metric_list.append(c_index_race)
    return float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))
# fmt: on



KWArgs = dict[str, Any]  # Keyword arguments for any function.
JSONDict = dict[str, Any]  # Must be JSON-serializable.

PROJECT_DIR = Path.cwd().resolve()
DATA_DIR = (
    Path('/kaggle/input/equity-post-HCT-survival-predictions')
    if KAGGLE
    else PROJECT_DIR / 'data'
)
assert DATA_DIR.exists()
print(f'{KAGGLE=}\n{PROJECT_DIR=}\n{DATA_DIR=}\n')

ID_COLUMN = 'ID'
EVENT_COLUMN = 'efs'
TIME_COLUMN = 'efs_time'
GROUP_COLUMN = 'race_group'

WORST_SCORE = -999999.0




def compute_score(df: pd.DataFrame, y_pred_risk: np.ndarray) -> float:
    df_true = df[[ID_COLUMN, EVENT_COLUMN, TIME_COLUMN, GROUP_COLUMN]].copy()
    df_pred = df_true[[ID_COLUMN]].copy()
    df_pred['prediction'] = y_pred_risk
    return score(df_pred, df_true, ID_COLUMN)


def ordinal_encoding(series: pd.Series) -> np.ndarray:
    has_missing_values = series.isna().any()
    values = series.factorize()[0].astype(np.int64)
    if has_missing_values:
        # By default, Pandas replaces unknown values with -1.
        assert values.min() == -1
        values = values + 1
    return values


def format_metrics(
    metrics: dict, train_loss: None | float = None, *, precision: int = 3
) -> str:
    message = []
    for part in metrics:
        message.append(f'[{part}] {round(metrics[part]["score"], precision)}')
    if train_loss is not None:
        message.append(f'[loss] {train_loss:.5f}')
    return ' '.join(message)


def print_sep():
    print('=' * 80)


def try_get_relative_path(path: str | Path) -> Path:
    path = Path(path).resolve()
    return path.relative_to(PROJECT_DIR) if PROJECT_DIR in path.parents else path


def dump_report(output: str | Path, report: JSONDict) -> None:
    Path(output).joinpath('report.json').write_text(json.dumps(report, indent=4))


def dump_kaggle_prediction(seed: int, test_predictions: np.ndarray) -> None:
    np.save(f'submission-{seed}.npy', test_predictions)


def create_kaggle_submission(average_raw_predictions: bool) -> None:
    """Average all kaggle predictions and dump the result in the required format."""
    submissions = []
    for seed in itertools.count():
        path = Path(f'submission-{seed}.npy')
        if path.exists():
            submissions.append(np.load(path))
        else:
            break
    assert submissions

    if average_raw_predictions:
        prediction = (
            submissions[0] if len(submissions) == 1 else np.stack(submissions).mean(0)
        )
        prediction = rankdata(prediction)
    else:
        submissions = [rankdata(x) for x in submissions]
        prediction = (
            submissions[0] if len(submissions) == 1 else np.stack(submissions).mean(0)
        )

    df = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    df['prediction'] = prediction
    df.to_csv('submission.csv', index=False)




class CustomModel(Model):
    """A modified version of TabM for the competition 'CIBMTR - Equity in post-HCT Survival Predictions'."""  # noqa: E501

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        assert self.arch_type == 'tabm'
        assert self.k is not None
        assert self.k % 2 == 0

        # Step 1. Remove the original prediction head.
        del self.output

        # Step 2. Take hyperparameters from the first block of the MLP backbone.
        first_block: nn.Sequential = self.backbone.blocks[0]  # type: ignore[code]
        d_block: int = first_block[0].out_features  # type: ignore[code]
        activation_layer = deepcopy(first_block[1])
        dropout_layer = deepcopy(first_block[2])

        # Step 3. The first half of submodels predict the event probability.
        #         In other words, no changes to the original head!
        self.output_event = NLinear(self.k // 2, d_block, 1)

        # Step 4. The second half of submodels predict the time.
        #         In fact, this is just one more TabM layer!
        #         This head takes the event flag as an additional feature,
        #         hence `d_block + 1` below.
        #
        #         P.S. It could be just NLinear as well, but it works better
        #         with an additional layer.
        self.output_time = nn.Sequential(
            # LinearEfficientEnsemble represents k linear layers
            # that share most (but not all) of their parameters.
            LinearEfficientEnsemble(
                d_block + 1,
                d_block,
                k=self.k // 2,
                ensemble_scaling_in=True,
                ensemble_scaling_out=True,
                ensemble_bias=True,
                scaling_init='ones',
            ),
            activation_layer,
            dropout_layer,
            # NLinear represents N fully independent linear layers.
            NLinear(self.k // 2, d_block, 1),
        )

    def forward(self, x_num: Tensor, x_cat: Tensor, y_event: Tensor) -> Tensor:
        # >>> Step 1. NO CHANGES
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
            if self.share_training_batches or not self.training:
                # (B, D) -> (B, K, D)
                x = x[:, None].expand(-1, self.k, -1)
            else:
                # (B * K, D) -> (B, K, D)
                x = x.reshape(len(x) // self.k, self.k, *x.shape[1:])
            if self.minimal_ensemble_adapter is not None:
                x = self.minimal_ensemble_adapter(x)
        else:
            assert self.minimal_ensemble_adapter is None

        x = self.backbone(x)
        # <<<

        # Split intermediate representations in two parts.
        x_event, x_time = x.chunk(2, dim=1)

        # Predict the event.
        x_event = self.output_event(x_event)

        # Predict the time.
        assert self.k is not None
        if not self.training:
            # During evaluation, set efs==1 for all objects.
            y_event = torch.ones_like(y_event)
        x_time = torch.cat(
            # Concatenate the intermediate representations with the event indicator.
            [x_time, y_event[:, None, None].expand(-1, self.k // 2, -1)],
            dim=-1,
        )
        x_time = self.output_time(x_time)

        # Combine the event and time predictions along the submodel dimension.
        return torch.cat([x_event, x_time], dim=1)




def tabm_multiloss(
    y_pred: Tensor,  # shape: (batch_size, k)
    y_event: Tensor,  # shape: (batch_size,)
    y_time: Tensor,  # shape: (batch_size,)
) -> Tensor:
    """Optimize one TabM for two tasks."""
    n_tasks = 2  # One classification and one regression.
    k = y_pred.shape[1]  # The number of submodels.
    k_per_task = k // n_tasks

    y_pred_logits, y_pred_time = y_pred.chunk(n_tasks, dim=1)
    # y_pred_logits.shape: (batch_size, k_per_task)
    # y_pred_time.shape:   (batch_size, k_per_task)

    # Compute the classification loss.
    cls_loss = nn.functional.binary_cross_entropy_with_logits(
        y_pred_logits.flatten(), y_event.float().repeat_interleave(k_per_task)
    )
    # Compute the regression loss.
    reg_loss = nn.functional.mse_loss(
        y_pred_time.flatten(), y_time.repeat_interleave(k_per_task)
    )

    # Make the magnitudes of the two losses equal.
    # I came up with this heuristically and found to work well for this competition.
    cls_loss = cls_loss * (reg_loss.detach().abs() / cls_loss.detach().abs())

    return (cls_loss + reg_loss) / 2.0




class Config(TypedDict):
    seed: int
    folds: str | Path | list[list[list[int]]]
    # Model
    bins: NotRequired[KWArgs]
    model: KWArgs
    # Training
    optimizer: KWArgs
    gradient_clipping_norm: NotRequired[float]
    batch_size: int
    eval_batch_size: NotRequired[int]
    patience: int
    n_epochs: int


def start(
    config: Config | str | Path, output: None | str | Path, *, force: bool = False
) -> None | tuple[Config, Path]:
    """Load the config and create the output directory.

    Returns:
        (config, output) if the caller should continue the execution.
        None if the caller should immediately return.
    """
    # >>> Load the config and infer the output path.
    if isinstance(config, str | Path):
        config = Path(config).resolve()
        assert config.suffix == '.toml', 'The config must be in TOML format'
        assert config.exists(), f'The config {config} does not exist'
        assert output is None, 'When `config` is a path, `output` must be None'
        output = config.with_suffix('')
        config = cast(
            Config, tomllib.loads(Path(output).with_suffix('.toml').read_text())
        )
    else:
        assert output is not None, (
            'If config is a dictionary, then the `output` directory must be provided'
        )
        output = Path(output).resolve()

    # >>> Check the config.
    if not KAGGLE:
        # The following checks are not possible on Kaggle because
        # of the old Python version.
        presented_keys = frozenset(config)
        required_keys = Config.__required_keys__  # type: ignore[code]
        optional_keys = Config.__optional_keys__  # type: ignore[code]
        assert presented_keys >= required_keys, (
            'The config is missing the following required keys:'
            f' {", ".join(required_keys - presented_keys)}'
        )
        assert set(config) <= (required_keys | optional_keys), (
            'The config has unknown keys:'
            f' {", ".join(presented_keys - required_keys - optional_keys)}'
        )

    # >>> Start the experiment.
    print_sep()
    print(f'[>>>] {try_get_relative_path(output)} | {datetime.datetime.now()}')
    print_sep()

    if output.exists():
        if force:
            print('Removing the existing output')
            time.sleep(2.0)  # Keep the above message visible for some time.
            shutil.rmtree(output)
            output.mkdir()
            return (config, output)
        elif output.joinpath('DONE').exists():
            print('Already done!')
            return None
        else:
            print('Continuing with the existing output')
            return (config, output)
    else:
        print('Creating the output')
        output.mkdir()
        return (config, output)




def main(
    config: Config | str | Path,
    output: None | str | Path = None,
    *,
    force: bool = False,
) -> None | JSONDict:
    # >>> Start
    config_and_output = start(config, output, force=force)
    if config_and_output is None:
        return None

    config, output = config_and_output

    print()
    print('Config')
    pprint(config, sort_dicts=False, width=100)

    report = {
        'function': 'bin.kaggle_public_notebook_v2.main',
        'config': deepcopy(config),
    }

    delu.cuda.free_memory()
    delu.random.seed(config['seed'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # >>> Data
    # Load the data.
    raw_df_train = pd.read_csv(DATA_DIR / 'train.csv')
    raw_df_test = pd.read_csv(DATA_DIR / 'test.csv')
    print(f'{raw_df_train.shape=}\n{raw_df_test.shape=}\n')

    raw_train_size = len(raw_df_train)
    raw_test_size = len(raw_df_test)
    raw_df_test['efs'] = [1.0] * raw_test_size
    raw_df_test['efs_time'] = np.arange(1, raw_test_size + 1).astype(np.float64)
    raw_df = pd.concat([raw_df_train, raw_df_test])

    # Preprocess the data.
    df = raw_df.copy()
    for column, series in df.items():
        if pandas.api.types.is_numeric_dtype(series.dtype):
            is_missing = series.isna()
            if is_missing.any():
                df[f'{column}:is_missing'] = is_missing.astype(np.int64)
            df[column] = series.astype(np.float64)
        else:
            df[column] = ordinal_encoding(series)
    df['efs'] = df['efs'].astype(np.int64)

    df_train = df.iloc[:raw_train_size].copy()
    df_test = df.iloc[raw_train_size:].copy()
    print(f'{df_train.shape=}\n{df_test.shape=}\n')

    # Prepare splits.
    # NOTE: yes, the random seed is also used as a repetition index,
    # so it must be in the valid range.
    folds: list[list[int]] = (
        config['folds']
        if isinstance(config['folds'], list)
        else json.loads(Path(config['folds']).read_text())
    )[config['seed']]
    n_splits = len(folds)

    splits = []
    for split_id in range(n_splits):
        val_fold_index = split_id
        train_idx = []
        for i, fold in enumerate(folds):
            if i != val_fold_index:
                train_idx.extend(fold)
            del i, fold
        val_idx = folds[val_fold_index]
        train_idx.sort()
        val_idx.sort()
        splits.append((train_idx, val_idx))
        del split_id, val_fold_index, train_idx, val_idx

    # Run the cross-validation.
    split_predictions = []
    time_elapsed = 0.0
    for split_id, (train_idx, val_idx) in enumerate(splits):
        print(f'\n{"-" * 80}\nSplit {split_id}\n{"-" * 80}\n')

        delu.random.seed(config['seed'] + split_id)
        split_report = {}

        # Split the data.
        dfs = {
            'train': df_train.iloc[train_idx].copy(),
            'val': df_train.iloc[val_idx].copy(),
            'test': df_test.copy(),
        }
        parts = list(dfs)

        # Convert the data to NumPy.
        num_columns = []
        cat_columns = []
        for column, series in dfs['train'].items():
            if column in (ID_COLUMN, EVENT_COLUMN, TIME_COLUMN) or column.startswith(
                'y_custom:'
            ):
                continue
            elif dfs['train'][column].nunique() == 1:
                continue
            elif pandas.api.types.is_float_dtype(series.dtype):
                num_columns.append(column)
            else:
                cat_columns.append(column)
            del column, series
        data_numpy: dict[str, dict[str, np.ndarray]] = {
            'x_num': {part: dfs[part][num_columns].values.copy() for part in parts},
            'x_cat': {part: dfs[part][cat_columns].values.copy() for part in parts},
            'y_event': {part: dfs[part][EVENT_COLUMN].values.copy() for part in parts},
            'y_time': {part: dfs[part][TIME_COLUMN].values.copy() for part in parts},
            'groups': {part: dfs[part][GROUP_COLUMN].values.copy() for part in parts},
        }

        # Preprocess the features.
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(data_numpy['x_num']['train'])
        for part in parts:
            data_numpy['x_num'][part] = scaler.transform(data_numpy['x_num'][part])  # type: ignore[code]
        for part in parts:
            data_numpy['x_num'][part] = np.nan_to_num(data_numpy['x_num'][part])

        # # Preprocess the time.
        # y_time_train = data_numpy['y_time']['train']
        # event_mask = data_numpy['y_event']['train'].astype(bool)
        # data_numpy['y_time']['train'] = (
        #     sklearn.preprocessing.QuantileTransformer(
        #         n_quantiles=len(y_time_train) // 30,
        #         output_distribution='normal',
        #         subsample=1000000,
        #     )
        #     .fit(y_time_train[event_mask][:, None])
        #     .transform(y_time_train[:, None])
        #     .squeeze(-1)  # type: ignore[code]
        # )
        # # y_time for "val" and "test" are not used.
        # data_numpy['y_time']['val'] = np.zeros_like(data_numpy['y_time']['val'])
        # data_numpy['y_time']['test'] = np.zeros_like(data_numpy['y_time']['test'])
        # del y_time_train, event_mask

        # Preprocess the time.
        y_time_train = data_numpy['y_time']['train'].copy()
        event_mask = data_numpy['y_event']['train'].astype(bool)
        for mask in [event_mask, ~event_mask]:
            y_time_train[mask] = (
                sklearn.preprocessing.QuantileTransformer(
                    n_quantiles=len(y_time_train) // 30,
                    output_distribution='normal',
                    subsample=1000000,
                )
                .fit(y_time_train[mask][:, None])
                .transform(y_time_train[mask][:, None])
                .squeeze(-1)  # type: ignore[code]
            )
        data_numpy['y_time']['train'] = y_time_train
        # y_time for "val" and "test" are not used.
        data_numpy['y_time']['val'] = np.zeros_like(data_numpy['y_time']['val'])
        data_numpy['y_time']['test'] = np.zeros_like(data_numpy['y_time']['test'])
        del y_time_train, event_mask

        # Convert the data to PyTorch.
        data = {
            key: {
                part: torch.as_tensor(
                    value,
                    dtype={'float64': torch.float32, 'int64': torch.int64}[
                        str(value.dtype)
                    ],
                    device=device,
                )
                for part, value in data_numpy[key].items()
            }
            for key in data_numpy.keys()
        }
        size = {part: len(data['y_time'][part]) for part in parts}

        # >>> Model
        if 'bins' in config:
            bins = rtdl_num_embeddings.compute_bins(
                data['x_num']['train'], **config['bins']
            )
            print(f'Bin counts: {[len(x) - 1 for x in bins]}')
        else:
            bins = None
        model: CustomModel = CustomModel(
            n_num_features=data['x_num']['train'].shape[1],
            cat_cardinalities=[len(torch.unique(x)) for x in data['x_cat']['train'].T],
            n_classes=None,
            **config['model'],
            bins=bins,
        )
        model.to(device)

        def apply_model(part: str, idx: None | Tensor = None) -> Tensor:
            x_num = data['x_num'][part]
            x_cat = data['x_cat'][part]
            y_event = data['y_event'][part]
            if idx is not None:
                x_num = x_num[idx]
                x_cat = x_cat[idx]
                y_event = y_event[idx]
            return model(x_num, x_cat, y_event).squeeze(-1)

        # >>> Training
        optimizer = torch.optim.AdamW(
            params=make_parameter_groups(model), **config['optimizer']
        )
        gradient_clipping_norm = config.get('gradient_clipping_norm')

        step = 0
        batch_size = config['batch_size']
        eval_batch_size = config.get('eval_batch_size')
        epoch_size = math.ceil(size['train'] / batch_size)

        batch_generator = torch.Generator(device).manual_seed(config['seed'] + split_id)
        training_log = []
        timer = delu.tools.Timer()
        early_stopping = delu.tools.EarlyStopping(
            patience=config['patience'], mode='max'
        )
        best_checkpoint = None

        @torch.inference_mode()
        def evaluate(
            parts: list[str],
        ) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
            model.eval()

            metrics = {}
            predictions = {}
            for part in parts:
                submodel_y_pred = (
                    apply_model(part)
                    if eval_batch_size is None
                    else torch.cat(
                        [
                            apply_model(part, idx)
                            for idx in torch.arange(size[part], device=device).split(
                                eval_batch_size
                            )
                        ]
                    )
                )
                submodel_y_pred_logits, submodel_y_pred_time = submodel_y_pred.chunk(
                    2, dim=1
                )
                y_pred_probs = submodel_y_pred_logits.sigmoid().mean(1)
                y_pred_time = submodel_y_pred_time.mean(1)
                y_pred_risk = y_pred_probs * (-y_pred_time).sigmoid()

                y_pred_risk = y_pred_risk.cpu().numpy()
                if part != 'test':
                    metrics[part] = {
                        'score': WORST_SCORE
                        if np.isnan(y_pred_risk).any()
                        else compute_score(dfs[part], y_pred_risk)
                    }

                predictions[part] = y_pred_risk

            return metrics, predictions

        def make_checkpoint() -> dict[str, Any]:
            return {
                'report': split_report,
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'random_state': delu.random.get_state(),
                'batch_generator': batch_generator.get_state(),
                'training_log': training_log,
                'timer': timer,
                'early_stopping': early_stopping,
            }

        timer.run()
        while config['n_epochs'] == -1 or step // epoch_size < config['n_epochs']:
            print(f'[...] {try_get_relative_path(output)} | {timer}')

            model.train()
            epoch_losses = []
            for batch_idx in tqdm(
                torch.randperm(size['train'], device=device).split(batch_size),
                desc=f'Split {split_id} Epoch {step // epoch_size}',
            ):
                optimizer.zero_grad()
                loss = tabm_multiloss(
                    y_pred=apply_model('train', batch_idx),
                    y_event=data['y_event']['train'][batch_idx],
                    y_time=data['y_time']['train'][batch_idx],
                )
                loss.backward()
                if gradient_clipping_norm is not None:
                    nn.utils.clip_grad.clip_grad_norm_(
                        model.parameters(), gradient_clipping_norm
                    )
                optimizer.step()
                epoch_losses.append(loss.detach())
                step += 1

            epoch_losses = torch.stack(epoch_losses).tolist()
            metrics, _ = evaluate(['val'])
            training_log.append({'train-loss': epoch_losses, **metrics})

            if (
                'metrics' not in split_report
                or metrics['val']['score'] > split_report['metrics']['val']['score']
            ):
                print('ğŸŒ¸ New best epoch ğŸŒ¸')
                split_report['best_step'] = step
                split_report['metrics'] = metrics
                best_checkpoint = deepcopy(make_checkpoint())

            early_stopping.update(metrics['val']['score'])
            if early_stopping.should_stop():
                break

            print(format_metrics(metrics, statistics.mean(epoch_losses)))
            print()
        split_report['time'] = str(timer)

        assert best_checkpoint is not None
        model.load_state_dict(best_checkpoint['model'])
        metrics, predictions = evaluate(['train', 'val', 'test'])
        print('\nSplit metrics: ' + format_metrics(metrics))
        split_report['metrics'] = metrics

        split_predictions.append(predictions)
        time_elapsed += timer.elapsed()
        report.setdefault('split_reports', []).append(split_report)
        dump_report(output, report)

    report['time'] = str(datetime.timedelta(seconds=time_elapsed))

    # NOTE
    # The 'train' and 'val' predictions correspond to the same original training data:
    # - 'train' corresponds to the prediction on the training folds (train_idx).
    # - 'val' corresponds to the prediction on validation folds (val_idx).
    predictions = {
        'train': np.zeros(raw_train_size, dtype=np.float32),
        'val': np.zeros(raw_train_size, dtype=np.float32),
        'test': np.zeros(raw_test_size, dtype=np.float32),
    }
    for (train_idx, val_idx), this_split_predictions in zip(splits, split_predictions):
        predictions['train'][train_idx] += this_split_predictions['train']
        predictions['val'][val_idx] += this_split_predictions['val']
        predictions['test'] += this_split_predictions['test']
    # How many models contribute to each part:
    # - train: n_splits - 1
    # - val:   1
    # - test:  n_splits
    predictions['train'] /= n_splits - 1
    predictions['test'] /= n_splits
    np.savez(output / 'predictions.npz', **predictions)  # type: ignore[code]

    if KAGGLE:
        dump_kaggle_prediction(config['seed'], predictions['test'])

    report['metrics'] = {
        part: {
            key: statistics.mean(
                x['metrics'][part][key] for x in report['split_reports']
            )
            for key in report['split_reports'][0]['metrics']['train']
        }
        for part in ['train', 'val']
    }
    print()
    print('Summary')
    print(f'\nTime: {report["time"]}')
    print()
    for split_id, split_report in zip(range(n_splits), report['split_reports']):
        print(
            f'Split {split_id}: ' + format_metrics(split_report['metrics'], precision=4)
        )
    print('CV:      ' + format_metrics(report['metrics'], precision=4))

    dump_report(output, report)
    output.joinpath('DONE').touch()

    print()
    print_sep()
    print(f'[<<<] {try_get_relative_path(output)} | {datetime.datetime.now()}')
    print_sep()

    return report




if __name__ == '__main__':
    torch.set_num_threads(1)
    # The following settings facilitate reproducibility.
    torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[code]
    torch.backends.cudnn.allow_tf32 = False  # type: ignore[code]
    torch.backends.cudnn.benchmark = False  # type: ignore[code]
    torch.backends.cudnn.deterministic = True  # type: ignore[code]

    if KAGGLE:
        n_folds = 10
        # One repetition == one cross-validation.
        # For example, n_repetitions=1 means that the final prediction
        # for the test data is done by one ensemble of n_folds models.
        n_repetitions = 1
        for seed in range(n_repetitions):
            config: Config = {
                'seed': seed,
                'folds': f'/kaggle/input/cibmtr-folds/stratified-group-{n_folds}.json',
                #
                'gradient_clipping_norm': 1.0,
                'batch_size': 1024,
                'eval_batch_size': 8192,
                'n_epochs': -1,  # "-1" means training until the early stopping.
                'patience': 16,
                'model': {
                    'arch_type': 'tabm',
                    'k': 48,
                    'backbone': {
                        'type': 'MLP',
                        'n_blocks': 2,
                        'd_block': 864,
                        'dropout': 0.024243092233188643,
                    },
                    'num_embeddings': {
                        'type': 'PeriodicEmbeddings',
                        'n_frequencies': 40,
                        'd_embedding': 24,
                        'frequency_init_scale': 0.2,
                        'lite': False,
                    },
                },
                'optimizer': {
                    'lr': 0.0015,
                    'weight_decay': 0.0,
                },
            }
            main(config, f'output-{seed}', force=True)

        create_kaggle_submission(True)

    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('config')
        parser.add_argument('--output')
        parser.add_argument('--force', action='store_true')
        if 'continue_' in inspect.signature(main).parameters:
            parser.add_argument('--continue', action='store_true', dest='continue_')
        main(**vars(parser.parse_args(sys.argv[1:])))





