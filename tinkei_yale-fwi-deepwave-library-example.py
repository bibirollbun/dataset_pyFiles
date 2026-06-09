!pip install deepwave pytorch_msssim -q


import os
import random
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Callable, Generator

import deepwave
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepwave import scalar
from deepwave.wavelets import ricker
from IPython.display import HTML
from matplotlib import gridspec
import matplotlib.animation as animation
from matplotlib.legend_handler import HandlerTuple
from pytorch_msssim import ssim
from scipy.ndimage import gaussian_filter
from scipy.fft import fft, fftfreq, fftshift
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm.notebook import tqdm


path_data = Path("/kaggle/input/waveform-inversion")
path_train = path_data / "train_samples"
path_test = path_data / "test"


class FamilyName(str, Enum):
    CurveFault_A = "CurveFault_A"
    CurveFault_B = "CurveFault_B"
    CurveVel_A   = "CurveVel_A"
    CurveVel_B   = "CurveVel_B"
    FlatFault_A  = "FlatFault_A"
    FlatFault_B  = "FlatFault_B"
    FlatVel_A    = "FlatVel_A"
    FlatVel_B    = "FlatVel_B"
    Style_A      = "Style_A"
    Style_B      = "Style_B"


def iter_sample_pairs(
    base_path: Path, family: FamilyName, random_sample: bool = False
) -> Generator[tuple[Path, Path], None, None]:
    """
    Yield (seismic, velocity) .npy file pairs from a given family.

    Parameters
    ----------
    base_path : Path
        Root path containing all family directories.
    family : FamilyName
        The dataset family to sample from.
    random_sample : bool, optional
        Whether to yield pairs in random order (default is sequential).

    Yields
    ------
    seismic_path : Path
        Path to seismic data .npy file.
    model_path : Path
        Path to velocity model .npy file.
    """
    family_data_path = base_path / family.value

    # Fault family of data has different directory structures, which requires different handling.
    if "fault" in family_name.name.lower():
        yield from _iter_fault_pairs(family_data_path, random_sample)
    else:
        yield from _iter_nonfault_pairs(family_data_path, random_sample)


def _iter_fault_pairs(family_data_path: Path, random_sample: bool) -> Generator[tuple[Path, Path], None, None]:
    seis_files = sorted(f for f in family_data_path.glob("seis*.npy"))
    vel_files = sorted(f for f in family_data_path.glob("vel*.npy"))

    if not seis_files or not vel_files:
        warnings.warn(f"No seis/vel .npy files found in {family_data_path}!")
        return

    pairs = []
    for s_file in seis_files:
        expected_v = s_file.name.replace("seis", "vel", 1)
        v_file = family_data_path / expected_v
        if v_file.exists():
            pairs.append((s_file, v_file))

    if not pairs:
        warnings.warn(f"No directly matched seis/vel pairs in {family_data_path}!")
        return

    if random_sample:
        while True:
            yield random.choice(pairs)
    else:
        for pair in pairs:
            yield pair


def _iter_nonfault_pairs(family_data_path: Path, random_sample: bool) -> Generator[tuple[Path, Path], None, None]:
    seis_path = family_data_path / "data"
    vel_path = family_data_path / "model"

    seis_files = sorted(f for f in seis_path.glob("*.npy"))
    vel_files = sorted(f for f in vel_path.glob("*.npy"))

    if not seis_files or not vel_files:
        warnings.warn(f"No data/model files found in {family_data_path}!")
        return

    pairs = []
    for s_file in seis_files:
        expected_v = s_file.name.replace("data", "model", 1)
        v_file = vel_path / expected_v
        if v_file.exists():
            pairs.append((s_file, v_file))

    if not pairs:
        warnings.warn(f"No matched data/model pairs in {family_data_path}!")
        return

    if random_sample:
        while True:
            yield random.choice(pairs)
    else:
        for pair in pairs:
            yield pair



def total_variation_loss(v: torch.Tensor) -> torch.Tensor:
    """
    Compute isotropic total variation (TV) loss to encourage piecewise smoothness and discourage noise/artifacts.

    $$ \mathcal{L}_{\text{TV}} = \sum_{i,j} \sqrt{(\partial_x v_{i,j})^2 + (\partial_y v_{i,j})^2} $$

    Parameters
    ----------
    v : torch.Tensor
        Velocity field tensor of shape (B, C, H, W). C is often == 1.

    Returns
    -------
    torch.Tensor
        Scalar TV loss.
    """
    dx = v[:, :, 1:, :] - v[:, :, :-1, :]
    dy = v[:, :, :, 1:] - v[:, :, :, :-1]
    return torch.mean(torch.sqrt(dx**2 + 1e-8)) + torch.mean(torch.sqrt(dy**2 + 1e-8))


def laplacian_loss(v: torch.Tensor) -> torch.Tensor:
    """
    Compute Laplacian loss to penalize curvature and promote smooth gradients.

    This discourages sharp, unrealistic jumps while preserving structure better than TV.

    $$ \mathcal{L}_{\text{Laplacian}} = \|\Delta v\|_1 \quad \text{or} \quad \|\nabla^2 v\|_2^2 $$

    Parameters
    ----------
    v : torch.Tensor
        Velocity field tensor of shape (B, C, H, W). C is often == 1.

    Returns
    -------
    torch.Tensor
        Scalar Laplacian loss.
    """
    kernel = (
        torch.tensor(
            [
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0],
            ],
            dtype=v.dtype,
            device=v.device,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    laplace = F.conv2d(v, kernel, padding=1)
    return torch.mean(torch.abs(laplace))


def monotonic_depth_loss(v: torch.Tensor) -> torch.Tensor:
    """
    Penalize decreases in velocity with depth to enforce monotonicity along vertical axis.

    Velocity typically increases with depth. Encourage ∂v/∂y ≥ 0.

    Parameters
    ----------
    v : torch.Tensor
        Velocity field tensor of shape (B, C, H, W). C is often == 1.

    Returns
    -------
    torch.Tensor
        Scalar monotonicity loss.
    """
    dy = v[:, :, 1:, :] - v[:, :, :-1, :]
    return torch.mean(F.relu(-dy))


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Structural Similarity Index (SSIM) loss.

    If ground truth velocity maps have consistent structure, SSIM loss can help preserve patterns.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted tensor of shape (B, C, H, W).
    target : torch.Tensor
        Ground truth tensor of same shape as pred.

    Returns
    -------
    torch.Tensor
        Scalar SSIM loss.
    """
    return 1 - ssim(pred, target, data_range=target.max() - target.min(), size_average=True)


def fourier_domain_loss(pred: torch.Tensor, target: torch.Tensor, p: int = 1) -> torch.Tensor:
    """
    Compute L_p loss between Fourier magnitudes of prediction and target.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted tensor of shape (B, 1, H, W).
    target : torch.Tensor
        Ground truth tensor of shape (B, 1, H, W).
    p : int, optional
        Norm degree (1 for L1, 2 for L2), by default 1.

    Returns
    -------
    torch.Tensor
        Scalar loss in frequency domain.
    """
    # Apply FFT2 to spatial dimensions
    pred_fft = torch.fft.fft2(pred.to(dtype=torch.float32).squeeze(1), norm="ortho")
    target_fft = torch.fft.fft2(target.to(dtype=torch.float32).squeeze(1), norm="ortho")

    # Take magnitude
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)

    loss = F.l1_loss(pred_mag, target_mag) if p == 1 else F.mse_loss(pred_mag, target_mag)
    return loss


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute L1 loss between Sobel edge magnitudes of prediction and target.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted tensor of shape (B, C, H, W).
    target : torch.Tensor
        Ground truth tensor of shape (B, C, H, W).

    Returns
    -------
    torch.Tensor
        Scalar edge loss.
    """
    B, C, H, W = pred.shape
    N = B * C
    pred = pred.view(N, 1, H, W)
    target = target.view(N, 1, H, W)

    sobel_x = (
        torch.tensor(
            [
                [1, 0, -1],
                [2, 0, -2],
                [1, 0, -1],
            ],
            dtype=pred.dtype,
            device=pred.device,
        ).view(1, 1, 3, 3)
        / 8.0
    )
    sobel_y = sobel_x.transpose(2, 3)

    pred_gx = F.conv2d(pred, sobel_x, padding=1)
    pred_gy = F.conv2d(pred, sobel_y, padding=1)

    sobel_x = sobel_x.to(target.device, dtype=target.dtype)
    sobel_y = sobel_x.transpose(2, 3)

    target_gx = F.conv2d(target, sobel_x, padding=1)
    target_gy = F.conv2d(target, sobel_y, padding=1)

    pred_grad = torch.sqrt(pred_gx**2 + pred_gy**2 + 1e-8).view(B, C, H, W)
    target_grad = torch.sqrt(target_gx**2 + target_gy**2 + 1e-8).view(B, C, H, W)

    return F.l1_loss(pred_grad, target_grad)


class LossAggregator(nn.Module):
    """
    A weighted loss aggregator for combining multiple scalar loss components.

    Methods
    -------
    register(name, fn, weight):
        Add a new named loss function with specified weight.

    forward(pred, target, return_components):
        Compute total loss and optionally return per-component losses.
    """

    def __init__(self):
        super().__init__()
        self.loss_fns: Dict[str, Callable] = {}
        self.weights: Dict[str, float] = {}

    def register(self, name: str, fn: Callable, weight: float = 1.0) -> None:
        """
        Register a named loss function and its weight.

        Parameters
        ----------
        name : str
            Unique name of the loss.
        fn : Callable
            Loss function taking (pred, target) or (pred) as arguments.
        weight : float, optional
            Weight for this loss component, by default 1.0.
        """
        self.loss_fns[name] = fn
        self.weights[name] = weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
        components_with_weight: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]] | torch.Tensor:
        """
        Compute total weighted loss from all registered loss functions.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted output tensor.
        target : torch.Tensor
            Ground truth tensor.
        return_components : bool, optional
            If True, return dictionary of individual component losses.

        Returns
        -------
        torch.Tensor or (torch.Tensor, Dict[str, float])
            Total scalar loss or (loss, component_dict).
        """
        loss_total = torch.tensor(0.0, device=pred.device)
        components = {}

        for name, fn in self.loss_fns.items():
            if name in {"monotonic", "tv", "laplacian"}:
                # Apply to pred only
                value = fn(pred)
            else:
                value = fn(pred, target)
            weighted_value = self.weights[name] * value
            loss_total += weighted_value
            if components_with_weight:
                components[name] = weighted_value.item()
            else:
                components[name] = value.item()

        return (loss_total, components) if return_components else loss_total



class WarmupPlateauCooldownLR(_LRScheduler):
    """
    Scheduler with linear warm-up, plateau, and linear cool-down learning rates.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose LR will be scheduled.
    lr_init : float
        Initial learning rate before warm-up.
    lr_peak : float
        Learning rate after warm-up.
    lr_final : float
        Learning rate after cool-down.
    steps_warmup : int
        Number of steps for linear warm-up.
    steps_plateau : int
        Number of steps to hold at lr_peak after warm-up.
    steps_cooldown : int
        Number of steps to linearly decay to lr_final.
    last_epoch : int, optional
        Index of last epoch. Default: -1.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        lr_init: float,
        lr_peak: float,
        lr_final: float,
        steps_warmup: int,
        steps_plateau: int,
        steps_cooldown: int,
        last_epoch: int = -1
    ):
        self.lr_init = lr_init
        self.lr_peak = lr_peak
        self.lr_final = lr_final
        self.steps_warmup = steps_warmup
        self.steps_plateau = steps_plateau
        self.steps_cooldown = steps_cooldown
        self.total_steps = steps_warmup + steps_plateau + steps_cooldown
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step < self.steps_warmup:
            # Linear warm-up: lr_init -> lr_peak
            scale = step / self.steps_warmup
            lr = self.lr_init + scale * (self.lr_peak - self.lr_init)
        elif step < self.steps_warmup + self.steps_plateau:
            # Hold at lr_peak
            lr = self.lr_peak
        elif step < self.total_steps:
            # Linear cooldown: lr_peak -> lr_final
            t = step - self.steps_warmup - self.steps_plateau
            scale = t / self.steps_cooldown
            lr = self.lr_peak + scale * (self.lr_final - self.lr_peak)
        else:
            # After schedule ends, hold lr_final
            lr = self.lr_final

        return [lr for _ in self.optimizer.param_groups]



def make_ricker_wavelet(freq: float, nt: float, dt: float, peak_time: float, n_sources: int) -> torch.Tensor:
    """
    Generate a Ricker wavelet tensor for all sources.

    Parameters
    ----------
    freq : float
        Ricker wavelet central frequency (Hz).
    nt : float
        Number of time steps.
    dt : float
        Temporal step size (seconds).
    peak_time : float
        Time in seconds with offset where the wavelet is at its peak.
    n_sources : int
        Number of independent sources.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n_sources, 1, nt) with repeated wavelet for all sources.
    """
    return -ricker(freq, nt, dt, peak_time).repeat(n_sources, 1).view(n_sources, 1, -1)


def make_src_locations(src_indices: list[int]) -> torch.Tensor:
    """
    Construct source coordinates at surface depth.

    Parameters
    ----------
    src_indices : list[int]
        Indices of horizontal source locations.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n_sources, 1, 2) containing (z, x) coordinates.
    """
    n_sources = len(src_indices)
    src_locations = torch.zeros(n_sources, 2, dtype=torch.float32)
    src_locations[:, 1] = torch.tensor(src_indices, dtype=torch.long)
    return src_locations.view(n_sources, 1, 2)


def make_rec_locations(n_receivers: int, n_sources: int) -> torch.Tensor:
    """
    Construct receiver coordinates for all sources.

    Parameters
    ----------
    n_receivers : int
        Number of receivers.
    n_sources : int
        Number of independent sources.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n_sources, n_receivers, 2) containing (z, x) coordinates.
    """
    rec_locations = torch.zeros(n_receivers, 2, dtype=torch.float32)
    rec_locations[:, 1] = torch.arange(n_receivers, dtype=torch.float32)
    return rec_locations.unsqueeze(0).repeat(n_sources, 1, 1)



@dataclass(frozen=True)
class OpenFWIConfig:
    """
    Configuration of physical parameters for seismic forward modeling and inversion.

    These configurations should be the same throughout our training dataset.

    Attributes
    ----------
    dx : float
        Spatial grid spacing in x-direction (meters).
    dz : float
        Spatial grid spacing in z-direction (meters).
    dt : float
        Temporal step size (seconds).
    nt : int
        Number of time steps.
    nx : int
        Number of horizontal grid points.
    nz : int
        Number of vertical grid points.
    freq : float
        Ricker wavelet central frequency (Hz).
    peak_index : int
        Time step where the wavelet is at its peak.
    unexplained_offset : int
        Offset used to determine peak time in wavelet generation relative to `peak_time` (index 76).
    src_indices : list[int]
        Indices of horizontal source locations.
    """

    dx: float = 10.0
    dz: float = 10.0
    dt: float = 0.001
    nt: int = 1000
    nx: int = 70
    nz: int = 70
    freq: float = 15.0
    peak_index: int = 76
    unexplained_offset: int = -4
    src_indices: list[int] = field(default_factory=lambda: [0, 17, 34, 52, 69])


class VelocityModel(torch.nn.Module):
    """
    Put velocity field through a sigmoid to resolve extreme velocity values, by mapping it within given bounds [v_min, v_max].

    This is how one can (potentially) include neural networks in the physics model.
    Here it is just a sigmoid function. But you can put any MLP/CNN etc here.

    Parameters
    ----------
    initial : torch.Tensor
        Initial guess for the velocity field. Must be in the range [v_min, v_max].
    v_min : float
        Minimum allowed velocity value.
    v_max : float
        Maximum allowed velocity value.
    """
    def __init__(self, initial: torch.Tensor, v_min: float, v_max: float):
        super().__init__()
        self.v_min = v_min
        self.v_max = v_max
        normalized = (initial - v_min) / (v_max - v_min)
        self.model = torch.nn.Parameter(torch.logit(normalized.clamp(1e-6, 1 - 1e-6)))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.model) * (self.v_max - self.v_min) + self.v_min


# A for-loop to demonstrate how to iterate through the entire dataset.
#  As this is just a code demo, I added `break` statements to prevent it from running.
for family_name in FamilyName:
    break  # TODO: Remove to use this loop.

    # Sequential iteration
    for i, (path_seis, path_vel) in enumerate(iter_sample_pairs(path_train, family_name, random_sample=False)):
        print(f"{i}-th sample of {family_name.name} -> Seismic: {path_seis.name}, Velocity: {path_vel.name}")
        if i == 2:
            break  # TODO: Remove to use this loop.

    # Random sampling
    gen = iter_sample_pairs(path_train, family_name, random_sample=True)
    path_seis, path_vel = next(gen)
    print(f"Random sample {family_name.name} -> Seismic: {path_seis.name}, Velocity: {path_vel.name}")

    # Batch of 500 input data per file pair!
    batch_seismic_data = np.load(path_seis)
    batch_velocity_data = np.load(path_vel)
    print(f"Shape of a batch of seismic data  : {batch_seismic_data.shape}")
    print(f"Shape of a batch of velocity data : {batch_velocity_data.shape}")


# Load random sample instead.
family_name = random.choice(list(FamilyName))
gen = iter_sample_pairs(path_train, family_name, random_sample=True)
path_seis, path_vel = next(gen)
print(f"Random sample {family_name.name} -> Seismic: {path_seis.name}, Velocity: {path_vel.name}")

batch_seismic_data = np.load(path_seis)
batch_velocity_data = np.load(path_vel)
print(f"Shape of a batch of seismic data  : {batch_seismic_data.shape}")
print(f"Shape of a batch of velocity data : {batch_velocity_data.shape}")


random_index = random.choice(range(batch_seismic_data.shape[0]))
print(f"Randomly select index {random_index}.")
sample_seis = torch.Tensor(batch_seismic_data[random_index])
sample_vel = torch.Tensor(batch_velocity_data[random_index].squeeze())
print(f"sample_seis: {sample_seis.shape}")
print(f"sample_vel: {sample_vel.shape}")


@dataclass
class TrainConfig:
    """
    Configuration for training loop.

    Attributes
    ----------
    init_with_true_vel : bool
        Whether to use a blurred version of ground truth to init the velocities. Demo only. Default: False.
    epochs : int
        Number of training epochs.
    lr_init : float
        Adam learning rate, initial.
    lr_peak : float
        Adam learning rate, peak.
        Use a crazy learning rate here in case the initial velocity guess is way off the target,
        if we use direct physics optimization..
    lr_final : float
        Adam learning rate, final.
    downscale_lr : float
        Should you choose to use the above VelocityModel, the learning rate should be way lower, like in normal ML.
    steps_warmup : int
        Steps for scheduler to warm up learning rate.
    steps_plateau : int
        Steps for scheduler to hold the learning rate constant.
    steps_cooldown : int
        Steps for scheduler to cool down learning rate to a final base rate.
    """

    init_with_true_vel: bool = False
    epochs: int = 1000
    lr_init: float = 1e-2
    lr_peak: float = 1e0
    lr_final: float = 1e-2
    downscale_lr = 1e3
    steps_warmup: int = 100
    steps_plateau: int = 500
    steps_cooldown: int = 400


def fit_one_sample(
    config_openfwi: OpenFWIConfig,
    config_train: TrainConfig,
    target_seis: torch.Tensor,
    target_vel: torch.Tensor | None = None,
    use_nn: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | torch.nn.Module, torch.Tensor, torch.Tensor, dict[str : list[float]]]:
    """
    Run full training loop for a single sample in the batch.

    Parameters
    ----------
    config_openfwi : OpenFWIConfig
        Configuration parameters for physical simulation grid.
    config_train : TrainConfig
        Configuration parameters for training loop.
    target_seis : torch.Tensor
        One sample from a batch of seismic data, shape (S, T, R).
    target_vel (optional) : torch.Tensor
        One sample from a batch of velocity models, shape (H, W). Only used for eval.
    use_nn : bool
        Whether to use the example `VelocityModel(nn.Module)` class.

    Returns
    -------
    pred_seis : torch.Tensor
        Predicted seismogram, shape (S, T, R).
    pred_vel : torch.Tensor | torch.nn.Module
        Final optimized velocity model, shape (H, W).
    init_vel : torch.Tensor
        Initial velocity model, shape (H, W), for debug.
    wavelet : torch.Tensor
        Generated wavelet, shape (S, 1, T), for debug.
    losses : dict[str: list[float]]]
        A dictionary of named losses over epochs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_seis = target_seis.to(device)
    if target_vel is not None:
        target_vel = target_vel.to(device)

    wavelet = make_ricker_wavelet(
        freq=config_openfwi.freq,
        nt=config_openfwi.nt,
        dt=config_openfwi.dt,
        peak_time=(config_openfwi.peak_index + config_openfwi.unexplained_offset) / config_openfwi.nt,
        n_sources=len(config_openfwi.src_indices),
    ).to(device)
    src_locations = make_src_locations(src_indices=config_openfwi.src_indices).to(device)
    rec_locations = make_rec_locations(n_receivers=config_openfwi.nx, n_sources=len(config_openfwi.src_indices)).to(
        device
    )

    if target_vel is not None and config_train.init_with_true_vel:
        # If `target_vel` is provided, this means we're not in test-time, and it's okay to cheat a little bit.
        #  We initialize the model with a blurred target in order to help the model converge easier.
        init_vel = torch.tensor(1 / gaussian_filter(1 / target_vel.cpu().numpy(), 5))  # .requires_grad_()
    else:
        # init_vel = torch.ones(config_openfwi.nz, config_openfwi.nx, dtype=torch.float32) * 3500
        init_vel = torch.linspace(3000, 4000, steps=config_openfwi.nz, dtype=torch.float32)
        init_vel = init_vel.unsqueeze(1).repeat(1, config_openfwi.nx)

    # You can choose to either optimize the velocity field directly as a trainable Tensor,
    #  or use a PyTorch nn.Module class.
    # Option 1) Wrap a model to help convergence. Naturally a model is trainable.
    if use_nn:
        init_vel = init_vel.requires_grad_()
        velocity_model = VelocityModel(init_vel, 1000, 8000).to(device)
        downscale_lr = config_train.downscale_lr
    # Option 2) `nn.Parameter` is still a Tensor and can be manipulated as such. But now it is trainable.
    else:
        velocity_model = nn.Parameter(init_vel.clone().detach().to(device))
        downscale_lr = 1.0

    if isinstance(velocity_model, nn.Module):
        optimizer = torch.optim.Adam(velocity_model.parameters(), lr=config_train.lr_init / downscale_lr)
    else:
        optimizer = torch.optim.Adam([velocity_model], lr=config_train.lr_init)

    scheduler = WarmupPlateauCooldownLR(
        optimizer=optimizer,
        lr_init=config_train.lr_init / downscale_lr,
        lr_peak=config_train.lr_peak / downscale_lr,
        lr_final=config_train.lr_final / downscale_lr,
        steps_warmup=config_train.steps_warmup,
        steps_plateau=config_train.steps_plateau,
        steps_cooldown=config_train.steps_cooldown,
    )

    criterion_seis = LossAggregator()
    criterion_seis.register("ssim", ssim_loss, weight=50.0)
    criterion_seis.register("l1", nn.L1Loss(), weight=10.0)
    criterion_seis.register("l2", nn.MSELoss(), weight=5.0)
    criterion_seis.register("fourier", fourier_domain_loss, weight=50.0)
    criterion_seis.register("edge", edge_loss, weight=10.0)

    criterion_vel = LossAggregator()
    criterion_vel.register("monotonic", monotonic_depth_loss, weight=0.5)
    criterion_vel.register("tv", total_variation_loss, weight=0.01)
    criterion_vel.register("laplacian", laplacian_loss, weight=0.005)

    # TODO: Upsample `wavelet` here when ground truth breaks CFL conditions.

    losses: dict[str, list[float]] = {}

    for step in (pbar := tqdm(range(config_train.epochs))):  # or desired number of iterations
        optimizer.zero_grad()

        # My own convention: model is normal, field is transposed.
        #  But this is probably me using Deepwave wrong.
        if isinstance(velocity_model, nn.Module):
            velocity_field = velocity_model()
        else:
            velocity_field = velocity_model

        # Forward modeling.
        #  It returns a tuple of 7 elements, only the last element is the relevant `receiver_amplitudes`.
        out = scalar(
            v=velocity_field,
            grid_spacing=config_openfwi.dx,  # Union[int, float, List[float], Tensor]
            dt=config_openfwi.dt,  # float
            source_amplitudes=wavelet,
            source_locations=src_locations,
            receiver_locations=rec_locations,
            # nt=nt,  # You cannot specify both the source amplitudes and `nt`.
            accuracy=8,  # Default: 4. Max: 8.
            # pml_width=20,  # Default: 20.
            pml_freq=config_openfwi.freq,
            freq_taper_frac=0.2,
            time_pad_frac=0.2,
            time_taper=True,
        )
        # Need to swap output dimensions because time step comes before spatial width in our target receiver amplitudes.
        pred_seis = out[-1]  # (5, 70, 1000)
        pred_seis = pred_seis.movedim(-2, -1)  # (5, 1000, 70)
        # wavefield_nt = out[0]
        # assert wavefield_nt.shape == (5, 110, 110)
        # wavefield_nt = wavefield_nt[..., 20:90, 20:90]

        seis_loss, seis_loss_components = criterion_seis(
            pred=pred_seis.unsqueeze(0),
            target=target_seis.unsqueeze(0),
            return_components=True,
            components_with_weight=True,
        )
        vel_loss, vel_loss_components = criterion_vel(
            pred=velocity_field.unsqueeze(0).unsqueeze(0),
            target=None,
            return_components=True,
            components_with_weight=True,
        )
        # Example scales of loss terms:
        # {'ssim': 0.03497380018234253, 'l1': 0.20045273005962372, 'fourier': 0.00616673706099391, 'edge': 0.049981553107500076}
        # {'monotonic': 1.7209473848342896, 'tv': 6.263477802276611, 'laplacian': 237.21453857421875}

        # We won't have `target_vel` during test-time.
        if target_vel is not None:
            vel_mae_loss = nn.L1Loss()(velocity_field, target_vel)
        else:
            vel_mae_loss = None

        loss = seis_loss + vel_loss
        loss.backward()

        # Clip gradients by quantile.
        if isinstance(velocity_model, nn.Module):
            all_grads = torch.cat([p.grad.detach().abs().flatten() for p in velocity_model.parameters() if p.grad is not None])
            global_clip_value = torch.quantile(all_grads, 0.98)
            torch.nn.utils.clip_grad_value_(velocity_model.parameters(), global_clip_value.item())
        else:
            torch.nn.utils.clip_grad_value_(velocity_field, torch.quantile(velocity_field.grad.detach().abs(), 0.98))

        optimizer.step()
        scheduler.step()

        pbar.set_description(f"Step {step:03d}")
        pbar.set_postfix(
            lr=f"{scheduler.get_last_lr()[0]:.1e}",
            seis_loss=f"{seis_loss.item():.6f}",
            vel_loss=f"{vel_loss.item():.6f}",
            vel_mae_loss=f"{vel_mae_loss.item():.6f}" if vel_mae_loss is not None else None,
        )

        for key, val in seis_loss_components.items():
            key = f"seis_{key}"
            if key not in losses:
                losses[key] = []
            losses[key].append(val)
        for key, val in vel_loss_components.items():
            key = f"vel_{key}"
            if key not in losses:
                losses[key] = []
            losses[key].append(val)
        if vel_mae_loss is not None:
            key = "vel_eval_mae"
            if key not in losses:
                losses[key] = []
            losses[key].append(vel_mae_loss.item())

    return (
        pred_seis.detach().cpu(),
        velocity_model.detach().cpu() if isinstance(velocity_model, torch.Tensor) else velocity_model.cpu(),
        init_vel.detach().cpu(),
        wavelet.detach().cpu(),
        losses,
    )



# Main simulation code here!
#  `sample_vel` is only used for reference (vel_mae_loss). Not used in training unless you force it in `TrainConfig`.
pred_seis, pred_vel, init_vel, wavelet, losses = fit_one_sample(
    config_openfwi=OpenFWIConfig(),
    config_train=TrainConfig(),
    target_seis=sample_seis,
    target_vel=sample_vel,
    use_nn=False,  # Set `False` to demonstrate direct physics optimization.
)


def plot_named_losses(losses: dict[str, list[float]], skip_first: int = 0) -> plt.Figure:
    """
    Plot named loss curves over epochs.

    Parameters
    ----------
    losses : dict of str to list of float
        A dictionary where each key is the name of a loss component
        and each value is a list of loss values per epoch.
    skip_first : int
        Number of initial epochs to skip.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the loss plots.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    for idx, (name, values) in enumerate(losses.items()):
        epochs = range(1, len(values) + 1)
        if name[:9] == "vel_eval_":
            ls = "-."
            values = [v / 100.0 for v in values]
            name = f"{name} / 100"
        elif name[:4] == "vel_":
            ls = "--"
        else:
            ls = "-"
        ax.plot(epochs[skip_first:], values[skip_first:], label=name, ls=ls, lw=3)

    ax.set_title("Loss Curves Over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    return fig


fig = plot_named_losses(losses, skip_first=200)
fig.show()


def compare_seismogram_traces(
    pred_seis: torch.Tensor,
    true_seis: torch.Tensor,
    family_name: str,
    source_idx_to_plot: int = 0,
    receiver_idx_to_plot: int = 0,
    num_traces_to_plot: int = 20,
    offset_trace: float = 10.0,
) -> plt.Figure:
    """
    Plot predicted and ground truth seismogram traces for a single source across multiple receivers.

    Parameters
    ----------
    pred_seis : torch.Tensor
        Predicted seismogram tensor of shape (num_sources, time_steps, num_receivers).
    true_seis : torch.Tensor
        Ground truth seismogram tensor with the same shape as pred_seis.
    family_name : str
        Identifier for the seismic shot family, used in the plot title.
    source_idx_to_plot : int, optional
        Index of the source to visualize (default is 0, max is 4).
    receiver_idx_to_plot : int, optional
        Starting index of receiver traces to visualize (default is 0).
    num_traces_to_plot : int, optional
        Number of receiver traces to visualize (default is 20).
    offset_trace : float, optional
        Vertical offset between traces for visual separation (default is 10.0).

    Returns
    -------
    matplotlib.figure.Figure
        The Matplotlib Figure object containing the seismogram plot.
    """

    num_sources, time_steps, num_receivers = pred_seis.shape

    assert source_idx_to_plot < num_sources, "Invalid source index"
    assert receiver_idx_to_plot + num_traces_to_plot <= num_receivers, "Receiver range exceeds bounds"

    fig, ax = plt.subplots(figsize=(16, max(4, num_traces_to_plot / 2)), constrained_layout=True)
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    legend_handles = []
    legend_labels = []

    for i in range(num_traces_to_plot):
        color = colors[i % len(colors)]

        trace_pred = pred_seis[source_idx_to_plot, :, receiver_idx_to_plot + i] - i * offset_trace
        trace_true = true_seis[source_idx_to_plot, :, receiver_idx_to_plot + i] - i * offset_trace
        line_pred = ax.plot(trace_pred, color=color, alpha=0.5)
        line_true = ax.plot(trace_true, color=color, alpha=0.7, ls="--")

        proxy_pred = mlines.Line2D([], [], color=color, alpha=0.5)
        proxy_true = mlines.Line2D([], [], color=color, alpha=0.7, linestyle="--")
        legend_handles.append((proxy_pred, proxy_true))
        legend_labels.append(f"Receiver {receiver_idx_to_plot + i:02d} (Pred vs GT)")

    ax.set_title(f"Seismogram from {family_name}: Source #{source_idx_to_plot}")
    ax.set_xlabel("Time Step (ms)")
    ax.set_ylabel("Amplitude (Offset)")
    ax.set_yticks([])
    ax.grid(True)
    ax.legend(
        legend_handles,
        legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        handlelength=4,
        loc="upper right",
    )
    return fig


fig = compare_seismogram_traces(
    pred_seis=pred_seis.detach().cpu(),
    true_seis=sample_seis.detach().cpu(),
    family_name=family_name.name,
    source_idx_to_plot=0,
    receiver_idx_to_plot=0,
    num_traces_to_plot=20,
    offset_trace=10.0,
)


def compare_seismograms(
    pred_seis: torch.Tensor,
    true_seis: torch.Tensor,
) -> plt.Figure:
    """
    Plot predicted vs ground truth seismograms for 5 shots, including absolute difference.

    Parameters
    ----------
    pred_seis : torch.Tensor
        Predicted seismogram tensor of shape (5, 1000, 70), where 5 is the number of shots,
        1000 is the number of time steps, and 70 is the number of receivers.
    true_seis : torch.Tensor
        Ground truth seismogram tensor with the same shape as pred_seis.

    Returns
    -------
    matplotlib.figure.Figure
        The Matplotlib Figure object with all subplots.
    """
    data_dict = {
        "Prediction": pred_seis,
        "Ground Truth": true_seis,
        "Abs Diff": torch.abs(pred_seis - true_seis),
    }
    n_shots, n_timesteps, n_receivers = next(iter(data_dict.values())).shape
    vmin = min(arr.min() for arr in data_dict.values())
    vmax = max(arr.max() for arr in data_dict.values())
    cmap = "seismic"

    fig = plt.figure(figsize=(24, 5 * len(data_dict)), constrained_layout=True)
    gs = gridspec.GridSpec(
        len(data_dict),
        n_shots + 1,
        figure=fig,
        width_ratios=[1] * n_shots + [0.05],
        height_ratios=[1] * len(data_dict),
        hspace=0.0,
        wspace=0.1,
    )

    for row_idx, (label, data) in enumerate(data_dict.items()):
        for col_idx in range(n_shots):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            im = ax.imshow(data[col_idx, :, :], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            if row_idx == 0:
                ax.set_title(f"Source #{col_idx + 1}")
            ax.set_xlabel("70x Receivers")
            if col_idx == 0:
                ax.set_ylabel(f"{label}\n1000x Time Steps")
            else:
                ax.set_yticks([])

    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Amplitude", fontsize=12)

    fig.suptitle(
        "Ground Truth vs Prediction of 5 Shots (observed from 70 receivers over 1000 time steps)", fontsize=18, y=1.02
    )

    return fig


fig = compare_seismograms(pred_seis=pred_seis.detach().cpu(), true_seis=sample_seis.detach().cpu())
fig.show()


def compare_velocity_models(
    init_vel: torch.Tensor,
    pred_vel: torch.Tensor,
    true_vel: torch.Tensor,
    family_name: str,
) -> plt.Figure:
    """
    Compare initial guess, prediction, and ground truth velocity models side-by-side.

    Parameters
    ----------
    init_vel : torch.Tensor
        Tensor containing the initial guess velocity model, shape (1, H, W) or (H, W).
    pred_vel : torch.Tensor
        Tensor containing the predicted velocity model, shape (1, H, W) or (H, W).
    true_vel : torch.Tensor
        Tensor containing the ground truth velocity model, shape (1, H, W) or (H, W).
    family_name : str
        Identifier for the seismic shot family, used in the plot title.

    Returns
    -------
    matplotlib.figure.Figure
        The Matplotlib Figure object containing the velocity model plots.
    """
    data_dict = {
        "Initial Guess": init_vel.squeeze(),
        "Prediction": pred_vel.squeeze(),
        "Ground Truth": true_vel.squeeze(),
    }
    dx = 10  # 10 meters per index
    height, width = next(iter(data_dict.values())).shape
    x_ticks = np.linspace(0, width - 1, 6)  # 6 ticks across width
    y_ticks = np.linspace(0, height - 1, 6)  # 6 ticks down height
    x_tick_labels = [f"{i * dx:.0f}" for i in x_ticks]
    y_tick_labels = [f"{-i * dx:.0f}" for i in y_ticks]
    vmin = min(arr.min() for arr in data_dict.values())
    vmax = max(arr.max() for arr in data_dict.values())

    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.05], wspace=0.1)

    for idx, (title, data) in enumerate(data_dict.items()):
        ax = fig.add_subplot(gs[0, idx])
        im = ax.imshow(data, cmap="seismic", aspect="equal", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Width (m)")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        if idx == 0:
            ax.set_ylabel("Depth (m)")
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_tick_labels)
        else:
            ax.set_yticks([])

    cax = fig.add_subplot(gs[:, 3])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Velocity (m/s)")

    fig.suptitle(f"Compare Velocity Models ({family_name})", fontsize=18, y=1.0)

    return fig


fig = compare_velocity_models(
    init_vel=init_vel.detach().cpu(),
    pred_vel=pred_vel.detach().cpu() if isinstance(pred_vel, torch.Tensor) else pred_vel().detach().cpu(),
    true_vel=sample_vel.detach().cpu(),
    family_name=family_name.name,
)
fig.show()


def plot_wavelet(wavelet_: np.ndarray) -> plt.Figure:
    """
    Plot a Ricker wavelet in time and frequency domains.

    Parameters
    ----------
    wavelet_ : np.ndarray
        1D array representing the Ricker wavelet amplitude over time.
        Assumes sampling rate of 1000 Hz and unit duration.

    Returns
    -------
    matplotlib.figure.Figure
        The Matplotlib Figure object containing the two subplots.
    """
    sampling_rate = 1000  # Hz
    dt = 1 / sampling_rate

    # Frequency spectrum via FFT
    N = len(wavelet_)
    t = np.arange(0, N * dt, dt)
    freq = fftshift(fftfreq(N, d=dt))
    spectrum = np.abs(fftshift(fft(wavelet_)))
    peak_freq = freq[spectrum.argmax()]

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, figure=fig, width_ratios=[1, 1])

    # Time domain plot
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, wavelet_, color="C0")
    ax0.set_title(f"Ricker Wavelet (Peak @ {abs(peak_freq)} Hz)")
    ax0.set_xlim(0, 0.2)  # restrict to relevant frequencies
    ax0.set_xlabel("Time [s]")
    ax0.set_ylabel("Amplitude")
    ax0.grid(True)

    # Frequency domain plot
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(freq, spectrum, color="C1")
    ax1.set_xlim(0, 100)  # restrict to relevant frequencies
    ax1.set_title("Frequency Spectrum")
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("Magnitude")
    ax1.grid(True)

    return fig


fig = plot_wavelet(wavelet.numpy().squeeze()[0])
fig.show()


def forward_propagation(
    config_openfwi: OpenFWIConfig,
    config_train: TrainConfig,
    target_seis: torch.Tensor,
    velocity_model: torch.Tensor | torch.nn.Module,
) -> torch.Tensor:
    """
    Forward propagate a fixed velocity model to obtain the wavefield history.

    Note how there is no backprop needed here.

    Parameters
    ----------
    config_openfwi : OpenFWIConfig
        Configuration parameters for physical simulation grid.
    config_train : TrainConfig
        Configuration parameters for training loop.
    target_seis : torch.Tensor
        One sample from a batch of seismic data, shape (S, T, R).
    velocity_model : torch.Tensor | torch.nn.Module
        One sample from a batch of ground truth velocity models, shape (H, W), or a fitted one.

    Returns
    -------
    wavefields : torch.Tensor
        Predicted wavefields given velocity model and receiver observations over T, shape (T, S, P + H + P, P + W + P).
        Here P is padding from `pml_width=20`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_seis = target_seis.to(device)
    velocity_model = velocity_model.clone().detach().requires_grad_(False).to(device)

    if isinstance(velocity_model, nn.Module):
        velocity_field = velocity_model()
    else:
        velocity_field = velocity_model

    wavelet = make_ricker_wavelet(
        freq=config_openfwi.freq,
        nt=config_openfwi.nt,
        dt=config_openfwi.dt,
        peak_time=(config_openfwi.peak_index + config_openfwi.unexplained_offset) / config_openfwi.nt,
        n_sources=len(config_openfwi.src_indices),
    ).to(device)
    src_locations = make_src_locations(src_indices=config_openfwi.src_indices).to(device)
    rec_locations = make_rec_locations(n_receivers=config_openfwi.nx, n_sources=len(config_openfwi.src_indices)).to(
        device
    )

    criterion_seis = LossAggregator()
    criterion_seis.register("l1", nn.L1Loss(), weight=1.0)

    # We'll need to upsample the wavelet (source_amplitudes) because
    #  ground truth max-velocity can be > 4242m/s, which breaks CFL condition.
    cfl_dt, step_ratio = deepwave.common.cfl_condition(
        config_openfwi.dx,
        config_openfwi.dx,
        config_openfwi.dt,
        max_vel=10000,
    )
    wavelet = deepwave.common.upsample(wavelet, step_ratio)

    losses: dict[str, list[float]] = {}

    # Initial states
    wavefield_nt, wavefield_ntm1 = None, None
    psiy_ntm1, psix_ntm1, zetay_ntm1, zetax_ntm1 = None, None, None, None
    wavefields = []

    for step in (pbar := tqdm(range(config_openfwi.nt))):
        # Forward propagation with previous time-step state (enables continuation).
        #  It returns a tuple of 7 elements, only the last element is the relevant `receiver_amplitudes`.
        step_ratio = 1
        wavelet_chunk = wavelet[..., step*step_ratio:(step+1)*step_ratio]
        out = scalar(
            v=velocity_field,
            grid_spacing=config_openfwi.dx,  # Union[int, float, List[float], Tensor]
            dt=cfl_dt,  # float
            source_amplitudes=wavelet_chunk,
            source_locations=src_locations,
            receiver_locations=rec_locations,
            # nt=nt,  # You cannot specify both the source amplitudes and `nt`.
            accuracy=8,  # Default: 4. Max: 8.
            # pml_width=20,  # Default: 20.
            pml_freq=config_openfwi.freq,
            # We have examples where v=2000
            freq_taper_frac=0.2,
            time_pad_frac=0.2,
            time_taper=True,
            # Here you reinitialize the simulation step from the previous step's wavefields.
            wavefield_0=wavefield_nt,
            wavefield_m1=wavefield_ntm1,
            psiy_m1=psiy_ntm1,
            psix_m1=psix_ntm1,
            zetay_m1=zetay_ntm1,
            zetax_m1=zetax_ntm1,
        )

        # Extract receiver predictions and updated state for next loop.
        wavefield_nt, wavefield_ntm1 = out[0].detach(), out[1].detach()
        psiy_ntm1, psix_ntm1, zetay_ntm1, zetax_ntm1 = out[2].detach(), out[3].detach(), out[4].detach(), out[5].detach()
        # Need to swap output dimensions because time step comes before spatial width in our target receiver amplitudes.
        pred_seis = out[-1]  # (5, 70, 1000)
        pred_seis = pred_seis.movedim(-2, -1)  # (5, 1000, 70)
        wavefields.append(wavefield_nt.detach())

        # Compute waveform misfit and backpropagate.
        loss = criterion_seis(
            pred=pred_seis.unsqueeze(0),
            target=target_seis[..., step*step_ratio:(step+1)*step_ratio, :].unsqueeze(0),
        )

        pbar.set_description(f"Step {step:03d}")
        pbar.set_postfix(
            loss=f"{loss.item():.6f}",
        )

    return torch.stack(wavefields, dim=0)



if isinstance(pred_vel, torch.Tensor):
    pred_vel_ = pred_vel
else:
    pred_vel_ = pred_vel().detach().cpu()
print("L1 loss:", nn.L1Loss()(pred_vel_, sample_vel).item())
print("Type:", type(sample_vel), type(pred_vel_))
print("Shape:", sample_vel.shape, pred_vel_.shape)
print("Grad:", sample_vel.requires_grad, pred_vel_.requires_grad)
print("Min:", sample_vel.min().item(), pred_vel_.min().item())
print("Max:", sample_vel.max().item(), pred_vel_.max().item())
print("Std:", sample_vel.std().item(), pred_vel_.std().item())


true_wavefields = forward_propagation(
    config_openfwi=OpenFWIConfig(),
    config_train=TrainConfig(),
    target_seis=sample_seis,
    velocity_model=sample_vel,
)  # True-ish. It's still a simulation, but based on ground-truth velocity fields.
pred_wavefields = forward_propagation(
    config_openfwi=OpenFWIConfig(),
    config_train=TrainConfig(),
    target_seis=sample_seis,
    velocity_model=pred_vel if isinstance(pred_vel, torch.Tensor) else pred_vel().detach().cpu(),
)


print("L1 loss:", nn.L1Loss()(pred_wavefields, true_wavefields).item())
print("Type:", type(true_wavefields), type(pred_wavefields))
print("Shape:", true_wavefields.shape, pred_wavefields.shape)
print("Grad:", true_wavefields.requires_grad, pred_wavefields.requires_grad)
print("Min:", true_wavefields.min().item(), pred_wavefields.min().item())
print("Max:", true_wavefields.max().item(), pred_wavefields.max().item())
print("Std:", true_wavefields.std().item(), pred_wavefields.std().item())


def animate_wavefields(
    pred_wave: torch.Tensor,
    true_wave: torch.Tensor,
    fps: int = 50,
) -> animation.FuncAnimation:
    """
    Animate predicted vs ground truth wavefields over time for multiple shots.

    Parameters
    ----------
    pred_wave : torch.Tensor
        Predicted wavefields of shape (1000, 5, 110, 110).
    true_wave : torch.Tensor
        Ground truth wavefields with the same shape.
    fps : int
        Frames per second for animation.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animated figure object.
    """
    assert pred_wave.shape == true_wave.shape
    n_timesteps, n_shots, H, W = pred_wave.shape

    data_dict = {
        "Prediction": pred_wave[..., 20:90, 20:90],
        "Ground Truth": true_wave[..., 20:90, 20:90],
        "Abs Diff": torch.abs(pred_wave - true_wave)[..., 20:90, 20:90],
    }

    vmin = min(arr.min().item() for arr in data_dict.values())
    vmax = max(arr.max().item() for arr in data_dict.values())
    # cmap = "seismic" gray_r, gray, bone
    cmap = "bone"

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, constrained_layout=True)
    gs = gridspec.GridSpec(
        len(data_dict),
        n_shots + 1,
        figure=fig,
        width_ratios=[1] * n_shots + [0.05],
        height_ratios=[1] * len(data_dict),
    )

    axes = []
    ims = []

    for row_idx, (label, data) in enumerate(data_dict.items()):
        row_axes = []
        row_ims = []
        for col_idx in range(n_shots):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            im = ax.imshow(data[0, col_idx], vmin=vmin, vmax=vmax, cmap=cmap, animated=True)
            if row_idx == 0:
                ax.set_title(f"Shot #{col_idx + 1}")
            if col_idx == 0:
                ax.set_ylabel(f"{label}")
            ax.set_xticks([])
            ax.set_yticks([])
            row_axes.append(ax)
            row_ims.append(im)
        axes.append(row_axes)
        ims.append(row_ims)

    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(ims[0][0], cax=cax)
    cbar.set_label("Amplitude", fontsize=12)

    fig.suptitle("Wavefield Evolution: Prediction vs Ground Truth", fontsize=16)

    pbar = tqdm(total=n_timesteps, desc="Animating frames", leave=False, dynamic_ncols=True)

    def update(frame_idx, pbar):
        for row_idx, data in enumerate(data_dict.values()):
            for col_idx in range(n_shots):
                ims[row_idx][col_idx].set_array(data[frame_idx, col_idx])
        pbar.n = frame_idx + 1
        pbar.refresh()
        return [im for row in ims for im in row]

    anim = animation.FuncAnimation(
        fig,
        partial(update, pbar=pbar),
        frames=n_timesteps,
        interval=1000 / fps,
        blit=True,
    )

    return anim



anim = animate_wavefields(pred_wave=pred_wavefields[::50].detach().cpu(), true_wave=true_wavefields[::50].detach().cpu(), fps=50)
_ = anim.save("wavefield_animation.mp4", writer="ffmpeg", fps=10, dpi=100)


HTML("""
<video width="1920" height="1080" controls>
  <source src="wavefield_animation.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
""")




