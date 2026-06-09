# PlantXMamba/mamba_block/pscan.py
import math
import pandas as pd
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm


def npo2(len):
    """
    Returns the next power of 2 above len
    """

    return 2 ** math.ceil(math.log2(len))


def pad_npo2(X):
    """
    Pads input length dim to the next power of 2

    Args:
        X : (B, L, D, N)

    Returns:
        Y : (B, npo2(L), D, N)
    """

    len_npo2 = npo2(X.size(1))
    pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))
    return F.pad(X, pad_tuple, "constant", 0)


class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # modifies X in place by doing a parallel scan.
        # more formally, X will be populated by these values :
        # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
        # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)

        # only supports L that is a power of two (mainly for a clearer code)

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        # we have only 4, 2 or 1 nodes left
        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])

            Xa[:, :, 3].add_(
                Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1]))
            )
        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
        Aa = A[:, :, 2 ** (num_steps - 2) - 1 : L : 2 ** (num_steps - 2)]
        Xa = X[:, :, 2 ** (num_steps - 2) - 1 : L : 2 ** (num_steps - 2)]
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Aa[:, :, 2].mul_(Aa[:, :, 1])

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 2**k - 1 : L : 2**k]
            Xa = X[:, :, 2**k - 1 : L : 2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def pscan_rev(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # the same function as above, but in reverse
        # (if you flip the input, call pscan, then flip the output, you get what this function outputs)
        # it is used in the backward pass

        # only supports L that is a power of two (mainly for a clearer code)

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]

        # we have only 4, 2 or 1 nodes left
        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])

            Xa[:, :, 0].add_(
                Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2])))
            )
        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
        Aa = A[:, :, 0 : L : 2 ** (num_steps - 2)]
        Xa = X[:, :, 0 : L : 2 ** (num_steps - 2)]
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Aa[:, :, 1].mul_(Aa[:, :, 2])

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 0 : L : 2**k]
            Xa = X[:, :, 0 : L : 2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        Applies the parallel scan operation, as defined above. Returns a new tensor.
        If you can, privilege sequence lengths that are powers of two.

        Args:
            A_in : (B, L, D, N)
            X_in : (B, L, D, N)

        Returns:
            H : (B, L, D, N)
        """

        L = X_in.size(1)

        # cloning is requiered because of the in-place ops
        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            # pad tensors (and clone btw)
            A = pad_npo2(A_in)  # (B, npo2(L), D, N)
            X = pad_npo2(X_in)  # (B, npo2(L), D, N)

        # prepare tensors
        A = A.transpose(2, 1)  # (B, D, npo2(L), N)
        X = X.transpose(2, 1)  # (B, D, npo2(L), N)

        # parallel scan (modifies X in-place)
        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)

        # slice [:, :L] (cut if there was padding)
        return X.transpose(2, 1)[:, :L]

    @staticmethod
    def backward(ctx, grad_output_in):
        """
        Flows the gradient from the output to the input. Returns two new tensors.

        Args:
            ctx : A_in : (B, L, D, N), X : (B, D, L, N)
            grad_output_in : (B, L, D, N)

        Returns:
            gradA : (B, L, D, N), gradX : (B, L, D, N)
        """

        A_in, X = ctx.saved_tensors

        L = grad_output_in.size(1)

        # cloning is requiered because of the in-place ops
        if L == npo2(L):
            grad_output = grad_output_in.clone()
            # the next padding will clone A_in
        else:
            grad_output = pad_npo2(grad_output_in)  # (B, npo2(L), D, N)
            A_in = pad_npo2(A_in)  # (B, npo2(L), D, N)

        # prepare tensors
        grad_output = grad_output.transpose(2, 1)
        A_in = A_in.transpose(2, 1)  # (B, D, npo2(L), N)
        A = torch.nn.functional.pad(
            A_in[:, :, 1:], (0, 0, 0, 1)
        )  # (B, D, npo2(L), N) shift 1 to the left (see hand derivation)

        # reverse parallel scan (modifies grad_output in-place)
        PScan.pscan_rev(A, grad_output)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])

        return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]


pscan = PScan.apply


# PlantXMamba/mamba_block/backbone.py
import math
from dataclasses import dataclass
from typing import Union


"""

This file closely follows the mamba_simple.py from the official Mamba implementation, and the mamba-minimal by @johnma2006.
The major differences are :
-the convolution is done with torch.nn.Conv1d
-the selective scan is done in PyTorch

A sequential version of the selective scan is also available for comparison. Also, it is possible to use the official Mamba implementation.

This is the structure of the torch modules :
- A Mamba model is composed of several layers, which are ResidualBlock.
- A ResidualBlock is composed of a MambaBlock, a normalization, and a residual connection : ResidualBlock(x) = mamba(norm(x)) + x
- This leaves us with the MambaBlock : its input x is (B, L, D) and its outputs y is also (B, L, D) (B=batch size, L=seq len, D=model dim).
First, we expand x into (B, L, 2*ED) (where E is usually 2) and split it into x and z, each (B, L, ED).
Then, we apply the short 1d conv to x, followed by an activation function (silu), then the SSM.
We then multiply it by silu(z).
See Figure 3 of the paper (page 8) for a visual representation of a MambaBlock.

"""


@dataclass
class MambaConfig:
    d_model: int  # D
    n_layers: int
    dt_rank: Union[int, str] = "auto"
    d_state: int = 16  # N in paper/comments
    expand_factor: int = 2  # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    dropout: float = 0.1

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False  # apply layernorms to internal activations

    mup: bool = False
    mup_base_width: float = 128  # width=d_model

    pscan: bool = True  # use parallel scan mode or sequential mode when training
    use_cuda: bool = False  # use official CUDA implementation when training (not compatible with (b)float16)

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)

        # muP
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width


class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList(
            [ResidualBlock(config) for _ in range(config.n_layers)]
        )

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        for layer in self.layers:
            x = layer(x)

        return x

    def step(self, x, caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches


class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)

    def forward(self, x):
        # x : (B, L, D)

        # output : (B, L, D)

        output = self.mixer(self.norm(x)) + x
        return output

    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
        # h : (B, ED, N)
        # inputs: (B, ED, d_conv-1)

        # output : (B, D)
        # cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )

        # projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(
            config.d_inner, config.dt_rank + 2 * config.d_state, bias=False
        )

        # projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # delta bias
        dt = torch.exp(
            torch.rand(config.d_inner)
            * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt)
        )  # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(
            config.d_inner, 1
        )
        self.A_log = nn.Parameter(
            torch.log(A)
        )  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # used in jamba
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(
                self.config.dt_rank, config.rms_norm_eps, config.mup
            )
            self.B_layernorm = RMSNorm(
                self.config.d_state, config.rms_norm_eps, config.mup
            )
            self.C_layernorm = RMSNorm(
                self.config.d_state, config.rms_norm_eps, config.mup
            )
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.config.use_cuda:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                self.config.use_cuda = False

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        _, L, _ = x.shape

        xz = self.in_proj(x)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)  # (B, L, ED), (B, L, ED)

        # x branch
        x = x.transpose(1, 2)  # (B, ED, L)
        x = self.conv1d(x)[
            :, :, :L
        ]  # depthwise convolution over time, with a short filter
        x = x.transpose(1, 2)  # (B, L, ED)

        x = F.silu(x)
        y = self.ssm(x, z)

        if self.config.use_cuda:
            output = self.out_proj(y)  # (B, L, D)
            return output  # the rest of the operations are done in the ssm function (fused with the CUDA pscan)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  # (B, L, D)

        return output

    def ssm(self, x, z):
        # x : (B, L, ED)

        # y : (B, L, ED)

        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x)  # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )  # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = self.dt_proj.weight @ delta.transpose(
            1, 2
        )  # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        # here we just apply the matrix mul operation of delta = softplus(dt_proj(delta))
        # the rest will be applied later (fused if using cuda)

        # choose which selective_scan function to use, according to config
        if self.config.use_cuda:
            # these are unfortunately needed for the selective_scan_cuda function
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)

            # "softplus" + "bias" + "y * silu(z)" operations are fused
            y = self.selective_scan_cuda(
                x,
                delta,
                A,
                B,
                C,
                D,
                z=z,
                delta_softplus=True,
                delta_bias=self.dt_proj.bias.float(),
            )
            y = y.transpose(1, 2)  # (B, L, ED)

        else:
            delta = delta.transpose(1, 2)
            delta = F.softplus(delta + self.dt_proj.bias)

            if self.config.pscan:
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y

    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(
            3
        )  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        h = torch.zeros(
            x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device
        )  # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)  # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(
            3
        )  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    # -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """

    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
        # h : (B, ED, N)
        # inputs : (B, ED, d_conv-1)

        # y : (B, D)
        # cache : (h, inputs)

        h, inputs = cache

        xz = self.in_proj(x)  # (B, 2*ED)
        x, z = xz.chunk(2, dim=1)  # (B, ED), (B, ED)

        # x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[
            :, :, self.config.d_conv - 1
        ]  # (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  # (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)  # (B, ED, d_conv-1)
        cache = (h, inputs)

        return output, cache

    def ssm_step(self, x, h):
        # x : (B, ED)
        # h : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)

        A = -torch.exp(
            self.A_log.float()
        )  # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()

        deltaBC = self.x_proj(x)  # (B, dt_rank+2*N)

        delta, B, C = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )  # (B, dt_rank), (B, N), (B, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta))  # (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  # (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, ED, N)

        if h is None:
            h = torch.zeros(
                x.size(0),
                self.config.d_inner,
                self.config.d_state,
                device=deltaA.device,
            )  # (B, ED, N)

        h = deltaA * h + BX  # (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2)  # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        return y, h


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()

        self.use_mup = use_mup
        self.eps = eps

        # https://arxiv.org/abs/2404.05728, RMSNorm gains prevents muTransfer (section 4.2.3)
        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        if not self.use_mup:
            return output * self.weight
        else:
            return output



# PlantXMamba/mamba_block/head.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaHead(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.dropout(x)
        return x  # (batch_size, seq_len, d_model)


# PlantXMamba/mamba_block/model.py
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaModule(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.d_model = self.args.d_model
        self.n_layers = self.args.n_layers

        config = MambaConfig(d_model=self.d_model, n_layers=self.n_layers,
                           d_state=self.args.d_state, d_conv=self.args.d_conv,
                           expand_factor=self.args.expand,dropout=self.args.dropout)
        self.backbone = Mamba(config)
        self.head = MambaHead(d_model=self.d_model, dropout=self.args.dropout)

    def forward(self, x):
        sequence_output = self.backbone(x)  # (batch_size, seq_len, d_model)
        output = self.head(sequence_output)  # (batch_size, seq_len, d_model)
        return output


import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights
from torchvision.models import inception_v3, Inception_V3_Weights


class InceptionBlock(nn.Module):
    def __init__(self, in_channels=128):
        super(InceptionBlock, self).__init__()
        # Nhánh 1: 1x1
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        # Nhánh 2: 1x1 -> 3x1 + 1x3
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        # Nhánh 3: 1x1 -> 3x1 + 1x3 -> 3x1 + 1x3
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 96, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 192, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 192, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(192)
        )

        # Nhánh 4: MaxPool -> 1x1
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        b1 = self.branch1x1(x)
        b2 = self.branch3x3(x)
        b3 = self.branch5x5(x)
        b4 = self.branch_pool(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


import torch
import torch.nn as nn

class InceptionOnlyBackbone(nn.Module):
    def __init__(self):
        super(InceptionOnlyBackbone, self).__init__()
        # --- STEM ---
        # Nhiệm vụ: Giảm kích thước ảnh từ 224x224 xuống 56x56
        # và tăng channels từ 3 (RGB) lên 128 để khớp với InceptionBlock của bạn.
        self.stem = nn.Sequential(
            # Conv1: 224x224x3 -> 112x112x64
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # MaxPool: 112x112x64 -> 56x56x64
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Conv2: 56x56x64 -> 56x56x128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # --- INCEPTION BLOCK ---
        # Gọi lại chính class InceptionBlock mà bạn đã định nghĩa
        self.inception = InceptionBlock(in_channels=128)

    def forward(self, x):
        x = self.stem(x)       # Output: (B, 128, 56, 56)
        x = self.inception(x)  # Output: (B, 512, 56, 56)
        return x


from torchvision.models import inception_v3, Inception_V3_Weights

class InceptionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        inception.aux_logits = False  

        # Lấy feature extractor từ inception
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
        )
        # Output feature map shape: (B, 2048, 8, 8)

    def forward(self, x):
        return self.features(x)



class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size=5, emb_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.proj = nn.Linear(in_channels * patch_size * patch_size, emb_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, -1, C * self.patch_size * self.patch_size)
        return self.proj(x)  # shape: (b,num patches,emb size)


class PlantXMamba(nn.Module):
    def __init__(self, num_classes=4, patch_size=4, emb_size=16, d_state=64,d_conv=64,expand=4,n_layers=2,num_blocks=4, dropout=0.1):
        super().__init__()

        # VGG16 (2 blocks)
        # vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
        # self.vgg_block = nn.Sequential(*list(vgg.features[:10]))
        # self.vgg_block = vgg.features

        # Inception-like block → (B, 512, 56, 56)
        # self.inception = InceptionBlock(in_channels=128)
        self.inception = InceptionOnlyBackbone() #Only inception

        # Patch Embedding → (B, 121, 16)
        self.patch_embed = PatchEmbedding(in_channels=512, patch_size=patch_size, emb_size=emb_size) 
        # self.patch_embed = PatchEmbedding(in_channels=128, patch_size=patch_size, emb_size=emb_size) #Only VGG
        # self.patch_embed = PatchEmbedding(in_channels=2048, patch_size=patch_size, emb_size=emb_size) #Only Inception
        
        # Mamba blocks
        mamba_args = type('Args', (), {
            'd_model': emb_size,
            'd_state': d_state,
            'd_conv': d_conv,
            'expand': expand,
            'n_layers': n_layers,
            'dropout': dropout
        })()
        self.mamba = nn.Sequential(*[MambaModule(mamba_args) for _ in range(num_blocks)])

        self.norm = nn.LayerNorm(emb_size)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
        self.classifier = nn.Linear(emb_size, num_classes) #Linear baseline
        # self.classifier = nn.Sequential(                   #Linear + LN + GELU
        #     nn.Linear(emb_size, emb_size),
        #     nn.LayerNorm(emb_size),
        #     nn.GELU(),
        #     nn.Linear(emb_size, num_classes)
        # )

        # hidden_dim = emb_size * 4                          #MLP embedding
        # self.classifier = nn.Sequential(
        #     nn.Linear(emb_size, hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, num_classes)
        # )
    def forward(self, x):
        # x = self.vgg_block(x)  # (B, 128, 56, 56)
        x = self.inception(x)  # (B, 512, 56, 56)
        x = self.patch_embed(x)  # (B, 121, 16)
        x = self.mamba(x)  # (B, 121, 16)
        x = self.norm(x)  # (B, 121, 16)
        x = x.permute(0, 2, 1)  # (B, 16, 121)
        x = self.global_pool(x).squeeze(-1)  # (B, 16)
        return self.classifier(x)  # (B, num_classes)


from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch


# Maize
class_to_idx = {
    'Healthy': 0,
    'Northern leaf blight': 1,
    'Common rust': 2,
    'Gray leaf spot': 3
}

train_labels = []
for f in os.listdir(train_dir):
    if f.endswith(('.jpg', '.png', '.jpeg')):
        train_labels.append(int(f.split('_')[0]) - 1)
train_labels = np.array(train_labels)

class_indices = np.array([0, 1, 2, 3])
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=class_indices,
    y=train_labels
)

class_weights = class_weights / class_weights.mean()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)


#apple

pp_labels = []

for _, label in train_dataset:
    pp_labels.append(label)

pp_labels = np.array(pp_labels)
pp_classes = np.unique(pp_labels)

pp_class_weights = compute_class_weight(
    class_weight='balanced',
    classes=pp_classes,
    y=pp_labels
)

pp_class_weights = pp_class_weights / pp_class_weights.mean()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pp_class_weights = torch.tensor(
    pp_class_weights,
    dtype=torch.float
).to(device)


#rice
rice_labels = [
    train_dataset.label_map[os.path.basename(p).split('_')[0]]
    for p in train_dataset.image_paths
]

rice_labels = np.array(rice_labels)
rice_classes = np.unique(rice_labels)

rice_class_weights = compute_class_weight(
    class_weight='balanced',
    classes=rice_classes,
    y=rice_labels
)

rice_class_weights = rice_class_weights / rice_class_weights.mean()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rice_class_weights = torch.tensor(
    rice_class_weights,
    dtype=torch.float
).to(device)

print("Rice class weights:", rice_class_weights)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PlantXMamba(num_classes=5, patch_size=5, num_blocks=4).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PlantXMamba(num_classes=38, patch_size=7, num_blocks=4).to(device)
if torch.cuda.device_count() > 1:
    print(f"{torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


classes = ['Healthy', 'Northern leaf blight', 'Common rust', 'Gray leaf spot']


from torch.utils.data import Dataset
from PIL import Image
import os

class MaizeDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert("RGB")

        label_index = int(filename.split('_')[0]) - 1

        if self.transform:
            image = self.transform(image)

        return image, label_index


from torch.utils.data import random_split, DataLoader

train_dir = "/kaggle/input/maize-dataset/Maize_dataset/train_smp"
test_dir  = "/kaggle/input/maize-dataset/Maize_dataset/pred_img"

full_train_dataset = MaizeDataset(train_dir, transform=train_transform)
test_dataset       = MaizeDataset(test_dir, transform=val_transform)

train_size = int(0.8 * len(full_train_dataset))
val_size   = len(full_train_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_train_dataset, [train_size, val_size]
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)


classes = ['Rice Stackburn','Rice Leaf Smut','Rice Leaf Scald','Rice White Tip','Bacterial Leaf Streak']


from torch.utils.data import Dataset
from PIL import Image
import os

class RiceDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.label_map = {
            '1': 0,
            '2': 1,
            '4': 2,
            '8': 3,
            '25': 4
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        filename = os.path.basename(img_path)
        prefix = filename.split('_')[0]
        label = self.label_map.get(prefix, -1)

        if label == -1:
            raise ValueError(f"Unknown label prefix '{prefix}' in filename '{filename}'")

        if self.transform:
            image = self.transform(image)

        return image, label


from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

data_dir = "/kaggle/input/rice-dataset/Rice_dataset/train_smp"
test_dir = "/kaggle/input/rice-dataset/Rice_dataset/pred_img"

all_images = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
              if f.endswith(('.jpg', '.jpeg', '.png'))]
test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
              if f.endswith(('.jpg', '.jpeg', '.png'))]

train_paths, val_paths = train_test_split(all_images, test_size=0.2, random_state=42, stratify=[
    f.split('_')[0] for f in all_images])

train_dataset = RiceDataset(train_paths, transform=train_transform)
val_dataset = RiceDataset(val_paths, transform=val_transform)
test_dataset = RiceDataset(test_images, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)


classes = ['healthy', 'multiple_diseases', 'rust', 'scab']


import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os

class PlantPathologyDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.classes = ['healthy', 'multiple_diseases', 'rust', 'scab']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_id'] + ".jpg")
        image = Image.open(img_path).convert("RGB")

        label = row[self.classes].values.astype(int)
        label_index = label.argmax()

        if self.transform:
            image = self.transform(image)

        return image, label_index


from torch.utils.data import DataLoader, random_split
import torch

csv_path  = "/kaggle/input/plant-pathology-2020-fgvc7/train.csv"
image_dir = "/kaggle/input/plant-pathology-2020-fgvc7/images"

full_dataset = PlantPathologyDataset(csv_path, image_dir, transform=train_transform)

total_size = len(full_dataset)

train_size = int(0.7 * total_size)
val_size   = int(0.15 * total_size)
test_size  = total_size - train_size - val_size 

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42) 
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)


from torchvision import datasets
from torch.utils.data import DataLoader
import os

embrapa_path = "/kaggle/input/datasets/poornimasinghthakur/embrapa/XDB"

train_dataset = datasets.ImageFolder(
    root=os.path.join(embrapa_path, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(embrapa_path, "val"),
    transform=val_transform
)

test_dataset = datasets.ImageFolder(
    root=os.path.join(embrapa_path, "test"),
    transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

classes = train_dataset.classes
print("Number of classes:", len(classes))


from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import torch

plantvillage_path = "/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color"

full_dataset = datasets.ImageFolder(
    root=plantvillage_path,
    transform=train_transform
)

total_size = len(full_dataset)

train_size = int(0.7 * total_size)
val_size   = int(0.15 * total_size)
test_size  = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

classes = full_dataset.classes
print("Number of classes:", len(classes))


from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch

def train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    all_preds, all_labels = [], []

    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc


from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score, accuracy_score
)

def val(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []  

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except:
        auc = float('nan')

    kappa = cohen_kappa_score(all_labels, all_preds)

    return avg_loss, acc, precision, recall, f1, auc, kappa, all_labels, all_preds



best_f1 = 0.0
save_path = "best_rice_model_vgg_only_inception.pth"
num_epochs = 30

train_metrics = {
    "loss": [], "acc": []
}
val_metrics = {
    "loss": [], "acc": [], "precision": [], "recall": [], "f1": [], "auc": [], "kappa": []
}

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")

    train_loss, train_acc = train(model, train_loader, optimizer, loss_function, device)

    val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, val_kappa, val_labels, val_preds = val(model, val_loader, loss_function, device)

    train_metrics["loss"].append(train_loss)
    train_metrics["acc"].append(train_acc)

    val_metrics["loss"].append(val_loss)
    val_metrics["acc"].append(val_acc)
    val_metrics["precision"].append(val_prec)
    val_metrics["recall"].append(val_rec)
    val_metrics["f1"].append(val_f1)
    val_metrics["auc"].append(val_auc)
    val_metrics["kappa"].append(val_kappa)

    print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}, Kappa: {val_kappa:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), save_path)
        print(f"Saved new best model with F1: {best_f1:.4f}")

print("\nTraining complete.")


import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

# --- Vẽ Loss ---
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(train_metrics["loss"], label='Train Loss', linewidth=2)
plt.plot(val_metrics["loss"], label='Val Loss', linewidth=2)
plt.legend()
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png", dpi=300, bbox_inches='tight')  # lưu hình
plt.show()

# --- Vẽ Accuracy ---
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(train_metrics["acc"], label='Train Accuracy', linewidth=2)
plt.plot(val_metrics["acc"], label='Validation Accuracy', linewidth=2)
plt.legend()
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_plot.png", dpi=300, bbox_inches='tight')  # lưu hình
plt.show()



import csv

def show_best_val_metrics(val_metrics, csv_path="best_val_metrics.csv"):
    print("===== Best Validation Metrics =====\n")
    metrics = ["loss", "acc", "precision", "recall", "f1", "auc", "kappa"]

    best_results = []

    for metric in metrics:
        if not val_metrics[metric]:
            continue
        if metric == "loss":
            best_val_value = min(val_metrics[metric])
        else:
            best_val_value = max(val_metrics[metric])
        best_val_epoch = val_metrics[metric].index(best_val_value) + 1

        print(f"{metric.upper()}:")
        print(f"{best_val_value:.4f} at Epoch {best_val_epoch}\n")

        best_results.append({
            "Metric": metric.upper(),
            "Best Value": round(best_val_value, 4),
            "Epoch": best_val_epoch
        })

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Metric", "Best Value", "Epoch"])
        writer.writeheader()
        writer.writerows(best_results)

    print(f"Best metrics saved to '{csv_path}'")

show_best_val_metrics(val_metrics)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names, label_numbers=None,
                          title="Confusion Matrix (%)", save_path=None):

    if label_numbers is None:
        label_numbers = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=label_numbers)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    # Thiết lập kiểu hiển thị font
    sns.set_context("notebook", font_scale=1.2)
    sns.set_style("white")

    fig = plt.figure(figsize=(8, 6), dpi=600)
    ax = sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
                     xticklabels=class_names, yticklabels=class_names,
                     cbar=True, linewidths=0.5, linecolor='gray', square=True)

    plt.title(title, fontsize=16)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.show()



classes = ['Rice Stackburn','Rice Leaf Smut','Rice Leaf Scald','Rice White Tip','Bacterial Leaf Streak']
# classes = ['Healthy', 'Northern leaf blight', 'Common rust', 'Gray leaf spot']
label_numbers = [0, 1, 2, 3, 4]

plot_confusion_matrix(val_labels, val_preds, class_names=classes, label_numbers=label_numbers, save_path="confusion_matrix.png")


from PIL import Image
import torch
import torchvision.transforms as transforms

def predict_image(image_path, model, class_names, device='cuda' if torch.cuda.is_available() else 'cpu'):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),      # Hoặc kích thước bạn đã dùng khi train
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # Mean/Std chuẩn ImageNet nếu dùng pretrained
                             [0.229, 0.224, 0.225])
    ])
    
    # Mở ảnh và chuyển sang tensor
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # Thêm batch dim

    # Model eval và dự đoán
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_class = torch.max(outputs, 1)
        predicted_class = predicted_class.item()

    predicted_label = class_names[predicted_class]
    return predicted_label



class_names = ['Healthy', 'Northern leaf blight', 'Common rust', 'Gray leaf spot']
# class_names = ['Rice Stackburn','Rice Leaf Smut','Rice Leaf Scald','Rice White Tip','Bacterial Leaf Streak']
model.load_state_dict(torch.load("/kaggle/input/models/phanlhuy/plantxmambamodelpath/pytorch/default/1/best_maize_model_vgg.pth"))
model = model.to(device)


image_path = "/kaggle/input/maize-dataset/Maize_dataset/pred_img/2_21.jpg"
result = predict_image(image_path, model, class_names, device=device)
print("Predicted label:", result)


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh gốc
img = cv2.imread("d7ff9f3f-7e3b-4a5a-88dc-95a43336a30e.png")

# Tọa độ thủ công xác định vùng plot (x, y, width, height)
# Bạn có thể tinh chỉnh các giá trị này nếu cần
x, y, w, h = 105, 60, 615, 510  # khớp viền đen vùng biểu đồ

plot_area = img[y:y+h, x:x+w].copy()

# Các tick giả định theo ảnh gốc
x_ticks = np.arange(0, 31, 5)
y_ticks = np.linspace(0.2, 1.4, 7)

# Vẽ grid dọc (Epoch)
for tick in x_ticks:
    x_pos = int((tick / 30) * w)
    cv2.line(plot_area, (x_pos, 0), (x_pos, h), (200, 200, 200), 1)

# Vẽ grid ngang (Loss)
for tick in y_ticks:
    y_pos = int((1 - (tick - 0.2) / (1.4 - 0.2)) * h)
    cv2.line(plot_area, (0, y_pos), (w, y_pos), (200, 200, 200), 1)

# Dán lại vào ảnh gốc
img[y:y+h, x:x+w] = plot_area

# Hiển thị
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Plot with Grid")
plt.show()



import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PlantXMamba(num_classes=5, patch_size=5, num_blocks=4).to(device)
model.load_state_dict(torch.load("/kaggle/working/best_rice_model_vgg_only_inception.pth", map_location=device))
model = model.to(device)
model.eval()


def evaluate_model(model, dataloader, device, save_path="evaluation_results.csv"):
    model.eval()
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_labels, all_preds)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Kappa    : {kappa:.4f}")
    
    results_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Kappa"],
        "Value": [acc, precision, recall, f1, kappa]
    })

    results_df.to_csv(save_path, index=False)

    print(f"\nSaved evaluation results to: {save_path}")

    return acc, precision, recall, f1, kappa


evaluate_model(model, test_loader, device, save_path="/kaggle/working/rice_only_inception_results.csv")


!pip install torchinfo thop


import torch
import torch.nn as nn
from torchinfo import summary
from thop import profile


model = PlantXMamba()
model.eval()


total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total params: {total_params/1e6:.2f} M")
print(f"Trainable params: {trainable_params/1e6:.2f} M")


input_tensor = torch.randn(1, 3, 224, 224)
flops, params = profile(model, inputs=(input_tensor,))
print(f"GFLOPs: {flops/1e9:.2f}")


param_size = sum(p.numel() * p.element_size() for p in model.parameters())
buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

size_mb = (param_size + buffer_size) / 1024**2
print(f"Model size: {size_mb:.2f} MB")

