import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from typing import Callable, List, Tuple, Any, Dict, Union, Optional
import textwrap 
import inspect

class SeismicDataHelper:
    """
    Manage OpenFWI seismic-velocity pairs under a root folder.

    Datasets scanned (keys):
      CurveFault_A, CurveFault_B, CurveVel_A, CurveVel_B,
      FlatFault_A,  FlatFault_B,  FlatVel_A,  FlatVel_B,
      Style_A,      Style_B

    Quick access:
      sd = SeismicDataHelper()                      # Default uses kaggle train folder path.
      OR sd = SeismicDataHelper("train_folder_path")

      sd.explain(folder_name)                -> print a brief, descriptive overview
      sd.datasets                            -> list of dataset names
      sd.CurveFault_A                        -> list of (seis_path, vel_path)
      len(sd.CurveFault_A)                   -> number of pairs in that set
      len(sd)                                -> total pairs across all sets
      sd['FlatVel_B']                        -> same as sd.FlatVel_B

    Common methods:
      folder_name refers to the CurveFault_A or CurveFault_B etc.
      
      sd.pairs(folder_name)                 -> all (seis_path, vel_path) pairs
      sd.sample(folder_name, n)             -> n random pairs (n='all')
      sd.get(folder_name, seis_file)        -> loads paired velocity array
      sd.pipeline(folder_name, fn)          -> apply fn(seis, vel) to every pair
      sd.verify(folder_name, verbose=True)  -> sanity-check all files/arrays
      sd.plot(folder_name, file=None)       -> plot seismic+velocity (random/file)
      sd.plot_fn(folder_name, fn, ...)      -> custom fn(X, y, ...) plotting
      sd.plot_signal_time_series(...)       -> plot seismic signal vs. time
      sd.sample_pairs(folder_name, n)      -> print random sample file pairs
    """

    def __init__(self, root_dir: str = "/kaggle/input/waveform-inversion/train_samples"):
        """
        Initialize SeismicDataHelper from a root directory of datasets.

        Args:
            root_dir (str): Directory path containing datasets. Defaults to Kaggle training samples path.
        """
        self.root_dir = root_dir
        self._pairs: Dict[str, List[Tuple[str, str]]] = self._scan_pairs()

    def _scan_pairs(self) -> Dict[str, List[Tuple[str, str]]]:
        pairs: Dict[str, List[Tuple[str, str]]] = {}
        for folder_name in os.listdir(self.root_dir):
            ds_dir = os.path.join(self.root_dir, folder_name)
            if not os.path.isdir(ds_dir):
                continue

            data_dir = os.path.join(ds_dir, 'data')
            model_dir = os.path.join(ds_dir, 'model')
            if os.path.isdir(data_dir) and os.path.isdir(model_dir):
                lst = []
                for fname in os.listdir(data_dir):
                    if fname.startswith('data') and fname.endswith('.npy'):
                        suf = fname[len('data'):-4]
                        seis = os.path.join(data_dir, fname)
                        vel = os.path.join(model_dir, f'model{suf}.npy')
                        if os.path.exists(vel):
                            lst.append((seis, vel))
                if lst:
                    pairs[folder_name] = sorted(lst)
                continue

            files = os.listdir(ds_dir)
            seis_fs = [f for f in files if f.startswith('seis') and f.endswith('.npy')]
            vel_fs = [f for f in files if f.startswith('vel') and f.endswith('.npy')]
            if seis_fs and vel_fs:
                vel_set = set(vel_fs)
                lst = []
                for sf in seis_fs:
                    suf = sf[len('seis'):-4]
                    vf = f'vel{suf}.npy'
                    if vf in vel_set:
                        lst.append((os.path.join(ds_dir, sf), os.path.join(ds_dir, vf)))
                if lst:
                    pairs[folder_name] = sorted(lst)
        return pairs

    @property
    def datasets(self) -> List[str]:
        """List all available dataset folder names."""
        return list(self._pairs.keys())

    def __repr__(self) -> str:
        return f"<SeismicDataHelper datasets={self.datasets}>"

    def __getattr__(self, key: str) -> Any:
        if key in self._pairs:
            return self._pairs[key]
        raise AttributeError(f"{self.__class__.__name__!r} has no attribute {key!r}")

    def __getitem__(self, key: str) -> List[Tuple[str, str]]:
        return self._pairs[key]

    def __len__(self) -> int:
        """Total number of (seis, vel) pairs across all datasets."""
        return sum(len(v) for v in self._pairs.values())

    def pairs(self, folder_name: str) -> List[Tuple[str, str]]:
        """Return all (seis_path, vel_path) for given dataset folder."""
        return self._pairs[folder_name]

    def sample(self, folder_name: str, n: Union[int, str] = 'all') -> List[Tuple[str, str]]:
        """Return n random (seis, vel) pairs or all if n='all'."""
        allp = self.pairs(folder_name)
        if isinstance(n, str) and n.lower() == 'all':
            return allp
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive int or 'all'")
        return allp if n >= len(allp) else random.sample(allp, n)

    def sample_pairs(self, folder_name: str, n: int = 4) -> None:
        """
        Print n random (seis, vel) pairs in a simple report format.
        """
        samples = self.sample(folder_name, n)
        print(f"{n} random {folder_name} samples:")
        for s, v in samples:
            print("  ", os.path.basename(s), "â†”", os.path.basename(v))

    def get(self, folder_name: str, seis_file: str) -> np.ndarray:
        """Load the velocity array paired with a given seismic filename."""
        for s, v in self.pairs(folder_name):
            if os.path.basename(s) == seis_file:
                return np.load(v)
        raise KeyError(f"{seis_file!r} not found in {folder_name!r}")

    def get_seis(self, folder_name: str, file: str) -> np.ndarray:
        """
        Load ONLY seismic data from the given folder_name + filename.
        """
        for s, v in self.pairs(folder_name):
            if os.path.basename(s) == file:
                return np.load(s)
        raise KeyError(f"{file!r} not found in {folder_name!r}")

    def get_vel(self, folder_name: str, file: str) -> np.ndarray:
        """
        Load ONLY velocity data from the given folder_name + filename.
        """
        for s, v in self.pairs(folder_name):
            if os.path.basename(s) == file:
                return np.load(v)
        raise KeyError(f"{file!r} not found in {folder_name!r}")
        
    def pipeline(
        self,
        folder_name: str,
        callbacks: Union[Callable[[np.ndarray, np.ndarray], Any],
                         List[Callable[[np.ndarray, np.ndarray], Any]]]
    ) -> List[Any]:
        """Apply one or more callbacks to every (seis, vel) pair in dataset."""
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        out = []
        for s, v in self.pairs(folder_name):
            seis = np.load(s)
            vel = np.load(v)
            for cb in callbacks:
                out.append(cb(seis, vel))
        return out

    def verify(self, folder_name: str, verbose: bool = True) -> bool:
        """Ensure all file pairs exist, load properly, and have expected dimensions."""
        for s, v in self.pairs(folder_name):
            if not os.path.exists(s) or not os.path.exists(v):
                raise FileNotFoundError(f"Missing {s} or {v}")
            a_s, a_v = np.load(s), np.load(v)
            if verbose:
                print(f"âœ” {os.path.basename(s)} <-> {os.path.basename(v)}")
            if a_s.ndim < 3 or a_v.ndim < 2:
                raise ValueError(f"Bad shapes: {a_s.shape}, {a_v.shape}")
        return True

    def plot(self, folder_name: str, file: str = None, index: int = 0):
        """Plot a seismic-velocity pair, selected randomly or by seismic filename."""
        pairs = self.pairs(folder_name)
        if file:
            for s, v in pairs:
                if os.path.basename(s) == file:
                    seis_p, vel_p = s, v
                    break
            else:
                raise KeyError(f"{file!r} not in {folder_name!r}")
        else:
            seis_p, vel_p = random.choice(pairs)
        self._plot_seismic(seis_p, index)
        self._plot_velocity(vel_p, index)

    def plot_signal_time_series(
        self,
        folder_name: str,
        file: str = None,
        index: int = 0,
        src: int = 0,
        recv: Union[int, None] = None
    ):
        """Plot a single seismic trace vs. time for a given (src, recv) pair."""
        pairs = self.pairs(folder_name)
        if file:
            for s, _ in pairs:
                if os.path.basename(s) == file:
                    seis_p = s
                    break
            else:
                raise KeyError(f"{file!r} not in {folder_name!r}")
        else:
            seis_p, _ = random.choice(pairs)
        arr = np.load(seis_p)
        trace_batch = arr[index] if arr.ndim == 4 else arr
        nrecv = trace_batch.shape[-1]
        r = recv if recv is not None else nrecv // 2
        trace = trace_batch[src, :, r]
        fname = os.path.basename(seis_p)
        plt.figure()
        plt.plot(trace)
        plt.title(f"{fname} â€” Trace src={src}, recv={r}")
        plt.xlabel("Time (sample step)")
        plt.ylabel("Amplitude")
        plt.show()

    def plot_fn(
        self,
        folder_name: str,
        fn: Callable[..., Any],
        file: str = None,
        index: int = 0
    ):
        """
        Load one sample, then call custom plotting function fn(seis, vel, [optional file]).
        If fn takes 3 args, we pass (X, y, seismic_path). Otherwise, just (X, y).
        """
        pairs = self.pairs(folder_name)
        if file:
            for s, v in pairs:
                if os.path.basename(s) == file:
                    seis_p, vel_p = s, v
                    break
            else:
                raise KeyError(f"{file!r} not in {folder_name!r}")
        else:
            seis_p, vel_p = random.choice(pairs)

        arr_X = np.load(seis_p)
        arr_y = np.load(vel_p)
        X = arr_X[index] if arr_X.ndim == 4 else arr_X
        if arr_y.ndim == 4:
            y = arr_y[index, 0]
        elif arr_y.ndim == 3:
            y = arr_y[0] if arr_y.shape[0] == 1 else arr_y[index]
        else:
            y = arr_y

        # Inspect how many parameters the user function wants:
        params = inspect.signature(fn).parameters
        if len(params) == 3:
            fig = fn(X, y, seis_p)
        else:
            fig = fn(X, y)

        if isinstance(fig, plt.Figure):
            plt.show()

    def _plot_seismic(self, path: str, index: int):
    
        arr   = np.load(path)
        batch = arr[index] if arr.ndim == 4 else arr     # â†’ shape = (nsrc, nt, nrec)
        nsrc, nt, nrec = batch.shape
        fname = os.path.basename(path)
    
        fig, axs = plt.subplots(1, nsrc, figsize=(3*nsrc, 3))
        if nsrc == 1:
            axs = [axs]
    
        # extent = [xmin, xmax, ymin, ymax] flips Y so time=0 is at the top
        extent = [0, nrec, nt, 0]
    
        for i, ax in enumerate(axs):
            ax.imshow(
                batch[i],          # rows=time, cols=receivers
                aspect='auto',
                cmap='gray',
                extent=extent
            )
            ax.set_title(f"{fname} â€” Src {i}")
            ax.set_xlabel("Offset (m)")
            ax.set_ylabel("Time (sample step)")
    
        plt.tight_layout()
        plt.show()




    def _plot_velocity(
        self,
        path: str | Path,
        index: int = 0,
        *,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "viridis",
    ) -> None:
        """
        Plot a single 2-D slice of an openWFI P-wave velocity cube.
        Units of the colors are in meters per second (shown in the title).
    
        Parameters
        ----------
        path : str | Path
            Path to the .npy file holding the velocity array.
        index : int, default 0
            Which slice (inlines or time steps) to visualise if arr.ndim >= 3.
        vmin, vmax : float, optional
            Colour limits in m/s.  Leave None for automatic scaling.
        cmap : str, default "viridis"
            Any Matplotlib colormap name.
        """
        arr = np.load(path)
        fname = Path(path).name
    
        # pick out the 2-D slice we actually want to draw
        if arr.ndim == 4:
            img = arr[index, 0]
        elif arr.ndim == 3:
            img = arr[0] if arr.shape[0] == 1 else arr[index]
        else:
            img = arr  # already 2-D
    
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.imshow(img, aspect="equal", vmin=vmin, vmax=vmax, cmap=cmap)
    
        # Updated title to include the color units
        ax.set_title(f"{fname} â€” velocity model\n(color = velocity m/s)")
        ax.set_xlabel("Horizontal index (sample #)")
        ax.set_ylabel("Depth index (sample #)")
    
        cbar = fig.colorbar(cax, ax=ax, orientation="vertical", fraction=0.04, pad=0.03)
        # we can leave the cbar ticks unlabeled since units are in the title
        cbar.ax.xaxis.set_ticks_position("none")
    
        fig.tight_layout(pad=3.0)
        plt.show()



    def _wrap_preserving_newlines(self, text, width=65):
        lines = text.splitlines()
        wrapped = [
            textwrap.fill(line, width=width, break_long_words=False, break_on_hyphens=False) if line.strip() else ''
            for line in lines
        ]
        return "\n".join(wrapped)

    
    def explain(self, folder_name: str):
        """
        Print a concise, beginner-friendly overview of the specified dataset,
        highlighting its significance and connection to other folders.
        """
    
        # Determine the main dataset type and version (A or B).
        name_upper = folder_name.upper()
        main_type = "Unknown"
        version = ""
    
        if "FLATVEL" in name_upper:
            main_type = "FlatVel"
        elif "CURVEVEL" in name_upper:
            main_type = "CurveVel"
        elif "FLATFAULT" in name_upper:
            main_type = "FlatFault"
        elif "CURVEFAULT" in name_upper:
            main_type = "CurveFault"
        elif "STYLE" in name_upper:
            main_type = "Style"
    
        if "_A" in name_upper:
            version = "A (easier version)"
        elif "_B" in name_upper:
            version = "B (harder version)"
    
        dataset_descriptions = {
            "FlatVel": (
                "This dataset contains horizontally-layered (flat) subsurface velocity "
                "models without any faults. Seismic signals reflect from layer boundaries "
                "in a more straightforward way, so it's considered simpler than faulted or "
                "curved scenarios."
            ),
            "FlatFault": (
                "Like FlatVel, but each velocity model has at least one major fault "
                "offsetting the flat layers. This creates more abrupt jumps in velocity "
                "and thus more complex waveforms for the inversion task."
            ),
            "CurveVel": (
                "These velocity models have undulating (curved) layers without faults. "
                "Waveforms reflect and refract in a more complicated manner compared to "
                "flat layers."
            ),
            "CurveFault": (
                "Combines curved layering with at least one fault. This is among the most "
                "challenging 2D cases, since faults add discontinuities on top of curved "
                "geometry."
            ),
            "Style": (
                "Velocity models generated via style transfer from natural or artistic "
                "images, producing irregular, non-layered patterns. Seismic signals here "
                "can be quite complex and unpredictable compared to layered scenarios."
            ),
            "Unknown": (
                "This dataset name doesn't match the typical OpenFWI naming pattern, "
                "so a specialized description isn't available."
            )
        }
    
        desc = dataset_descriptions.get(main_type, dataset_descriptions["Unknown"])
    
        explanation_raw = f"""
    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
    â”‚          EXPLANATION FOR:  {folder_name}           â”‚
    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
    
    This folder is classified as:  {main_type}
    Version:                       {version or "N/A"}
    
    {desc}
    
    In OpenFWI, {main_type} is connected to other families like:
      - 'Flat' or 'Curve' indicate the layer geometry (horizontal vs. undulating).
      - 'Vel' or 'Fault' tell us if there's a fault present or not.
      - 'Style' stands for more irregular or style-transferred velocity fields.
      - 'A' versions are often simpler, while 'B' versions add complexity.
    
    The /data and /model subfolders hold .npy files that line up one-to-one,
    where each seismic recording file in /data corresponds to a velocity map
    in /model with the same file number. Together, they form the input-output
    pairs used in deep-learning-based Full Waveform Inversion.
    
    Enjoy exploring {folder_name}!
    """.strip("\n")
    
        # Print the explanation directly (no text wrapping).
        print(explanation_raw)
    
        print(f"Plotting example of {folder_name}")
        self.plot(folder_name)



kaggle_train_path = "/kaggle/input/waveform-inversion/train_samples"
sd = SeismicDataHelper(kaggle_train_path)
print(sd)


sd.explain("CurveFault_A")


sd.explain("CurveFault_B")


sd.explain("CurveVel_A")


sd.explain("CurveVel_B")


sd.explain("FlatFault_A")


sd.explain("FlatFault_B")


sd.explain("FlatVel_A")


sd.explain("FlatVel_B")


sd.explain("Style_A")


sd.explain("Style_B")


print("Datasets available:")
for name in sd.datasets:
    print(" â€¢", name)


count_cf = len(sd.CurveFault_A)
print(f"CurveFault_A has {count_cf} seismic/velocity pairs.")


sd.get_seis("FlatVel_A", "data1.npy")[0]


# Will work with model soon.


total = len(sd)
print(f"Total seismic/velocity pairs across all datasets: {total}")


flatvel_b_pairs = sd.pairs("FlatVel_B")
print("First 3 FlatVel_B pairs:")
for seis_path, vel_path in flatvel_b_pairs[:3]:
    print("  ", seis_path, "<->", vel_path)


sd.sample_pairs("CurveVel_A", 4)


try:
    ok = sd.verify("CurveFault_B", verbose=True)
    print("CurveFault_B verification passed:", ok)
except Exception as e:
    print("Verification error:", e)


sd.plot("Style_A")


def line_center(X: np.ndarray, y: np.ndarray):
    seismic_energy = X.sum(axis=(-1, -2))  # sum over time & receivers
    center_row = y[y.shape[0]//2, :]       # middle row of velocity
    fig, ax = plt.subplots()
    ax.plot(seismic_energy, label="Total energy per source")
    ax.plot(center_row,   label="Velocity at mid-row")
    ax.set_title("Energy vs. Mid-row Velocity")
    ax.set_xlabel("Index")  # dimensionless index
    ax.set_ylabel("Value")
    ax.legend()
    return fig

sd.plot_fn("CurveFault_B", line_center)


def max_amplitude(seis: np.ndarray, vel: np.ndarray) -> float:
    return float(np.abs(seis).max())

max_vals = sd.pipeline("FlatFault_A", max_amplitude)
print("Computed max amplitudes for", len(max_vals), "batches.")
print("First five:", max_vals[:5])


def spectrogram_plot(X: np.ndarray, y: np.ndarray, seis_path: str):
    trace = X[0, :, X.shape[-1]//2]   # source 0, mid-receiver
    fname = os.path.basename(seis_path)
    fig, ax = plt.subplots()
    ax.specgram(
        trace,
        NFFT=256,
        Fs=1.0,        # sample rate can be treated as 1 if unknown
        noverlap=128,
        cmap='magma'
    )
    ax.set_title(f"Spectrogram: {fname} (src=0, mid-receiver)")
    ax.set_xlabel("Time (sample step)")
    ax.set_ylabel("Frequency (sample bin)")
    return fig

sd.plot_fn("CurveFault_B", spectrogram_plot)


sd.plot_signal_time_series("CurveFault_A", "seis4_1_0.npy")

