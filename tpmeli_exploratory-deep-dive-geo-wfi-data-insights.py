%%bash
mkdir -p ~/.jupyter/custom
cat > ~/.jupyter/custom/custom.css <<'CSS'
/* scale absolutely everything inside the notebook iframe */
html, body       { font-size: 18px !important; line-height: 1.6 !important; }
div.input_area   { font-size: 16px !important; }
div.output_area  { font-size: 16px !important; }
.prompt          { font-size: 12px !important; }   /* â€œInÂ [ ]:â€� labels */
h1 { font-size: 2.0em !important; }
h2 { font-size: 1.75em !important; }
h3 { font-size: 1.50em !important; }
CSS



!pip install -q itables


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from pathlib import Path
from typing import List, Tuple, Dict

###############################################################################
# 1) LIGHT SEISMIC HELPER
###############################################################################
class SeismicDataHelperLight:
    """
    Minimal version of the Seismic Data Helper:
      - Scans a root folder for subdirectories (datasets).
      - Each dataset has matching .npy files: dataXXXX.npy <-> modelXXXX.npy.
      - Access them through .datasets (list of dataset names) or pairs(dataset).
    """

    def __init__(self, root_dir: str):
        """
        Args:
            root_dir (str): Directory path containing subfolders with
                            seismic/velocity .npy files.
        """
        self.root_dir = root_dir
        self._pairs: Dict[str, List[Tuple[str, str]]] = self._scan_pairs()

    def _scan_pairs(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Look for subdirectories under root_dir. 
        For each subdirectory, try:
          - If /data and /model exist, match dataXXXX.npy with modelXXXX.npy
          - Otherwise, match seisXXXX.npy with velXXXX.npy.
        Return a dict:  {"folder_name": [ (seis_path, vel_path), ... ], ... }
        """
        pairs: Dict[str, List[Tuple[str, str]]] = {}
        for folder_name in os.listdir(self.root_dir):
            ds_dir = os.path.join(self.root_dir, folder_name)
            if not os.path.isdir(ds_dir):
                continue

            data_dir = os.path.join(ds_dir, 'data')
            model_dir = os.path.join(ds_dir, 'model')

            # If /data and /model exist:
            if os.path.isdir(data_dir) and os.path.isdir(model_dir):
                matched_list = []
                for fname in os.listdir(data_dir):
                    if fname.startswith('data') and fname.endswith('.npy'):
                        suf = fname[len('data'):-4]  # part after 'data' before '.npy'
                        seis_file = os.path.join(data_dir, fname)
                        vel_file  = os.path.join(model_dir, f'model{suf}.npy')
                        if os.path.exists(vel_file):
                            matched_list.append((seis_file, vel_file))
                if matched_list:
                    pairs[folder_name] = sorted(matched_list)
                continue

            # Otherwise, try to match seisXXXX.npy with velXXXX.npy
            all_files = os.listdir(ds_dir)
            seis_fs = [f for f in all_files if f.startswith('seis') and f.endswith('.npy')]
            vel_fs  = [f for f in all_files if f.startswith('vel')  and f.endswith('.npy')]
            if seis_fs and vel_fs:
                vel_set = set(vel_fs)
                matched_list = []
                for sf in seis_fs:
                    suf = sf[len('seis'):-4]
                    vf  = f'vel{suf}.npy'
                    if vf in vel_set:
                        matched_list.append((os.path.join(ds_dir, sf),
                                             os.path.join(ds_dir, vf)))
                if matched_list:
                    pairs[folder_name] = sorted(matched_list)

        return pairs

    @property
    def datasets(self) -> List[str]:
        """List all available subfolders that contain matched (seis, vel) pairs."""
        return list(self._pairs.keys())

    def pairs(self, folder_name: str) -> List[Tuple[str, str]]:
        """All (seis_path, vel_path) pairs belonging to a named dataset folder."""
        return self._pairs[folder_name]



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_first_arrivals(
    seis_path: str,
    vel_path: str,
    family_name: str = "",
    sample_idx: int = 0,
    src_idx: int = 0,
    thr_ratio: float = 0.05
):
    """
    1) Loads seismic & velocity arrays from disk.
    2) Computes first arrivals from the selected gather.
    3) Plots them side by side (seismic gather + velocity map).
    4) Places a large, clearly visible annotation for earliest arrival well inside the chart.
    5) Increases figure size; changes velocity-map annotation text to white for clarity.
    6) Optionally includes family_name in the figure title if provided.
    """

    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    seis = np.load(seis_path)
    vel  = np.load(vel_path)

    if vel.ndim == 4 and vel.shape[1] == 1:
        vel = vel[:, 0]  # remove singular dimension: (N,1,H,W) -> (N,H,W)

    # shape => (time_len, nrec)
    if seis.ndim == 4:
        gather = seis[sample_idx, src_idx]
    else:
        gather = seis[src_idx]
    time_len, nrec = gather.shape

    # -------------------------------------------------------------------------
    # Compute first-breaks for each receiver
    # -------------------------------------------------------------------------
    first_breaks = []
    for r in range(nrec):
        trace = gather[:, r]
        peak_amp = np.max(np.abs(trace))
        fb = np.argmax(np.abs(trace) >= thr_ratio * peak_amp)
        first_breaks.append(fb)
    first_breaks = np.array(first_breaks)

    # Find the earliest arrival across receivers
    earliest_idx = np.argmin(first_breaks)
    earliest_fb  = first_breaks[earliest_idx]

    # -------------------------------------------------------------------------
    # Set Figure + Title
    # -------------------------------------------------------------------------
    title_str = "CUE 1 â€“ Earliest received signal â�œ Shallow bulk velocity"
    if family_name.strip():
        title_str = f"[{family_name}] {title_str}"

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))  # bigger figure
    fig.suptitle(title_str, fontsize=14, weight='bold')

    # -------------------------------------------------------------------------
    # LEFT: Seismic Gather
    # -------------------------------------------------------------------------
    im0 = axes[0].imshow(
        gather.T,
        aspect='auto',
        cmap='seismic',
        origin='lower'
    )
    axes[0].scatter(first_breaks, np.arange(nrec), c='yellow', s=20, label="First arrival")

    # Annotate earliest arrival. Place text well inside the plot:
    # We'll offset the text 50 samples to the right and 30 receivers up
    # from the earliest arrival point (in data coordinates).
    axes[0].annotate(
        f"Earliest arrival = {earliest_fb}",
        xy=(earliest_fb, earliest_idx),
        xytext=(earliest_fb + 50, earliest_idx + 30),
        textcoords='data',
        arrowprops=dict(arrowstyle='->', color='black', lw=2),
        fontsize=14,    # bigger font
        color='black',
        ha='left',
        va='bottom',
        clip_on=False
    )

    axes[0].set_xlabel("Time sample", fontsize=12)
    axes[0].set_ylabel("Receiver index", fontsize=12)
    axes[0].set_title("Seismic gather", fontsize=13)
    axes[0].legend(fontsize=10)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, label="Amplitude")

    # -------------------------------------------------------------------------
    # RIGHT: Velocity Map
    # -------------------------------------------------------------------------
    if vel.ndim == 3:
        vel_slice = vel[sample_idx] if sample_idx < vel.shape[0] else vel[0]
    else:
        vel_slice = vel  # 2D
    im1 = axes[1].imshow(vel_slice, aspect='auto', cmap='viridis', origin='upper')
    axes[1].set_xlabel("X grid", fontsize=12)
    axes[1].set_ylabel("Depth grid", fontsize=12)
    axes[1].set_title("Velocity map (m/s)", fontsize=13)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, label="Velocity (m/s)")

    # Shallow layer rectangle
    rect_height = 8
    rect = patches.Rectangle(
        (0, 0), vel_slice.shape[1], rect_height,
        ec='red', fc='none', ls='--', lw=2
    )
    axes[1].add_patch(rect)

    # Center "Shallow layer" text inside the red rectangle, color=white
    text_x = vel_slice.shape[1] / 2.0
    text_y = rect_height / 2.0
    axes[1].text(
        text_x,
        text_y,
        "Shallow layer\n(high influence)",
        color='white',
        fontsize=14,
        ha='center',
        va='center',
        clip_on=False
    )

    plt.tight_layout()
    plt.show()



###############################################################################
# 3) LOOP OVER ALL DATASETS & PLOT ONE RANDOM SAMPLE PER DATASET
###############################################################################
def loop_over_all_families_and_plot(helper: SeismicDataHelperLight, 
                                    thr_ratio: float = 0.05, 
                                    ):
    """
    For every dataset folder in the helper, randomly choose one (seis, vel) pair
    and plot the first arrivals.
    """
    for dataset_name in helper.datasets:
        pairs = helper.pairs(dataset_name)
        if not pairs:
            print(f"No pairs found in dataset: {dataset_name}")
            continue

        print(f"\n--- Dataset: {dataset_name} ---")
        # Grab a random pair
        seis_path, vel_path = random.choice(pairs)

        # Show which files we picked
        print("Seismic file:", os.path.basename(seis_path))
        print("Velocity file:", os.path.basename(vel_path))

        # Plot first arrivals
        plot_first_arrivals(
            seis_path=seis_path,
            vel_path=vel_path,
            sample_idx=0,
            src_idx=0,
            thr_ratio=thr_ratio,
        )


###############################################################################
# EXAMPLE USAGE (comment out if needed)
###############################################################################

root_path = "/kaggle/input/waveform-inversion/train_samples" 
helper = SeismicDataHelperLight(root_path)
loop_over_all_families_and_plot(helper, thr_ratio=0.05)


import pandas as pd
from itables import init_notebook_mode, show

init_notebook_mode(all_interactive=True)

# ------------------------------------------------------------------
# 1) Hardcode the outcome data
# ------------------------------------------------------------------
stats_data = {
    "FlatFault_A":   {"mean": 150.261539, "median": 133.0, "std": 91.125188,  "var": 8303.799908,  "mode": 27},
    "FlatFault_B":   {"mean": 158.973959, "median": 143.0, "std": 94.326665,  "var": 8897.519799,  "mode": 27},
    "FlatVel_A":     {"mean": 163.750418, "median": 145.0, "std": 99.752426,  "var": 9950.546542,  "mode": 27},
    "FlatVel_B":     {"mean": 120.575377, "median": 104.0, "std": 74.968204,  "var": 5620.231563,  "mode": 31},
    "CurveFault_A":  {"mean": 149.355445, "median": 132.0, "std": 90.428030,  "var": 8177.228680,  "mode": 27},
    "CurveFault_B":  {"mean": 159.578760, "median": 143.0, "std": 94.642768,  "var": 8957.253621,  "mode": 27},
    "CurveVel_A":    {"mean": 163.157360, "median": 144.0, "std": 99.494013,  "var": 9899.058663,  "mode": 27},
    "CurveVel_B":    {"mean": 120.351633, "median": 104.0, "std": 74.444502,  "var": 5541.983890,  "mode": 31},
    "Style_A":       {"mean": 173.776879, "median": 158.0, "std": 101.937806, "var": 10391.316337, "mode": 32},
    "Style_B":       {"mean": 168.990238, "median": 156.0, "std": 96.733206,  "var": 9357.313050,  "mode": 32},

}

# ------------------------------------------------------------------
# 2) Convert to a DataFrame for further display and analysis
# ------------------------------------------------------------------
df = pd.DataFrame.from_dict(stats_data, orient="index")
df.index.name = "family"
df = df[["mean", "median", "std", "var", "mode"]]

# ------------------------------------------------------------------
# 3) Display a quick analysis (text-based)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# 4) Display with itables
# ------------------------------------------------------------------
show(df, scrollX=True, scrollY="400px")



# A Clear Explanation of The Dataset
# I had a hard time understanding what was going on with the data. Here's a potentially better/simpler explanation for those new to the domain (like me).

# 1. What are we even talking about?

# Weâ€™re dealing with a seismic imaging problem. In seismic imaging, energy waves are sent into the ground (from sources like vibroseis trucks or air guns), and then the signals that bounce back are recorded by receivers (geophones or hydrophones). We want to figure out how fast those waves travel at different points below the surfaceâ€”this speed is called the subsurface velocity.

# Now, imagine we cut the Earth in half so we can look at it from the side. Along that vertical slice, each point in that 2D cross-section has some speed (velocity) with which sound waves (seismic waves) travel. This 2D arrangement of speeds is called a velocity map. Meanwhile, up on the surface, we have instruments recording the wave signals over time, which we call seismic data.

# So the fundamental pair is:

# Seismic data: all the waveforms recorded over time.
# Velocity map: a 2D â€œpictureâ€� of how speed varies with depth and horizontal position.
# 2. Why are there three â€œfamiliesâ€�?

# Short answer: each â€œfamilyâ€� is a different type of 2D velocity cross-section (and corresponding seismic data) that was artificially generated to represent distinct geological scenarios.

# Vel Family

# What it looks like: Think of relatively orderly, layered sediments. â€œVelâ€� stands for â€œvelocity,â€� but effectively this family has simpler layered structures (like a cake with flat or gently curved layers).
# Why itâ€™s simpler: The layers might be horizontal (FlatVel) or smoothly dipping/curving (CurveVel), but no major breaks in the rock.
# What the seismic data show: You typically get clear, continuous reflection events.
# Fault Family

# What it looks like: Same idea of layered geology, but now there are faultsâ€”places where the layers are cut and shifted relative to each other.
# Why itâ€™s more complex: A fault is basically a break in the Earth where one side of a layer is displaced relative to the other. This adds complexity.
# What the seismic data show: Discontinuous reflections, sudden jumps, or â€œdiffractionsâ€� at the fault lines.
# Style Family

# What it looks like: Not your typical â€œlayer cake.â€� Instead, itâ€™s more chaotic or irregular shapes, because the creators used â€œstyle transferâ€� from arbitrary images to generate velocity patterns. Imagine velocity maps that might have swirls, blobs, or texturesâ€”no clean layering.
# Why itâ€™s the most unpredictable: It can mimic random or exotic geologies.
# What the seismic data show: More scattered wave energy, less straightforward layering.
# Essentially, these three â€œfamiliesâ€� cover a spectrum of geological complexity:

# Vel: simpler stratified layers
# Fault: layered but with big breaks (faults)
# Style: all sorts of random or organic shapes
# 3. So, these are synthetic or real?

# Theyâ€™re synthetic datasets. That means people used a physics simulator to generate the seismic data given an artificially created velocity map. This approach ensures we have a â€œtrue velocity mapâ€� for every sample, which is extremely valuable for training or testing machine learning algorithms, because real field data usually doesnâ€™t have perfect â€œground truth.â€�

# 4. Why do we need so many subfolders?

# Each family can have sub-variations:

# â€œAâ€� vs. â€œBâ€� versions: typically â€œAâ€� is somewhat simpler (fewer random variations, fewer layers, gentler changes), while â€œBâ€� is more complex (more layers, more variability).
# â€œFlatâ€� vs. â€œCurveâ€� (for Vel or Fault): indicates whether layers are relatively flat or curved/folded.
# For example, FlatVel_A means the simplest version of layered velocity with almost no major complexity. CurveVel_B means a more complex version of layered velocity with significant curving/folding. FlatFault_A means a simpler fault scenario; CurveFault_B means a complex fault scenario with curved layers. And so on.

# 5. Whatâ€™s actually in each folder?

# In each subfolder, you see .npy files (NumPy format) containing:

# Seismic data â€” the time series recordings at the surface. Each seismic .npy file holds a batch of 500 samples.
# Velocity maps â€” the 2D grid of wave speeds for those same 500 samples.
# Concretely, if you open, say, FlatVel_B, you might see something like:

# FlatVel_B/
#    data/
#       data1.npy  <-- 500 seismic samples
#       data2.npy  <-- another 500 seismic samples
#    model/
#       model1.npy <-- 500 velocity maps
#       model2.npy <-- another 500 velocity maps
# The index in the file (like data1.npy[i] and model1.npy[i]) lines up the i-th seismic sample with the i-th velocity map.

# For the Fault family, itâ€™s the same pairing but with filenames like seis4_1_0.npy (seismic) and vel4_1_0.npy (velocity). Each still has 500 paired samples.

# 6. Shape of the Data: 4D vs. 3D

# Seismic data is in 4 dimensions:

# The â€œbatchâ€� dimension (500) for the number of samples.
# The number of sources (like 5 different shot points).
# The number of time steps (like 1000 samples in time).
# The number of receivers (like 70 recording positions).
# So if the shape is (500, 5, 1000, 70), that means:

# 500 different subsurface scenarios
# Each scenario has 5 seismic sources
# Recorded for 1000 time steps
# At 70 receiver positions.
# Velocity map is in 3 dimensions:

# The â€œbatchâ€� dimension (500) for the number of samples.
# The height (70 grid points from top to bottom).
# The width (70 grid points horizontally).
# So if the shape is (500, 70, 70), that means each of the 500 scenarios has a 70Ã—70 â€œimageâ€� representing velocity in the ground.

# 7. The Competition Setup

# You have a training folder with many .npy files (like data1.npy/model1.npy pairs). This is what you can use to train or test your approach.
# You also get a test folder that only has the seismic data (no velocity). The competition wants you to predict the velocity maps for those unknown examples.
# You submit your predictions in a particular CSV or NumPy format so the organizers can compare them to the real velocity maps (which they keep hidden for evaluation).
# Essentially, the competition is: â€œGiven 3D seismic waveforms, can you predict the 2D velocity cross-section for each sample?â€�

# 8. Why does it matter?

# In real life, if you record seismic data in the field, you donâ€™t automatically know the exact velocity structure underground (and thatâ€™s often what you want to find). Having synthetic data with known â€œanswersâ€� (velocity maps) helps us benchmark algorithms. The final step is to see who can do the best â€œinversionâ€� from seismic to velocity.

# Recap in Super-Simple Terms

# 3 Families:
# Vel = simpler, layered Earth.
# Fault = layered Earth but broken by faults.
# Style = more random, weird patterns.
# Each family has subfolders (e.g., FlatVel_A, CurveVel_B) indicating more or less complexity.
# Inside each subfolder are .npy files holding (a) seismic data and (b) velocity maps. Each .npy holds 500 â€œexamplesâ€� (pairs).
# The competition: train a model on these known pairs to learn to predict velocity from seismic, then apply it to new test seismic data.
# Thatâ€™s the entire structure in a nutshell. Each folder is basically a chunk of data with 500 training examples. The difference is just how the underground geology was generated (flat layers, curved layers, faults, or random images). The ultimate goal is to see if your method can handle them all!

 
# Data Paths:
# On Kaggle: 
# Training: /kaggle/input/waveform-inversion/train_samples
# Test: /kaggle/input/waveform-inversion/test

# /kaggle/input/waveform-inversion/
# â”œâ”€â”€ test/
# â”‚    â””â”€â”€ [Test seismic data .npy files]
# â”‚
# ------------------------------------------------------------------
#  Complete snapshot of the *train_samples* tree (combined views)
# ------------------------------------------------------------------
#
#  Folder          â”‚  Layout detected by the script  â”‚  Contents
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  CurveFault_A    â”‚  â€œseis / velâ€� prefix layout     â”‚  seis*.npy   +   vel*.npy
#  CurveFault_B    â”‚  â€œseis / velâ€� prefix layout     â”‚  seis*.npy   +   vel*.npy
#
#  CurveVel_A      â”‚  twoâ€‘folder layout              â”‚  data/  â†’ *.npy (seismic)
#                  â”‚                                 â”‚  model/ â†’ *.npy (velocity)
#  CurveVel_B      â”‚  twoâ€‘folder layout              â”‚  data/  â†’ *.npy (seismic)
#                  â”‚                                 â”‚  model/ â†’ *.npy (velocity)
#
#  FlatFault_A     â”‚  â€œseis / velâ€� prefix layout     â”‚  seis*.npy   +   vel*.npy
#  FlatFault_B     â”‚  â€œseis / velâ€� prefix layout     â”‚  seis*.npy   +   vel*.npy
#
#  FlatVel_A       â”‚  twoâ€‘folder layout              â”‚  data/  â†’ *.npy (seismic)
#                  â”‚                                 â”‚  model/ â†’ *.npy (velocity)
#  FlatVel_B       â”‚  twoâ€‘folder layout              â”‚  data/  â†’ *.npy (seismic)
#                  â”‚                                 â”‚  model/ â†’ *.npy (velocity)
#
#  Style_A         â”‚  twoâ€‘folder layout              â”‚  data/  â†’ *.npy (seismic)
#                  â”‚                                 â”‚  model/ â†’ *.npy (velocity)
#  Style_B         â”‚  twoâ€‘folder layout              â”‚  data/  â†’ *.npy (seismic)
#                  â”‚                                 â”‚  model/ â†’ *.npy (velocity)
#
#  test            â”‚  seismicâ€‘only folder            â”‚  seis*.npy   (no models)
#
#  Notes
#  â€¢ â€œTwoâ€‘folderâ€�  â†’ subâ€‘directories named data/ (waveforms) and model/ (velocity).
#  â€¢ â€œPrefixâ€�      â†’ files live directly in the style folder and are
#                    distinguished by filename prefix:  seis*.npy vsÂ vel*.npy.
#  â€¢ Any other pattern would fall back to a single â€œFilesâ€� bar in the scriptâ€™s chart.


# %% [code]  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SETâ€‘UP  (imports, paths, helper)     â”€â”€Â DROP THIS CELL IN ONCEÂ â”€â”€
# --------------------------------------------------------------------------
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import fft
from scipy.signal import hilbert
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import skimage.measure
import scipy.ndimage as ndi
import matplotlib.gridspec as gridspec
from IPython.display import display, HTML

# --------------------------------------------------------------------------
# Data roots
# --------------------------------------------------------------------------
TRAIN_DIR = Path("/kaggle/input/waveform-inversion/train_samples")
if not isinstance(TRAIN_DIR, Path):
    raise TypeError(f"CRITICAL FAILURE: TRAIN_DIR must be a pathlib.Path, got {type(TRAIN_DIR)}")
print(f"SETUP CELL CONFIRMED â€“ TRAIN_DIR: {TRAIN_DIR}")

# (optional) test folder
TEST_DIR  = None   # or Path("/kaggle/input/waveform-inversion/test")

# --------------------------------------------------------------------------
# The ORIGINAL lowâ€‘level finder (unchanged)
# --------------------------------------------------------------------------
def _find_first_file_internal(base_dir: Path,
                              family_name: str,
                              file_type: str = 'model',
                              specific_batch: str | None = None):
    """
    Lowâ€‘level implementation â€“Â expects *base_dir* to be a Path object.
    """
    if not isinstance(base_dir, Path):
        print(f"CRITICAL ERROR in find_first_file: base_dir must be a Path, got {type(base_dir)}")
        return None

    family_dir = base_dir / family_name
    if not family_dir.is_dir():
        print(f"Warning: Directory not found â€“ {family_dir}")
        return None

    # --- twoâ€‘folder layout? -------------------------------------------------
    type_subdir         = family_dir / file_type
    is_two_folder       = type_subdir.is_dir() and file_type in {'data', 'model'}

    if is_two_folder:
        search_dir      = type_subdir
        prefix          = ""
    else:                               # prefix layout
        search_dir      = family_dir
        if file_type in {'model', 'vel'}:   prefix = "vel"
        elif file_type in {'data',  'seis'}:prefix = "seis"
        else:                               prefix = ""

    npy_files = sorted(search_dir.glob(f"{prefix}*.npy")) if search_dir.is_dir() else []

    # --- broaden search if nothing found -----------------------------------
    if not npy_files:
        broad = sorted(p for p in family_dir.glob("*.npy") if p.is_file())
        if broad:
            cand = [p for p in broad if p.name.startswith(prefix)] or broad
            npy_files = cand

    if not npy_files:
        print(f"Warning: No '{prefix}*.npy' in {search_dir} or {family_dir}")
        return None

    # --- honour specific_batch if requested --------------------------------
    if specific_batch:
        for p in npy_files:
            if p.name == specific_batch:
                return p
        print(f"Warning: '{specific_batch}' not found â€“ returning {npy_files[0].name}")

    return npy_files[0]


# --------------------------------------------------------------------------
# Public helper â€“Â BACKWARDS COMPATIBLE + SHORTâ€‘FORM
# --------------------------------------------------------------------------
def find_first_file(*args,
                    file_type: str = 'model',
                    specific_batch: str | None = None):
    """
    Flexible callâ€‘signatures
    ------------------------
    1) Legacy (explicit path)  :  find_first_file(base_dir, family, file_type, specific_batch)
    2) New   (implicit path)   :  find_first_file(family,          file_type, specific_batch)

    The second form automatically inserts TRAIN_DIR as *base_dir*.
    """
    if not args:
        raise TypeError("find_first_file() missing required positional argument 'family_name'")

    # If first arg is a directory â†’ legacy style
    if isinstance(args[0], (str, Path)) and Path(args[0]).is_dir():
        base_dir, family_name, *positional_tail = args
    else:   # short form â€“Â assume TRAIN_DIR
        base_dir       = TRAIN_DIR
        family_name, *positional_tail = args

    # allow positional overrides for file_type / specific_batch
    if positional_tail:
        file_type      = positional_tail[0] if len(positional_tail) >= 1 else file_type
    if len(positional_tail) >= 2:
        specific_batch = positional_tail[1]

    return _find_first_file_internal(base_dir, Path(family_name),
                                     file_type=file_type,
                                     specific_batch=specific_batch)



# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Global Matplotlib / Seaborn preset + bulletâ€‘proof helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import os, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from matplotlib import cycler

# â”€â”€ rcParams â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.rcParams.update({
    "figure.figsize"   : (14, 8),
    "figure.dpi"       : 110,
    "figure.titlesize" : 18,
    "axes.titlesize"   : 18,
    "axes.labelsize"   : 14,
    "xtick.labelsize"  : 14,
    "ytick.labelsize"  : 14,
    "legend.fontsize"  : 12,
    "axes.prop_cycle"  : cycler("color",
        ["#008080", "#e07b39", "#6ba3d6", "#dd345f",
         "#2ca02c", "#9467bd", "#ffbf00"]),
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "grid.linestyle"   : "--",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "image.cmap"       : "viridis",
    "savefig.bbox"     : "tight",
})

sns.set_theme(context="paper", style="whitegrid", rc=plt.rcParams)

# Optional custom .mplstyle
STYLE = "./my_poster.mplstyle"
if os.path.isfile(STYLE):
    plt.style.use(STYLE)
else:
    print(f"âš ï¸�  custom style '{STYLE}' not found â€“ using rcParams")

# â”€â”€ tiny helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def add_caption(fig, txt, gap=0.04, **kw):
    fig.subplots_adjust(bottom=gap + 0.02)
    fig.text(0.01, gap, txt, ha="left", va="top", **kw)

def plot_mean_std_example(path_to_npy):
    """Load one velocity cube and show its mean + std maps (demo)."""
    cube = np.load(path_to_npy, mmap_mode="r").squeeze()   # (500,70,70) â†’ (500,70,70)
    mean, std = cube.mean(axis=0), cube.std(axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    vmin, vmax = cube.min(), cube.max()
    im0 = ax[0].imshow(mean, vmin=vmin, vmax=vmax, origin="upper", aspect="auto")
    ax[0].set_title("mean velocity (m/s)")
    im1 = ax[1].imshow(std, origin="upper", aspect="auto")
    ax[1].set_title("stdâ€‘dev velocity (m/s)")

    for a, im in zip(ax, (im0, im1)):
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04).set_label("m/s")

    add_caption(fig,
        "â€¢ **Mean** â€“ an average underground view.\n"
        "â€¢ **Stdâ€‘Dev** â€“ bright bands flag layers that shift across samples.",
        gap=0.08)
    plt.show()

# â”€â”€ quick smokeâ€‘test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SAMPLE_FILE = "/kaggle/input/waveform-inversion/train_samples/FlatVel_A/model/model1.npy"
# plot_mean_std_example(SAMPLE_FILE)


# â•”â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•—
# â•‘   DatasetÂ SanityÂ CheckÂ â€“ Getting Acquainted with Training .npy Files  â•‘
# â•šâ•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
import pathlib, pandas as pd, plotly.express as px, plotly.io as pio

# â€“â€“â€“â€“â€“ 0) renderer that works reliably on Kaggle â€“â€“â€“â€“â€“
pio.renderers.default = "iframe"

# â€“â€“â€“â€“â€“ 1) locate â€œâ€¦/train_samplesâ€� folder â€“â€“â€“â€“â€“
ROOT = next(pathlib.Path("/kaggle/input").rglob("train_samples"), None)
assert ROOT and ROOT.is_dir(), "Could not find a /train_samples folder anywhere under /kaggle/input"
print("âœ”ï¸� dataset path detected:", ROOT)

# â€“â€“â€“â€“â€“ 2) classify .npy files â€“â€“â€“â€“â€“
def join_names(names, limit=20):
    names = sorted(names)
    if len(names) > limit:
        names = names[:limit] + [f"â€¦ ({len(names)-limit} more)"]
    return "<br>".join(names)

records = []
for f in sorted(ROOT.iterdir()):
    if not f.is_dir():
        continue
    data_dir, model_dir = f/"data", f/"model"

    # (a) explicit data/model subâ€‘dirs
    if data_dir.is_dir() or model_dir.is_dir():
        for kind, sub in [("Data", data_dir), ("Model", model_dir)]:
            files = [x.name for x in sub.glob("*.npy")] if sub.is_dir() else []
            records.append({"Folder": f.name, "Kind": kind,
                            "Count": len(files), "Files": join_names(files)})
        continue

    # (b) plain files split by prefix
    seis = [x.name for x in f.glob("seis*.npy")]
    vel  = [x.name for x in f.glob("vel*.npy")]
    if seis or vel:
        if seis: records.append({"Folder": f.name, "Kind": "Seis",
                                 "Count": len(seis), "Files": join_names(seis)})
        if vel:  records.append({"Folder": f.name, "Kind": "Vel",
                                 "Count": len(vel),  "Files": join_names(vel)})
        continue

    # (c) fallback
    other = [x.name for x in f.glob("*.npy")]
    records.append({"Folder": f.name, "Kind": "Files",
                    "Count": len(other), "Files": join_names(other)})

df = pd.DataFrame(records)

# â€“â€“â€“â€“â€“ 3) interactive groupedâ€‘bar plot â€“â€“â€“â€“â€“
fig = px.bar(
    df, x="Folder", y="Count", color="Kind",
    hover_data={"Files": True, "Count": True},
    title="DatasetÂ SanityÂ CheckÂ â€”Â Getting Acquainted with Training .npy Files",
    barmode="group"
)

# rotate xâ€‘labels, tidy axes/legend
fig.update_layout(
    xaxis_tickangle=-45,
    yaxis=dict(title="# of .npy files", dtick=1),
    legend_title_text="File category",
    title_x=0.5,                   # center the title
)

# add a subtle caption just above the chart
fig.add_annotation(
    text="A quick visual audit of file counts by folder and category",
    xref="paper", yref="paper", x=0.5, y=1.08, showarrow=False,
    font=dict(size=12)
)

fig.show()


# â•”â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•—
# â•‘   Folder Explorer with Inline Explanations                               â•‘
# â•šâ•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
import pathlib, html, numpy as np
from IPython.display import HTML

ROOT = next(pathlib.Path("/kaggle/input").rglob("train_samples"), None)
assert ROOT and ROOT.is_dir(), "â�Œ  /train_samples not found"

# ---------- helpers ----------
def human_bytes(n):
    units = ['B','KB','MB','GB','TB']; i = 0
    while n >= 1024 and i < len(units)-1: n /= 1024; i += 1
    return f"{n:.1f}{units[i]}"

def npy_info(p):
    arr = np.load(p, mmap_mode='r'); shp, dt = arr.shape, arr.dtype; arr._mmap.close()
    return shp, dt

def folder_desc(name):
    if name.startswith("FlatVel"):   base = "Velâ€”flat, gently layered"; 
    elif name.startswith("CurveVel"): base = "Velâ€”curved/folded layers";
    elif name.startswith("FlatFault"): base = "Faultâ€”flat layers with breaks";
    elif name.startswith("CurveFault"):base = "Faultâ€”curved layers with breaks";
    elif name.startswith("Style"):   base = "Styleâ€”random texture pattern";
    else:                             base = "Unknown pattern"
    level = "A simpler" if name.endswith("_A") else "B more complex" if name.endswith("_B") else ""
    return f"{base} ({level})".strip()

KIND_INFO = {
    "Data":  "4â€‘D seismic waveforms (sourcesÂ Ã—Â timeÂ Ã—Â receivers)",
    "Model": "2â€‘D velocity ground truth",
    "Seis":  "Seismic recordings (prefix layout)",
    "Vel":   "Velocity maps (prefix layout)",
    "Files": "Misc. .npy files"
}

# ---------- gather ----------
tree = {}
for fld in sorted(ROOT.iterdir()):
    if not fld.is_dir(): continue
    kinds={}
    def add(k, paths):
        items=[(p.name, f"shape={shp}, dtype={dt}, size={human_bytes(p.stat().st_size)}")
               for p in paths if (shp:=npy_info(p)[0]) or True for dt in [npy_info(p)[1]]]
        if items: kinds[k]={"count":len(items),"files":items}
    add("Data",  (fld/"data").glob("*.npy")  if (fld/"data").is_dir()  else [])
    add("Model", (fld/"model").glob("*.npy") if (fld/"model").is_dir() else [])
    add("Seis", fld.glob("seis*.npy"));  add("Vel", fld.glob("vel*.npy"))
    if not kinds: add("Files", fld.glob("*.npy"))
    tree[fld.name]={"desc":folder_desc(fld.name),"kinds":kinds}

# ---------- html ----------
html_parts=["""
<style>
:root{--accent:#136efd;--bg:#fafafa;--border:#ddd;--font:system-ui,sans-serif}
#exp{font-family:var(--font);background:var(--bg);padding:1rem;border:1px solid var(--border);
     border-radius:8px;max-width:1050px;margin:auto}
#exp h2{margin:0 0 1rem;text-align:center;font-size:1.35rem}
.folder{border:1px solid var(--border);border-radius:6px;margin:.7rem 0;background:#fff}
.folder summary{cursor:pointer;display:flex;gap:.6rem;align-items:center;padding:.5rem .75rem}
.fname{font-weight:600}
.fdesc{font-size:.8rem;color:#555}
.kind-line{margin-left:1.3rem;margin-top:.45rem}
.kind-tag{background:var(--accent);color:#fff;border-radius:4px;padding:.18rem .55rem;font-size:.75rem}
.kind-info{font-size:.78rem;color:#222;margin-left:.45rem}
.count{color:#555;font-size:.78rem;margin-left:.25rem}
.file-list{margin:.25rem 0 .8rem 2.5rem;font-size:.85rem;color:#333}
.file-list li{margin:.04rem 0;list-style-type:disc}
</style>
<div id="exp">
  <h2>TrainingÂ .npyÂ ExplorerÂ â€”Â InlineÂ Explanations</h2>
"""]

for name,info in tree.items():
    kinds=info["kinds"]; desc=html.escape(info["desc"])
    html_parts.append("<details class='folder'>")
    html_parts.append(f"<summary><span class='fname'>{html.escape(name)}</span>"
                      f"<span class='fdesc'>â€” {desc}</span></summary>")
    for kind,meta in kinds.items():
        html_parts.append(f"<div class='kind-line'><span class='kind-tag'>{kind}</span>"
                          f"<span class='count'>({meta['count']})</span>"
                          f"<span class='kind-info'>{html.escape(KIND_INFO.get(kind,''))}</span></div>")
        html_parts.append("<ul class='file-list'>")
        for fn,tip in meta["files"]:
            html_parts.append(f"<li title='{html.escape(tip)}'>{html.escape(fn)}</li>")
        html_parts.append("</ul>")
    html_parts.append("</details>")

html_parts.append("</div>")
HTML("".join(html_parts))


def analyze_batch_velocity_range(family_name, file_type='model', specific_batch=None):
    """Calculates and plots the distribution of min/max velocity within a batch."""
    vel_path = find_first_file(family_name, file_type, specific_batch)
    if not vel_path: return

    print(f"Analyzing velocity range within batch: {vel_path.name} from {family_name}")
    try:
        # Use mmap_mode for potentially large files
        velocity_batch = np.load(vel_path, mmap_mode='r') # Shape e.g., (500, 70, 70)

        if velocity_batch.ndim != 3:
             print(f"Warning: Expected 3D array (batch, H, W), got {velocity_batch.shape}. Skipping.")
             return

        batch_size = velocity_batch.shape[0]
        per_sample_min = np.min(velocity_batch, axis=(1, 2))
        per_sample_max = np.max(velocity_batch, axis=(1, 2))
        per_sample_range = per_sample_max - per_sample_min

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Velocity Range Distribution within Batch: {family_name} ({vel_path.name})', fontsize=16)

        axs[0].hist(per_sample_min, bins=30, color='skyblue', edgecolor='black')
        axs[0].set_title('Distribution of Min Velocities per Sample')
        axs[0].set_xlabel('Min Velocity (m/s)')
        axs[0].set_ylabel('Frequency')

        axs[1].hist(per_sample_max, bins=30, color='salmon', edgecolor='black')
        axs[1].set_title('Distribution of Max Velocities per Sample')
        axs[1].set_xlabel('Max Velocity (m/s)')

        axs[2].hist(per_sample_range, bins=30, color='lightgreen', edgecolor='black')
        axs[2].set_title('Distribution of Velocity Ranges per Sample')
        axs[2].set_xlabel('Velocity Range (Max - Min) (m/s)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    except Exception as e:
        print(f"Error processing {vel_path}: {e}")

# Example usage: Analyze the first model file in FlatVel_A
analyze_batch_velocity_range('FlatVel_A', file_type='model')
# Example usage: Analyze a specific file if needed (name depends on layout)
# analyze_batch_velocity_range('CurveFault_A', file_type='model', specific_batch='vel4_1_0.npy')


# %%code
def compare_A_B_variants(family_base):
    """Compares velocity distributions and stats for _A and _B variants of a family."""
    family_A = f"{family_base}_A"
    family_B = f"{family_base}_B"

    path_A = find_first_file(family_A, 'model')
    path_B = find_first_file(family_B, 'model')

    if not path_A or not path_B:
        print(f"Could not find files for both {family_A} and {family_B}. Skipping comparison.")
        return

    print(f"Comparing {family_A} ({path_A.name}) vs {family_B} ({path_B.name})")

    try:
        vel_A = np.load(path_A, mmap_mode='r').flatten()
        vel_B = np.load(path_B, mmap_mode='r').flatten()

        # Calculate basic stats
        stats_A = {'mean': np.mean(vel_A), 'std': np.std(vel_A), 'min': np.min(vel_A), 'max': np.max(vel_A)}
        stats_B = {'mean': np.mean(vel_B), 'std': np.std(vel_B), 'min': np.min(vel_B), 'max': np.max(vel_B)}

        print(f"Stats for {family_A}: {stats_A}")
        print(f"Stats for {family_B}: {stats_B}")

        # Plot overlaid histograms
        plt.figure(figsize=(12, 6))
        plt.hist(vel_A, bins=50, alpha=0.7, label=f'{family_A} (Std: {stats_A["std"]:.0f})', density=True, color='blue')
        plt.hist(vel_B, bins=50, alpha=0.7, label=f'{family_B} (Std: {stats_B["std"]:.0f})', density=True, color='red')
        plt.title(f'Velocity Distribution Comparison: {family_A} vs {family_B}')
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.show()

        # Optional: Compare mean/std maps (requires loading full batch and more plotting)
        # mean_A = np.mean(np.load(path_A, mmap_mode='r'), axis=0)
        # std_A = np.std(np.load(path_A, mmap_mode='r'), axis=0)
        # mean_B = np.mean(np.load(path_B, mmap_mode='r'), axis=0)
        # std_B = np.std(np.load(path_B, mmap_mode='r'), axis=0)
        # ... (add plotting code for mean/std maps side-by-side)

    except Exception as e:
        print(f"Error comparing {family_A} and {family_B}: {e}")

# Example usage: Compare FlatVel_A and FlatVel_B
compare_A_B_variants("FlatVel")
# Example usage: Compare CurveFault_A and CurveFault_B
compare_A_B_variants("CurveFault")


# %%code
def analyze_seismic_zeros(family_name, file_type='data', specific_batch=None):
    """Analyzes the occurrence of zero values in a seismic data batch."""
    seis_path = find_first_file(family_name, file_type, specific_batch)
    if not seis_path: return

    print(f"Analyzing seismic zeros in: {seis_path.name} from {family_name}")
    try:
        seismic_batch = np.load(seis_path, mmap_mode='r') # Shape e.g., (500, 5, 1000, 70)

        if seismic_batch.ndim != 4:
             print(f"Warning: Expected 4D array (batch, src, time, rec), got {seismic_batch.shape}. Skipping.")
             return

        num_samples, num_sources, num_timesteps, num_receivers = seismic_batch.shape
        total_elements = seismic_batch.size
        zero_mask = (seismic_batch == 0)
        num_zeros = np.sum(zero_mask)
        zero_percentage = (num_zeros / total_elements) * 100

        print(f"Total elements: {total_elements}")
        print(f"Number of zero values: {num_zeros} ({zero_percentage:.2f}%)")

        # Analyze zeros across dimensions
        zeros_per_source = np.sum(zero_mask, axis=(0, 2, 3)) # Sum over batch, time, receivers
        zeros_per_timestep = np.sum(zero_mask, axis=(0, 1, 3)) # Sum over batch, sources, receivers
        zeros_per_receiver = np.sum(zero_mask, axis=(0, 1, 2)) # Sum over batch, sources, time

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Zero Value Distribution Across Dimensions: {family_name} ({seis_path.name})', fontsize=16)

        axs[0].bar(range(num_sources), zeros_per_source, color='purple')
        axs[0].set_title('Zeros per Source Index')
        axs[0].set_xlabel('Source Index')
        axs[0].set_ylabel('Total Zero Count')
        axs[0].set_xticks(range(num_sources))

        axs[1].plot(range(num_timesteps), zeros_per_timestep, color='orange')
        axs[1].set_title('Zeros per Time Step')
        axs[1].set_xlabel('Time Step Index')
        axs[1].set_ylabel('Total Zero Count')

        axs[2].plot(range(num_receivers), zeros_per_receiver, color='green')
        axs[2].set_title('Zeros per Receiver Index')
        axs[2].set_xlabel('Receiver Index')
        axs[2].set_ylabel('Total Zero Count')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Further analysis could involve checking if entire traces/time steps are zero, etc.
        # E.g., are zeros clustered at the beginning/end of time axis?
        print(f"Zeros in first 10 timesteps: {np.sum(zeros_per_timestep[:10])}")
        print(f"Zeros in last 10 timesteps: {np.sum(zeros_per_timestep[-10:])}")


    except Exception as e:
        print(f"Error processing {seis_path}: {e}")

# Example usage: Analyze zeros in the first data file of CurveVel_A
analyze_seismic_zeros('CurveVel_A', file_type='data')
# Example usage: Analyze zeros in a fault family seismic file
# analyze_seismic_zeros('FlatFault_A', file_type='seis', specific_batch='seis4_1_0.npy')


# %%code
def check_seismic_artifacts_clipping(family_name, file_type='data', specific_batch=None):
    """Checks seismic data for potential clipping or numerical artifacts via histogram."""
    seis_path = find_first_file(family_name, file_type, specific_batch)
    if not seis_path: return

    print(f"Checking for artifacts/clipping in: {seis_path.name} from {family_name}")
    try:
        seismic_batch = np.load(seis_path, mmap_mode='r').flatten() # Flatten for global histogram

        min_val, max_val = np.min(seismic_batch), np.max(seismic_batch)
        mean_val, std_val = np.mean(seismic_batch), np.std(seismic_batch)
        # Calculate kurtosis (Fisher's definition, where 0 is normal)
        # Use pandas for easy calculation, handling potential NaNs if any
        kurt_val = pd.Series(seismic_batch).kurtosis()

        print(f"Min: {min_val:.2f}, Max: {max_val:.2f}, Mean: {mean_val:.4f}, StdDev: {std_val:.2f}, Kurtosis: {kurt_val:.2f}")

        plt.figure(figsize=(12, 6))
        # Use a large number of bins to see details, adjust range if needed
        plt.hist(seismic_batch, bins=200, range=(min_val - 1, max_val + 1), color='gray', edgecolor='black', lw=0.5)
        plt.title(f'Detailed Seismic Amplitude Histogram: {family_name} ({seis_path.name})')
        plt.xlabel('Amplitude Value')
        plt.ylabel('Frequency (Log Scale)')
        plt.yscale('log') # Log scale helps see tail behavior and potential clipping
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        # Highlight potential clipping zones (adjust thresholds based on observed min/max)
        plt.axvline(min_val, color='red', linestyle='--', label=f'Observed Min ({min_val:.2f})')
        plt.axvline(max_val, color='red', linestyle='--', label=f'Observed Max ({max_val:.2f})')
        plt.legend()
        plt.show()

        # Look for:
        # 1. Sharp vertical drop-offs at min_val or max_val (suggests clipping).
        # 2. Unexpected isolated spikes or gaps in the histogram.
        # 3. Very high kurtosis suggests extremely heavy tails/outliers.

    except Exception as e:
        print(f"Error processing {seis_path}: {e}")

# Example usage:
check_seismic_artifacts_clipping('FlatVel_A', file_type='data')
check_seismic_artifacts_clipping('Style_B', file_type='data')


# %%code
from scipy import fft

def plot_average_seismic_spectrum(families_to_compare, file_type='data', sample_index=0, source_index=0, receiver_index=0):
    """Compares the average frequency spectrum of seismic traces from different families."""
    plt.figure(figsize=(12, 7))
    plt.title(f'Average Seismic Frequency Spectrum Comparison (Sample {sample_index}, Src {source_index}, Rec {receiver_index})')

    all_spectra = {}

    for family_name in families_to_compare:
        seis_path = find_first_file(family_name, file_type=file_type)
        if not seis_path: continue

        print(f"Processing spectrum for: {family_name} ({seis_path.name})")
        try:
            seismic_batch = np.load(seis_path, mmap_mode='r') # Shape e.g. (500, 5, 1000, 70)

            if seismic_batch.ndim != 4:
                 print(f"Warning: Expected 4D array for {family_name}, got {seismic_batch.shape}. Skipping.")
                 continue

            # Select a specific trace across the batch, or average traces first
            # Here, we take one specific trace from each sample in the batch
            # Adjust indices as needed
            traces = seismic_batch[:, source_index, :, receiver_index] # Shape (500, 1000)

            # Calculate FFT for each trace in the batch
            magnitudes = []
            num_timesteps = traces.shape[1]
            freqs = fft.fftfreq(num_timesteps) # Assuming dt=1 sample, adjust if time step is known

            for trace in traces:
                fft_vals = fft.fft(trace)
                fft_mag = np.abs(fft_vals)
                magnitudes.append(fft_mag)

            # Average the magnitude spectra across the batch
            avg_magnitude = np.mean(np.array(magnitudes), axis=0)

            # Only plot positive frequencies
            positive_freq_indices = np.where(freqs >= 0)[0]
            positive_freqs = freqs[positive_freq_indices]
            avg_magnitude_positive = avg_magnitude[positive_freq_indices]

            all_spectra[family_name] = (positive_freqs, avg_magnitude_positive)

            plt.plot(positive_freqs, avg_magnitude_positive, label=f'{family_name}')

        except Exception as e:
            print(f"Error processing {family_name} ({seis_path}): {e}")

    plt.xlabel('Frequency (cycles/sample)') # Adjust if sampling rate known
    plt.ylabel('Average Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.xlim(0, 0.5) # Show up to Nyquist frequency (0.5 cycles/sample)
    plt.show()

# Example usage: Compare Vel, Fault, and Style families
families = ['FlatVel_B', 'CurveFault_B', 'Style_B']
plot_average_seismic_spectrum(families, file_type='data') # Assumes 'data' type exists for Style_B too

# Adjust file_type if necessary for fault families
# plot_average_seismic_spectrum(['FlatVel_B', 'CurveFault_B', 'Style_B'], file_type='seis', ...) # If fault uses 'seis'


# %%code
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- new similarity utilities -------------------------------------------------
def phase_only_corr(g1: np.ndarray, g2: np.ndarray) -> float:
    """Phaseâ€“only correlation (shiftâ€‘invariant)."""
    G1 = np.fft.fft2(g1)
    G2 = np.fft.fft2(g2)
    eps = 1e-9
    cross_phase = G1 * np.conj(G2)
    cross_phase /= (np.abs(cross_phase) + eps)
    poc = np.fft.ifft2(cross_phase)
    return np.abs(poc).max()

def fk_magnitude_corr(g1: np.ndarray, g2: np.ndarray) -> float:
    """Pearson correlation between |Fâ€‘K| spectra (shift + sign invariant)."""
    F1 = np.abs(np.fft.fft2(g1))
    F2 = np.abs(np.fft.fft2(g2))
    v1, v2 = F1.flatten(), F2.flatten()
    return np.corrcoef(v1, v2)[0, 1]

try:
    from skimage.metrics import structural_similarity as ssim
    def ssim_similarity(g1: np.ndarray, g2: np.ndarray) -> float:
        """Structuralâ€�similarity index averaged over image."""
        # ssim expects (H,W); data_range needed since seismic can be >1
        rng = g1.max() - g1.min() + 1e-6
        val = ssim(g1, g2, data_range=rng)
        return float(val)
except Exception:                       # skimage not present
    def ssim_similarity(g1, g2):       # graceful fallback
        return np.nan

# -----------------------------------------------------------------------------


def compare_source_gathers(family_name, file_type='data', specific_batch=None, sample_index=0):
    """Visualises gathers per source AND reports multiple similarity metrics."""
    seis_path = find_first_file(family_name, file_type, specific_batch)
    if not seis_path:
        return

    print(f"Comparing source gathers for sample {sample_index} in: {seis_path.name} from {family_name}")
    try:
        seismic_batch = np.load(seis_path, mmap_mode='r')   # shape: (N, S, T, R)

        if seismic_batch.ndim != 4:
            print(f"Warning: Expected 4â€‘D array, got {seismic_batch.shape}. Skipping.")
            return
        if sample_index >= seismic_batch.shape[0]:
            print(f"Warning: sample_index {sample_index} out of bounds; using 0.")
            sample_index = 0

        sample_data  = seismic_batch[sample_index]          # (S, T, R)
        num_sources  = sample_data.shape[0]

        # ------------- visualisation --------------------------------------------------
        fig = plt.figure(figsize=(5 * num_sources, 7))
        gs  = gridspec.GridSpec(1, num_sources, figure=fig)
        fig.suptitle(f'Seismic Gathers from Different Sources '
                     f'(Sample {sample_index}, {family_name})', fontsize=16, y=1.02)

        vmin = np.percentile(sample_data, 1)
        vmax = np.percentile(sample_data, 99)

        for i in range(num_sources):
            ax = fig.add_subplot(gs[0, i])
            im = ax.imshow(sample_data[i], aspect='auto', cmap='seismic',
                           vmin=vmin, vmax=vmax)
            ax.set_title(f'Source {i}')
            ax.set_xlabel('Receiver Index')
            if i == 0:
                ax.set_ylabel('Time Step')
            else:
                ax.set_yticklabels([])

        plt.tight_layout(rect=[0, 0, 0.98, 1])
        cbar_ax = fig.add_axes([0.985, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.show()

        # ------------- similarity metrics --------------------------------------------
        if num_sources > 1:
            ref = sample_data[0]
            print("\nSimilarity against SourceÂ 0")
            print("{:<10}{:>12}{:>12}{:>12}{:>12}".format(
                "Source",
                                "Pixel Corr (Ï�)",
                                "Phaseâ€‘Only Corr",
                                "SSIM (structure)",
                                "Fâ€‘K MagÂ Corr"))
            for i in range(1, num_sources):
                tgt = sample_data[i]

                # raw Pearson correlation (as before)
                corr = np.corrcoef(ref.flatten(), tgt.flatten())[0, 1]

                # phaseâ€‘only correlation (shiftâ€‘invariant)
                poc  = phase_only_corr(ref, tgt)

                # structural similarity
                ssim_val = ssim_similarity(ref, tgt)

                # Fâ€‘K magnitude correlation
                fk_r = fk_magnitude_corr(ref, tgt)

                print(f"{i:<10}{corr:12.4f}{poc:12.4f}{ssim_val:12.4f}{fk_r:12.4f}")

    except Exception as e:
        print(f"Error processing {seis_path} for sample {sample_index}: {e}")


# Example usage
compare_source_gathers('CurveFault_B', file_type='seis', sample_index=10)




import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import traceback

# Assuming TRAIN_DIR (Path object) and find_first_file (modified) are defined

def estimate_gather_snr_proxy(base_dir, family_name, file_type='data', specific_batch=None, noise_window_end_time=50):
    """Estimates a proxy for SNR for each gather in a batch using RMS amplitude."""
    if not isinstance(base_dir, Path):
        print("CRITICAL ERROR: base_dir passed to estimate_gather_snr_proxy must be a Path object.")
        return None

    seis_path = find_first_file(base_dir, family_name, file_type, specific_batch)
    if not seis_path:
        print(f"Skipping SNR analysis for {family_name}, file not found.")
        return None

    print(f"Estimating SNR proxy for gathers in: {seis_path.name} from {family_name}")
    try:
        seismic_batch = np.load(seis_path, mmap_mode='r')

        # Seismic data is expected to be 4D (samples, sources, time, receivers)
        # No squeeze needed here.
        if seismic_batch.ndim != 4:
             print(f"  âœ˜ FAIL {family_name}: Expected 4D seismic array, got {seismic_batch.shape}. Skipping.")
             return None

        num_samples, num_sources, num_timesteps, num_receivers = seismic_batch.shape
        if num_samples == 0:
             print(f"  âœ˜ FAIL {family_name}: Batch contains no samples.")
             return None

        snr_proxies = []
        epsilon = 1e-12

        for i in range(num_samples):
            gather_data = seismic_batch[i] # Shape (src, time, rec)
            if not np.all(np.isfinite(gather_data)):
                 snr_proxies.append(np.nan)
                 continue

            noise_window_end = min(noise_window_end_time, num_timesteps)
            if noise_window_end > 0:
                noise_part = gather_data[:, :noise_window_end, :]
                noise_rms = np.sqrt(np.mean(noise_part**2)) if noise_part.size > 0 and np.any(noise_part != 0) else 0.0
            else:
                noise_rms = 0.0

            signal_window_start = noise_window_end
            if signal_window_start < num_timesteps:
                signal_part = gather_data[:, signal_window_start:, :]
                signal_rms = np.sqrt(np.mean(signal_part**2)) if signal_part.size > 0 and np.any(signal_part != 0) else 0.0
            else:
                signal_rms = 0.0

            snr_proxy_ratio = signal_rms / (noise_rms + epsilon)
            snr_proxies.append(snr_proxy_ratio)

        snr_proxies = np.array(snr_proxies)
        valid_snr_mask = np.isfinite(snr_proxies)
        if not np.any(valid_snr_mask):
            print("Warning: No valid SNR proxies calculated (all NaN/Inf).")
            return None

        snr_proxies_valid = snr_proxies[valid_snr_mask]

        # Visualization
        plt.figure(figsize=(10, 6))
        plot_data = snr_proxies_valid[snr_proxies_valid < np.percentile(snr_proxies_valid, 99.5)] if len(snr_proxies_valid) > 0 else snr_proxies_valid
        if len(plot_data) > 0: plt.hist(plot_data, bins=50, alpha=0.8)
        else: plt.hist([], bins=50, alpha=0.8)

        plt.title(f'Distribution of Estimated SNR Proxy (Signal RMS / Noise RMS)\n{family_name} ({seis_path.name})')
        plt.xlabel('SNR Proxy Value (Ratio)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.4)

        if len(snr_proxies_valid)>0:
            q25 = np.percentile(snr_proxies_valid, 25)
            plt.axvline(q25, color='r', linestyle='--', label=f'Bottom Quartile ({q25:.2f})')
            plt.legend()
            print(f"\nAnalysis complete for {num_samples} samples ({len(snr_proxies_valid)} valid).")
            print(f"SNR Proxy (Signal/Noise RMS Ratio) Statistics (Valid Only):")
            print(f"  Min: {np.min(snr_proxies_valid):.2f}, Max: {np.max(snr_proxies_valid):.2f}")
            print(f"  Mean: {np.mean(snr_proxies_valid):.2f}, Median: {np.median(snr_proxies_valid):.2f}")
            print(f"  25th Percentile (Q1): {q25:.2f}")
            print(f"Action: Consider down-weighting or temporarily excluding gathers with SNR proxy below ~{q25:.2f} during early training epochs.")
        else:
            print("\nNo valid SNR proxies to calculate statistics.")

        plt.show()
        return snr_proxies

    except Exception as e:
        print(f"  âœ˜ FAIL {family_name}: Error processing {seis_path}: {e}")
        traceback.print_exc()
        return None

# Example Usage:
if 'TRAIN_DIR' in globals() and isinstance(TRAIN_DIR, Path):
    snr_results = estimate_gather_snr_proxy(TRAIN_DIR, 'CurveVel_A', file_type='data', noise_window_end_time=50)
else:
    print("Cannot run SNR analysis: TRAIN_DIR not correctly defined.")


import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from pathlib import Path
import traceback

# Assuming TRAIN_DIR (Path object) and find_first_file (modified) are defined

def analyze_frequency_drift(base_dir, family_name, file_type='data', specific_batch=None,
                            sample_index=0, source_index=0,
                            time_window_size=100, time_step=50,
                            spatial_time_center=500, spatial_time_half_width=50):
    """Analyzes dominant frequency drift over time and space for a seismic gather."""
    if not isinstance(base_dir, Path):
        print("CRITICAL ERROR: base_dir passed to analyze_frequency_drift must be a Path object.")
        return

    seis_path = find_first_file(base_dir, family_name, file_type, specific_batch)
    if not seis_path:
        print(f"Skipping frequency drift analysis for {family_name}, file not found.")
        return

    print(f"Analyzing frequency drift for Sample {sample_index}, Source {source_index} in: {seis_path.name} ({family_name})")
    try:
        seismic_batch = np.load(seis_path, mmap_mode='r')
        # Seismic data is expected to be 4D (samples, sources, time, receivers)
        # No squeeze needed here.
        if seismic_batch.ndim != 4:
             print(f"  âœ˜ FAIL {family_name}: Expected 4D seismic array, got {seismic_batch.shape}. Skipping.")
             return

        num_samples, num_sources, num_timesteps, num_receivers = seismic_batch.shape
        if sample_index >= num_samples: sample_index = 0
        if source_index >= num_sources: source_index = 0

        gather = seismic_batch[sample_index, source_index] # Shape (time, receivers)
        current_num_timesteps, current_num_receivers = gather.shape # Use actual shape after indexing

        if current_num_timesteps < time_window_size or current_num_receivers == 0:
             print(f"  âœ˜ FAIL {family_name}: Insufficient data dimensions (Time={current_num_timesteps}, Rec={current_num_receivers}). Min Time needed: {time_window_size}.")
             return

        mid_receiver_idx = current_num_receivers // 2
        trace = gather[:, mid_receiver_idx]

        temporal_dominant_freqs = []
        window_centers = []
        for start in range(0, current_num_timesteps - time_window_size + 1, time_step):
            end = start + time_window_size
            window = trace[start:end]
            center_time = start + time_window_size // 2
            if not np.all(np.isfinite(window)) or np.ptp(window) < 1e-9: dom_freq = np.nan
            else:
                fft_vals = fft.fft(window); fft_freq = fft.fftfreq(time_window_size)
                pos_mask = fft_freq > 0
                if not np.any(pos_mask): dom_freq = np.nan; continue
                magnitudes = np.abs(fft_vals[pos_mask]); frequencies = fft_freq[pos_mask]
                if len(magnitudes) == 0: dom_freq = np.nan; continue
                dom_freq_index = np.argmax(magnitudes); dom_freq = frequencies[dom_freq_index]
            temporal_dominant_freqs.append(dom_freq)
            window_centers.append(center_time)

        spatial_dominant_freqs = []
        receiver_indices = list(range(current_num_receivers))
        t_start = max(0, spatial_time_center - spatial_time_half_width)
        t_end = min(current_num_timesteps, spatial_time_center + spatial_time_half_width)
        spatial_window_size = t_end - t_start

        if spatial_window_size <= 1:
            print(f"  Warning: Spatial time window [{t_start}, {t_end}] too small. Skipping spatial drift.")
            spatial_dominant_freqs = [np.nan] * current_num_receivers
        else:
            fft_freq_spatial = fft.fftfreq(spatial_window_size)
            pos_mask_spatial = fft_freq_spatial > 0
            if not np.any(pos_mask_spatial):
                 print(f"  Warning: No positive frequencies in spatial time window. Skipping spatial drift.")
                 spatial_dominant_freqs = [np.nan] * current_num_receivers
            else:
                 frequencies_spatial = fft_freq_spatial[pos_mask_spatial]
                 for rcv_idx in range(current_num_receivers):
                    window = gather[t_start:t_end, rcv_idx]
                    if not np.all(np.isfinite(window)) or np.ptp(window) < 1e-9: dom_freq = np.nan
                    else:
                        fft_vals = fft.fft(window)
                        magnitudes = np.abs(fft_vals[pos_mask_spatial])
                        if len(magnitudes) == 0: dom_freq = np.nan
                        else: dom_freq_index = np.argmax(magnitudes); dom_freq = frequencies_spatial[dom_freq_index]
                    spatial_dominant_freqs.append(dom_freq)

        # Visualization
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
        fig.suptitle(f'Frequency Content Drift Analysis\n{family_name} - Sample {sample_index}, Source {source_index}', fontsize=16)
        valid_temporal = np.isfinite(temporal_dominant_freqs)
        if np.any(valid_temporal):
             axs[0].plot(np.array(window_centers)[valid_temporal], np.array(temporal_dominant_freqs)[valid_temporal], marker='o', linestyle='-', markersize=4)
             axs[0].set_ylim(bottom=0, top=max(0.01, np.nanmax(temporal_dominant_freqs)*1.1))
        else: axs[0].set_ylim(bottom=0, top=0.5)
        axs[0].set_title(f'Temporal Drift (Receiver {mid_receiver_idx})'); axs[0].set_xlabel('Time Window Center (samples)'); axs[0].set_ylabel('Dominant Frequency (cycles/sample)'); axs[0].grid(True, alpha=0.4)
        valid_spatial = np.isfinite(spatial_dominant_freqs)
        if np.any(valid_spatial):
             axs[1].plot(np.array(receiver_indices)[valid_spatial], np.array(spatial_dominant_freqs)[valid_spatial], marker='o', linestyle='-', markersize=4)
             axs[1].set_ylim(bottom=0, top=max(0.01, np.nanmax(spatial_dominant_freqs)*1.1))
        else: axs[1].set_ylim(bottom=0, top=0.5)
        axs[1].set_title(f'Spatial Drift (Time Window: {t_start}-{t_end} samples)'); axs[1].set_xlabel('Receiver Index'); axs[1].set_ylabel('Dominant Frequency (cycles/sample)'); axs[1].grid(True, alpha=0.4)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()
        print("\nInterpretation hints:\n  - Temporal drift: Decreasing frequency over time suggests attenuation (Q effects).\n  - Spatial drift: Systematic changes across receivers might indicate geometric spreading or ray path differences.\nAction: Design band-pass filters considering the lowest common bandwidth across relevant time/space ranges.")

    except Exception as e:
        print(f"  âœ˜ FAIL {family_name}: Error processing {seis_path}: {e}")
        traceback.print_exc()

# Example Usage:
if 'TRAIN_DIR' in globals() and isinstance(TRAIN_DIR, Path):
    analyze_frequency_drift(TRAIN_DIR, 'FlatVel_B', file_type='data', sample_index=10, source_index=2, time_window_size=150, time_step=25, spatial_time_center=600)
else:
    print("Cannot run frequency drift analysis: TRAIN_DIR not correctly defined.")


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import traceback

# Assuming TRAIN_DIR (Path object) and find_first_file (modified) are defined

def analyze_receiver_channel_health(base_dir, family_name, file_type='data', max_files_to_load=5):
    """Analyzes receiver channel health by calculating variance across samples/files."""
    if not isinstance(base_dir, Path):
        print("CRITICAL ERROR: base_dir passed to analyze_receiver_channel_health must be a Path object.")
        return None

    family_dir = base_dir / family_name
    if not family_dir.is_dir():
        print(f"Directory not found: {family_dir}. Skipping analysis.")
        return None

    # Logic to find files based on layout
    type_subdir = family_dir / file_type
    is_two_folder_layout = type_subdir.is_dir() and file_type in ['data', 'model']
    prefix = "seis" if file_type == 'seis' or file_type == 'data' else ""
    search_dir = type_subdir if is_two_folder_layout else family_dir
    if not search_dir.is_dir(): search_dir = family_dir # Fallback if subdir doesn't exist

    try: # Wrap globbing in try-except for permission errors etc.
        npy_files = sorted(list(search_dir.glob(f"{prefix}*.npy")))
        if not npy_files and search_dir != family_dir: # Try family dir if subdir failed
            npy_files = sorted(list(family_dir.glob(f"{prefix}*.npy")))

        if not npy_files: # Fallback broad search
             all_npy_files = sorted([p for p in family_dir.glob("*.npy") if p.is_file()])
             if file_type in ['data', 'seis']:
                  candidates = [f for f in all_npy_files if f.name.startswith('seis') or f.name.startswith('data')]
                  if candidates: npy_files = candidates
                  else: npy_files = [f for f in all_npy_files if not f.name.startswith('vel') and not f.name.startswith('model')]
             else: npy_files = all_npy_files # Take any if still failing
    except Exception as e:
         print(f"Error finding files in {search_dir} or {family_dir}: {e}")
         return None


    if not npy_files:
        print(f"No suitable {file_type} files found in {family_dir} or subdirs. Skipping analysis.")
        return None

    files_to_process = npy_files[:max_files_to_load]
    print(f"Analyzing receiver health for {family_name} using {len(files_to_process)} file(s): {[f.name for f in files_to_process]}")

    num_receivers = None
    for fpath in files_to_process:
         try:
             with open(fpath, 'rb') as f: version = np.lib.format.read_magic(f); shape, _, _ = np.lib.format._read_array_header(f, version)
             if len(shape) == 4: num_receivers = shape[3]; print(f"  Determined dimensions from {fpath.name}: Receivers={num_receivers}"); break
             else: print(f"  Warning: Skipping {fpath.name} for dim check (not 4D).")
         except Exception as e: print(f"  Warning: Could not read header {fpath.name}: {e}")

    if num_receivers is None: print("Error: Could not determine receiver count. Aborting."); return None

    sum_per_receiver = np.zeros(num_receivers, dtype=np.float64)
    sum_sq_per_receiver = np.zeros(num_receivers, dtype=np.float64)
    total_count_per_receiver = np.zeros(num_receivers, dtype=np.int64)
    files_actually_processed = 0

    for file_path in files_to_process:
        try:
            seismic_batch = np.load(file_path, mmap_mode='r')
            # Seismic data is expected to be 4D (samples, sources, time, receivers)
            # No squeeze needed here.
            if seismic_batch.ndim != 4 or seismic_batch.shape[3] != num_receivers:
                 print(f"  Warning: Skipping {file_path.name} due to shape {seismic_batch.shape}.")
                 continue
            if seismic_batch.shape[0] == 0: continue

            if not np.all(np.isfinite(seismic_batch)):
                print(f"  Warning: Non-finite values found in {file_path.name}. Skipping this file.")
                continue

            batch_sum = np.sum(seismic_batch, axis=(0, 1, 2), dtype=np.float64)
            batch_sum_sq = np.sum(seismic_batch**2, axis=(0, 1, 2), dtype=np.float64)
            batch_count = seismic_batch.shape[0] * seismic_batch.shape[1] * seismic_batch.shape[2]

            sum_per_receiver += batch_sum
            sum_sq_per_receiver += batch_sum_sq
            total_count_per_receiver += batch_count
            files_actually_processed += 1

        except Exception as e:
            print(f"  âœ˜ FAIL {family_name}: Error processing file {file_path}: {e}")
            traceback.print_exc()
            continue

    if files_actually_processed == 0 or np.sum(total_count_per_receiver) == 0:
        print("No data successfully processed. Cannot calculate variance.")
        return None

    valid_counts_mask = total_count_per_receiver >= 2
    variance_per_receiver = np.zeros(num_receivers, dtype=np.float64)
    if np.any(valid_counts_mask):
         mean_per_receiver_valid = sum_per_receiver[valid_counts_mask] / total_count_per_receiver[valid_counts_mask]
         mean_sq_per_receiver_valid = sum_sq_per_receiver[valid_counts_mask] / total_count_per_receiver[valid_counts_mask]
         variance_per_receiver[valid_counts_mask] = mean_sq_per_receiver_valid - (mean_per_receiver_valid**2)
         variance_per_receiver = np.maximum(variance_per_receiver, 0)

    # Visualization
    plt.figure(figsize=(12, 6))
    receiver_indices = np.arange(num_receivers)
    plt.bar(receiver_indices, variance_per_receiver + 1e-12, width=0.8) # Add epsilon for log scale
    plt.title(f'Receiver Channel Variance ({family_name}, using {files_actually_processed} files)')
    plt.xlabel('Receiver Index'); plt.ylabel('Variance (across all samples, sources, time)')
    plt.yscale('log'); plt.grid(True, axis='y', alpha=0.4)

    low_threshold = 1e-9
    valid_variances = variance_per_receiver[valid_counts_mask & (variance_per_receiver > low_threshold)]
    if len(valid_variances) > 10: high_threshold = np.percentile(valid_variances, 98)
    elif len(valid_variances) > 0: high_threshold = np.max(valid_variances)
    else: high_threshold = low_threshold
    high_threshold = max(high_threshold, low_threshold * 10)

    dead_channels = receiver_indices[variance_per_receiver < low_threshold]
    hyper_channels = receiver_indices[variance_per_receiver > high_threshold]

    print(f"\nAnalysis Results:")
    median_variance = np.median(variance_per_receiver[valid_counts_mask]) if np.any(valid_counts_mask) else 0
    min_variance = np.min(variance_per_receiver) if len(variance_per_receiver)>0 else 0
    max_variance = np.max(variance_per_receiver) if len(variance_per_receiver)>0 else 0
    print(f"  Median Variance (active channels): {median_variance:.2e}")
    print(f"  Variance Range: [{min_variance:.2e}, {max_variance:.2e}]")
    print(f"  Potentially Dead Channels (Variance < {low_threshold:.1e}): {list(dead_channels)}")
    print(f"  Potentially Hyper-active Channels (Variance > {high_threshold:.2e} [~98th percentile]): {list(hyper_channels)}")

    if len(dead_channels) > 0: plt.scatter(dead_channels, np.full_like(dead_channels, 1e-11, dtype=float), color='red', zorder=5, label=f'Low Var (<{low_threshold:.1e})', s=50)
    if len(hyper_channels) > 0: plt.scatter(hyper_channels, np.maximum(variance_per_receiver[hyper_channels], low_threshold*10), color='orange', zorder=5, label=f'High Var (>P98)', s=50)
    if len(dead_channels) > 0 or len(hyper_channels) > 0: plt.legend()
    plt.show()
    print(f"Action: Review flagged channels ({list(dead_channels)} + {list(hyper_channels)}). Consider zeroing out or interpolating them before batching data to the GPU.")
    return variance_per_receiver

# Example Usage:
if 'TRAIN_DIR' in globals() and isinstance(TRAIN_DIR, Path):
    variance_results = analyze_receiver_channel_health(TRAIN_DIR, 'Style_A', file_type='data', max_files_to_load=10)
else:
     print("Cannot run receiver health analysis: TRAIN_DIR not correctly defined.")


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path
import traceback

# Assuming TRAIN_DIR (Path object) and find_first_file (modified) are defined

def analyze_shot_repeatability(base_dir, family_name, file_type='data', specific_batch=None,
                               source_index=0, receiver_index=35): # Use a mid receiver
    """Estimates shot-to-shot repeatability by correlating consecutive samples in a batch."""
    if not isinstance(base_dir, Path):
        print("CRITICAL ERROR: base_dir passed to analyze_shot_repeatability must be a Path object.")
        return None

    print("WARNING: This analysis ASSUMES samples i and i+1 in the batch are 'consecutive shots'.")
    print("         This might not be true for the dataset generation process.")

    seis_path = find_first_file(base_dir, family_name, file_type, specific_batch)
    if not seis_path:
        print(f"Skipping repeatability analysis for {family_name}, file not found.")
        return None

    print(f"\nAnalyzing shot-to-shot repeatability for Source {source_index}, Receiver {receiver_index}")
    print(f"File: {seis_path.name} ({family_name})")
    try:
        seismic_batch = np.load(seis_path, mmap_mode='r')
        # Seismic data is expected to be 4D (samples, sources, time, receivers)
        # No squeeze needed here.
        if seismic_batch.ndim != 4:
             print(f"  âœ˜ FAIL {family_name}: Expected 4D seismic array, got {seismic_batch.shape}. Skipping.")
             return None

        num_samples, num_sources, num_timesteps, num_receivers = seismic_batch.shape
        if num_samples < 2:
             print(f"  âœ˜ FAIL {family_name}: Need at least 2 samples for comparison. Found {num_samples}.")
             return None

        if source_index >= num_sources: print(f"  Warning: Source index {source_index} invalid. Using 0."); source_index = 0
        if receiver_index >= num_receivers: print(f"  Warning: Receiver index {receiver_index} invalid. Using 0."); receiver_index = 0
        print(f"  Using Source Index: {source_index}, Receiver Index: {receiver_index}")

        correlations = []
        for i in range(num_samples - 1):
            trace1 = seismic_batch[i, source_index, :, receiver_index]
            trace2 = seismic_batch[i + 1, source_index, :, receiver_index]

            if not np.all(np.isfinite(trace1)) or not np.all(np.isfinite(trace2)): corr = np.nan
            elif np.ptp(trace1) < 1e-9 or np.ptp(trace2) < 1e-9: corr = np.nan
            else:
                try: corr, _ = pearsonr(trace1, trace2); corr = corr if np.isfinite(corr) else np.nan
                except ValueError as e: print(f"  ValueError: pair {i}-{i+1}: {e}. Skip."); corr = np.nan
            if not np.isnan(corr): correlations.append(corr)

        if not correlations:
            print("  No valid correlations calculated.")
            return None

        correlations = np.array(correlations)

        # Visualization
        plt.figure(figsize=(10, 6))
        plt.hist(correlations, bins=50, range=(-1, 1), alpha=0.8, density=True)
        plt.title(f'Shot-to-Shot Repeatability (Correlation of Consecutive Traces)\n{family_name} - Src {source_index}, Rec {receiver_index}')
        plt.xlabel('Pearson Correlation Coefficient'); plt.ylabel('Density')
        plt.grid(True, alpha=0.4)
        mean_corr = np.mean(correlations); median_corr = np.median(correlations)
        plt.axvline(mean_corr, color='r', linestyle='--', label=f'Mean Corr: {mean_corr:.2f}')
        plt.axvline(median_corr, color='g', linestyle=':', label=f'Median Corr: {median_corr:.2f}')
        plt.legend(); plt.xlim(-1.05, 1.05); plt.show()

        print(f"\nAnalysis complete for {len(correlations)} pairs.")
        print(f"Correlation Statistics:\n  Min: {np.min(correlations):.2f}, Max: {np.max(correlations):.2f}\n  Mean: {mean_corr:.2f}, Median: {median_corr:.2f}")
        if median_corr < 0.7: print("Action: Repeatability low/moderate. Be cautious using consecutive shots for contrastive tasks.\n        Consider treating shots independently or using unstable shots for augmentation.")
        else: print("Action: Repeatability relatively high. Consecutive shots might be suitable for contrastive pairs.\n        Mix stable shots for contrastive pairs, potentially separate unstable ones for augmentation.")
        return correlations

    except Exception as e:
        print(f"  âœ˜ FAIL {family_name}: Error processing {seis_path}: {e}")
        traceback.print_exc()
        return None

# Example Usage:
if 'TRAIN_DIR' in globals() and isinstance(TRAIN_DIR, Path):
    repeatability_results = analyze_shot_repeatability(TRAIN_DIR, 'CurveVel_A', file_type='data', source_index=1, receiver_index=50)
else:
     print("Cannot run repeatability analysis: TRAIN_DIR not correctly defined.")


# %%code
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def evaluate_velocity_scaling(families_to_include, file_type='model'):
    """Applies different scalers to velocity data and compares distributions."""
    all_velocities = []
    print(f"Loading velocities from families: {families_to_include}")
    for family_name in families_to_include:
        vel_path = find_first_file(family_name, file_type=file_type)
        if not vel_path: continue
        try:
            # Load and flatten; append to list
            vel_data = np.load(vel_path, mmap_mode='r').flatten()
            all_velocities.append(vel_data)
        except Exception as e:
            print(f"Error loading {vel_path}: {e}")

    if not all_velocities:
        print("No velocity data loaded. Aborting.")
        return

    # Concatenate all velocity data into a single 1D array
    combined_velocities = np.concatenate(all_velocities).reshape(-1, 1) # Reshape for scaler
    print(f"Combined data shape for scaling: {combined_velocities.shape}")

    # Initialize scalers
    scalers = {
        "MinMax": MinMaxScaler(),
        "Standard": StandardScaler(),
        "Robust": RobustScaler() # Uses median and IQR, good for outliers
    }

    scaled_data = {"Original": combined_velocities.flatten()} # Keep original for comparison
    for name, scaler in scalers.items():
        print(f"Applying {name} scaler...")
        scaled_data[name] = scaler.fit_transform(combined_velocities).flatten()

    # Plot histograms of scaled data
    num_scalers = len(scaled_data)
    fig, axs = plt.subplots(1, num_scalers, figsize=(6 * num_scalers, 5), sharey=True)
    fig.suptitle('Comparison of Velocity Scaling Methods', fontsize=16)

    for i, (name, data) in enumerate(scaled_data.items()):
        axs[i].hist(data, bins=100, density=True, alpha=0.8)
        axs[i].set_title(name)
        axs[i].set_xlabel('Scaled Velocity Value')
        if i == 0:
            axs[i].set_ylabel('Density')
        axs[i].grid(True, alpha=0.5)
        # Optionally set xlim for better comparison, e.g., (-3, 3) for Standard/Robust
        # if name != "Original": axs[i].set_xlim(-4, 4)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Observe:
    # - MinMaxScaler: Puts everything between 0-1, outliers squash the main distribution.
    # - StandardScaler: Centers data around 0, std dev 1. Outliers can still be far out.
    # - RobustScaler: Less affected by outliers, main distribution might be better represented near 0.

# Example Usage: Use diverse families
diverse_families = ['FlatVel_A', 'CurveFault_B', 'Style_B']
evaluate_velocity_scaling(diverse_families, file_type='model')


## Try twoscale
# %%code
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np
import matplotlib.pyplot as plt

def evaluate_velocity_scaling(families_to_include, file_type='model'):
    """
    For **each** family: 
      â€¢ load its velocity cube              (shape â‰ˆ [Z, Y, X] or flattened)  
      â€¢ apply three common scalers *independently*  
      â€¢ plot the before/after histograms sideâ€‘byâ€‘side so we can see how
        the scaler reshapes THAT family's distribution.
    
    Nothing is aggregated across families â€“ every family keeps its own scale.
    """
    scalers = {
        "Minâ€‘Max\n[0,â€¯1]": MinMaxScaler(),
        "Standard\n(Î¼=0,â€¯Ïƒ=1)": StandardScaler(),
        "Robust\n(median/IQR)": RobustScaler()
    }
    
    for fam in families_to_include:
        vel_path = find_first_file(fam, file_type=file_type)
        if not vel_path:
            print(f"[skip] could not locate {fam} ({file_type})")
            continue
        
        try:
            vel = np.load(vel_path, mmap_mode='r').astype(np.float32).reshape(-1, 1)
        except Exception as e:
            print(f"[error] loading {vel_path}: {e}")
            continue
        
        # --- scale -------------------------------------------------------------
        scaled = {"Original": vel.flatten()}
        for name, scaler in scalers.items():
            scaled[name] = scaler.fit_transform(vel).flatten()
        
        # --- plot --------------------------------------------------------------
        cols = len(scaled)
        fig, axs = plt.subplots(1, cols, figsize=(5 * cols, 4), sharey=True)
        fig.suptitle(f'Velocity Scaling â€“ {fam}', fontsize=14)
        
        for ax, (name, data) in zip(axs, scaled.items()):
            ax.hist(data, bins=100, density=True, alpha=0.85, color='steelblue')
            ax.set_title(name)
            ax.set_xlabel('Scaled value')
            ax.grid(alpha=0.4)
            if name == "Original":
                ax.set_ylabel('Density')
            else:
                # zoom in for scaled views
                if "Standard" in name or "Robust" in name:
                    ax.set_xlim(-4, 4)
        
        plt.tight_layout()
        plt.show()

# -------------------------------------------------------------------------
diverse_families = ['FlatVel_A', 'CurveFault_B', 'Style_B']
evaluate_velocity_scaling(diverse_families, file_type='model')




# %%code
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def evaluate_seismic_scaling_single(family_name, file_type='data', sample_index=0):
    """Compare three scaling strategies for one family & one gather."""
    seis_path = find_first_file(family_name, file_type)
    if seis_path is None:
        print(f"[skip] no file found for {family_name}")
        return

    print(f"\nEvaluating seismic scaling â€“ {family_name}  ({seis_path.name})")
    try:
        # expected shape (N_samples, N_src, N_time, N_rcv)
        batch = np.load(seis_path, mmap_mode='r')
        if batch.ndim != 4:
            print(f"[warn] expected 4â€‘D, got {batch.shape}; skipping.")
            return
        if sample_index >= batch.shape[0]:
            sample_index = 0

        sample = batch[sample_index]        # (src, t, r)
        S, T, R = sample.shape

        # ---------- 1) Global Standard scaler ---------------------------
        scaler = StandardScaler()
        subset = batch[::10].reshape(-1, 1)   # subsample for fit speed
        scaler.fit(subset)
        sample_global = scaler.transform(sample.reshape(-1, 1)).reshape(S, T, R)

        # ---------- 2) Perâ€‘gather MaxAbs -------------------------------
        max_abs = np.max(np.abs(sample))
        sample_gather = sample / max_abs if max_abs > 0 else sample

        # ---------- 3) Perâ€‘trace Standard ------------------------------
        sample_trace = sample.copy()
        for s in range(S):
            for r in range(R):
                tr = sample_trace[s, :, r]
                m, sd = tr.mean(), tr.std()
                if sd > 1e-6:
                    sample_trace[s, :, r] = (tr - m) / sd
                else:
                    sample_trace[s, :, r] = tr - m

        # ---------- quick stats ----------------------------------------
        print(f"Global std  -> Î¼={sample_global.mean():.3f}, Ïƒ={sample_global.std():.3f}")
        print(f"Perâ€‘gather  -> min={sample_gather.min():.2f}, max={sample_gather.max():.2f}")
        print(f"Perâ€‘trace   -> Î¼â‰ˆ{sample_trace.mean():.3f}, Ïƒâ‰ˆ{sample_trace.std():.3f}")

        # ---------- plot first source ----------------------------------
        src = 0
        figs = {
            "Original": sample[src],
            "Global Std": sample_global[src],
            "Perâ€‘Gather MaxAbs": sample_gather[src],
            "Perâ€‘Trace Std": sample_trace[src],
        }

        fig, axs = plt.subplots(1, 4, figsize=(22, 4), sharey=True)
        fig.suptitle(f'Seismic Scaling â€“ {family_name}  (sample {sample_index}, src {src})')

        for ax, (ttl, dat) in zip(axs, figs.items()):
            vmin, vmax = np.percentile(dat, [1, 99])
            im = ax.imshow(dat, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax)
            ax.set_title(ttl, fontsize=10)
            ax.set_xlabel('Receiver')
            if ttl == "Original":
                ax.set_ylabel('Time')
            fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"[error] {e}")


def evaluate_seismic_scaling(families, file_type='data', sample_index=0):
    """Iterate over families, calling the singleâ€‘family routine."""
    for fam in families:
        evaluate_seismic_scaling_single(fam, file_type=file_type, sample_index=sample_index)


# ------------------------------------------------------------------
families = ['FlatVel_A', 'CurveFault_B', 'Style_B']
evaluate_seismic_scaling(families, file_type='data', sample_index=10)




import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from pathlib import Path
import os
import traceback

# Assuming TRAIN_DIR (Path object) and find_first_file (modified) are defined

def analyze_curvature_dip_angles(base_dir, curve_families, gradient_threshold_percentile=75):
    """Estimates and plots the distribution of dip angles in Curve family velocity maps."""
    if not isinstance(base_dir, Path):
        print("CRITICAL ERROR: base_dir passed to analyze_curvature_dip_angles must be a Path object.")
        return

    all_dip_angles = []
    print(f"Analyzing dip angles for Curve families: {curve_families}")

    for family_name in curve_families:
        if not (base_dir / family_name).is_dir():
            print(f"  Skipping {family_name}, directory not found in {base_dir}.")
            continue

        file_type = 'vel' if 'Fault' in family_name else 'model'
        vel_path = find_first_file(base_dir, family_name, file_type=file_type)
        if not vel_path:
            print(f"  Skipping {family_name}, velocity file not found.")
            continue

        try:
            vel_batch_raw = np.load(vel_path, mmap_mode='r')
            vel_batch = np.squeeze(vel_batch_raw) # Squeeze potential singleton dimension

            if vel_batch.ndim != 3:
                 print(f"  âœ˜ FAIL {family_name}: Expected 3D after squeeze from {vel_batch_raw.shape}, got {vel_batch.ndim}D â†’ skipping family")
                 continue
            if vel_batch.shape[0] == 0:
                 print(f"  âœ˜ FAIL {family_name}: Batch is empty â†’ skipping family")
                 continue

            family_angles = []
            for i in range(vel_batch.shape[0]):
                vel_map = vel_batch[i] # Should be 2D

                if vel_map.ndim != 2: continue # Safeguard

                if not np.all(np.isfinite(vel_map)): continue

                grad_y = ndimage.sobel(vel_map, axis=0)
                grad_x = ndimage.sobel(vel_map, axis=1)
                magnitude = np.sqrt(grad_y**2 + grad_x**2)

                if np.ptp(magnitude) < 1e-9: continue

                threshold = np.percentile(magnitude, gradient_threshold_percentile)
                significant_mask = magnitude > threshold
                if not np.any(significant_mask): continue

                gy_sig = grad_y[significant_mask]
                gx_sig = grad_x[significant_mask]
                dip_angles_deg = np.degrees(np.arctan(np.abs(gy_sig / (gx_sig + 1e-12))))
                family_angles.extend(dip_angles_deg)

            all_dip_angles.extend(family_angles)

        except Exception as e:
            print(f"  âœ˜ FAIL {family_name}: Error processing {vel_path}: {e} â†’ skipping family")
            traceback.print_exc()

    if not all_dip_angles:
        print("No dip angles collected. Cannot plot histogram.")
        return

    # Plot histogram
    plt.figure(figsize=(10, 6))
    all_dip_angles = np.array(all_dip_angles)
    plt.hist(all_dip_angles, bins=90, range=(0, 90), density=True, alpha=0.8)
    plt.title(f'Distribution of Estimated Dip Angles (Steepness) in Curve Families\n(Gradient Mag > {gradient_threshold_percentile}th percentile)')
    plt.xlabel('Structure Dip Angle relative to Horizontal (degrees)')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.4)
    steep_threshold = 25
    fraction_steep = np.sum(all_dip_angles >= steep_threshold) / len(all_dip_angles) if len(all_dip_angles) > 0 else 0
    plt.axvline(steep_threshold, color='r', linestyle='--', label=f'>{steep_threshold}Â° threshold')
    plt.legend()
    plt.show()
    print(f"\nFraction of significant structure points with dip >= {steep_threshold}Â°: {fraction_steep:.2%}")
    if fraction_steep > 0.1: print(f"Action: Dips frequently exceed {steep_threshold}Â°. Consider wider spatial CNN kernels (e.g., 5x5 or 7x7) or direction-aware filters.")
    else: print(f"Action: Steep dips (>= {steep_threshold}Â°) are relatively rare. Standard kernels (e.g., 3x3) might suffice.")


# Example Usage:
if 'TRAIN_DIR' in globals() and isinstance(TRAIN_DIR, Path) and TRAIN_DIR.is_dir():
    try:
         curve_family_list = [d.name for d in TRAIN_DIR.iterdir() if d.is_dir() and d.name.startswith('Curve')]
         if not curve_family_list:
             print("Warning: No 'Curve...' families found automatically in TRAIN_DIR.")
             curve_family_list = ['CurveVel_A', 'CurveVel_B', 'CurveFault_A', 'CurveFault_B'] # Manual fallback
         analyze_curvature_dip_angles(TRAIN_DIR, curve_family_list)
    except Exception as e:
        print(f"Error listing directories in TRAIN_DIR ({TRAIN_DIR}): {e}")
else:
    print("Cannot run curvature analysis: TRAIN_DIR not correctly defined as a Path object or does not exist.")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import random
import traceback # Import for detailed error logging

# Assuming TRAIN_DIR (Path object), TEST_DIR (optional Path object), and find_first_file (modified) are defined

def analyze_pca_compressibility(target_dir, family_name, data_source_label='train', n_samples_max=500, target_variance=0.95):
    """Performs PCA on flattened velocity maps for a family and finds components for target variance."""
    if not isinstance(target_dir, Path):
        print(f"CRITICAL ERROR: target_dir passed to analyze_pca_compressibility must be a Path object for {family_name}.")
        return None, None

    file_type = 'vel' if 'Fault' in family_name else 'model'
    vel_path = find_first_file(target_dir, family_name, file_type=file_type)

    if not vel_path or not vel_path.exists():
        print(f"  Skipping {family_name} ({data_source_label}), velocity file not found in {target_dir / family_name}.")
        return None, None

    try:
        vel_batch_raw = np.load(vel_path, mmap_mode='r')
        # REMOVED: print(f"â–¶ ENTER  {family_name:<15} ({data_source_label})")
        # REMOVED: print(f"   â€¢ vel_path: {vel_path}")
        # REMOVED: print(f"   â€¢ vel_batch_raw.shape: {vel_batch_raw.shape} dtype: {vel_batch_raw.dtype}")

        vel_batch = np.squeeze(vel_batch_raw)
        # REMOVED: print(f"   â€¢ vel_batch_squeezed.shape: {vel_batch.shape}")

        if vel_batch.ndim != 3:
             print(f"  âœ˜ FAIL {family_name} ({data_source_label}): Expected 3-D after squeeze from {vel_batch_raw.shape}, got {vel_batch.ndim}D â†’ skipping") # Print shape only on error
             return None, None
        if vel_batch.shape[0] == 0:
             print(f"  âœ˜ FAIL {family_name} ({data_source_label}): Batch is empty â†’ skipping")
             return None, None

        num_available = vel_batch.shape[0]
        n_samples = min(num_available, n_samples_max)

        if data_source_label == 'test' and num_available > n_samples:
             indices = random.sample(range(num_available), n_samples)
             vel_samples = vel_batch[indices]
             # print(f"  â†’ Using {n_samples} random samples from test batch {vel_path.name}.") # Optional: Keep for info
        else:
             vel_samples = vel_batch[:n_samples]
             # print(f"  â†’ Using {n_samples} samples from {data_source_label} batch {vel_path.name}.") # Optional: Keep for info


        if not np.all(np.isfinite(vel_samples)):
            print(f"  âœ˜ FAIL {family_name} ({data_source_label}): Non-finite values detected â†’ skipping PCA.")
            return None, None

        H, W = vel_samples.shape[1], vel_samples.shape[2]
        flattened_vel = vel_samples.reshape(n_samples, H * W)

        scaler = StandardScaler()
        flattened_vel_scaled = scaler.fit_transform(flattened_vel)

        n_pca_components = min(n_samples, H * W)
        if n_pca_components < 1:
             print(f"  âœ˜ FAIL {family_name} ({data_source_label}): Insufficient data for PCA (n_components={n_pca_components}) â†’ skipping")
             return None, None

        pca = PCA(n_components=n_pca_components)
        pca.fit(flattened_vel_scaled)

        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        if cumulative_variance[-1] < target_variance:
             components_needed = n_pca_components
             print(f"  âœ“ {family_name} ({data_source_label}): Target {target_variance:.1%} variance not reached. Using all {components_needed} components ({cumulative_variance[-1]:.1%} explained).")
        else:
             components_needed = np.argmax(cumulative_variance >= target_variance) + 1
             print(f"  âœ“ {family_name} ({data_source_label}): {components_needed} components explain {cumulative_variance[components_needed-1]:.1%} variance.")


        return components_needed, cumulative_variance

    except FileNotFoundError:
         print(f"  âœ˜ FAIL {family_name} ({data_source_label}): File not found error for: {vel_path} â†’ skipping")
         return None, None
    except Exception as e:
        print(f"  âœ˜ FAIL {family_name} ({data_source_label}) processing {vel_path}: {e} â†’ skipping")
        traceback.print_exc()
        return None, None

# --- Execution (No changes needed here, relies on corrected function) ---
families_for_pca = ['FlatVel_A', 'CurveVel_B', 'FlatFault_A', 'CurveFault_B', 'Style_A', 'Style_B']
results_pca = {}
max_components_overall_train = 0
max_components_overall_test = 0

print("\n--- Analyzing PCA Compressibility (Train Data) ---")
if 'TRAIN_DIR' in globals() and isinstance(TRAIN_DIR, Path) and TRAIN_DIR.is_dir():
    for family in families_for_pca:
        comps, curve = analyze_pca_compressibility(TRAIN_DIR, family, data_source_label='train')
        if comps is not None:
            results_pca[f'{family}_train'] = {'components': comps, 'curve': curve}
            max_components_overall_train = max(max_components_overall_train, comps)
else:
    print("TRAIN_DIR not valid. Skipping Train PCA analysis.")

print("\n--- Analyzing PCA Compressibility (Test Data Subset) ---")
# Check TEST_DIR type and existence
if 'TEST_DIR' in globals() and TEST_DIR is not None and isinstance(TEST_DIR, Path) and TEST_DIR.is_dir():
     print("Analyzing Hypothetical Test Velocity Data (if files exist)...")
     # Adapt this list if test families are named differently
     test_families = families_for_pca
     for family in test_families:
         comps, curve = analyze_pca_compressibility(TEST_DIR, family, data_source_label='test', n_samples_max=100)
         if comps is not None:
             results_pca[f'{family}_test'] = {'components': comps, 'curve': curve}
             max_components_overall_test = max(max_components_overall_test, comps)
else:
    print("Skipping PCA analysis on test data (TEST_DIR not set, not Path, does not exist, or test velocity unavailable).")


# --- Visualization (No changes needed here) ---
if results_pca:
    plt.figure(figsize=(12, 7))
    max_comps_plot = 0
    plotted_something = False
    for name, result in results_pca.items():
        if result is None or 'curve' not in result or result['curve'] is None or len(result['curve'])==0:
            continue
        curve = result['curve']
        label = f"{name} ({result['components']} comps for 95%)"
        plt.plot(range(1, len(curve) + 1), curve, marker='.', linestyle='-', label=label)
        max_comps_plot = max(max_comps_plot, len(curve))
        plotted_something = True

    if plotted_something:
        plt.axhline(0.95, color='r', linestyle='--', label='95% Variance Threshold')
        plt.title('Family-wise PCA Cumulative Explained Variance (Velocity Maps)')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.4)
        plt.xlim(0, min(max_comps_plot + 1, 150))
        plt.ylim(0, 1.05)
        plt.show()

        print(f"\nAcross analyzed train families, max components for 95% variance: {max_components_overall_train}")
        if max_components_overall_test > 0:
             print(f"Across analyzed test families, max components for 95% variance: {max_components_overall_test}")
        print("Action: Consider sizing latent space bottlenecks (e.g., in Autoencoders) based on the relevant maximum.")
    else:
         print("\nNo valid PCA results to plot.")
else:
    print("\nNo PCA results generated.")


# %%code
# Question 9: Do simple seismic attributes show clear correlations with velocity features?
from scipy.signal import hilbert

def analyze_attribute_velocity_correlation(family_name, sample_index=0, source_index=0, receiver_index=35): # Mid receiver
    """Calculates seismic envelope and compares with velocity profile. NOTE: Requires careful interpretation due to Time-Depth Mismatch."""
    # Adjust file_type based on family convention
    if 'Fault' in family_name:
        file_type_seis='seis'
        file_type_vel='vel'
    elif 'Vel' in family_name or 'Style' in family_name:
        file_type_seis='data'
        file_type_vel='model'
    else: # Default guess
        file_type_seis='data'
        file_type_vel='model'

    seis_path = find_first_file(family_name, file_type_seis)
    vel_path = find_first_file(family_name, file_type_vel)

    if not seis_path or not vel_path:
        print(f"Missing seismic (type: {file_type_seis}, path: {seis_path}) or "
              f"velocity (type: {file_type_vel}, path: {vel_path}) file. Skipping.")
        return

    print(f"Analyzing attribute correlation for sample {sample_index} in: {family_name}")
    print(f"  Seismic file: {seis_path.name}")
    print(f"  Velocity file: {vel_path.name}")
    print(f"  Using Source Index: {source_index}, Receiver Index: {receiver_index}")
    print("  WARNING: Visual comparison assumes approximate vertical alignment between seismic time and velocity depth. This is inaccurate and for illustration only.")

    try:
        seismic_batch = np.load(seis_path, mmap_mode='r')
        velocity_batch = np.load(vel_path, mmap_mode='r')

        # --- Dimension Checks ---
        if seismic_batch.ndim != 4:
             print(f"  Error: Expected 4D seismic batch, got {seismic_batch.ndim}D. Skipping.")
             return
        if velocity_batch.ndim != 3:
             print(f"  Error: Expected 3D velocity batch, got {velocity_batch.ndim}D. Skipping.")
             return

        num_samples_s, num_sources, num_timesteps, num_receivers = seismic_batch.shape
        num_samples_v, num_depth_pixels, num_width_pixels = velocity_batch.shape

        if num_samples_s == 0 or num_samples_v == 0:
             print("  Error: One or both batches contain 0 samples.")
             return

        # --- Index Validation ---
        if sample_index >= num_samples_s or sample_index >= num_samples_v:
             print(f"  Warning: Sample index {sample_index} out of bounds. Using 0.")
             sample_index = 0
        if source_index >= num_sources:
             print(f"  Warning: Source index {source_index} out of bounds. Using 0.")
             source_index = 0
        if receiver_index >= num_receivers:
             print(f"  Warning: Receiver index {receiver_index} out of bounds for seismic. Using 0.")
             receiver_index = 0
        # Use same index for velocity width, adjust if out of bounds
        receiver_index_vel = receiver_index
        if receiver_index_vel >= num_width_pixels:
             print(f"  Warning: Receiver index {receiver_index_vel} out of bounds for velocity width. Using {num_width_pixels-1}.")
             receiver_index_vel = num_width_pixels-1

        # Select seismic trace and velocity profile
        seismic_trace = seismic_batch[sample_index, source_index, :, receiver_index]
        velocity_profile = velocity_batch[sample_index, :, receiver_index_vel]

        if seismic_trace.size == 0 or velocity_profile.size == 0:
             print("  Error: Selected seismic trace or velocity profile is empty.")
             return

        # Calculate envelope (instantaneous amplitude)
        analytic_signal = hilbert(seismic_trace)
        envelope = np.abs(analytic_signal)

        # --- Visualization ---
        fig, ax1 = plt.subplots(figsize=(8, 10))
        fig.suptitle(f'Seismic Trace/Envelope vs. Velocity Profile (Sample {sample_index}, Src {source_index}, Rec {receiver_index})\n'
                     f'{family_name} - Approx. Time/Depth Alignment', y=0.98)

        color = 'tab:red'
        ax1.set_xlabel('Amplitude / Envelope')
        time_axis = np.arange(num_timesteps)
        ax1.plot(seismic_trace, time_axis, color=color, label='Seismic Trace', alpha=0.6)
        ax1.plot(envelope, time_axis, color='darkred', label='Envelope', linewidth=1.5)
        ax1.tick_params(axis='x', labelcolor=color)
        ax1.set_ylabel('Time Step Index')
        ax1.invert_yaxis() # Time increases downwards typically

        ax2 = ax1.twiny()  # instantiate a second axes that shares the same y-axis

        color = 'tab:blue'
        ax2.set_xlabel('Velocity (m/s)', color=color)
        # Crude linear mapping of depth pixels to time axis range
        depth_axis_mapped_to_time = np.linspace(0, num_timesteps - 1, num_depth_pixels)
        ax2.plot(velocity_profile, depth_axis_mapped_to_time, color=color, label='Velocity Profile', linestyle='--')
        ax2.tick_params(axis='x', labelcolor=color)

        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')

        plt.grid(True, alpha=0.4)
        plt.show()

        print("Reminder: Direct quantitative correlation requires accurate Time-Depth Conversion.")

    except FileNotFoundError as e:
        print(f"  Error: File not found - {e}")
    except Exception as e:
        print(f"  An error occurred processing sample {sample_index} from {family_name}: {e}")
        import traceback
        traceback.print_exc()

# Example usage:
analyze_attribute_velocity_correlation('CurveVel_A', sample_index=20, receiver_index=40)
analyze_attribute_velocity_correlation('FlatFault_B', sample_index=30, receiver_index=30)


# %%code
# Question 10: Would augmenting the training data improve model generalization?
import scipy.ndimage as ndi

def demonstrate_augmentations(family_name='FlatVel_A', sample_index=0):
    """Demonstrates simple augmentation on velocity and seismic data."""
    # Adjust file_type based on family convention
    if 'Fault' in family_name:
        file_type_seis='seis'
        file_type_vel='vel'
    else:
        file_type_seis='data'
        file_type_vel='model'

    vel_path = find_first_file(family_name, file_type_vel)
    seis_path = find_first_file(family_name, file_type_seis)

    if not vel_path or not seis_path:
        print(f"Missing velocity (type: {file_type_vel}, path: {vel_path}) or "
              f"seismic (type: {file_type_seis}, path: {seis_path}) file. Skipping.")
        return

    print(f"Demonstrating augmentations for sample {sample_index} from {family_name}")
    try:
        velocity_batch = np.load(vel_path, mmap_mode='r')
        seismic_batch = np.load(seis_path, mmap_mode='r')

        # Basic dimension checks
        if velocity_batch.ndim != 3 or seismic_batch.ndim != 4:
             print("  Error: Incorrect dimensions for velocity or seismic batch. Skipping.")
             return
        if velocity_batch.shape[0] == 0 or seismic_batch.shape[0] == 0:
             print("  Error: Empty batch found. Skipping.")
             return

        # Index validation
        if sample_index >= velocity_batch.shape[0] or sample_index >= seismic_batch.shape[0]:
             print(f"  Warning: Sample index {sample_index} out of bounds. Using 0.")
             sample_index = 0

        # --- Velocity Augmentation ---
        original_vel = velocity_batch[sample_index].copy() # Shape (70, 70)
        if original_vel.ndim != 2:
             print("  Error: Velocity sample is not 2D. Skipping velocity augmentation.")
             original_vel = None # Mark as unavailable

        if original_vel is not None:
            # Aug 1: Horizontal Flip
            flipped_vel = np.fliplr(original_vel)
            # Aug 2: Random Shift
            shift_y, shift_x = np.random.randint(-5, 6, size=2)
            shifted_vel = ndi.shift(original_vel, (shift_y, shift_x), mode='reflect')
            # Aug 3: Add Gaussian Noise
            noise_std_vel = 50 # Adjust noise level (m/s)
            noisy_vel = original_vel + np.random.normal(0, noise_std_vel, original_vel.shape)
            vel_plots = {"Original": original_vel, "FlipLR": flipped_vel, f"Shift({shift_y},{shift_x})": shifted_vel, f"Noise(std={noise_std_vel})": noisy_vel}
            vmin_v, vmax_v = np.percentile(original_vel, [1, 99]) if original_vel.size > 0 else (0, 1)
        else:
            vel_plots = {}


        # --- Seismic Augmentation (on one source gather) ---
        source_index = 0
        num_sources = seismic_batch.shape[1]
        if source_index >= num_sources:
             print(f"  Warning: Source index {source_index} out of bounds for seismic. Using 0.")
             source_index = 0

        if num_sources > 0:
            original_seis_gather = seismic_batch[sample_index, source_index].copy() # Shape (1000, 70)
            if original_seis_gather.ndim == 2 and original_seis_gather.shape[0] > 0:
                # Aug 1: Add Gaussian Noise
                noise_std_seis = 0.05 * np.std(original_seis_gather) # Noise relative to data std dev
                noisy_seis = original_seis_gather + np.random.normal(0, noise_std_seis, original_seis_gather.shape)
                # Aug 2: Random Time Shift (applied to all traces)
                time_shift = np.random.randint(-20, 21)
                shifted_seis = np.roll(original_seis_gather, time_shift, axis=0)
                if time_shift > 0: shifted_seis[:time_shift, :] = 0 # Zero-pad start
                if time_shift < 0: shifted_seis[time_shift:, :] = 0 # Zero-pad end
                seis_plots = {"Original": original_seis_gather, f"Noise(std={noise_std_seis:.2f})": noisy_seis, f"TimeShift({time_shift})": shifted_seis}
                vmin_s, vmax_s = np.percentile(original_seis_gather, [1, 99]) if original_seis_gather.size > 0 else (-1, 1)
                # Ensure vmin/vmax differ
                if abs(vmax_s - vmin_s) < 1e-6: vmin_s, vmax_s = vmin_s - 0.5, vmax_s + 0.5
            else:
                 print("  Error: Seismic gather for augmentation is not 2D or is empty.")
                 seis_plots = {}
        else:
             print("  Error: Seismic batch has 0 sources. Skipping seismic augmentation.")
             seis_plots = {}


        # --- Visualization ---
        if vel_plots:
            fig_vel, axs_vel = plt.subplots(1, len(vel_plots), figsize=(5 * len(vel_plots), 5))
            if len(vel_plots) == 1: axs_vel = [axs_vel] # Make iterable
            fig_vel.suptitle(f'Velocity Augmentations (Sample {sample_index}, {family_name})', fontsize=16)
            for i, (name, data) in enumerate(vel_plots.items()):
                im = axs_vel[i].imshow(data, aspect='image', vmin=vmin_v, vmax=vmax_v)
                axs_vel[i].set_title(name)
                axs_vel[i].axis('off')
            # Add a single colorbar - tricky for multiple subplots, might need GridSpec
            # fig_vel.colorbar(im, ax=axs_vel.ravel().tolist(), shrink=0.7) # Simple attempt
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        if seis_plots:
            fig_seis, axs_seis = plt.subplots(1, len(seis_plots), figsize=(5 * len(seis_plots), 6))
            if len(seis_plots) == 1: axs_seis = [axs_seis] # Make iterable
            fig_seis.suptitle(f'Seismic Augmentations (Sample {sample_index}, Source {source_index}, {family_name})', fontsize=16)
            num_receivers = seis_plots.get("Original", np.zeros((1,1))).shape[1] # Get width for extent
            num_timesteps = seis_plots.get("Original", np.zeros((1,1))).shape[0] # Get height for extent
            for i, (name, data) in enumerate(seis_plots.items()):
                im = axs_seis[i].imshow(data, aspect='auto', cmap='seismic', vmin=vmin_s, vmax=vmax_s,
                                        extent=[0, num_receivers-1, num_timesteps-1, 0])
                axs_seis[i].set_title(name)
                axs_seis[i].set_xlabel("Receiver Index")
                if i == 0: axs_seis[i].set_ylabel("Time Step")
            # fig_seis.colorbar(im, ax=axs_seis.ravel().tolist(), shrink=0.6) # Simple attempt
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    except FileNotFoundError as e:
        print(f"  Error: File not found - {e}")
    except Exception as e:
        print(f"  Error during augmentation demonstration: {e}")
        import traceback
        traceback.print_exc()

# Example usage:
demonstrate_augmentations(family_name='CurveVel_B', sample_index=5)
demonstrate_augmentations(family_name='FlatFault_A', sample_index=10)


# %%code
import skimage.measure
import scipy.ndimage as ndi # Keep ndi for upsampling later

def simulate_cnn_effect_on_fault(family_name='CurveFault_B', file_type_vel='vel', sample_index=0):
    """Visualizes a fault and simulates blurring effect of pooling."""
    vel_path = find_first_file(family_name, file_type_vel)
    if not vel_path:
        print(f"Velocity file not found for family {family_name} with type {file_type_vel}. Skipping.")
        return

    print(f"Visualizing fault and simulated pooling effect: Sample {sample_index} from {family_name} ({vel_path.name})")
    try:
        velocity_batch = np.load(vel_path, mmap_mode='r') # Expect (500, 70, 70)

        # --- Input Dimension Check ---
        if velocity_batch.ndim != 3:
            print(f"Error: Expected 3D velocity batch (samples, H, W), got {velocity_batch.ndim}D with shape {velocity_batch.shape}. Skipping.")
            return
        # --- End Input Dimension Check ---

        if sample_index >= velocity_batch.shape[0]:
            print(f"Warning: sample_index {sample_index} out of bounds for batch size {velocity_batch.shape[0]}. Using 0.")
            sample_index = 0

        original_vel = velocity_batch[sample_index] # Expect (70, 70)

        # --- Check Velocity Sample Dimension ---
        if original_vel.ndim != 2:
            print(f"Error: Expected 2D velocity sample (H, W) for pooling, got {original_vel.ndim}D with shape {original_vel.shape}. Skipping pooling simulation.")
            pooled_vel_available = False
        else:
            # Simulate Max Pooling (2x2 kernel, stride 2)
            pool_size = (2, 2)
            # block_reduce expects block_size tuple length == input array ndim
            pooled_vel = skimage.measure.block_reduce(original_vel, block_size=pool_size, func=np.max)
            # Upsample back to original size for comparison (using nearest neighbor)
            # Calculate zoom factor needed
            zoom_factor = tuple(o / p for o, p in zip(original_vel.shape, pooled_vel.shape))
            upsampled_pooled_vel = ndi.zoom(pooled_vel, zoom=zoom_factor, order=0) # order=0 is nearest neighbor
            pooled_vel_available = True
        # --- End Check and Pooling ---


        # --- Visualization ---
        num_plots = 3 if pooled_vel_available else 2
        fig, axs = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
        if num_plots == 1: axs = [axs] # Make iterable
        fig.suptitle(f'Fault Visualization & Simulated Pooling Effect (Sample {sample_index}, {family_name})', fontsize=16)

        vmin, vmax = np.percentile(original_vel, [1, 99]) if original_vel.size > 0 else (0, 1)

        # Plot Original Velocity
        im0 = axs[0].imshow(original_vel, aspect='image', vmin=vmin, vmax=vmax) # Use aspect='image' for velocity maps
        axs[0].set_title('Original Velocity Map')
        axs[0].axis('off')
        fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04, label='m/s')

        # Plot Gradient Magnitude (if original is 2D)
        if original_vel.ndim == 2:
            grads_y, grads_x = np.gradient(original_vel)
            gradient_mag = np.sqrt(grads_y**2 + grads_x**2)
            im1 = axs[1].imshow(gradient_mag, aspect='image', cmap='hot')
            axs[1].set_title('Gradient Magnitude (Highlights Fault)')
            axs[1].axis('off')
            fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label='Gradient')
        else:
             axs[1].set_title('Gradient (Skipped - Input not 2D)')
             axs[1].axis('off')

        # Plot Pooled/Upsampled (if available)
        if pooled_vel_available:
            im2 = axs[2].imshow(upsampled_pooled_vel, aspect='image', vmin=vmin, vmax=vmax)
            axs[2].set_title(f'Simulated Max Pooling {pool_size} Effect')
            axs[2].axis('off')
            fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04, label='m/s')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        if pooled_vel_available:
            print("Observe how pooling (right panel) can blur/thicken the sharp edges of the fault seen in the original (left) and gradient (middle).")
        else:
            print("Pooling simulation skipped due to unexpected input dimensions.")


    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"An error occurred during visualization for sample {sample_index}: {e}")
        import traceback
        traceback.print_exc()


# Example usage: Use a fault family, ensure file_type_vel matches layout ('vel' for prefix)
simulate_cnn_effect_on_fault(family_name='CurveFault_B', file_type_vel='vel', sample_index=10)


# %%code
def visualize_style_vs_others(vel_family='FlatVel_A', fault_family='CurveFault_B', style_family='Style_B',
                               sample_index=0):
    """Visualizes sample velocity maps from Vel, Fault, and Style families."""

    # Define file types based on common patterns (adjust if needed)
    file_types = {
        vel_family: 'model',
        fault_family: 'vel', # Fault families often use 'vel' prefix
        style_family: 'model'
    }

    families = {
        "Vel": vel_family,
        "Fault": fault_family,
        "Style": style_family
    }

    paths = {}
    for name, family_name in families.items():
        file_type = file_types.get(family_name, 'model') # Default to 'model' if not specified
        paths[name] = find_first_file(family_name, file_type)

    images = {}
    titles = {}
    shapes = {}

    print(f"Loading samples for comparison (Index: {sample_index})")
    for name, path in paths.items():
        family_name = families[name] # Get the family name back
        if not path:
            print(f"Skipping {name} ({family_name}), file not found.")
            continue
        try:
            batch = np.load(path, mmap_mode='r')
            if batch.ndim != 3: # Expect (samples, H, W)
                print(f"Warning: Skipping {name} ({family_name}). Expected 3D batch, got {batch.ndim}D shape {batch.shape} from {path.name}.")
                continue

            current_index = sample_index if sample_index < batch.shape[0] else 0
            if sample_index >= batch.shape[0]:
                 print(f"Warning: Sample index {sample_index} out of bounds for {name} ({family_name}, size {batch.shape[0]}). Using index 0.")

            img = batch[current_index]
            if img.ndim != 2: # Expect (H, W)
                 print(f"Warning: Skipping {name} ({family_name}). Expected 2D sample, got {img.ndim}D shape {img.shape}.")
                 continue

            images[name] = img
            shapes[name] = img.shape
            titles[name] = f"{name} ({family_name})\nShape: {img.shape}" # Add shape to title
            print(f"  Loaded {name} ({family_name}) sample {current_index}, Shape: {img.shape}")

        except FileNotFoundError as e:
            print(f"Error loading {path}: {e}")
        except Exception as e:
            print(f"Generic error loading {path}: {e}")
            import traceback
            traceback.print_exc()


    if len(images) < 1:
        print("No valid images loaded to compare.")
        return
    elif len(images) < 2:
        print("Only one image loaded. Plotting it.")


    # --- Visualization ---
    num_images = len(images)
    fig, axs = plt.subplots(1, num_images, figsize=(6 * num_images, 6)) # Adjusted fig height slightly
    if num_images == 1: axs = [axs] # Make it iterable if only one subplot
    fig.suptitle(f'Velocity Map Comparison (Sample {sample_index})', fontsize=16, y=0.98) # Adjust title y-pos

    # --- Robust Global Color Scaling ---
    if num_images > 0:
        # Flatten all loaded images and concatenate them into a single 1D array
        all_flat_data = np.concatenate([img.flatten() for img in images.values()])
        if all_flat_data.size > 0:
             # Compute percentiles on the combined data
             vmin, vmax = np.percentile(all_flat_data, [1, 99])
             print(f"Global color scale (1st-99th percentile): vmin={vmin:.0f}, vmax={vmax:.0f}")
        else:
             vmin, vmax = 0, 1 # Fallback
    else:
        vmin, vmax = 0, 1 # Fallback

    # --- Plotting Loop ---
    i = 0
    for name, img in images.items():
        ax = axs[i]
        im = ax.imshow(img, aspect='image', vmin=vmin, vmax=vmax) # Use 'image' aspect ratio
        ax.set_title(titles[name])
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='m/s')
        i += 1

    # Adjust layout to prevent title overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust rect to make space for main title
    plt.show()

# Example usage:
visualize_style_vs_others(sample_index=25)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
from pathlib import Path
import traceback

# Assuming TRAIN_DIR (Path object) and find_first_file (modified) are defined

def visualize_tsne_embedding(base_dir, families_to_include, samples_per_family=100, use_pca=True, n_pca_components=50):
    """Creates a t-SNE embedding of velocity maps from multiple families."""
    if not isinstance(base_dir, Path):
        print("CRITICAL ERROR: base_dir passed to visualize_tsne_embedding must be a Path object.")
        return

    all_flattened_maps = []
    family_labels = []

    print("Loading data for t-SNE embedding...")
    for family_name in families_to_include:
        if not (base_dir / family_name).is_dir():
            print(f"  Skipping {family_name}, directory not found in {base_dir}.")
            continue

        file_type = 'vel' if 'Fault' in family_name else 'model'
        vel_path = find_first_file(base_dir, family_name, file_type=file_type)
        if not vel_path:
            print(f"  Skipping {family_name}, file not found.")
            continue

        try:
            vel_batch_raw = np.load(vel_path, mmap_mode='r')
            vel_batch = np.squeeze(vel_batch_raw) # Squeeze potential singleton dimension

            if vel_batch.ndim != 3:
                 print(f"  âœ˜ FAIL {family_name}: Expected 3D after squeeze from {vel_batch_raw.shape}, got {vel_batch.ndim}D â†’ skipping family")
                 continue

            num_available = vel_batch.shape[0]
            n_samples = min(num_available, samples_per_family)
            if n_samples == 0: continue

            vel_samples = vel_batch[:n_samples]

            is_finite_mask = np.all(np.isfinite(vel_samples.reshape(n_samples, -1)), axis=1)
            if not np.all(is_finite_mask):
                 n_finite = np.sum(is_finite_mask)
                 print(f"  Warning: Found non-finite values in {family_name}. Using {n_finite} finite samples.")
                 vel_samples = vel_samples[is_finite_mask]
                 n_samples = vel_samples.shape[0]
                 if n_samples == 0: continue

            H, W = vel_samples.shape[1], vel_samples.shape[2]
            flattened = vel_samples.reshape(n_samples, H * W)

            all_flattened_maps.append(flattened)
            family_labels.extend([family_name] * n_samples)
            print(f"  Loaded {n_samples} samples from {family_name}.")

        except Exception as e:
            print(f"  âœ˜ FAIL {family_name}: Error loading {vel_path}: {e}")
            traceback.print_exc()

    if not all_flattened_maps:
        print("No data loaded for t-SNE. Aborting.")
        return

    X = np.concatenate(all_flattened_maps, axis=0)
    y = np.array(family_labels)
    print(f"\nTotal samples for t-SNE: {X.shape[0]}")
    if X.shape[0] < 2: print("Need at least 2 samples for t-SNE."); return

    # --- Preprocessing ---
    print("Applying StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if np.any(np.isnan(X_scaled)): print("Warning: NaNs detected after scaling. Replacing NaNs with 0."); X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    if use_pca:
        print(f"Applying PCA to reduce dimensions to {n_pca_components}...")
        n_components = min(n_pca_components, X_scaled.shape[0], X_scaled.shape[1])
        if n_components < 2: print(f"Error: Cannot perform PCA with n_components={n_components}. Need >= 2."); return
        if n_components < n_pca_components: print(f"  Adjusted PCA components to {n_components}.")
        pca = PCA(n_components=n_components, random_state=42)
        X_reduced = pca.fit_transform(X_scaled)
        print(f"  PCA done. Explained variance by {n_components} components: {np.sum(pca.explained_variance_ratio_):.2%}")
    else:
        X_reduced = X_scaled
        print("Skipping PCA.")
    if X_reduced.shape[0] < 2: print("Not enough samples remaining after preprocessing for t-SNE."); return

    # --- t-SNE ---
    perplexity_value = min(30, X_reduced.shape[0] - 1)
    perplexity_value = max(1 if X_reduced.shape[0] < 6 else 5, perplexity_value)
    print(f"Applying t-SNE (perplexity={perplexity_value})... (this may take a while)")
    tsne = TSNE(n_components=2, perplexity=perplexity_value, n_iter=1000, learning_rate='auto', init='pca', random_state=42, n_jobs=-1)
    try: X_tsne = tsne.fit_transform(X_reduced); print("t-SNE embedding complete.")
    except ValueError as e: print(f"Error during t-SNE: {e}"); return

    # --- Visualization ---
    tsne_df = pd.DataFrame(data=X_tsne, columns=['TSNE1', 'TSNE2'])
    tsne_df['Family'] = y
    plt.figure(figsize=(12, 10))
    unique_families = sorted(tsne_df['Family'].unique())
    sns.scatterplot(x="TSNE1", y="TSNE2", hue="Family", hue_order=unique_families, palette=sns.color_palette("husl", len(unique_families)), data=tsne_df, legend="full", alpha=0.7, s=50)
    plt.title('t-SNE Embedding of Velocity Maps by Family')
    plt.xlabel('t-SNE Dimension 1'); plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, alpha=0.3); plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
    print("\nAnalysis:")
    print(" - Observe the scatter plot: Do families form distinct clusters or overlap significantly?")
    print(" - Separated clusters suggest family-specific models or curriculum learning might be beneficial.")
    print(" - Significant overlap suggests a single universal model may generalize well.")
    print("Action: Based on cluster separability, choose between a single U-Net vs. an ensemble/specialized architecture.")


# Example Usage: (No changes needed here, relies on corrected function)
families_for_tsne = ['FlatVel_A', 'FlatVel_B', 'CurveVel_A', 'CurveVel_B',
                     'FlatFault_A', 'FlatFault_B', 'CurveFault_A', 'CurveFault_B',
                     'Style_A', 'Style_B']
if 'TRAIN_DIR' in globals() and isinstance(TRAIN_DIR, Path):
    visualize_tsne_embedding(TRAIN_DIR, families_for_tsne, samples_per_family=150, use_pca=True, n_pca_components=60)
else:
     print("Cannot run t-SNE analysis: TRAIN_DIR not correctly defined.")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Velocityâ€‘family histograms
#   â€¢ Small 2â€‘D panels (Matplotlib, as before)
#   â€¢ NEW: interactive 3â€‘D overlay built with Matplotlibâ€™s mplot3d.bar3d
#     (works in Kaggle; no unavailable PlotlyÂ API calls)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import os, numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D   # registers 3â€‘D projection

# Paths & settings
TRAIN_DIR = "/kaggle/input/waveform-inversion/train_samples"
FAMILIES  = ["FlatVel_A", "CurveVel_A", "FlatFault_A",
             "CurveFault_A", "Style_A", "Style_B"]
BINS = 60

CAPTION = {
    "FlatVel_A"   : "Simple, flatâ€‘layered sediments; baseline reference.",
    "CurveVel_A"  : "Gently curved layers (no breaks); adds mild complexity.",
    "FlatFault_A" : "Flat layers cut by a fault (vertical break).",
    "CurveFault_A": "Curved layers *and* a fault; mixed difficulties.",
    "Style_A"     : "Organic patterns from styleâ€‘transfer images; broad velocity spread.",
    "Style_B"     : "Even wilder style patterns; largest velocity range."
}

def first_velocity_file(folder):
    root = os.path.join(TRAIN_DIR, folder)
    model_dir = os.path.join(root, "model")
    if os.path.isdir(model_dir):                 # twoâ€‘folder layout
        f = sorted([x for x in os.listdir(model_dir) if x.endswith(".npy")])[0]
        return os.path.join(model_dir, f)
    f = sorted([x for x in os.listdir(root)
                if x.startswith(("vel", "model")) and x.endswith(".npy")])[0]
    return os.path.join(root, f)

# â”€â”€ 1) Small 2â€‘D histogram panels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13
})

fig = plt.figure(figsize=(20, 24))
gs  = GridSpec(5, 3, height_ratios=[1, 1, 0.07, 0.0, 0.05],
               hspace=1.4, wspace=0.55)
fig.suptitle("Velocity Distributions of Synthetic Seismic Families",
             fontsize=24, y=0.97)

positions = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
for (r, c), fam in zip(positions, FAMILIES):
    ax = fig.add_subplot(gs[r, c])
    v  = np.load(first_velocity_file(fam), mmap_mode="r").ravel()
    ax.hist(v, bins=BINS, histtype="step", linewidth=1.8)
    ax.set_title(fam, pad=12)
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Count")
    ax.grid(alpha=.25, lw=.4)
    ax.text(0.5, -0.40, CAPTION[fam],
            transform=ax.transAxes,
            ha='center', va='top', fontsize=14, wrap=True)

fig.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()

# â”€â”€ 2) Interactiveâ€‘like 3â€‘D overlay (Matplotlib mplot3d) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# NB: Matplotlib 3â€‘D allows rotation & zoom in Kaggleâ€™s viewer.

fig3d = plt.figure(figsize=(14, 7))
ax3d  = fig3d.add_subplot(111, projection='3d')
ax3d.set_title("3â€‘D Histogram Overlay: All Velocity Families")

# gather a consistent colour per family
cmap = plt.get_cmap('tab10')
dx   = None        # will store common bin width
dy   = 0.8         # bar â€œthicknessâ€� along Y (family index)

for y_idx, fam in enumerate(FAMILIES):
    v = np.load(first_velocity_file(fam), mmap_mode="r").ravel()
    counts, edges = np.histogram(v, bins=BINS)
    if dx is None:
        dx = edges[1] - edges[0]
    x_centres = 0.5 * (edges[:-1] + edges[1:])
    # zâ€‘base is zero for all bars
    ax3d.bar3d(
        x_centres,                           # x positions (velocity bins)
        np.full_like(x_centres, y_idx),      # y positions (family row)
        np.zeros_like(counts),               # zâ€‘base = 0
        dx,                                  # bar width (x)
        dy,                                  # bar depth (y)
        counts,                              # bar height (z)
        color=cmap(y_idx), alpha=0.9, shade=True
    )

# axis labels & ticks
ax3d.set_xlabel("Velocity (m/s)")
ax3d.set_ylabel("Family")
ax3d.set_zlabel("Pixel Count")
ax3d.set_yticks(range(len(FAMILIES)))
ax3d.set_yticklabels(FAMILIES, fontsize=10)

plt.show()


# CODEÂ CELLÂ 4Â â€”  **ULTRAâ€‘LARGE â€œpictureÂ |Â explanationâ€� layout**
#
# â€¢ Canvas:  32Â in Ã—Â 44Â in, DPIÂ =Â 150  â‡’ fills the whole screen / PDF page  
# â€¢ Fonts:   36Â pt everywhere, 50Â pt superâ€‘title  
# â€¢ Layout:  4 rows Ã— 2 columns  (image | explanation)  
#            Every element is huge and legible.

import os, numpy as np, matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 36})        # colossal base font

TRAIN_DIR = "/kaggle/input/waveform-inversion/train_samples"
FAM_A, FAM_B = "FlatVel_A", "CurveFault_B"    # â†� change if needed

def first_file(folder, kind):
    """kind='seis' or 'vel' â†’ first .npy path (handles both layouts)."""
    root = os.path.join(TRAIN_DIR, folder)
    if os.path.isdir(os.path.join(root, "data")):            # twoâ€‘folder
        sub   = "data" if kind == "seis" else "model"
        fname = sorted(f for f in os.listdir(os.path.join(root, sub)) if f.endswith(".npy"))[0]
        return os.path.join(root, sub, fname)
    prefix = "seis" if kind == "seis" else "vel"
    fname  = sorted(f for f in os.listdir(root) if f.startswith(prefix) and f.endswith(".npy"))[0]
    return os.path.join(root, fname)

SEIS_PATHS = [first_file(FAM_A, "seis"), first_file(FAM_B, "seis")]
VEL_PATHS  = [first_file(FAM_A, "vel"),  first_file(FAM_B, "vel")]

fig, axes = plt.subplots(
    4, 2, figsize=(32, 44), dpi=150,        # MASSIVE canvas
    gridspec_kw={"width_ratios": [4, 3]}
)

IMG_FS  = 36
TXT_FS  = 36
SUP_FS  = 50

explain = [
    "Straight, evenlyâ€‘spaced reflections dipping linearly â�œ flat, layerâ€‘cake stratigraphy with no lateral velocity variation.",
    "Overall linear moveâ€‘out, but subtle time shifts and amplitude jitter due to curved layers and a fault cutting the model.",
    "Horizontal velocity bands stepping from slow (purple) near the surface to fast (yellowâ€‘green) at depth â€” textbook layering.",
    "Undulating velocity bands plus a vertical fault offset. Lateral contrasts drive the extra complexity in the seismic gather."
]

def add_text(ax, txt):
    ax.axis("off")
    ax.text(0, 0.5, txt, wrap=True, va="center", fontsize=TXT_FS)

# RowÂ 0 â”€ seismic gather (flat)
ax_img, ax_txt = axes[0]
sg = np.load(SEIS_PATHS[0], mmap_mode="r")[0, 0].copy()
sg /= np.max(np.abs(sg), axis=0, keepdims=True)
for r in range(sg.shape[1]):
    ax_img.plot(sg[:, r] + r, lw=2)
ax_img.invert_yaxis()
ax_img.set_title(f"Seismic â€“ {os.path.basename(SEIS_PATHS[0])}", fontsize=IMG_FS, pad=20)
ax_img.set_xlabel("receiver", fontsize=IMG_FS); ax_img.set_ylabel("time sample", fontsize=IMG_FS)
add_text(ax_txt, explain[0])

# RowÂ 1 â”€ seismic gather (faulted)
ax_img, ax_txt = axes[1]
sg = np.load(SEIS_PATHS[1], mmap_mode="r")[0, 0].copy()
sg /= np.max(np.abs(sg), axis=0, keepdims=True)
for r in range(sg.shape[1]):
    ax_img.plot(sg[:, r] + r, lw=2)
ax_img.invert_yaxis()
ax_img.set_title(f"Seismic â€“ {os.path.basename(SEIS_PATHS[1])}", fontsize=IMG_FS, pad=20)
ax_img.set_xlabel("receiver", fontsize=IMG_FS); ax_img.set_ylabel("time sample", fontsize=IMG_FS)
add_text(ax_txt, explain[1])

# RowÂ 2 â”€ velocity model (flat)
ax_img, ax_txt = axes[2]
vm = np.load(VEL_PATHS[0], mmap_mode="r")[0].squeeze()
im = ax_img.imshow(vm, origin="upper", aspect="auto")
plt.colorbar(im, ax=ax_img, fraction=0.03, pad=0.01)
ax_img.set_title(f"Velocity â€“ {os.path.basename(VEL_PATHS[0])}", fontsize=IMG_FS, pad=20)
add_text(ax_txt, explain[2])

# RowÂ 3 â”€ velocity model (faulted)
ax_img, ax_txt = axes[3]
vm = np.load(VEL_PATHS[1], mmap_mode="r")[0].squeeze()
im = ax_img.imshow(vm, origin="upper", aspect="auto")
plt.colorbar(im, ax=ax_img, fraction=0.03, pad=0.01)
ax_img.set_title(f"Velocity â€“ {os.path.basename(VEL_PATHS[1])}", fontsize=IMG_FS, pad=20)
add_text(ax_txt, explain[3])

fig.suptitle(
    f"Flatâ€‘layered model ({FAM_A}) vs. curved, faulted model ({FAM_B}):\n"
    "how velocity structure shapes their seismic gathers",
    fontsize=SUP_FS, y=0.993
)

plt.tight_layout(rect=[0, 0, 1, 0.985])
plt.show()


import os
import numpy as np
import matplotlib.pyplot as plt

TRAIN_DIR = "/kaggle/input/waveform-inversion/train_samples"
FAMILIES = ["CurveVel_A", "CurveVel_B"]
MAX_FILES_FAM = 4
SAMPLE_SHOW_N = 6


def load_family(folder: str, max_files: int = 4) -> np.ndarray:
    """Load up to max_files velocity-cube .npy files from *folder* (or its
    optional 'model' subfolder) and stack them along axis 0."""
    model_dir = os.path.join(folder, "model")
    if os.path.isdir(model_dir):
        files = sorted(f for f in os.listdir(model_dir) if f.endswith(".npy"))
        roots = [os.path.join(model_dir, f) for f in files[:max_files]]
    else:
        files = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
        roots = [os.path.join(folder, f) for f in files[:max_files]]

    stacks = []
    for path in roots:
        arr = np.load(path, mmap_mode="r").squeeze()
        if arr.ndim == 2:  # single map â†’ add dummy stackâ€‘axis
            arr = arr[None, ...]
        stacks.append(arr)
    return np.concatenate(stacks, axis=0)


def plot_mean_std(stack: np.ndarray) -> None:
    """Display mean and standardâ€‘deviation velocity maps (no titles, no captions)."""
    mean_map, std_map = stack.mean(0), stack.std(0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(mean_map, origin="upper", aspect="auto")
    axes[1].imshow(std_map, origin="upper", aspect="auto")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_samples(stack: np.ndarray, n_show: int = 6) -> None:
    """Display *n_show* sample cubes in a grid (no text, no colour bars)."""
    n = min(n_show, len(stack))
    cols, rows = 3, int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            if idx < n:
                ax.imshow(stack[idx], origin="upper", aspect="auto")
            ax.axis("off")
            idx += 1

    plt.tight_layout()
    plt.show()


def main() -> None:
    for fam in FAMILIES:
        folder = os.path.join(TRAIN_DIR, fam)
        if not os.path.isdir(folder):
            print(f"{fam} missing â€“ skipped")
            continue
        stack = load_family(folder, MAX_FILES_FAM)
        plot_mean_std(stack)
        plot_samples(stack, SAMPLE_SHOW_N)


if __name__ == "__main__":
    main()

