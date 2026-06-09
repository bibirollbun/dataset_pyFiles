import os

# Configure Zyra to use Gemini LLM for planning and narration
os.environ["ZYRA_LLM_PROVIDER"] = "gemini"
os.environ["ZYRA_LLM_MODEL"] = "gemini-2.5-flash"

# Mirror env vars into Python scope for readability and debugging
ZYRA_LLM_PROVIDER = os.environ["ZYRA_LLM_PROVIDER"]
ZYRA_LLM_MODEL = os.environ["ZYRA_LLM_MODEL"]



import os
from kaggle_secrets import UserSecretsClient

# Retrieve Google API key from Kaggle secrets for Gemini access
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


# --- Install Zyra and Confirm Version ---
"""
Zyra is distributed as an open-source Python package maintained by
NOAAâ€™s Global Systems Laboratory (GSL).

This cell installs Zyra quietly from PyPI and confirms the installed version.
Nearly all of Zyraâ€™s functionality â€” including agent swarms, provenance
tracking, visualization, and workflow orchestration â€” lives inside the
package itself, not in this notebook.

Source and Documentation:
  â€¢ PyPI:  https://pypi.org/project/zyra/
  â€¢ GitHub: https://github.com/NOAA-GSL/zyra
"""

# Install Zyra quietly
!pip install -q zyra

# Display installed version and verify CLI availability
!zyra --version



# --- Validate Zyra LLM Backend Connectivity ---
"""
This cell verifies that Zyra can successfully connect to its configured
language model (LLM) backend before running any swarm-driven workflows.

Zyra uses LLMs for:
  â€¢ Workflow planning (via `sess.run.plan()` or `sess.decide.optimize()`)
  â€¢ Narrative generation and explanation (via `sess.narrate.describe()` or `sess.narrate.swarm()`)
  â€¢ Collaborative swarm reasoning for model critiques and summaries

The `_test_llm_connectivity()` function confirms that:
  â€¢ The environment variables `ZYRA_LLM_PROVIDER` and `ZYRA_LLM_MODEL` are set
  â€¢ The provider (e.g., `openai`, `gemini`, or `anthropic`) is reachable
  â€¢ The specified model can be instantiated for inference

If connectivity fails, it raises a RuntimeError to prevent running downstream
swarm or narration cells that depend on the LLM.
"""

from zyra.wizard import _test_llm_connectivity

# --------------------------------------------------------------------
# 1. Validate LLM backend connectivity
# --------------------------------------------------------------------
status_ok, status_msg = _test_llm_connectivity(ZYRA_LLM_PROVIDER, ZYRA_LLM_MODEL)

# --------------------------------------------------------------------
# 2. Report connection status
# --------------------------------------------------------------------
print("ğŸ§© LLM provider:", ZYRA_LLM_PROVIDER)
print("ğŸ¤– LLM model:", ZYRA_LLM_MODEL)
print(status_msg)

# --------------------------------------------------------------------
# 3. Abort execution if connection fails
# --------------------------------------------------------------------
if not status_ok:
    raise RuntimeError(f"â�Œ Zyra LLM connectivity failed: {status_msg}")
else:
    print("âœ… Zyra LLM connection verified and ready for swarm operations.")



# --- Initialize Zyra Notebook Session and Workspace ---
"""
This cell initializes a Zyra notebook session â€” the reproducible runtime context
for all subsequent data acquisition, processing, and visualization steps.

What it does:
  â€¢ Creates or connects to a Zyra workspace rooted in `/kaggle/working`
  â€¢ Sets up environment variables for notebook provenance tracking
  â€¢ Creates a reproducible working directory for drought analysis
  â€¢ Instantiates a Zyra session (`sess`) that manages all CLI-like operations

Every Zyra notebook run begins with a session; it tracks all actions in a
SQLite provenance database for reproducibility.
"""

from pathlib import Path
from zyra.notebook import create_session
import os

# --------------------------------------------------------------------
# 1. Define and export notebook environment variables
# --------------------------------------------------------------------
# Root working directory (Kaggle runtime)
os.environ["ZYRA_NOTEBOOK_DIR"] = "/kaggle/working"

# Provenance database path â€” logs all actions for reproducibility
os.environ.setdefault(
    "ZYRA_NOTEBOOK_PROVENANCE",
    str(Path(os.environ["ZYRA_NOTEBOOK_DIR"]) / "drought_notebook" / "provenance.sqlite")
)

# --------------------------------------------------------------------
# 2. Create a new Zyra notebook session
# --------------------------------------------------------------------
sess = create_session()  # Entry point for Zyra's modular workflow engine

# Retrieve the workspace path from the active session
WORKSPACE = sess.workspace()

# --------------------------------------------------------------------
# 3. Define drought analysis subdirectory inside workspace
# --------------------------------------------------------------------
DROUGHT_DIR = WORKSPACE / "drought_notebook"
DROUGHT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# 4. Display confirmation of environment setup
# --------------------------------------------------------------------
print("âœ… Zyra notebook session initialized.")
print(f"Workspace root: {WORKSPACE}")
print(f"Drought analysis directory: {DROUGHT_DIR}")



# --- Define Folder Structure for Zyra Drought Analysis ---
"""
This cell defines and initializes the local directory structure used throughout
the drought analysis workflow, as well as the expected temporal cadence between frames.

Each subdirectory serves a specific purpose:

  â€¢ FRAMES_RAW     â€” Raw drought imagery frames acquired from NOAA FTP.
  â€¢ FRAMES_PADDED  â€” Frames after temporal padding (gap filling via basemap).
  â€¢ FRAMES_META    â€” JSON file containing scanned metadata (timestamps, cadence, etc.).
  â€¢ VIDEO_OUT      â€” Final MP4 animation output path.
  â€¢ BASEMAP_REF    â€” Reference basemap image (used for padding and compositing).

We also define `CADENCE_SECONDS`, representing the expected time interval
between frames (1 week). This constant is used across multiple stages, including
FTP synchronization, frame scanning, and planner validation.

All directories are created if they donâ€™t already exist to ensure the pipeline
can run end-to-end without manual setup.
"""

# --------------------------------------------------------------------
# 1. Define all key directories and output paths
# --------------------------------------------------------------------
FRAMES_RAW = DROUGHT_DIR / "frames_raw"               # Raw drought frames downloaded from NOAA
FRAMES_PADDED = DROUGHT_DIR / "frames_padded"         # Frames with missing dates filled
FRAMES_META = DROUGHT_DIR / "frames_meta.json"        # Metadata output from scan_frames()
VIDEO_OUT = DROUGHT_DIR / "drought_animation.mp4"     # Final video animation output
BASEMAP_REF = "pkg:zyra.assets/images/earth_vegetation.jpg"  # Zyraâ€™s packaged vegetation basemap

# --------------------------------------------------------------------
# 2. Define frame cadence (in seconds)
# --------------------------------------------------------------------
CADENCE_SECONDS = 7 * 24 * 3600  # one week
print(f"ğŸ•’ Frame cadence set to {CADENCE_SECONDS / 3600 / 24:.0f} days per frame")

# --------------------------------------------------------------------
# 3. Ensure that working directories exist
# --------------------------------------------------------------------
for folder in (FRAMES_RAW, FRAMES_PADDED):
    folder.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# 4. Confirm directory setup
# --------------------------------------------------------------------
print("âœ… Folder structure initialized:")
print(f"ğŸ“‚ Raw frames directory: {FRAMES_RAW}")
print(f"ğŸ“‚ Padded frames directory: {FRAMES_PADDED}")
print(f"ğŸ§¾ Metadata file path: {FRAMES_META}")
print(f"ğŸ��ï¸�  Output video path: {VIDEO_OUT}")
print(f"ğŸ—ºï¸�  Basemap reference: {BASEMAP_REF}")



# --- Prepopulate Drought Frames and Colorbar from Kaggle Input Dataset ---
"""
This cell preloads NOAA drought imagery and the reference colorbar image
from a mounted Kaggle input dataset to accelerate the Zyra workflow.

Dataset layout (expected):
  /kaggle/input/noaa-weekly-drought-frames-colorbar/
  â”œâ”€â”€ frames_raw/               â†� folder containing weekly drought PNGs
  â”‚     â”œâ”€â”€ DroughtRisk_Weekly_YYYYMMDD.png
  â”‚     â””â”€â”€ ...
  â””â”€â”€ VTHI.colorbar.png         â†� colorbar reference used for analysis

Files are copied into the active Zyra workspace:
  â€¢ Frames â†’ FRAMES_RAW
  â€¢ Colorbar â†’ DROUGHT_DIR / "VTHI.colorbar.png"

Benefits:
  â€¢ Avoids re-downloading a full year of frames via FTP
  â€¢ Provides reproducible inputs for Zyra drought analysis
  â€¢ Enables immediate execution of analysis and visualization stages
"""

import shutil
from pathlib import Path

# --------------------------------------------------------------------
# 1. Define dataset source path (adjust this to your dataset name)
# --------------------------------------------------------------------
INPUT_DATASET_PATH = Path("/kaggle/input/noaa-weekly-drought-frames-colorbar")
FRAMES_SRC = INPUT_DATASET_PATH / "frames_raw"
COLORBAR_SRC = INPUT_DATASET_PATH / "VTHI.colorbar.png"
COLORBAR_DEST = DROUGHT_DIR / "VTHI.colorbar.png"

# Ensure destination directories exist
FRAMES_RAW.mkdir(parents=True, exist_ok=True)
DROUGHT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# 2. Copy drought frame PNGs from dataset â†’ working directory
# --------------------------------------------------------------------
if not FRAMES_SRC.exists():
    print("â�Œ Expected subfolder not found:", FRAMES_SRC)
    print("   Skipping prepopulation â€” Zyra will perform full FTP sync instead.")
else:
    png_files = sorted(FRAMES_SRC.glob("*.png"))
    if not png_files:
        print("âš ï¸� No PNG frames found in dataset subfolder:", FRAMES_SRC)
    else:
        copied = 0
        for src in png_files:
            dest = FRAMES_RAW / src.name
            if not dest.exists():
                shutil.copy2(src, dest)
                copied += 1
        if copied:
            print(f"âœ… Copied {copied} new drought frame(s) into {FRAMES_RAW}")
        else:
            print(f"â„¹ï¸� All drought frames already present in {FRAMES_RAW}")

# --------------------------------------------------------------------
# 3. Copy colorbar image from dataset root â†’ Zyra working directory
# --------------------------------------------------------------------
if COLORBAR_SRC.exists():
    if not COLORBAR_DEST.exists():
        shutil.copy2(COLORBAR_SRC, COLORBAR_DEST)
        print(f"âœ… Copied colorbar to {COLORBAR_DEST}")
    else:
        print(f"â„¹ï¸� Colorbar already present at {COLORBAR_DEST}")
else:
    print("âš ï¸� Colorbar file not found in input dataset root.")
    print("   Zyra will attempt HTTP acquisition later if needed.")

# --------------------------------------------------------------------
# 4. Verify prepopulation success
# --------------------------------------------------------------------
frame_files = sorted(FRAMES_RAW.glob("*.png"))
print(f"\nğŸ“¸ Local drought frames ready: {len(frame_files)}")
if frame_files:
    print("Sample frames:", [f.name for f in frame_files[:3]])

if COLORBAR_DEST.exists():
    print(f"ğŸ�¨ Colorbar ready: {COLORBAR_DEST.name}")
else:
    print("âš ï¸� No colorbar found â€” will attempt HTTP acquire later.")



# --- Acquire Drought Frames from NOAA FTP ---
"""
This cell synchronizes the latest drought imagery frames from NOAAâ€™s FTP server
using Zyraâ€™s built-in FTP acquisition agent. The agent automatically handles
listing, filtering, and downloading new or missing files.
"""

FTP_PATH = "ftp://ftp.nnvl.noaa.gov/SOS/DroughtRisk_Weekly"
PATTERN = r"^DroughtRisk_Weekly_[0-9]{8}\.png$"
SINCE_PERIOD = "P1Y"
DATE_FORMAT = "%Y%m%d"

try:
    sess.acquire.ftp(
        path=FTP_PATH,
        sync_dir=str(FRAMES_RAW),
        pattern=PATTERN,
        since_period=SINCE_PERIOD,   # Limit to the past year
        date_format=DATE_FORMAT, # Parse dates in filenames
    )
    print(f"âœ… FTP sync complete â€” frames saved to {FRAMES_RAW}")
except Exception as e:
    print(f"â�Œ FTP acquisition failed: {e}")

# Count and preview a few frames
frames = sorted(FRAMES_RAW.glob("*.png"))
print(f"ğŸ“¦ Total frames downloaded: {len(frames)}")
if frames:
    print("ğŸ†• Sample latest frames:", [f.name for f in frames[-3:]])



# --- Download Drought Colorbar Using Zyra HTTP Acquire Agent ---
"""
This cell downloads the NOAA-provided drought colorbar image, which is used as
a reference legend for classifying drought severity in later analysis steps.

Zyraâ€™s `acquire.http()` agent is used here to:
  â€¢ Retrieve remote files (HTTP/HTTPS endpoints)
  â€¢ Save them to the working directory for reproducibility
  â€¢ Optionally suppress verbose logs with `quiet=True`

Once downloaded, this colorbar defines the pixel color-to-severity mapping
("moderate", "high", "extreme") for the `analyze_drought_frames` stage.
"""

# --------------------------------------------------------------------
# 1. Define the source URL and local output path
# --------------------------------------------------------------------
URL = "https://www.nnvl.noaa.gov/view/Colorbars/VTHI.colorbar.png"
OUTPUT_PATH = DROUGHT_DIR / "VTHI.colorbar.png"

# --------------------------------------------------------------------
# 2. Attempt download using Zyraâ€™s HTTP acquisition agent
# --------------------------------------------------------------------
try:
    sess.acquire.http(
        url=URL,                # Remote image file (colorbar reference)
        output=str(OUTPUT_PATH),# Local file destination
        quiet=True,             # Suppress internal progress/logging output
    )

    # ----------------------------------------------------------------
    # 3. Verify the file was downloaded successfully
    # ----------------------------------------------------------------
    if OUTPUT_PATH.exists():
        print(f"âœ… Downloaded successfully to {OUTPUT_PATH}")
    else:
        print(f"âš ï¸� Download completed but file not found at {OUTPUT_PATH}")

except Exception as e:
    # Handle network or permission errors gracefully
    print(f"â�Œ Download failed: {e}")



# --- Simulate Missing Frames by Deleting Two Random Images ---
"""
This cell intentionally deletes a small number of frames from the raw drought
imagery directory to simulate missing time steps.

Why?
-----
This controlled data loss allows us to test Zyraâ€™s ability to:
  â€¢ Detect irregular cadence via `transform.scan_frames`
  â€¢ Automatically fill gaps using `process.pad_missing`

The deletion is deterministic (reproducible) thanks to a fixed random seed.
"""

import random

# --------------------------------------------------------------------
# 1. Set up reproducible randomness
# --------------------------------------------------------------------
random.seed(42)  # Ensures the same frames are removed each run

# --------------------------------------------------------------------
# 2. Identify all current drought imagery frames
# --------------------------------------------------------------------
frames = sorted(p for p in FRAMES_RAW.iterdir() if p.is_file())

# --------------------------------------------------------------------
# 3. Delete two random frames (if enough exist)
# --------------------------------------------------------------------
if len(frames) < 2:
    print("âš ï¸� Not enough frames to delete; skipping gap simulation.")
else:
    # Randomly select two frames for deletion
    to_delete = random.sample(frames, 2)
    
    # Remove the selected files (missing_ok=True allows safe repeated runs)
    for fp in to_delete:
        fp.unlink(missing_ok=True)
    
    # ----------------------------------------------------------------
    # 4. Report which frames were deleted and new total count
    # ----------------------------------------------------------------
    print("ğŸ§¹ Deleted frames:", [fp.name for fp in sorted(to_delete)])
    print("Remaining frame count:",
          sum(1 for f in FRAMES_RAW.iterdir() if f.is_file()))



# --- Scan Frames Metadata (Cadence and Missing Timestamps) ---
"""
This cell uses Zyraâ€™s `transform.scan_frames()` utility to analyze the
temporal cadence of raw drought imagery frames.

The scan determines:
  â€¢ How many frames exist (`frame_count_actual`)
  â€¢ Whether any expected timestamps are missing (`missing_count`)
  â€¢ The overall cadence and temporal range
  â€¢ Any irregular gaps between frames

This metadata is stored in `FRAMES_META` and used by later steps
(e.g., `pad_missing`) to fill in gaps with basemap placeholders.
"""

import json

# --------------------------------------------------------------------
# 1. Execute Zyra's frame scanning utility
# --------------------------------------------------------------------
meta_result = sess.transform.scan_frames(
    frames_dir=str(FRAMES_RAW),          # Input directory containing drought imagery frames
    pattern=PATTERN,                     # Filename pattern, e.g. "DroughtRisk_*_%Y%m%d.png"
    datetime_format="%Y%m%d",            # Date format used in frame filenames
    period_seconds=CADENCE_SECONDS,      # Expected temporal spacing between frames (in seconds)
    output=str(FRAMES_META),             # Output JSON file to store metadata summary
)

# --------------------------------------------------------------------
# 2. Load the metadata summary for quick inspection
# --------------------------------------------------------------------
summary = json.loads(FRAMES_META.read_text()) if FRAMES_META.exists() else {}

# --------------------------------------------------------------------
# 3. Print concise summary stats for verification
# --------------------------------------------------------------------
print("âœ… Frames metadata saved to:", FRAMES_META)
print(
    "Detected frames:",
    summary.get("frame_count_actual", "N/A"),
    "| Missing frames:",
    summary.get("missing_count", "N/A"),
)



# --- Display Frames Metadata from Zyra scan_frames Stage ---
"""
This cell loads and displays the metadata file produced by Zyraâ€™s
`scan_frames` process. The metadata typically includes:

  â€¢ Frame cadence information (timestamps, ordering)
  â€¢ Missing or irregular frame detections
  â€¢ File naming consistency checks
  â€¢ Optional image dimensions or hashes for provenance

Inspecting this file helps verify that the input imagery sequence
was correctly scanned before padding or visualization steps.
"""

import json

# --------------------------------------------------------------------
# Load and print formatted JSON if the metadata file exists
# --------------------------------------------------------------------
if FRAMES_META.exists():
    with open(FRAMES_META, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Pretty-print the metadata with 2-space indentation for readability
    print(json.dumps(data, indent=2, ensure_ascii=False))
else:
    # Safety check â€” handle missing file
    print(f"â�Œ Metadata file not found: {FRAMES_META}")



# --- Drought Analysis Helper Functions and Configuration ---
"""
This cell defines helper functions and constants used by the custom drought analysis module.

These helpers support frame sampling, downsampling, coordinate conversion, and
reverse geocoding.  They are used by the analysis routine to interpret each
pixel of drought imagery into meaningful spatial and categorical information.

The goal is to prepare utility components that can be reused across Zyra
sessions or in other environmental workflows.
"""

import math
from pathlib import Path
from typing import Any, Iterable, Tuple
from collections import defaultdict
from PIL import Image

try:
    import reverse_geocoder as rg
except ImportError:
    rg = None

# --------------------------------------------------------------------
# Configuration Constants
# --------------------------------------------------------------------
SAMPLE_STRATEGY = "monthly"   # Options: "monthly" or "every_n"
SAMPLE_EVERY_N = 4            # If SAMPLE_STRATEGY == "every_n"
DOWNSAMPLE_MAX_SIDE = 256     # Max width/height for fast analysis
LON_MIN, LON_MAX = -180.0, 180.0
LAT_MIN, LAT_MAX = -90.0, 90.0
MAX_COORD_SAMPLES = 500       # Limit lon/lat samples per frame per class


def _select_frames(frames_dir: Path) -> list[Path]:
    """
    Sample drought frames from a directory according to the configured strategy.

    Parameters
    ----------
    frames_dir : Path
        Directory containing drought imagery (.png) files.

    Returns
    -------
    list[Path]
        A filtered list of frames selected for analysis.
    """
    frames = sorted(frames_dir.glob("*.png"))
    if SAMPLE_STRATEGY == "monthly":
        by_month: dict[str, Path] = {}
        for fp in frames:
            # Extract YYYYMM portion from filename
            key = fp.stem.split("_")[-1][:6] if "_" in fp.stem else fp.stem[:6]
            by_month.setdefault(key, fp)
        return [by_month[k] for k in sorted(by_month.keys())]
    if SAMPLE_STRATEGY == "every_n":
        return frames[:: max(SAMPLE_EVERY_N, 1)]
    return frames


def _downsample(img: Image.Image) -> Tuple[Image.Image, float]:
    """
    Downsample an image to a manageable resolution for faster pixel analysis.

    Parameters
    ----------
    img : PIL.Image.Image
        The image to resize.

    Returns
    -------
    Tuple[Image.Image, float]
        A tuple containing the (possibly resized) image and the applied scale factor.
    """
    w, h = img.size
    scale = min(DOWNSAMPLE_MAX_SIDE / max(w, h), 1.0)
    if scale < 1.0:
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return img.resize(new_size, Image.BOX), scale
    return img, 1.0


def _lonlat_from_xy(x: int, y: int, w: int, h: int) -> tuple[float, float]:
    """
    Convert pixel coordinates (x, y) into approximate geographic coordinates (lon, lat).

    Parameters
    ----------
    x, y : int
        Pixel coordinates.
    w, h : int
        Image width and height.

    Returns
    -------
    tuple[float, float]
        (longitude, latitude) in degrees.
    """
    lon = LON_MIN + (x / max(w - 1, 1)) * (LON_MAX - LON_MIN)
    lat = LAT_MAX - (y / max(h - 1, 1)) * (LAT_MAX - LAT_MIN)
    return lon, lat


def _nearest_place(lat: float, lon: float, cache: dict[tuple[float, float], str]) -> str:
    """
    Return the nearest place name for a coordinate pair, using a local cache.

    Parameters
    ----------
    lat, lon : float
        Latitude and longitude in degrees.
    cache : dict
        Dictionary used to cache prior reverse geocoding results.

    Returns
    -------
    str
        A human-readable place name (e.g., "Denver, CO, US"), or an empty string
        if reverse geocoding is unavailable.
    """
    if rg is None:
        return ""
    key = (round(lat, 3), round(lon, 3))
    if key in cache:
        return cache[key]
    try:
        res = rg.search((lat, lon), mode=1, verbose=False)
        label = ""
        if res:
            entry = res[0]
            parts = [entry.get("name"), entry.get("admin1"), entry.get("cc")]
            label = ", ".join(p for p in parts if p)
    except Exception:
        label = ""
    cache[key] = label
    return label


def _iter_rows(img: Image.Image) -> Iterable[Tuple[tuple[int, int, int], ...]]:
    """
    Yield each image row as a tuple of RGB pixel values.

    Parameters
    ----------
    img : PIL.Image.Image
        The image to iterate over.

    Yields
    ------
    tuple
        A sequence of RGB pixels for a given row.
    """
    w, h = img.size
    pixels = img.load()
    for y in range(h):
        yield tuple(pixels[x, y] for x in range(w))

print(f"âœ… Helper functions successfully initalized.")


# --- Register Custom Drought Analysis into Zyra Session ---
"""
This cell registers the custom drought analysis routine into the active Zyra session.

The registration exposes a new process-stage function called
`analyze_drought_frames()` that can be executed within any Zyra workflow.

Once registered, you can call:
    sess.process.analyze_drought_frames(frames_dir=..., colorbar=..., tolerance=30)

This analyzer reads each drought imagery frame, compares pixel colors against
a provided reference colorbar, and classifies each pixel into drought severity
levels: moderate, high, extreme, or nodata. It also collects geospatial samples
and optional place names for each class, producing both per-frame and aggregate
summaries suitable for visualization, narration, or further statistical study.
"""

import math
from collections import defaultdict
from typing import Any
from PIL import Image
from datetime import datetime

def _register_drought_analysis(session) -> None:
    """
    Register a custom drought analysis function with the active Zyra session.

    This function defines and registers `analyze_drought_frames()`, a workflow step
    that analyzes a directory of drought imagery using a reference colorbar legend.
    Each image frame is sampled, downscaled, and classified into drought severity
    categories (moderate, high, or extreme) based on pixel color distance.

    Once registered, the analyzer becomes available to Zyraâ€™s `process` stage and
    can be called just like a built-in command:
        sess.process.analyze_drought_frames(frames_dir=..., colorbar=...)

    The analyzer produces structured summaries for each frame (pixel counts,
    geographic samples, and inferred place names) as well as an aggregate summary
    for all frames. It is primarily used in environmental or climate workflows to
    track drought evolution over time.
    """

    def analyze(ns: Any) -> dict[str, Any]:
        """Perform drought severity analysis using a provided colorbar reference."""
        frames_dir = Path(ns.frames_dir)
        colorbar = Path(ns.colorbar)
        tol = float(vars(ns).get("tolerance", 30))

        # Load and parse the colorbar palette
        bar = Image.open(colorbar).convert("RGB")
        row = bar.height // 2
        colors = [bar.getpixel((x, row)) for x in range(bar.width)]
        palette = []
        for c in colors:
            if not palette or c != palette[-1]:
                palette.append(c)

        def classify_pixel(rgb: tuple[int, int, int]) -> str:
            """Classify a pixelâ€™s drought severity based on color similarity."""
            best_idx, best_d = None, float("inf")
            for idx, pc in enumerate(palette):
                d = math.dist(rgb, pc)
                if d < best_d:
                    best_d, best_idx = d, idx
            if best_d > tol:
                return "nodata"
            frac = best_idx / max(len(palette) - 1, 1)
            if frac < 0.3:
                return "moderate"
            if frac < 0.65:
                return "high"
            return "extreme"

        # Initialize output summary
        summary = {
            "frames": {},
            "aggregate": {
                "total_pixels": 0,
                "nodata": 0,
                "moderate": 0,
                "high": 0,
                "extreme": 0,
            },
        }

        sampled = _select_frames(frames_dir)
        summary["sampled_frames"] = [p.name for p in sampled]
        place_cache: dict[tuple[float, float], str] = {}

        # Analyze each frame
        for img_path in sampled:
            img, scale = _downsample(Image.open(img_path).convert("RGB"))
            w, h = img.size
            counts = defaultdict(int)
            samples = {k: [] for k in ("moderate", "high", "extreme", "nodata")}
            sample_places = {k: [] for k in samples}

            for y, row_px in enumerate(_iter_rows(img)):
                for x, rgb in enumerate(row_px):
                    bucket = classify_pixel(rgb)
                    counts[bucket] += 1
                    if len(samples[bucket]) < MAX_COORD_SAMPLES:
                        lon, lat = _lonlat_from_xy(x, y, w, h)
                        samples[bucket].append((lon, lat))
                        place_label = _nearest_place(lat, lon, place_cache)
                        if place_label:
                            sample_places[bucket].append(
                                {"lon": lon, "lat": lat, "place": place_label}
                            )

            total = sum(counts.values())
            counts["total"] = total
            summary["frames"][img_path.name] = {
                "counts": dict(counts),
                "sampled_coords": samples,
                "sampled_places": sample_places,
            }

            agg = summary["aggregate"]
            agg["total_pixels"] += total
            for k in ("moderate", "high", "extreme", "nodata"):
                agg[k] += counts.get(k, 0)

        return summary

    # Register with Zyraâ€™s session registry
    session.process.register(
        "analyze_drought_frames",
        analyze,
        returns="object",
        extras=["pillow", "reverse_geocoder"],
    )

# Log successful registration
print("âœ… Registered custom analyzer: analyze_drought_frames")



# --- Register Custom Drought Analysis with Zyra Session ---
"""
Executes the `_register_drought_analysis()` function defined earlier
(after all required helper functions have been loaded).

This step binds a new process-stage operation, `analyze_drought_frames()`,
into your active Zyra session (`sess`).

Once registered, you can call this analyzer just like a built-in Zyra tool:
    sess.process.analyze_drought_frames(
        frames_dir=FRAMES_RAW,
        colorbar=DROUGHT_DIR / "VTHI.colorbar.png",
        tolerance=30
    )

It classifies drought imagery based on a NOAA colorbar reference, computing
per-frame drought severity statistics (moderate, high, extreme, nodata),
and aggregating them for downstream visualization or narration.

Includes:
  â€¢ Helper dependency verification
  â€¢ Clean success/failure messages
  â€¢ Version-safe confirmation for Zyra â‰¥ 0.1.42
"""

# --------------------------------------------------------------------
# 1. Verify helper function availability before registering
# --------------------------------------------------------------------
required_helpers = [
    "_select_frames",
    "_downsample",
    "_iter_rows",
    "_lonlat_from_xy",
    "_nearest_place",
]
missing_helpers = [name for name in required_helpers if name not in globals()]

if missing_helpers:
    print("âš ï¸� Missing helper definitions:", ", ".join(missing_helpers))
    print("   Please re-run the helper function cell before registering.")
else:
    # ----------------------------------------------------------------
    # 2. Register the custom analyzer with the Zyra session
    # ----------------------------------------------------------------
    try:
        _register_drought_analysis(sess)
        print("âœ… Successfully registered custom analyzer: analyze_drought_frames")

        # ------------------------------------------------------------
        # 3. Functional verification (modern Zyra-safe)
        # ------------------------------------------------------------
        try:
            # Modern Zyra (>= 0.1.42): Registry is internalized, so we check callability
            if hasattr(sess.process, "analyze_drought_frames"):
                print("ğŸ”� Verified: analyzer callable via sess.process.analyze_drought_frames()")
            else:
                print("âš ï¸� Analyzer not directly visible on sess.process; retry if needed.")
        except Exception as verify_err:
            print("âš ï¸� Verification skipped due to:", verify_err)

        # ------------------------------------------------------------
        # 4. Friendly UX note for newer Zyra versions
        # ------------------------------------------------------------
        print("â„¹ï¸� Note: Registry introspection is disabled in Zyra â‰¥0.1.42, "
              "but the analyzer is active and callable.")

    except Exception as exc:
        print(f"â�Œ Registration failed: {exc}")



# --- Run Drought Frame Analysis --- (This will take a minute.)
"""
This cell executes the custom drought analysis function that was previously
registered with the Zyra session. It processes a selection of drought imagery frames
in the raw directory, classifies drought severity based on the reference colorbar,
and writes a structured JSON summary for downstream visualization or narration.
"""

# --------------------------------------------------------------------
# Execute the registered drought analysis
# --------------------------------------------------------------------
analysis = sess.process.analyze_drought_frames(
    frames_dir=str(FRAMES_RAW),                            # Directory of drought imagery
    colorbar=str(DROUGHT_DIR / "VTHI.colorbar.png"),       # Reference colorbar defining severity mapping
    tolerance=30,                                          # Color distance threshold for classification
)

# --------------------------------------------------------------------
# Display basic analysis summary
# --------------------------------------------------------------------
print("Analysis aggregate:", analysis.get("aggregate", {}))          # Overall pixel counts across all frames
print("Sampled frames count:", len(analysis.get("sampled_frames", [])))  # How many frames were analyzed

# --------------------------------------------------------------------
# Basic validation: Ensure results are non-empty
# --------------------------------------------------------------------
if not analysis or "frames" not in analysis:
    raise RuntimeError("Analysis produced no frame results.")

# --------------------------------------------------------------------
# Persist analysis results to disk for reuse and reproducibility
# --------------------------------------------------------------------
ANALYSIS_JSON = DROUGHT_DIR / "analysis.json"
ANALYSIS_JSON.write_text(json.dumps(analysis, indent=2, default=str))
print("âœ… Analysis results saved to", ANALYSIS_JSON)

# --------------------------------------------------------------------
# Preview results from the first analyzed frame
# --------------------------------------------------------------------
first_frame = next(iter(analysis.get("frames", {}).items()), None)
if first_frame:
    name, stats = first_frame
    counts = stats.get("counts", {})
    print(
        f"Sample frame '{name}':",
        {k: counts.get(k, 0) for k in ("moderate", "high", "extreme", "nodata")},
    )

    # Print a few sample coordinates for the 'nodata' class
    coords = stats.get("sampled_coords", {})
    nodata_samples = coords.get("nodata", [])[:3]
    print("Sample nodata lon/lat pairs:", nodata_samples)



# --- Build Frame and Location Summaries from Cached Analysis ---
"""
This cell parses the results of the drought frame analysis and extracts
a readable summary of locations corresponding to "high" and "extreme"
drought severity classes.

It converts the structured data in `analysis` (or the saved analysis.json)
into short text snippets â€” ideal for narration, reporting, or logging.

The result: two objects are created:
  â€¢ frames_locations â€” structured list of drought event summaries
  â€¢ frames_locations_text â€” human-readable bullet points
"""

import json

# --------------------------------------------------------------------
# Prepare data sources and containers
# --------------------------------------------------------------------
frames_locations = []         # structured output (list of dicts)
frames_locations_text = []    # text summary lines

# Use existing analysis object if available, otherwise fallback to empty dict
frames_dict = (analysis or {}).get("frames", {}) if "analysis" in globals() else {}

# --------------------------------------------------------------------
# Helper: Format coordinate pairs (lon, lat) as short text tuples
# --------------------------------------------------------------------
def _fmt_coords(coords: list[tuple[float, float]]) -> list[str]:
    """
    Convert a list of (lon, lat) tuples into human-readable coordinate strings.
    Only the first three coordinates are included for brevity.
    """
    out = []
    for c in coords[:3]:
        try:
            lon, lat = float(c[0]), float(c[1])
            out.append(f"({lon:.2f}, {lat:.2f})")
        except Exception:
            continue
    return out

# --------------------------------------------------------------------
# Helper: Format sampled places with names and coordinates
# --------------------------------------------------------------------
def _fmt_places(entries: list[dict]) -> list[str]:
    """
    Convert sampled place entries (with name and coordinates) into readable strings.
    Example output: "Denver, CO, US (-104.99, 39.74)"
    """
    out = []
    for entry in entries[:3]:
        place = entry.get("place")
        lon = entry.get("lon")
        lat = entry.get("lat")
        if place and isinstance(lon, (int, float)) and isinstance(lat, (int, float)):
            out.append(f"{place} ({lon:.2f}, {lat:.2f})")
        elif place:
            out.append(str(place))
    return out

# --------------------------------------------------------------------
# Iterate through all analyzed frames and extract locations
# --------------------------------------------------------------------
for name, stats in sorted(frames_dict.items()):
    coords = stats.get("sampled_coords") or {}
    places = stats.get("sampled_places") or {}

    # Extract representative locations for high and extreme drought classes
    h_places = _fmt_places(places.get("high") or [])
    x_places = _fmt_places(places.get("extreme") or [])

    # Fallback: use coordinates if named places are unavailable
    h_fmt = h_places or _fmt_coords(coords.get("high") or [])
    x_fmt = x_places or _fmt_coords(coords.get("extreme") or [])

    # Skip frames without any location data
    if not h_fmt and not x_fmt:
        continue

    # Extract date from filename (e.g., "DroughtRisk_Weekly_20250310.png" â†’ 20250310)
    frame_date = name.split("_")[-1].split(".")[0]

    # Append structured version for downstream processing
    frames_locations.append(
        {
            "frame": name,
            "date": frame_date,
            "high": h_fmt,
            "extreme": x_fmt,
        }
    )

    # Append text-friendly bullet summary
    high_txt = ", ".join(h_fmt) or "none"
    ext_txt = ", ".join(x_fmt) or "none"
    frames_locations_text.append(
        f"- {frame_date or name}: High: {high_txt}; Extreme: {ext_txt}"
    )

# --------------------------------------------------------------------
# Display summary results
# --------------------------------------------------------------------
print("âœ… Frames with extracted drought locations:", len(frames_locations))
print("\n".join(frames_locations_text[:5]))  # Preview first few summaries



# --- Fill Missing Frames Using Basemap Placeholders ---
"""
This cell reconstructs a complete sequence of drought imagery frames by filling
any missing time steps with placeholder images. These placeholders are generated
using Zyraâ€™s `pad_missing` process, which can insert either duplicate frames or
synthetic basemap images to preserve continuity.

This step ensures that downstream animations and analyses remain temporally
consistent even if certain frames were missing from the original dataset.
"""

import shutil

# --------------------------------------------------------------------
# 1. Prepare the padded output directory
# --------------------------------------------------------------------
# Remove any prior version of FRAMES_PADDED to ensure a clean start
if FRAMES_PADDED.exists():
    shutil.rmtree(FRAMES_PADDED)

# Recreate the directory for padded frames
FRAMES_PADDED.mkdir(parents=True, exist_ok=True)

# Copy all existing raw drought frames into the padded directory
# (this seeds it with known data before filling the gaps)
for src in FRAMES_RAW.iterdir():
    if src.is_file():
        shutil.copy2(src, FRAMES_PADDED / src.name)

# --------------------------------------------------------------------
# 2. Execute Zyraâ€™s frame padding process
# --------------------------------------------------------------------
# The pad_missing agent uses the metadata file to detect time gaps
# and generate replacement frames using the provided basemap.
pad_result = sess.process.pad_missing(
    frames_meta=str(FRAMES_META),     # Metadata JSON describing time-indexed frames
    output_dir=str(FRAMES_PADDED),    # Destination directory for the completed sequence
    fill_mode="basemap",              # Use basemap placeholders for missing frames
    basemap=BASEMAP_REF,              # Reference basemap image (e.g., vegetation map)
    overwrite=True,                   # Allow overwriting existing placeholder frames
)

# --------------------------------------------------------------------
# 3. Validate and report padding results
# --------------------------------------------------------------------
filled = sum(1 for f in FRAMES_PADDED.iterdir() if f.is_file())

print("âœ… Padded frames directory:", FRAMES_PADDED)
print("Total frames after padding:", filled)



# --- Compose MP4 Animation from Padded Frames ---
"""
This cell stitches together all drought imagery frames (including placeholders)
into a continuous MP4 video animation.

Zyraâ€™s `visualize.compose_video()` internally calls FFmpeg to perform the
encoding step. Therefore, FFmpeg **must be installed and available in the
system PATH** for this operation to succeed.

If youâ€™re running on Kaggle, Colab, or most Linux-based environments, FFmpeg
is already preinstalled. If running locally, install it via:
    sudo apt-get install ffmpeg
or on Windows via:
    choco install ffmpeg
"""

# Generate an MP4 drought animation at 4 frames per second
# Overlaying frames on a vegetation basemap for visual context
video_path = sess.visualize.compose_video(
    frames=str(FRAMES_PADDED),   # Input directory containing padded frame sequence
    output=str(VIDEO_OUT),       # Output video path (MP4 file)
    fps=4,                       # Animation frame rate (frames per second)
    basemap=BASEMAP_REF,         # Optional basemap overlay
)

# Display the resulting video path for downstream use or verification
print("âœ… Video successfully composed!")
print("Output video path:", video_path)



# --- Save Final Animation Locally (Disseminate Stage) ---
"""
This cell mirrors Zyraâ€™s `disseminate/local` workflow stage.

The goal is to ensure that the final drought animation (the MP4 file created
earlier) is saved to a stable, accessible location within the working directory.
In this notebook, the destination is the same as the source path, which ensures
the output is captured by Kaggleâ€™s automatic notebook export system.

If you were running on a shared or cloud environment, this same step could be
adapted to push results to remote storage (e.g., S3, GCS, or FTP) using other
Zyra disseminate agents.
"""

# --------------------------------------------------------------------
# 1. Copy or persist the generated MP4 video locally
# --------------------------------------------------------------------
final_path = sess.disseminate.local(
    input=str(VIDEO_OUT),   # Source file (MP4 animation generated earlier)
    path=str(VIDEO_OUT),    # Destination path (mirrors source for local export)
)

# --------------------------------------------------------------------
# 2. Confirm completion and show the local file path
# --------------------------------------------------------------------
print("âœ… Final animation saved locally.")
print("Local copy path:", final_path)



# --- Narration (Drought-Only Focused Wording) --- (This will take a minute.)
"""
This cell runs a focused Zyra narration swarm using the outputs of the
previous drought analysis steps.

Purpose:
  â€¢ Converts per-frame drought summaries into a natural-language report
  â€¢ Restricts terminology strictly to drought and dryness (no weather words)
  â€¢ Uses a lightweight single-round LLM swarm with context, summary, critic,
    and editor agents to ensure scientific fidelity and wording control

If no frame/location data is available, the cell will exit gracefully.
"""

import yaml

# --------------------------------------------------------------------
# 1. Prepare data references from prior cells
# --------------------------------------------------------------------
# Metadata from scan_frames (includes missing timestamps)
narration_meta = summary if isinstance(summary, dict) else {}
missing_dates_raw = narration_meta.get("missing_timestamps") or []
missing_dates = [str(ts).split("T")[0] for ts in missing_dates_raw if ts is not None]

# Analysis metadata (includes frame-level drought stats)
analysis_meta = analysis if isinstance(analysis, dict) else {}
analysis_frames = analysis_meta.get("frames") if isinstance(analysis_meta, dict) else {}

# Frame summaries (places and severity levels)
frames_locations = frames_locations if "frames_locations" in globals() else []
frames_locations_text = (
    frames_locations_text if "frames_locations_text" in globals() else []
)

print("frames_locations_text count:", len(frames_locations_text))
if not frames_locations_text:
    print("âš ï¸� No frame/location bullets found. Rerun the analysis and build-bullets cells before narrate.")
    narration = None
else:
    # ----------------------------------------------------------------
    # 2. Build the structured input payload for the swarm
    # ----------------------------------------------------------------
    narrative_text = "\n".join(frames_locations_text)
    narration_input = {
        "title": "Weekly Drought Risk Animation",
        "description": (
            "Start first sentence with 'Drought risk:' and summarize drought risk changes over time. "
            "Use only drought/dryness terminology; avoid storms, weather, precipitation, solar, "
            "or geomagnetic references. Include all per-frame drought risk bullets verbatim."
        ),
        "narrative": narrative_text,
        "data": {
            "frames_locations_text": frames_locations_text,
            "frames_metadata": narration_meta,
        },
    }

    # Paths for swarm configuration and rubric
    PACK_PATH = DROUGHT_DIR / "narrate_pack.yaml"
    RUBRIC_PATH = DROUGHT_DIR / "critic_rubric.yaml"

    # ----------------------------------------------------------------
    # 3. Write rubric for the critic/editor agents
    # ----------------------------------------------------------------
    RUBRIC_PATH.write_text(
        "- Reject any storms/weather/wind/precip/solar/geomagnetic/event wording; "
        "require explicit drought/dryness phrasing and fidelity to provided locations.\n"
        '- Summary must start with "Drought risk:".\n'
        "- Editor must preserve drought/dryness wording and all locations from the bullets.\n",
        encoding="utf-8",
    )

    # ----------------------------------------------------------------
    # 4. Execute Zyraâ€™s narration swarm
    # ----------------------------------------------------------------
    try:
        narration = sess.narrate.swarm(
            provider=ZYRA_LLM_PROVIDER,         # LLM backend provider (e.g., gemini, openai)
            model=ZYRA_LLM_MODEL,               # Specific model variant
            preset="scientific_rigorous",       # Balanced tone for clarity + scientific accuracy
            max_workers=1,                      # Single-threaded swarm (lightweight)
            max_rounds=3,                       # Up to 3 refinement rounds
            agents="context,summary,critic,editor",
            strict_grounding=False,             # Allow flexible interpretation of text data
            critic_structured=True,             # Enforce structured critic feedback
            rubric=str(RUBRIC_PATH),            # Custom rubric file for drought-only focus
            style="detailed",                   # Output style
            input_data=narration_input,         # The structured input prepared above
            pack=str(PACK_PATH),                # Where to write the full narration pack YAML
            memory="-",                         # No persistent memory between runs
        )
        print("âœ… Narration swarm executed successfully.")
        print("Narration summary:", narration)
    except Exception as exc:
        print(f"â�Œ Narration swarm not executed: {exc}")
        narration = None

    # ----------------------------------------------------------------
    # 5. Parse and display swarm output pack
    # ----------------------------------------------------------------
    try:
        pack_doc = yaml.safe_load(PACK_PATH.read_text()) or {}
        pack = pack_doc.get("narrative_pack") or {}
        outputs = pack.get("outputs") or {}
        errors = pack.get("errors") or []
        provenance = pack.get("provenance") or []

        print("ğŸ§© Agent outputs:")
        for k, v in outputs.items():
            print(f"  {k}: {str(v)[:200]}...")  # Print a short preview of each output

        if errors:
            print("âš ï¸� Errors:", errors)
        if provenance:
            print("ğŸ§¾ Provenance entries:")
            for entry in provenance:
                print("  ", entry)

        preview = pack.get("input_preview") or {}
        print("ğŸ“‹ Input preview:", preview)

        # Choose the final edited narration (or fallback to summary)
        final_narration = outputs.get("edited") or outputs.get("summary") or narration
        print("\nğŸ—£ï¸� Final drought narration:\n", final_narration)

    except Exception as exc:
        print(f"âš ï¸� Narration pack parse failed: {exc}")



# --- Display Final Drought Animation and Narration --- 
# (This takes a minute. the notebook is trying to encode the MP4 into the output stream.)
"""
This cell embeds the generated MP4 drought animation and prints the narration summary.

It uses IPythonâ€™s native `Video()` display utility, which properly encodes and embeds
the MP4 for inline playback inside Kaggle and Colab notebooks.

The animation was generated automatically by Zyraâ€™s `compose_video()` agent, while the
narration summary (if available) comes from Zyraâ€™s multi-agent Narrate Swarm.
"""

from IPython.display import Video, display
import os

# --------------------------------------------------------------------
# 1. Display the drought animation inline
# --------------------------------------------------------------------
if VIDEO_OUT.exists():
    print("ğŸ��ï¸� Weekly Drought Risk Animation")
    print(f"ğŸ“� Path: {VIDEO_OUT}\n")
    display(Video(str(VIDEO_OUT), embed=True, width=720, height=480))
else:
    print("âš ï¸� Drought animation not found. Please re-run the visualization cell to generate it.")

# --------------------------------------------------------------------
# 2. Display the narration summary if available
# --------------------------------------------------------------------
print("\nğŸ—£ï¸� Narration Summary:\n" + "-" * 80)
if "final_narration" in globals() and final_narration:
    print(final_narration)
elif "narration" in globals() and narration:
    print(narration)
else:
    print("â„¹ï¸� No narration available. Re-run the narrate swarm cell to generate an interpretation.")



# --- Define Zyra AI Workflow Planner Intent ---
"""
This cell defines the high-level *intent* for Zyraâ€™s workflow planner or swarm agents.

Zyra uses this intent string when reasoning about the workflow structure, planning
execution steps, or generating narration context. It provides a concise, human-
readable summary of what the notebook is meant to accomplish.

The intent is later passed to:
  â€¢ sess.run.plan() â€” to auto-generate a workflow DAG
  â€¢ sess.narrate.describe() â€” to produce a human explanation of the workflow
  â€¢ sess.decide.optimize() â€” to evaluate alternate execution strategies
"""

intent = (
    "Download the last year of Weekly Drought Risk PNG frames from NOAAâ€™s FTP, "
    "analyze frames with the registered drought analyzer, "
    "fill missing frames using basemap placeholders, "
    "compose an MP4 drought animation, "
    "and save the final output to disk for dissemination."
)

# Optional: confirm that the intent was stored correctly
print("ğŸ§­ Workflow intent defined:")
print(intent)



# --- Zyra AI Workflow Planner (Step 1: Plan + Fill + Augment) ---
# (This may take a minute, keep an eye out for questions from the agent.)
"""
This cell initializes Zyraâ€™s AI workflow planner, generates a draft execution plan,
and interactively fills in missing parameters *if running in an interactive session*.

In non-interactive Kaggle runs (e.g., during â€œSaveâ€� or competition submission),
it auto-fills known defaults and skips input() prompts.

Typical known values:
  â€¢ FTP path: ftp://ftp.nnvl.noaa.gov/SOS/DroughtRisk_Weekly
  â€¢ Pattern:  ^DroughtRisk_Weekly_[0-9]{8}\.png$
"""

import contextlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from zyra.swarm import planner as planner_mod
from zyra.swarm import suggest_augmentations

plan_path = DROUGHT_DIR / "plan_session_inline.json"
overlay_path = DROUGHT_DIR / "notebook_capabilities_overlay.json"
os.environ["ZYRA_NOTEBOOK_OVERLAY"] = str(overlay_path)

with contextlib.suppress(Exception):
    planner_mod.planner._caps = None  # reset overlays

# Detect if notebook is running in non-interactive mode (e.g., Kaggle headless)
IS_INTERACTIVE = sys.stdin and sys.stdin.isatty()

# --------------------------------------------------------------------
# 1. Generate initial plan
# --------------------------------------------------------------------
manifest = planner_mod.planner.plan(intent)
manifest = deepcopy(manifest)

# --------------------------------------------------------------------
# 2. Collect argument gaps and handle interactively (or fill defaults)
# --------------------------------------------------------------------
gaps = planner_mod._collect_arg_gaps(manifest)  # noqa: SLF001
GAP_REASONS = {"missing_arg", "placeholder", "resolver_hint", "confirm_choice"}

print("ğŸ§­ Zyra Planner: reviewing argument gaps...")

for gap in gaps:
    if gap.get("reason") not in GAP_REASONS:
        continue
    field = gap.get("field")
    agent_ref = gap.get("agent_ref")
    if not field or not isinstance(agent_ref, dict):
        continue

    stage = gap.get("stage") or ""
    command = gap.get("command") or ""
    label = gap.get("agent_id") or stage
    current = gap.get("current")
    suffix = f" (current: {current})" if current else ""
    help_text = planner_mod._field_help_text(stage, command, field)  # noqa: SLF001

    print(f"\nğŸ”§ Clarify [{label} â€” {stage} {command}] for '{field}'{suffix}")
    if help_text:
        print(f"   hint: {help_text}")

    if IS_INTERACTIVE:
        # Ask user only if running interactively
        resp = input("   Enter value (leave blank to skip): ").strip()
    else:
        # Non-interactive: apply known defaults or skip gracefully
        if field == "path":
            resp = "ftp://ftp.nnvl.noaa.gov/SOS/DroughtRisk_Weekly"
        elif field == "pattern":
            resp = r"^DroughtRisk_Weekly_[0-9]{8}\.png$"
        else:
            resp = current or ""

        if resp:
            print(f"   (auto-filled: {resp})")

    if not resp:
        continue
    args = agent_ref.setdefault("args", {})
    args[field] = resp

# --------------------------------------------------------------------
# 3. Apply LLM-suggested augmentations
# --------------------------------------------------------------------
auto_suggestions = suggest_augmentations(manifest, intent=intent)
accepted = []
if auto_suggestions and IS_INTERACTIVE:
    print("\nâœ¨ Suggestions (type 'y' to accept):")
    for sug in auto_suggestions:
        stage = sug.get("stage")
        desc = sug.get("description") or sug.get("text") or ""
        choice = input(f"  Accept {stage}: {desc}? [y/N] ").strip().lower()
        if choice == "y":
            accepted.append(sug)
elif auto_suggestions:
    print(f"\nâš™ï¸� Auto-applying {len(auto_suggestions)} planner suggestions (non-interactive mode).")
    accepted = auto_suggestions

if accepted:
    manifest = planner_mod._apply_suggestion_templates(manifest, accepted)  # noqa: SLF001
    manifest["accepted_suggestions"] = accepted
else:
    manifest["suggestions"] = auto_suggestions

# --------------------------------------------------------------------
# 4. Save manifest
# --------------------------------------------------------------------
plan_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"\nâœ… Planning complete. Manifest written to: {plan_path}")



# --- Zyra AI Workflow Planner (Step 2: Normalize + Validate + Save) ---
"""
This cell completes the Zyra planning workflow by normalizing agent arguments,
ensuring required dependencies exist (scan, pad, compose, disseminate), validating
the manifest, and writing it to disk as a JSON file.

Use this section after running the initial planning cell.
"""

# --------------------------------------------------------------------
# 5. Normalize custom analyzer args and dependencies
# --------------------------------------------------------------------
agents = manifest.setdefault("agents", [])
pad_id = next((a.get("id") for a in agents if a.get("command") == "pad-missing"), None)

for agent in agents:
    if agent.get("command") == "analyze_drought_frames":
        args = agent.setdefault("args", {})
        args["frames_dir"] = str(FRAMES_PADDED)
        args["colorbar"] = str(DROUGHT_DIR / "VTHI.colorbar.png")
        args.setdefault("tolerance", 30)
        deps = agent.setdefault("depends_on", [])
        if pad_id and pad_id not in deps:
            deps.append(pad_id)

# --------------------------------------------------------------------
# 6. Ensure required pipeline agents exist (scan, pad, compose, local)
# --------------------------------------------------------------------
ids = {a.get("id") for a in agents if isinstance(a, dict)}

ftp_agent = next((a for a in agents if a.get("command") == "ftp"), None)
if ftp_agent:
    ftp_args = ftp_agent.setdefault("args", {})
    ftp_args.setdefault("sync_dir", str(FRAMES_RAW))
    ftp_args.setdefault("pattern", PATTERN)
    ftp_args.setdefault("date_format", "%Y%m%d")

# Ensure process/scan stage
scan_agent = next(
    (a for a in agents if a.get("stage") == "process" and a.get("command") == "scan-frames"),
    None,
)
if not scan_agent:
    scan_agent = {
        "id": "scan_frames" if "scan_frames" not in ids else "scan_frames_1",
        "stage": "process",
        "command": "scan-frames",
        "depends_on": [ftp_agent.get("id")] if ftp_agent else [],
        "args": {
            "frames_dir": str(FRAMES_RAW),
            "pattern": PATTERN,
            "datetime_format": "%Y%m%d",
            "period_seconds": CADENCE_SECONDS,
            "output": str(FRAMES_META),
        },
    }
    agents.append(scan_agent)

# Ensure pad-missing stage
pad_agent = next(
    (a for a in agents if a.get("stage") == "process" and a.get("command") == "pad-missing"),
    None,
)
if not pad_agent:
    pad_agent = {
        "id": "pad_missing" if "pad_missing" not in ids else "pad_missing_1",
        "stage": "process",
        "command": "pad-missing",
        "depends_on": [scan_agent.get("id")],
        "args": {
            "frames_meta": str(FRAMES_META),
            "output_dir": str(FRAMES_PADDED),
            "fill_mode": "basemap",
            "basemap": BASEMAP_REF,
            "overwrite": True,
        },
    }
    agents.append(pad_agent)

# Ensure compose-video and local dissemination dependencies
compose_agent = next(
    (a for a in agents if a.get("stage") == "visualize" and a.get("command") == "compose-video"),
    None,
)
if compose_agent:
    args = compose_agent.setdefault("args", {})
    args.setdefault("frames", str(FRAMES_PADDED))
    args.setdefault("output", str(VIDEO_OUT))
    deps = compose_agent.setdefault("depends_on", [])
    if pad_agent and pad_agent.get("id") not in deps:
        deps.append(pad_agent.get("id"))

local_agent = next(
    (a for a in agents if a.get("command") == "local" and a.get("stage") in {"decimate", "disseminate", "export"}),
    None,
)
if compose_agent and local_agent:
    la_args = local_agent.setdefault("args", {})
    la_args.setdefault("input", compose_agent.get("args", {}).get("output", str(VIDEO_OUT)))
    la_args.setdefault("path", str(VIDEO_OUT))
    deps = local_agent.setdefault("depends_on", [])
    if compose_agent.get("id") not in deps:
        deps.append(compose_agent.get("id"))

# --------------------------------------------------------------------
# 7. Validate and finalize manifest
# --------------------------------------------------------------------
manifest = planner_mod._propagate_inferred_args(manifest)  # noqa: SLF001
errors = planner_mod._validate_manifest(manifest)  # noqa: SLF001

if errors:
    print("\nâš ï¸� Validation warnings:")
    for err in errors:
        print("  -", err)

planner_mod._ensure_auto_verify_agent(manifest)       # noqa: SLF001
planner_mod._ensure_verify_agents_materialized(manifest)  # noqa: SLF001
planner_mod._trace_agent_reasoning(manifest, intent)  # noqa: SLF001

# Embed workflow intent for provenance
manifest.setdefault("metadata", {})["intent"] = intent
manifest["intent"] = intent

# --------------------------------------------------------------------
# 8. Save manifest and print summary
# --------------------------------------------------------------------
plan_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("ğŸ“„ Plan written to:", plan_path)
print(plan_path.read_text()[:1500])  # Preview the start of the manifest

print("\nğŸ’¡ Suggestions summary:")
print("Accepted:", len(manifest.get("accepted_suggestions") or []))
print("Remaining:", len(manifest.get("suggestions") or []))



# --- Visualize and Verify Final Workflow Plan ---
"""
This cell summarizes the finalized Zyra workflow manifest produced by the
planner cells above. It prints a readable execution DAG, showing each
agentâ€™s stage, command, and dependencies.

This helps verify that all required steps (acquire â†’ process â†’ visualize â†’ disseminate)
are properly linked before execution.
"""

import json

# Load manifest if not in memory
if "manifest" not in globals() and plan_path.exists():
    manifest = json.loads(plan_path.read_text())

# Guard clause
if not manifest or "agents" not in manifest:
    raise RuntimeError("No workflow manifest found â€” run planner cells first.")

agents = manifest.get("agents", [])
if not agents:
    print("âš ï¸� No agents found in manifest.")
else:
    print("âœ… Workflow Plan Overview")
    print("=" * 70)

    # Helper: build lookup for dependencies
    id_map = {a.get("id"): a for a in agents if isinstance(a, dict)}

    for a in agents:
        stage = a.get("stage", "unknown")
        cmd = a.get("command", "unknown")
        agent_id = a.get("id", "unnamed")
        deps = a.get("depends_on") or []
        args = a.get("args") or {}

        # Header line
        print(f"\nğŸ”¹ [{stage.upper()}] {agent_id} â†’ {cmd}")
        if deps:
            dep_labels = ", ".join(deps)
            print(f"   â”œâ”€ Depends on: {dep_labels}")
        else:
            print("   â”œâ”€ Depends on: (none)")

        # Key arguments
        for k, v in list(args.items())[:6]:
            val_str = str(v)
            if len(val_str) > 80:
                val_str = val_str[:77] + "..."
            print(f"   â”œâ”€ {k}: {val_str}")

    print("\n" + "=" * 70)
    print(f"Total agents: {len(agents)}")
    stages = sorted(set(a.get("stage", "unknown") for a in agents))
    print("Pipeline stages:", " â†’ ".join(stages))



# --- Generate Zyra Pipeline + CLI Command Summary ---
"""
This cell extracts and displays the final workflow pipeline as understood by
the active Zyra session.

Outputs:
  â€¢ A structured Python representation of the pipeline (ordered stages + agents)
  â€¢ The equivalent CLI commands that Zyra would execute if run via command line

Purpose:
  â€¢ Validate that the in-notebook session is correctly materialized into a
    reproducible, command-line-compatible pipeline.
  â€¢ Provide human-readable reference for external automation (e.g., CI/CD jobs).
"""

from pprint import pprint

# --------------------------------------------------------------------
# 1. Convert active Zyra session to a structured pipeline
# --------------------------------------------------------------------
pipeline = sess.to_pipeline()  # Returns ordered pipeline structure
cli_cmds = sess.to_cli()       # Returns equivalent CLI command list

# --------------------------------------------------------------------
# 2. Display the pipeline summary
# --------------------------------------------------------------------
print("ğŸ§± Pipeline Stages & Agents:")
print("=" * 80)
pprint(pipeline, sort_dicts=False)

# --------------------------------------------------------------------
# 3. Display the equivalent CLI commands
# --------------------------------------------------------------------
print("\nğŸ’» Equivalent Zyra CLI Commands:")
print("=" * 60)
for cmd in cli_cmds:
    print(cmd)


# --- Inspect Zyra Provenance SQLite Logs ---
"""
This cell inspects the Zyra provenance database, which automatically tracks
executed runs, agent events, timestamps, and statuses.

Purpose:
  â€¢ Verify that recent workflow runs were logged correctly
  â€¢ Inspect the `runs` and `events` tables for traceability
  â€¢ Confirm event payloads, agent activity, and execution metadata

By default, this database is located at:
  ${ZYRA_NOTEBOOK_PROVENANCE:-<workspace>/provenance.sqlite}
"""

import os
import sqlite3
from contextlib import suppress
from pathlib import Path

# --------------------------------------------------------------------
# 1. Resolve provenance database path
# --------------------------------------------------------------------
prov_path = Path(
    os.environ.get("ZYRA_NOTEBOOK_PROVENANCE", WORKSPACE / "provenance.sqlite")
)
print(f"ğŸ§­ Provenance DB Path: {prov_path}")

if not prov_path.exists():
    print("â�Œ Provenance database not found. Run a few Zyra commands first to log activity.")
else:
    try:
        # ----------------------------------------------------------------
        # 2. Connect to the SQLite database and enumerate tables
        # ----------------------------------------------------------------
        conn = sqlite3.connect(prov_path)
        cur = conn.cursor()

        tables = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t[0] for t in tables]
        print("\nğŸ“‹ Tables:", ", ".join(table_names) if table_names else "(none)")

        # ----------------------------------------------------------------
        # 3. Show the latest recorded runs (if available)
        # ----------------------------------------------------------------
        if "runs" in table_names:
            runs = cur.execute(
                "SELECT run_id, started, completed, status FROM runs ORDER BY started DESC LIMIT 5"
            ).fetchall()
            print("\nğŸ§¾ Recent Runs (latest 5):")
            if not runs:
                print("  (no runs logged yet)")
            for row in runs:
                run_id, started, completed, status = row
                print(f"  â–¶ {run_id} | {status} | {started} â†’ {completed}")

        # ----------------------------------------------------------------
        # 4. Show recent event records (if available)
        # ----------------------------------------------------------------
        if "events" in table_names:
            cols = cur.execute("PRAGMA table_info(events)").fetchall()
            col_names = [c[1] for c in cols]
            print("\nğŸ“‘ Event Columns:", ", ".join(col_names))

            rows = cur.execute(
                "SELECT id, run_id, event, agent, created, payload "
                "FROM events ORDER BY id DESC LIMIT 10"
            ).fetchall()

            if not rows:
                print("  (no events recorded yet)")
            else:
                print("\nğŸ”¹ Last 10 Events (most recent first):")
                for (event_id, run_id, event, agent, created, payload) in rows:
                    payload_str = str(payload)
                    if len(payload_str) > 120:
                        payload_str = payload_str[:117] + "..."
                    print(
                        f"  â€¢ #{event_id} | run {run_id} | {agent} | {event} | {created}\n"
                        f"    payload: {payload_str}"
                    )

    except Exception as exc:
        print("âš ï¸� Failed to read provenance DB:", exc)
    finally:
        with suppress(Exception):
            conn.close()



# --- Export Zyra Workflow Artifacts for Competition Submission ---
"""
This cell prepares Zyraâ€™s complete output package for competition submission.

Included files:
  â€¢ drought_animation.mp4 â€” final animation generated by visualize.compose_video()
  â€¢ analysis.json â€” structured drought analysis results
  â€¢ plan_session_inline.json â€” AI workflow manifest from the planner
  â€¢ frames_meta.json â€” frame cadence and missing timestamp metadata
  â€¢ provenance.sqlite â€” Zyraâ€™s full provenance and execution log
  â€¢ VTHI.colorbar.png â€” NOAA colorbar reference used for drought classification

All files are copied to /kaggle/outputs and additionally zipped into zyra_submission.zip
for convenience and reproducibility.
"""

import shutil
from pathlib import Path

OUTPUT_DIR = Path("/kaggle/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

artifacts = [
    DROUGHT_DIR / "drought_animation.mp4",
    DROUGHT_DIR / "analysis.json",
    DROUGHT_DIR / "plan_session_inline.json",
    DROUGHT_DIR / "frames_meta.json",
    Path(os.environ.get("ZYRA_NOTEBOOK_PROVENANCE", WORKSPACE / "provenance.sqlite")),
    DROUGHT_DIR / "VTHI.colorbar.png",
]

print("ğŸ“¦ Preparing Zyra submission package...\n")
for artifact in artifacts:
    if artifact.exists():
        shutil.copy2(artifact, OUTPUT_DIR / artifact.name)
        print(f"âœ… Copied: {artifact.name}")
    else:
        print(f"âš ï¸� Missing: {artifact.name} (skipped)")

# Optional: also export a zip bundle
shutil.make_archive(str(OUTPUT_DIR / "zyra_submission"), "zip", DROUGHT_DIR)
print("\nğŸ�� Created archive: zyra_submission.zip")

# Confirm
exported_files = list(OUTPUT_DIR.glob("*"))
print("\nğŸ�‰ Export complete â€” files ready for competition submission:\n")
for f in exported_files:
    print(" â€¢", f.name)


