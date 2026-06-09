# ====================================================================
# ğŸ“š IMPORTS AND SETUP
# ====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.signal import medfilt
from scipy import stats
from IPython.display import display, Markdown
import warnings
warnings.filterwarnings('ignore')


# ====================================================================
# âš™ï¸� Configuration
# ====================================================================
BASE_PATH = Path('/kaggle/input/ariel-data-challenge-2025/')
SAMPLE_PLANET_ID = '1010375142'

# Styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150

def display_header(title):
    """Helper function to display styled headers."""
    display(Markdown(f"### {title}"))


# ====================================================================
# ğŸ�¯ SECTION 1: COMPETITION OVERVIEW & CORE TASK
# ====================================================================

display_header("ğŸ�¯ Competition Overview")
display(Markdown(
    "**The Ariel Data Challenge 2025** is a ML competition focused on extracting "
    "exoplanet atmospheric spectra from noisy telescope observations. This is a **supervised "
    "denoising problem** where we can predict 283 spectral values and their uncertainties "
    "from complex time-series data."
))

print("ğŸ”� CORE CHALLENGE:")
print("â€¢ Extract faint exoplanet atmospheric signals from noisy telescope data")
print("â€¢ Extract 283 spectral values + uncertainties from complex 3D time-series data")
print("â€¢ Handle complex 3D data cubes (time Ã— spatial Ã— spectral dimensions)")
print("â€¢ Predict both mean spectrum values AND uncertainty estimates")
print("â€¢ Score optimization using Gaussian Log-Likelihood (GLL) metric")


# ====================================================================
# ğŸ�¯ SECTION 2: METADATA DEEP DIVE
# ====================================================================

# --- 2.1 The Ground Truth: train.csv ---
display_header("`train.csv`: The Ground Truth (Target)")
train_df = pd.read_csv(BASE_PATH / 'train.csv')

print(f"ğŸ“Š Training Data Shape: {train_df.shape}")
print(f"ğŸ“Š Sample Planet ID: {SAMPLE_PLANET_ID}")

# Sample data
sample_data = train_df[train_df['planet_id'] == int(SAMPLE_PLANET_ID)]
display(sample_data)

display(Markdown(
    "**ğŸ’¡ Significance:** This is the **target variable**. Each row represents one planet, "
    "with 283 `wl_*` columns containing the true, clean atmospheric spectrum values. "
    "The model must learn to predict these values from the raw observational data."
))

# --- 2.2 The Spectral Grid: wavelengths.csv ---
display_header("`wavelengths.csv`: The Spectral Grid")
wavelengths_df = pd.read_csv(BASE_PATH / 'wavelengths.csv')

print(f"ğŸ“Š Wavelength Grid Shape: {wavelengths_df.shape}")
display(wavelengths_df.iloc[:, :5])  # first 5 wavelengths

# Wavelength array
wavelengths = wavelengths_df.iloc[0].values
print(f"ğŸ“Š Wavelength Statistics:")
print(f"   â€¢ Range: {wavelengths.min():.3f} - {wavelengths.max():.3f} Î¼m")
print(f"   â€¢ Mean spacing: {np.diff(wavelengths).mean():.6f} Î¼m")
print(f"   â€¢ Total points: {len(wavelengths)}")

display(Markdown(
    "**ğŸ’¡ Significance:** This wavelengths.csv file provides the **physical wavelength mapping** for each "
    "spectral point. Essential for physics-informed feature engineering and instrument-specific "
    "processing (FGS1 vs AIRS-CH0 ranges)."
))

# --- 2.3 The Time Axis: axis_info.parquet ---
display_header("`axis_info.parquet`: The Time Axis")
axis_info_df = pd.read_parquet(BASE_PATH / 'axis_info.parquet')

print(f"ğŸ“Š Axis Info Shape: {axis_info_df.shape}")
print(f"ğŸ“Š Column Names: {axis_info_df.columns.tolist()}")
print(f"ğŸ“Š Data Types: {axis_info_df.dtypes.to_dict()}")

display(axis_info_df.head())

print("\nğŸ”� TIME AXIS INSPECTION:")
for col in axis_info_df.columns:
    col_data = axis_info_df[col]
    print(f"   â€¢ {col}:")
    print(f"     - Range: {col_data.min():.6f} to {col_data.max():.6f}")
    print(f"     - Mean: {col_data.mean():.6f}")
    print(f"     - Non-null count: {col_data.notna().sum()}")

display(Markdown(
    "**ğŸ’¡ Significance:** This axis_info.parquet defines the **temporal dimension** of the raw data. "
    "The structure shows separate time axes for different instruments and observation modes. "
    "Critical for time-series analysis and transit event detection."
))

# --- 2.4 The Conversion Key: adc_info.csv ---
display_header("`adc_info.csv`: The ADC Conversion Key")
adc_info_df = pd.read_csv(BASE_PATH / 'adc_info.csv')

print(f"ğŸ“Š ADC Info Shape: {adc_info_df.shape}")
display(adc_info_df)

# Conversion parameters
fgs1_gain = adc_info_df['FGS1_adc_gain'].iloc[0]
fgs1_offset = adc_info_df['FGS1_adc_offset'].iloc[0]
airs_gain = adc_info_df['AIRS-CH0_adc_gain'].iloc[0]
airs_offset = adc_info_df['AIRS-CH0_adc_offset'].iloc[0]

print(f"ğŸ“Š Conversion Parameters:")
print(f"   â€¢ FGS1: gain={fgs1_gain:.6f}, offset={fgs1_offset:.6f}")
print(f"   â€¢ AIRS-CH0: gain={airs_gain:.6f}, offset={airs_offset:.6f}")

display(Markdown(
    "**ğŸ’¡ Significance:** Raw observational data is stored as integers to save space. "
    "These parameters convert to physical flux units using: "
    "`physical_value = (raw_value Ã— gain) + offset`. "
))

# --- 2.5 Physical Context: train_star_info.csv ---
display_header("`train_star_info.csv`: Physical Context for Features")
train_star_info_df = pd.read_csv(BASE_PATH / 'train_star_info.csv')

print(f"ğŸ“Š Star Info Shape: {train_star_info_df.shape}")
sample_star_info = train_star_info_df[train_star_info_df['planet_id'] == int(SAMPLE_PLANET_ID)]
display(sample_star_info)

# Statistical overview of stellar parameters
print(f"ğŸ“Š Stellar Parameter Statistics:")
numeric_cols = train_star_info_df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != 'planet_id':
        values = train_star_info_df[col]
        print(f"   â€¢ {col}: {values.min():.3f} - {values.max():.3f} (mean: {values.mean():.3f})")

display(Markdown(
    "**ğŸ’¡ Significance:** **Metadata** containing stellar and planetary parameters. "
    "Properties like star temperature (`Ts`), planetary mass (`Mp`), and orbital inclination (`i`) "
    "may influence transit signals. **Powerful features for model enhancement.**"
))

# --- 2.6 Sample_submission.csv ---
display_header("`sample_submission.csv`: The Final Goal")
sample_submission_df = pd.read_csv(BASE_PATH / 'sample_submission.csv')

print(f"ğŸ“Š Submission Format Shape: {sample_submission_df.shape}")
print(f"ğŸ“Š Expected Columns: {sample_submission_df.shape[1]} (1 + 283 + 283)")
print(f"ğŸ“Š Test Planets: {len(sample_submission_df)}")

display(sample_submission_df.head())

# Submission format
expected_cols = 1 + 283 + 283  # planet_id + spectrum + uncertainty
actual_cols = sample_submission_df.shape[1]
format_check = "âœ… CORRECT" if actual_cols == expected_cols else "â�Œ MISMATCH"

print(f"ğŸ“Š Format Validation: {format_check}")
print(f"   â€¢ Expected: {expected_cols} columns")
print(f"   â€¢ Actual: {actual_cols} columns")

display(Markdown(
    "**ğŸ’¡ Significance:** This defines the **exact submission format**. For each test planet (only one is there though), "
    "we must predict 283 mean spectrum values (`wl_*`) AND 283 uncertainty values (`sigma_*`). "
))


# ====================================================================
# ğŸ”§ SECTION 3: SIGNAL PROCESSING FUNCTIONS
# ====================================================================

display_header("ğŸ”§ Signal Processing Functions")

def adc_calibration(raw_df, gain, offset):
    """Convert raw ADC uint16 values to physical flux units"""
    return raw_df * gain + offset

def differential_reading(signal_df):
    """Extract differential signal using up-the-ramp sampling"""
    # Extract even (start) and odd (end) frames
    start_frames = signal_df.iloc[0::2].reset_index(drop=True)
    end_frames = signal_df.iloc[1::2].reset_index(drop=True)
    
    # differential signal
    diff_signal = end_frames - start_frames
    return diff_signal

def create_light_curve(diff_signal):
    """Generate light curve by summing across all pixels"""
    return diff_signal.sum(axis=1)

def sigma_clip(data, window_size=51, sigma=5):
    """ sigma clipping for cosmic ray removal"""
    local_median = medfilt(data, kernel_size=window_size)
    residual = data - local_median
    mad = np.median(np.abs(residual))
    robust_std = mad * 1.4826
    outliers = np.abs(residual) > (sigma * robust_std)
    clean_data = data.copy()
    clean_data[outliers] = np.nan
    return clean_data, outliers

def polynomial_detrend(data, order=2, transit_mask=None):
    """Remove systematic trends using polynomial fitting"""
    x_axis = np.arange(len(data))
    
    if transit_mask is not None:
        fit_mask = ~transit_mask & ~np.isnan(data)
    else:
        fit_mask = ~np.isnan(data)
    
    poly_coeffs = np.polyfit(x_axis[fit_mask], data[fit_mask], order)
    baseline_model = np.polyval(poly_coeffs, x_axis)
    detrended = data - baseline_model
    
    return detrended, baseline_model

def detect_transit(light_curve, buffer_size=500):
    """Detect transit event in light curve"""
    min_idx = np.nanargmin(light_curve)
    min_depth = light_curve[min_idx]
    
    transit_start = max(0, min_idx - buffer_size)
    transit_end = min(len(light_curve) - 1, min_idx + buffer_size)
    
    return transit_start, transit_end, min_depth

print("âœ… Signal processing functions loaded successfully")


# ====================================================================
# ğŸ“Š SECTION 4: MERGE TRAINING DATA
# ====================================================================

display_header("ğŸ“Š Merge training data for comprehensive analysis")

merged_train_df = pd.merge(train_df, train_star_info_df, on='planet_id')

# Reshape wavelengths
wavelengths_long_df = wavelengths_df.T.reset_index()
wavelengths_long_df.columns = ['wl_id', 'wavelength_um']
wavelengths_long_df['instrument'] = np.where(wavelengths_long_df['wavelength_um'] < 1.0, 'FGS1', 'AIRS-CH0')

# Extract key parameters
wavelengths = wavelengths_df.iloc[0].values
gain = adc_info_df['FGS1_adc_gain'].iloc[0]
offset = adc_info_df['FGS1_adc_offset'].iloc[0]

print(f"ğŸ“Š Dataset Overview:")
print(f"   â€¢ Training planets: {len(train_df):,}")
print(f"   â€¢ Test planets: {len(sample_submission_df):,}")
print(f"   â€¢ Wavelength points: {len(wavelengths)}")
print(f"   â€¢ Wavelength range: {wavelengths.min():.3f} - {wavelengths.max():.3f} Î¼m")
print(f"   â€¢ ADC parameters: gain={gain:.4f}, offset={offset:.1f}")
    
# Instrument analysis
fgs1_count = (wavelengths_long_df['instrument'] == 'FGS1').sum()
airs_count = (wavelengths_long_df['instrument'] == 'AIRS-CH0').sum()

print(f"   â€¢ FGS1 points: {fgs1_count}")
print(f"   â€¢ AIRS-CH0 points: {airs_count}")


# ====================================================================
# ğŸ”¬ SECTION 5: OBSERVATIONAL DATA EXPLORATION
# ====================================================================

# --- 5.1 Instrument & Observation Set Analysis ---
display_header("ğŸ›°ï¸� Instrument & Observation Set Analysis")

planet_folders = [p for p in (BASE_PATH / 'train').iterdir() if p.is_dir()]
print(f"ğŸ“Š Total Planet Folders: {len(planet_folders)}")

sample_size = max(100, len(planet_folders)) # Total 1100 are there or sample 100 (by changing min or max)
sample_folders = planet_folders[:sample_size]

# Counting structure
observation_analysis = {
    'single_observation_set': 0,      # Planets with _0 files only
    'double_observation_set': 0,      # Planets with _0 and _1 files
}

# Planet information storage
planet_details = {}

print(f"ğŸ”� Analyzing observation sets for {sample_size} planets...")

for planet_folder in sample_folders:
    planet_id = planet_folder.name
    
    # Find all calibration folders
    calibration_folders = [f for f in planet_folder.iterdir() 
                          if f.is_dir() and ('calibration' in f.name.lower())]
    
    # Find all signal files
    signal_files = [f for f in planet_folder.iterdir() 
                   if f.is_file() and f.name.endswith('.parquet') and 'signal' in f.name]
    
    # Extract observation set numbers from calibration folders
    calib_suffixes = set()
    for calib_folder in calibration_folders:
        # Extracting suffix from names like "AIRS-CH0_calibration_0" or "FGS1_calibration_1"
        suffix = calib_folder.name.split('_')[-1]
        calib_suffixes.add(suffix)
    
    # Extract observation set numbers from signal files
    signal_suffixes = set()
    for signal_file in signal_files:
        # Extracting suffix from names like "FGS1_signal_0.parquet" or "AIRS-CH0_signal_1.parquet"
        parts = signal_file.name.split('_')
        if len(parts) >= 3:
            suffix = parts[-1].replace('.parquet', '')
            signal_suffixes.add(suffix)
    
    # Find common observation sets (both calibration and signal data available)
    common_sets = calib_suffixes.intersection(signal_suffixes)
    num_observation_sets = len(common_sets)
    
    # Store detailed information
    planet_details[planet_id] = {
        'observation_sets': sorted(list(common_sets)),
        'num_sets': num_observation_sets,
        'calibration_folders': len(calibration_folders),
        'signal_files': len(signal_files),
        'all_folders': [f.name for f in planet_folder.iterdir() if f.is_dir()],
        'all_files': [f.name for f in planet_folder.iterdir() if f.is_file()]
    }
    
    # Count observation sets
    if num_observation_sets == 1:
        observation_analysis['single_observation_set'] += 1
    elif num_observation_sets == 2:
        observation_analysis['double_observation_set'] += 1

print(f"\nğŸ“Š OBSERVATION SET ANALYSIS (sample of {sample_size} planets):")
print(f"   â€¢ Single observation set (e.g., _0 only): {observation_analysis['single_observation_set']} planets ({observation_analysis['single_observation_set']/sample_size*100:.1f}%)")
print(f"   â€¢ Double observation sets (e.g., _0 + _1): {observation_analysis['double_observation_set']} planets ({observation_analysis['double_observation_set']/sample_size*100:.1f}%)")

# Examples of different observation set patterns
print(f"\nğŸ”� EXAMPLE OBSERVATION PATTERNS:")
single_example = next((pid for pid, details in planet_details.items() if details['num_sets'] == 1), None)
double_example = next((pid for pid, details in planet_details.items() if details['num_sets'] == 2), None)

if single_example:
    details = planet_details[single_example]
    print(f"   â€¢ Single set example (Planet {single_example}):")
    print(f"     - Observation sets: {details['observation_sets']}")
    print(f"     - Calibration folders: {details['calibration_folders']}")
    print(f"     - Signal files: {details['signal_files']}")
    print(f"     - All folders: {details['all_folders']}")

if double_example:
    details = planet_details[double_example]
    print(f"   â€¢ Double set example (Planet {double_example}):")
    print(f"     - Observation sets: {details['observation_sets']}")
    print(f"     - Calibration folders: {details['calibration_folders']}")
    print(f"     - Signal files: {details['signal_files']}")
    print(f"     - All folders: {details['all_folders']}")

# First few planet details for debugging
print(f"\nğŸ”� FIRST FEW PLANET DETAILS (for debugging):")
for i, (pid, details) in enumerate(list(planet_details.items())[:3]):
    print(f"   Planet {pid}:")
    print(f"     - Folders: {details['all_folders']}")
    print(f"     - Files: {details['all_files']}")
    print(f"     - Observation sets found: {details['observation_sets']}")

display(Markdown(
    "**ğŸ’¡ Insights:**\n\n"
    "- **Multiple observation sets** may increase data volume and complexity\n"
    "- **Double observation planets** provide more training data but require careful handling\n"
    "- **Memory estimation** needs to factor in observation set multiplicity\n\n"
))


# --- 5.2 Visualizing a Flattened Image ---
display_header(f"ğŸ–¼ï¸� Telescope Image Visualization (Planet {SAMPLE_PLANET_ID})")

fgs1_signal = None
try:
    fgs1_signal = pd.read_parquet(BASE_PATH / f'train/{SAMPLE_PLANET_ID}/FGS1_signal_0.parquet')
    print(f"ğŸ“Š FGS1 Signal Shape: {fgs1_signal.shape}")
    print(f"ğŸ“Š Time Steps: {len(fgs1_signal):,}")
    print(f"ğŸ“Š Pixels per Frame: {fgs1_signal.shape[1]}")
    
    # Take a frame from the middle of the observation
    frame_idx = len(fgs1_signal) // 2
    flattened_image = fgs1_signal.iloc[frame_idx].values
    
    # Reshape from 1D vector (1024,) to 2D image (32, 32)
    image_2d = flattened_image.reshape(32, 32)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Raw image
    im1 = ax1.imshow(image_2d, cmap='viridis', aspect='equal')
    ax1.set_title(f'FGS1 Detector Image\n(Planet {SAMPLE_PLANET_ID}, Frame {frame_idx:,})')
    ax1.set_xlabel('Pixel Column')
    ax1.set_ylabel('Pixel Row')
    fig.colorbar(im1, ax=ax1, label='Flux (Raw ADC Units)')
    
    # Plot 2: Cross-section showing the stellar trace
    middle_row = image_2d[16, :]  # Middle row
    ax2.plot(range(32), middle_row, 'b-', linewidth=2, label='Stellar Trace')
    ax2.set_title('Stellar Trace (Middle Row)')
    ax2.set_xlabel('Pixel Column')
    ax2.set_ylabel('Flux (Raw ADC Units)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"ğŸ“Š Image Statistics:")
    print(f"   â€¢ Min flux: {flattened_image.min():.1f}")
    print(f"   â€¢ Max flux: {flattened_image.max():.1f}")
    print(f"   â€¢ Mean flux: {flattened_image.mean():.1f}")
    print(f"   â€¢ Dynamic range: {flattened_image.max()/flattened_image.min():.2f}x")
    
    display(Markdown(
        "**ğŸ’¡ Significance:** This shows what the telescope **actually observes**. "
        "The raw data is a time series of these 32Ã—32 pixel images, flattened into 1024-element vectors. "
        "The bright horizontal band is **starlight dispersed by the spectrograph**. "
        "**The planet's atmospheric signal is hidden in the subtle changes of this trace over time.**"
    ))

except FileNotFoundError:
    print(f"âš ï¸� Could not find FGS1 data for planet {SAMPLE_PLANET_ID}")
    print("This planet may not have FGS1 observations available.")


# --- 5.3 Time Series Analysis: Signal vs. Noise ---
display_header("ğŸ“ˆ Signal vs. Noise Analysis")

def extract_time_data(axis_info_df, signal_length, instrument='FGS1'):
    """
    Time data extraction handling various column name formats
    """
    print(f"ğŸ”� Extracting time data for {instrument}...")
    print(f"ğŸ“Š Available columns: {axis_info_df.columns.tolist()}")
    
    # Looking for exact instrument-specific column
    target_patterns = [
        f'{instrument}-axis0-h',
        f'{instrument}-axis0',
        f'time_{instrument}',
        f'{instrument}_time'
    ]
    
    for pattern in target_patterns:
        if pattern in axis_info_df.columns:
            print(f"âœ… Found time column: {pattern}")
            time_data = axis_info_df[pattern].values
            return time_data[:signal_length] if len(time_data) >= signal_length else time_data
    
    print("âš ï¸� Generating synthetic time array (dummy data)")
    return np.linspace(0, signal_length-1, signal_length)


if fgs1_signal is not None:
    # Extract time data
    time_data = extract_time_data(axis_info_df, len(fgs1_signal), 'FGS1')
    
    # Converting to seconds if in hours
    if time_data.max() < 1.0:  # Likely in hours
        time_data_seconds = time_data * 3600
        time_unit = "seconds"
    else:
        time_data_seconds = time_data
        time_unit = "time units"
    
    # Select pixel for analysis (center of detector)
    pixel_idx = 512  # Center pixel
    pixel_flux = fgs1_signal.iloc[:, pixel_idx].values
    
    # Quality check
    valid_indices = ~np.isnan(time_data_seconds) & ~np.isnan(pixel_flux)
    n_valid = valid_indices.sum()
    
    print(f"ğŸ“Š Time Series Statistics:")
    print(f"   â€¢ Total time points: {len(time_data_seconds):,}")
    print(f"   â€¢ Valid data points: {n_valid:,}")
    print(f"   â€¢ Observation duration: {time_data_seconds.max():.1f} {time_unit}")
    print(f"   â€¢ Pixel analyzed: {pixel_idx} (center)")
    
    if n_valid > 10:  # trying only 10 data for analysis
        # Fit polynomial trend (instrument systematics)
        poly_degree = 3  # Slightly higher order for better fit
        poly_coeffs = np.polyfit(time_data_seconds[valid_indices], pixel_flux[valid_indices], poly_degree)
        poly_fit = np.poly1d(poly_coeffs)
        trend_line = poly_fit(time_data_seconds[valid_indices])
        
        # Calculate detrended signal
        detrended_signal = pixel_flux - poly_fit(time_data_seconds)
        
        # Transit detection (approximate)
        min_flux_idx = np.nanargmin(pixel_flux)
        transit_buffer = min(20000, len(pixel_flux) // 10)
        transit_start = max(0, min_flux_idx - transit_buffer)
        transit_end = min(len(pixel_flux) - 1, min_flux_idx + transit_buffer)
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Plot 1: Raw signal with trend
        ax1 = axes[0, 0]
        ax1.plot(time_data_seconds, pixel_flux, 'b-', alpha=0.7, linewidth=1, 
                label='Raw Signal')
        ax1.plot(time_data_seconds[valid_indices], trend_line, 'g--', 
                linewidth=2, label='Systematic Trend')
        ax1.axvspan(time_data_seconds[transit_start], time_data_seconds[transit_end], 
                   color='orange', alpha=0.2, label='Transit Region')
        ax1.set_xlabel(f'Time ({time_unit})')
        ax1.set_ylabel('Flux (ADC Units)')
        ax1.set_title('Raw Signal + Instrumental Trend')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Detrended signal
        ax2 = axes[0, 1]
        ax2.plot(time_data_seconds, detrended_signal, 'r-', alpha=0.7, linewidth=1,
                label='Detrended Signal')
        ax2.axvspan(time_data_seconds[transit_start], time_data_seconds[transit_end], 
                   color='orange', alpha=0.2, label='Transit Region')
        ax2.set_xlabel(f'Time ({time_unit})')
        ax2.set_ylabel('Relative Flux')
        ax2.set_title('Detrended Signal (Transit Visible)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Transit zoom-in
        ax3 = axes[1, 0]
        zoom_margin = transit_buffer // 2
        zoom_start = max(0, min_flux_idx - zoom_margin)
        zoom_end = min(len(pixel_flux) - 1, min_flux_idx + zoom_margin)
        
        ax3.plot(time_data_seconds[zoom_start:zoom_end], 
                detrended_signal[zoom_start:zoom_end], 
                'r-', linewidth=2, label='Transit Signal')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel(f'Time ({time_unit})')
        ax3.set_ylabel('Relative Flux')
        ax3.set_title('Transit Event (Zoomed)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Noise analysis
        ax4 = axes[1, 1]
        residuals = pixel_flux[valid_indices] - trend_line
        ax4.hist(residuals, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Residuals (ADC Units)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Noise Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate performance metrics
        signal_range = np.ptp(pixel_flux)
        noise_std = np.std(residuals)
        transit_depth = np.min(detrended_signal[transit_start:transit_end])
        
        print(f"ğŸ“Š Signal Quality Metrics:")
        print(f"   â€¢ Signal range: {signal_range:.1f} ADC units")
        print(f"   â€¢ Noise std: {noise_std:.2f} ADC units")
        print(f"   â€¢ Approximate SNR: {signal_range/noise_std:.1f}")
        print(f"   â€¢ Transit depth: {abs(transit_depth):.4f} (relative)")
        print(f"   â€¢ Transit duration: {time_data_seconds[transit_end] - time_data_seconds[transit_start]:.1f} {time_unit}")
        
        display(Markdown(
            "**ğŸ’¡ Significance:** This analysis reveals the **core challenge**:\n\n"
            "- **Blue line (top-left)**: Raw telescope data showing instrumental drift\n"
            "- **Red line (top-right)**: Detrended signal revealing the transit\n"
            "- **Bottom-left**: Zoomed transit showing the atmospheric signature\n"
            "- **Bottom-right**: Noise characteristics for filter design\n\n"
            "**The task is to separate the tiny planetary signal from complex systematic noise.**"
        ))
    else:
        print("âš ï¸� Insufficient valid data points for time series analysis")
            
else:
    print("âš ï¸� No FGS1 signal data available for time series analysis")


# ====================================================================
# ğŸ�¨ SECTION 6: VISUALIZATIONS
# ====================================================================

display_header("ğŸ�¨ Visualizations")

def plot_feature_distributions():
    """Plot distributions of stellar/planetary parameters"""
    feature_cols = ['Rs', 'Ms', 'Ts', 'Mp', 'P', 'sma', 'i']
    palette = sns.color_palette("viridis", len(feature_cols))
    
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(feature_cols):
        ax = axes[i]
        sns.histplot(
            data=merged_train_df,
            x=col,
            ax=ax,
            kde=True,
            color=palette[i]
        )
        ax.set_title(f'Distribution of {col}', fontsize=14)
        ax.set_xlabel('')
    
    fig.suptitle('Stellar and Planetary Parameter Distributions', fontsize=20, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_sample_spectrum():
    """Plot ground truth spectrum for sample planet"""
    # Sample planet data
    planet_sample = merged_train_df[merged_train_df['planet_id'] == int(SAMPLE_PLANET_ID)]
    
    if len(planet_sample) == 0:
        planet_sample = merged_train_df.iloc[[0]]
        sample_id = planet_sample['planet_id'].values[0]
    else:
        sample_id = int(SAMPLE_PLANET_ID)
    
    # Spectrum data
    wl_cols = [col for col in train_df.columns if col.startswith('wl_')]
    spectrum_data = planet_sample.melt(
        id_vars=['planet_id'], 
        value_vars=wl_cols,
        var_name='wl_id', 
        value_name='spectrum_value'
    )
    
    # Merge with wavelengths
    spectrum_plot = pd.merge(spectrum_data, wavelengths_long_df, on='wl_id')
    
    # Plot
    plt.figure(figsize=(16, 8))
    
    instrument_colors = {"FGS1": "#0077b6", "AIRS-CH0": "#d62728"}
    
    sns.lineplot(
        data=spectrum_plot,
        x='wavelength_um',
        y='spectrum_value',
        hue='instrument',
        style='instrument',
        markers=True,
        dashes=False,
        palette=instrument_colors,
        linewidth=2,
        markersize=8
    )
    
    plt.title(f'Ground Truth Spectrum - Planet ID: {sample_id}', fontsize=18)
    plt.xlabel('Wavelength (Î¼m)', fontsize=14)
    plt.ylabel('Transit Depth', fontsize=14)
    plt.legend(title='Instrument', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()

# Visualizations
plot_feature_distributions()
plot_sample_spectrum()


# ====================================================================
# ğŸ”¬ SECTION 7: SIGNAL PROCESSING ANALYSIS
# ====================================================================
display_header("ğŸ”¬ Signal Processing Analysis")

def differential_reading(light_curve):
    """differential reading with comprehensive bounds checking"""
    
    # Ensureing we have enough data for differential reading
    if len(light_curve) < 2:
        return np.array([])
    
    # Calculate valid start indices (even positions)
    max_start_idx = len(light_curve) - 2  # some room for end_idx
    start_indices = np.arange(0, max_start_idx + 1, 2)
    
    # Calculate corresponding end indices
    end_indices = start_indices + 1
    
    # ensuring all indices are valid
    valid_mask = (end_indices < len(light_curve)) & (start_indices >= 0)
    start_indices = start_indices[valid_mask]
    end_indices = end_indices[valid_mask]
    
    # Perform differential reading
    if len(start_indices) > 0:
        diff = light_curve.iloc[end_indices].values - light_curve.iloc[start_indices].values
        return diff
    else:
        return np.array([])

def process_raw_signal(planet_id):
    """ Signal processing pipeline for a single planet"""
    
    print(f"ğŸ”„ Processing Planet {planet_id}...")
    
    # Load raw signal data
    fgs1_path = BASE_PATH / f"train/{planet_id}/FGS1_signal_0.parquet"
    airs_path = BASE_PATH / f"train/{planet_id}/AIRS-CH0_signal_0.parquet"
    
    try:
        fgs1_raw = pd.read_parquet(fgs1_path)
        airs_raw = pd.read_parquet(airs_path)
        
        print(f"   âœ… Raw data loaded: FGS1 {fgs1_raw.shape}, AIRS-CH0 {airs_raw.shape}")
        
        # ADC calibration
        fgs1_calibrated = adc_calibration(fgs1_raw, gain, offset)
        airs_calibrated = adc_calibration(airs_raw, gain, offset)
        
        # Create light curves (sum all pixels)
        fgs1_light_curve = fgs1_calibrated.sum(axis=1)
        airs_light_curve = airs_calibrated.sum(axis=1)
        
        print(f"   âœ… Light curves created")
        
        fgs1_diff = differential_reading(fgs1_light_curve)
        airs_diff = differential_reading(airs_light_curve)
        
        print(f"   âœ… Differential reading applied")
        print(f"       â€¢ FGS1 differential points: {len(fgs1_diff)}")
        print(f"       â€¢ AIRS-CH0 differential points: {len(airs_diff)}")
        
        # Proceed with signal cleaning for AIRS-CH0 (better signal)
        if len(airs_diff) > 0:
            airs_cleaned, outliers = sigma_clip(airs_diff)
            
            # Transit detection
            transit_start, transit_end, transit_depth = detect_transit(airs_cleaned)
            
            print(f"   âœ… Transit detected: frames {transit_start}-{transit_end}, depth {abs(transit_depth):.6f}")
            
            # Detrending
            out_of_transit_mask = np.ones_like(airs_cleaned, dtype=bool)
            out_of_transit_mask[transit_start:transit_end] = False
            out_of_transit_mask[np.isnan(airs_cleaned)] = False
            
            x_axis = np.arange(len(airs_cleaned))
            poly_coeffs = np.polyfit(x_axis[out_of_transit_mask], airs_cleaned[out_of_transit_mask], 2)
            baseline_model = np.polyval(poly_coeffs, x_axis)
            airs_normalized = airs_cleaned / baseline_model
            
            print(f"   âœ… Signal processing complete")
            
            return {
                'fgs1_raw': fgs1_raw,
                'airs_raw': airs_raw,
                'fgs1_diff': fgs1_diff,
                'airs_diff': airs_diff,
                'airs_cleaned': airs_cleaned,
                'airs_normalized': airs_normalized,
                'baseline_model': baseline_model,
                'transit_range': (transit_start, transit_end),
                'outliers': outliers
            }
        else:
            print("   â�Œ No differential data points available")
            return None
            
    except Exception as e:
        print(f"   â�Œ Error processing planet {planet_id}: {e}")
        return None


def visualize_detector_images(fgs1_raw, airs_raw, planet_id, frame_idx=5000):
    """Visualize detector images"""
    # Get sample frames
    fgs1_frame = adc_calibration(fgs1_raw.iloc[frame_idx], gain, offset)
    airs_frame = adc_calibration(airs_raw.iloc[frame_idx], gain, offset)
    
    # Reshape to 2D
    fgs1_2d = fgs1_frame.values.reshape(32, 32)
    airs_2d = airs_frame.values.reshape(32, 356)
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))
    
    # FGS1 detector
    sns.heatmap(fgs1_2d, ax=axes[0], cmap='viridis', cbar_kws={'label': 'Flux'})
    axes[0].set_title(f'FGS1 Detector Image (32Ã—32) - Planet {planet_id}, Frame {frame_idx}', fontsize=16)
    axes[0].set_xlabel('Pixel Column')
    axes[0].set_ylabel('Pixel Row')
    
    # AIRS-CH0 detector
    sns.heatmap(airs_2d, ax=axes[1], cmap='viridis', cbar_kws={'label': 'Flux'})
    axes[1].set_title(f'AIRS-CH0 Detector Image (32Ã—356) - Planet {planet_id}, Frame {frame_idx}', fontsize=16)
    axes[1].set_xlabel('Wavelength Axis (Pixel Column)')
    axes[1].set_ylabel('Spatial Axis (Pixel Row)')
    
    plt.tight_layout()
    plt.show()

def visualize_signal_processing(results, planet_id):
    """Visualize complete signal processing pipeline"""
    
    if results is None:
        print("â�Œ No results to visualize")
        return
    
    # Extract data
    fgs1_diff = results['fgs1_diff']
    airs_diff = results['airs_diff']
    airs_cleaned = results['airs_cleaned']
    airs_normalized = results['airs_normalized']
    baseline_model = results['baseline_model']
    transit_range = results['transit_range']
    outliers = results['outliers']
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Raw differential signals
    # Create separate x-axes for each instrument
    fgs1_x_axis = np.arange(len(fgs1_diff))
    airs_x_axis = np.arange(len(airs_diff))
    
    # Plot FGS1 and AIRS-CH0 separately with their own x-axes
    axes[0,0].plot(fgs1_x_axis, fgs1_diff, color='#0077b6', linewidth=0.8, alpha=0.7, label='FGS1')
    axes[0,0].plot(airs_x_axis, airs_diff, color='#d62728', linewidth=0.8, alpha=0.7, label='AIRS-CH0')
    
    # Only showing transit region on AIRS-CH0
    axes[0,0].axvspan(transit_range[0], transit_range[1], color='orange', alpha=0.3, label='Transit (AIRS-CH0)')
    axes[0,0].set_title('Raw Differential Signals', fontsize=14)
    axes[0,0].set_xlabel('Frame Number')
    axes[0,0].set_ylabel('Differential Flux')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Sigma clipping and baseline - Use AIRS-CH0 x-axis
    axes[0,1].plot(airs_x_axis, airs_diff, color='lightgray', linewidth=0.5, alpha=0.7, label='Raw')
    axes[0,1].plot(airs_x_axis, baseline_model, color='orange', linewidth=2, label='Baseline')
    axes[0,1].scatter(airs_x_axis[outliers], airs_diff[outliers], color='red', s=20, label='Outliers', zorder=5)
    axes[0,1].set_title('Sigma Clipping & Baseline Fitting', fontsize=14)
    axes[0,1].set_xlabel('Frame Number')
    axes[0,1].set_ylabel('Flux')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Final normalized light curve - Use AIRS-CH0 x-axis
    axes[1,0].plot(airs_x_axis, airs_normalized, color='dodgerblue', linewidth=1.5, label='Normalized')
    axes[1,0].axvspan(transit_range[0], transit_range[1], color='orange', alpha=0.3, label='Transit')
    axes[1,0].set_title('Final Normalized Light Curve', fontsize=14)
    axes[1,0].set_xlabel('Frame Number')
    axes[1,0].set_ylabel('Normalized Flux')
    axes[1,0].set_ylim(0.98, 1.01)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Transit zoom - Use AIRS-CH0 x-axis
    buffer = (transit_range[1] - transit_range[0]) // 3
    zoom_start = max(0, transit_range[0] - buffer)
    zoom_end = min(len(airs_normalized) - 1, transit_range[1] + buffer)
    
    axes[1,1].plot(airs_x_axis[zoom_start:zoom_end], airs_normalized[zoom_start:zoom_end], 
                   color='red', linewidth=2, label='Transit Signal')
    axes[1,1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    axes[1,1].set_title('Transit Event (Zoomed)', fontsize=14)
    axes[1,1].set_xlabel('Frame Number')
    axes[1,1].set_ylabel('Normalized Flux')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    fig.suptitle(f'Complete Signal Processing Pipeline - Planet {planet_id}', fontsize=18)
    plt.tight_layout()
    plt.show()

# Run complete processing on sample planet
try:
    # Find a planet that exists in the data
    planet_folders = [p.name for p in (BASE_PATH / 'train').iterdir() if p.is_dir()]
    
    if SAMPLE_PLANET_ID in planet_folders:
        demo_planet = SAMPLE_PLANET_ID
    else:
        demo_planet = planet_folders[0]  # Use first available planet
    
    print(f"Running demo on Planet {demo_planet}...")
    
    # Process the planet
    results = process_raw_signal(demo_planet)
    
    if results is not None:
        # Visualize detector images
        visualize_detector_images(results['fgs1_raw'], results['airs_raw'], demo_planet)
        
        # Visualize signal processing
        visualize_signal_processing(results, demo_planet)
        
        # Calculate quality metrics
        transit_signal = results['airs_normalized'][results['transit_range'][0]:results['transit_range'][1]]
        transit_depth = 1.0 - np.nanmean(transit_signal)
        noise_level = np.nanstd(results['airs_normalized'])
        snr = transit_depth / noise_level if noise_level > 0 else 0
        
        print(f"ğŸ“Š Quality Metrics for Planet {demo_planet}:")
        print(f"   â€¢ Transit depth: {transit_depth:.6f}")
        print(f"   â€¢ Noise level: {noise_level:.6f}")
        print(f"   â€¢ SNR: {snr:.2f}")
        print(f"   â€¢ Data points: {len(results['airs_normalized'])}")
        print(f"   â€¢ Outliers removed: {results['outliers'].sum()}")
        
except Exception as e:
    print(f"â�Œ Error in signal processing demo: {e}")


# ====================================================================
# ğŸ”� SECTION 8: Summary
# ====================================================================

display_header("ğŸ�¯ Summary")

# Dataset scale analysis
total_planets = len(train_df)
total_wavelengths = len([col for col in train_df.columns if col.startswith('wl_')])
test_planets = len(sample_submission_df)

print("ğŸ“Š DATASET CHARACTERISTICS:")
print(f"   â€¢ Training planets: {len(train_df):,}")
print(f"   â€¢ Test planets: {test_planets:,}")
print(f"   â€¢ Spectral points: {len(wavelengths)}")
print(f"   â€¢ Wavelength range: {wavelengths.min():.3f} - {wavelengths.max():.3f} Î¼m")
print(f"   â€¢ Time axis shape: {axis_info_df.shape}")
print(f"   â€¢ Estimated dataset size: ~263 GB")

print(f"\nğŸ›°ï¸� INSTRUMENT ANALYSIS:")
print(f"   â€¢ FGS1 (photometer): {fgs1_count} points at visible wavelengths")
print(f"   â€¢ AIRS-CH0 (spectrometer): {airs_count} points at infrared wavelengths")
print(f"   â€¢ Multi-instrument processing required")

# Instrument coverage analysis
wavelength_ranges = {
    'FGS1': (0.6, 0.8),
    'AIRS-CH0': (1.95, 3.9)
}

for instrument, (min_wl, max_wl) in wavelength_ranges.items():
    mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)
    coverage = mask.sum()
    print(f"   â€¢ {instrument} coverage: {coverage} points ({min_wl}-{max_wl} Î¼m)")

print("=" * 70)

print("ğŸ“‹ FINDINGS:")
print(f"   âœ… Dataset scale: {total_planets:,} planets, {total_wavelengths} wavelengths")
print(f"   âœ… Time structure: {axis_info_df.shape} with instrument-specific axes")
print(f"   âœ… Dual instruments: FGS1 + AIRS-CH0 coverage confirmed")
print(f"   âœ… ADC conversion: Parameters available for physical units")
print(f"   âœ… Rich metadata: Stellar/planetary features ready for use")
print(f"   âœ… Submission format: 283 + 283 dual predictions required")

print(f"Reference:")
print(f"   https://www.kaggle.com/code/lordpatil/perfect-eda-doesn-t-exist")

print("=" * 70)

