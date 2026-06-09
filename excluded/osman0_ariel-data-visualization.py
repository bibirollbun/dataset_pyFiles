#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Scientific computing
from scipy import stats
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

# Color palette settings
plt.style.use('default')
sns.set_palette("husl")
print("=" * 60)


class VisualizationConfig:
    """Configuration for Kaggle environment"""
    
    # Kaggle Paths
    INPUT_DIR = Path('/kaggle/input/ariel-data-challenge-2025')
    WORKING_DIR = Path('/kaggle/working')
    OUTPUT_DIR = Path('/kaggle/working/visualizations')
    
    # Local paths for testing (development)
    LOCAL_INPUT_DIR = Path('./Data')
    LOCAL_OUTPUT_DIR = Path('./visualizations')
    
    def __init__(self, use_kaggle=True):
        self.use_kaggle = use_kaggle
        
        if use_kaggle:
            self.input_dir = self.INPUT_DIR
            self.output_dir = self.OUTPUT_DIR
        else:
            self.input_dir = self.LOCAL_INPUT_DIR  
            self.output_dir = self.LOCAL_OUTPUT_DIR
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Data paths
        self.train_csv = self.input_dir / 'train.csv'
        self.wavelengths_csv = self.input_dir / 'wavelengths.csv'
        self.train_star_info = self.input_dir / 'train_star_info.csv'
        self.test_star_info = self.input_dir / 'test_star_info.csv'
        self.adc_info = self.input_dir / 'adc_info.csv'
        
        # Signal directories
        self.train_dir = self.input_dir / 'train'
        self.test_dir = self.input_dir / 'test'
        
        print(f"ğŸ“� Input dir: {self.input_dir}")
        print(f"ğŸ“� Output dir: {self.output_dir}")

# Check if running in Kaggle
try:
    # Check for Kaggle file structure
    if Path('/kaggle/input/ariel-data-challenge-2025').exists():
        config = VisualizationConfig(use_kaggle=True)
        print("ğŸŒ� Kaggle environment detected!")
    else:
        config = VisualizationConfig(use_kaggle=False)
        print("ğŸ�  Local environment detected!")
except:
    config = VisualizationConfig(use_kaggle=False)
    print("ğŸ�  Using local environment!")


class ArielDataVisualizer:
    """Ariel data visualization class"""
    
    def __init__(self, config):
        self.config = config
        self.adc_info = None
        self.wavelengths = None
        self.train_spectra = None
        self.star_info = None
        
        # Color palettes
        self.colors = {
            'airs': '#FF6B6B',      # Red (Infrared)
            'fgs1': '#4ECDC4',      # Turquoise (Visible)
            'spectrum': '#45B7D1',   # Blue
            'prediction': '#96CEB4', # Green
            'uncertainty': '#FFEAA7' # Yellow
        }
        
        print("ğŸ”§ ArielDataVisualizer initialized!")
    
    def load_basic_data(self):
        """Load basic data files"""
        print("\nğŸ“Š Loading basic data...")
        
        try:
            # Wavelengths
            if self.config.wavelengths_csv.exists():
                self.wavelengths = pd.read_csv(self.config.wavelengths_csv)
                print(f"âœ… Wavelengths: {self.wavelengths.shape}")
            
            # ADC information
            if self.config.adc_info.exists():
                self.adc_info = pd.read_csv(self.config.adc_info)
                print(f"âœ… ADC Info: {self.adc_info.shape}")
            
            # Training spectra
            if self.config.train_csv.exists():
                self.train_spectra = pd.read_csv(self.config.train_csv)
                print(f"âœ… Train Spectra: {self.train_spectra.shape}")
            
            # Star information
            if self.config.train_star_info.exists():
                self.star_info = pd.read_csv(self.config.train_star_info)
                print(f"âœ… Star Info: {self.star_info.shape}")
                
        except Exception as e:
            print(f"âš ï¸� Data loading error: {e}")
    
    def apply_adc_conversion(self, data, instrument):
        """Apply ADC conversion"""
        if self.adc_info is None:
            return data.astype(np.float64)
            
        try:
            if instrument == 'FGS1':
                gain = self.adc_info['FGS1_adc_gain'].iloc[0]
                offset = self.adc_info['FGS1_adc_offset'].iloc[0]
            else:  # AIRS-CH0
                gain = self.adc_info['AIRS-CH0_adc_gain'].iloc[0]
                offset = self.adc_info['AIRS-CH0_adc_offset'].iloc[0]
            
            return (data.astype(np.float64) / gain) + offset
        except:
            return data.astype(np.float64)
    
    def load_sample_planet_data(self, planet_id=None, max_frames=100):
        """Load sample planet data"""
        print(f"\nğŸª� Loading planet data...")
        
        # If planet_id not specified, take the first planet
        if planet_id is None:
            if self.config.train_dir.exists():
                planet_dirs = [d for d in self.config.train_dir.iterdir() if d.is_dir()]
                if planet_dirs:
                    planet_id = planet_dirs[0].name
                else:
                    print("â�Œ Planet data not found!")
                    return None, None
        
        planet_dir = self.config.train_dir / str(planet_id)
        if not planet_dir.exists():
            print(f"â�Œ Planet {planet_id} not found!")
            return None, None
        
        signals = {}
        
        # AIRS-CH0 data
        airs_files = list(planet_dir.glob('AIRS-CH0_signal_*.parquet'))
        if airs_files:
            try:
                airs_data = pd.read_parquet(airs_files[0])
                # Take first max_frames and reshape
                airs_raw = airs_data.values[:max_frames]
                airs_converted = self.apply_adc_conversion(airs_raw, 'AIRS-CH0')
                # Reshape: (frames, 32, 356)
                airs_images = airs_converted.reshape(-1, 32, 356)
                signals['AIRS-CH0'] = airs_images
                print(f"âœ… AIRS-CH0: {airs_images.shape}")
            except Exception as e:
                print(f"â�Œ AIRS-CH0 loading error: {e}")
        
        # FGS1 data
        fgs1_files = list(planet_dir.glob('FGS1_signal_*.parquet'))
        if fgs1_files:
            try:
                fgs1_data = pd.read_parquet(fgs1_files[0])
                # Take first max_frames and reshape
                fgs1_raw = fgs1_data.values[:max_frames]
                fgs1_converted = self.apply_adc_conversion(fgs1_raw, 'FGS1')
                # Reshape: (frames, 32, 32)
                fgs1_images = fgs1_converted.reshape(-1, 32, 32)
                signals['FGS1'] = fgs1_images
                print(f"âœ… FGS1: {fgs1_images.shape}")
            except Exception as e:
                print(f"â�Œ FGS1 loading error: {e}")
        
        return signals, planet_id

    def plot_overview_dashboard(self):
        """Create overview dashboard"""
        print("\nğŸ“Š Creating overview dashboard...")
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Dataset summary
        if self.train_spectra is not None and self.star_info is not None:
            
            # Subplot grid
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # 1. Spectrum distribution
            ax1 = fig.add_subplot(gs[0, :2])
            if self.wavelengths is not None:
                # Wavelengths data is in a single row with 283 columns
                # Check if we have multiple rows, if not use the first (and only) row
                if len(self.wavelengths) > 1:
                    wavelength_vals = self.wavelengths.iloc[1, :].values  # Second row has the actual wavelength values
                else:
                    wavelength_vals = self.wavelengths.iloc[0, :].values  # First row has the wavelength values
                spectrum_cols = [col for col in self.train_spectra.columns if col != 'planet_id']
                sample_spectra = self.train_spectra[spectrum_cols[:283]].iloc[:5]
                
                for i, (idx, spectrum) in enumerate(sample_spectra.iterrows()):
                    # Make sure we have the same number of wavelengths and spectrum points
                    n_points = min(len(wavelength_vals), len(spectrum.values))
                    ax1.plot(wavelength_vals[:n_points], spectrum.values[:n_points], 
                           alpha=0.7, label=f'Planet {idx}')
                
                ax1.set_xlabel('Wavelength (Âµm)')
                ax1.set_ylabel('Spectral Intensity')
                ax1.set_title('ğŸŒŸ Sample Planet Spectra')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # 2. Planet parameters distribution
            ax2 = fig.add_subplot(gs[0, 2:])
            if 'temperature' in self.star_info.columns:
                ax2.hist(self.star_info['temperature'], bins=30, alpha=0.7, 
                        color=self.colors['airs'], edgecolor='black')
                ax2.set_xlabel('Star Temperature (K)')
                ax2.set_ylabel('Frequency')
                ax2.set_title('ğŸŒ¡ï¸� Star Temperature Distribution')
                ax2.grid(True, alpha=0.3)
            
            # 3. ADC information
            ax3 = fig.add_subplot(gs[1, :2])
            if self.adc_info is not None:
                instruments = []
                gains = []
                offsets = []
                
                for col in self.adc_info.columns:
                    if 'gain' in col:
                        inst = col.replace('_adc_gain', '')
                        instruments.append(inst)
                        gains.append(self.adc_info[col].iloc[0])
                        offsets.append(self.adc_info[col.replace('gain', 'offset')].iloc[0])
                
                x = np.arange(len(instruments))
                width = 0.35
                
                ax3_twin = ax3.twinx()
                bars1 = ax3.bar(x - width/2, gains, width, label='Gain', 
                              color=self.colors['airs'], alpha=0.7)
                bars2 = ax3_twin.bar(x + width/2, offsets, width, label='Offset', 
                                   color=self.colors['fgs1'], alpha=0.7)
                
                ax3.set_xlabel('Instrument')
                ax3.set_ylabel('Gain Value', color=self.colors['airs'])
                ax3_twin.set_ylabel('Offset Value', color=self.colors['fgs1'])
                ax3.set_title('ğŸ”§ ADC Conversion Parameters')
                ax3.set_xticks(x)
                ax3.set_xticklabels(instruments)
                ax3.legend(loc='upper left')
                ax3_twin.legend(loc='upper right')
            
            # 4. Dataset statistics
            ax4 = fig.add_subplot(gs[1, 2:])
            stats_data = {
                'Total Planets': len(self.train_spectra),
                'Spectral Points': len([col for col in self.train_spectra.columns if col != 'planet_id']),
                'Wavelength Range': f"{self.wavelengths.iloc[0 if len(self.wavelengths) == 1 else 1, 0]:.2f}-{self.wavelengths.iloc[0 if len(self.wavelengths) == 1 else 1, -1]:.2f} Âµm" if self.wavelengths is not None else "N/A"
            }
            
            stats_text = "\n".join([f"{k}: {v}" for k, v in stats_data.items()])
            ax4.text(0.1, 0.5, stats_text, fontsize=14, transform=ax4.transAxes,
                    verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor=self.colors['uncertainty'], alpha=0.7))
            ax4.set_title('ğŸ“ˆ Dataset Statistics')
            ax4.axis('off')
            
            # 5. Spectral summary
            ax5 = fig.add_subplot(gs[2, :])
            if len(spectrum_cols) >= 283:
                all_spectra = self.train_spectra[spectrum_cols[:283]].values
                
                # Mean and standard deviation
                mean_spectrum = np.mean(all_spectra, axis=0)
                std_spectrum = np.std(all_spectra, axis=0)
                
                # Safe wavelength access
                if self.wavelengths is not None:
                    if len(self.wavelengths) > 1:
                        wavelength_vals = self.wavelengths.iloc[1, :283].values
                    else:
                        wavelength_vals = self.wavelengths.iloc[0, :283].values
                else:
                    wavelength_vals = range(283)
                
                ax5.fill_between(wavelength_vals, 
                               mean_spectrum - std_spectrum,
                               mean_spectrum + std_spectrum,
                               alpha=0.3, color=self.colors['spectrum'], label='Â±1Ïƒ')
                ax5.plot(wavelength_vals, mean_spectrum, 
                        color=self.colors['spectrum'], linewidth=2, label='Mean')
                
                ax5.set_xlabel('Wavelength (Âµm)')
                ax5.set_ylabel('Spectral Intensity')
                ax5.set_title('ğŸ“Š Overall Spectral Profile (Mean Â± Std Dev)')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
        
        plt.suptitle('ğŸš€ ARIEL DATA CHALLENGE 2025 - DATASET OVERVIEW', 
                    fontsize=16, fontweight='bold')
        plt.savefig(self.config.output_dir / 'overview_dashboard.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        print(f"âœ… Dashboard saved: {self.config.output_dir / 'overview_dashboard.png'}")

    def plot_telescope_images(self, signals, planet_id, num_frames=6):
        """Visualize telescope images"""
        print(f"\nğŸ“¸ Plotting telescope images - Planet {planet_id}...")
        
        if not signals:
            print("â�Œ Image data not found!")
            return
        
        # Separate figure for each instrument
        for instrument, images in signals.items():
            if len(images) == 0:
                continue
                
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            # Show first num_frames images
            frames_to_show = min(num_frames, len(images))
            
            for i in range(frames_to_show):
                ax = axes[i]
                
                # Display image
                im = ax.imshow(images[i], cmap='viridis', aspect='auto')
                ax.set_title(f'Frame {i+1}')
                ax.set_xlabel('X Pixel')
                ax.set_ylabel('Y Pixel')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Hide empty subplots
            for i in range(frames_to_show, len(axes)):
                axes[i].axis('off')
            
            plt.suptitle(f'ğŸ“¡ {instrument} Telescope Images - Planet {planet_id}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.config.output_dir / f'{instrument}_images_planet_{planet_id}.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"âœ… {instrument} images saved")

    def plot_signal_analysis(self, signals, planet_id):
        """Signal analysis plots"""
        print(f"\nğŸ“ˆ Signal analysis - Planet {planet_id}...")
        
        if not signals:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for idx, (instrument, images) in enumerate(signals.items()):
            if len(images) == 0:
                continue
            
            # Average signal level per frame
            mean_signals = []
            std_signals = []
            
            for frame in images:
                mean_signals.append(np.mean(frame))
                std_signals.append(np.std(frame))
            
            # 1. Time series - average signal
            ax1 = axes[0, idx] if idx < 2 else axes[0, 0]
            time_steps = np.arange(len(mean_signals))
            
            ax1.plot(time_steps, mean_signals, color=self.colors['airs'] if 'AIRS' in instrument else self.colors['fgs1'])
            ax1.fill_between(time_steps, 
                           np.array(mean_signals) - np.array(std_signals),
                           np.array(mean_signals) + np.array(std_signals),
                           alpha=0.3)
            ax1.set_title(f'{instrument} - Average Signal vs Time')
            ax1.set_xlabel('Frame Number')
            ax1.set_ylabel('Signal Level')
            ax1.grid(True, alpha=0.3)
            
            # 2. Signal distribution histogram
            ax2 = axes[1, idx] if idx < 2 else axes[1, 0]
            all_pixels = images.flatten()
            
            ax2.hist(all_pixels, bins=50, alpha=0.7, density=True,
                    color=self.colors['airs'] if 'AIRS' in instrument else self.colors['fgs1'])
            ax2.set_title(f'{instrument} - Pixel Value Distribution')
            ax2.set_xlabel('Pixel Value')
            ax2.set_ylabel('Density')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f'Mean: {np.mean(all_pixels):.2f}\nStd: {np.std(all_pixels):.2f}\nMin: {np.min(all_pixels):.2f}\nMax: {np.max(all_pixels):.2f}'
            ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        plt.suptitle(f'ğŸ“Š Signal Analysis - Planet {planet_id}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.config.output_dir / f'signal_analysis_planet_{planet_id}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"âœ… Signal analysis saved")

    def plot_spectral_comparison(self, num_planets=5):
        """Spectral comparison"""
        print(f"\n Spectral comparison - {num_planets} PLANET...")
        
        if self.train_spectra is None or self.wavelengths is None:
            print("â�Œ No spectrum data found!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Spektrum sÃ¼tunlarÄ±nÄ± al
        spectrum_cols = [col for col in self.train_spectra.columns if col != 'planet_id'][:283]
        # Safe wavelength access
        if len(self.wavelengths) > 1:
            wavelength_vals = self.wavelengths.iloc[1, :len(spectrum_cols)].values
        else:
            wavelength_vals = self.wavelengths.iloc[0, :len(spectrum_cols)].values
        
        # 1. Bireysel spektrumlar
        ax1 = axes[0, 0]
        for i in range(min(num_planets, len(self.train_spectra))):
            spectrum = self.train_spectra[spectrum_cols].iloc[i]
            ax1.plot(wavelength_vals, spectrum.values, alpha=0.8, 
                    label=f'Planet {self.train_spectra.iloc[i]["planet_id"]}')
        
        ax1.set_xlabel('Wavelength (Âµm)')
        ax1.set_ylabel('Spectral Density')
        ax1.set_title('ğŸª� Individual Planet Spectra')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Normalize edilmiÅŸ spektrumlar
        ax2 = axes[0, 1]
        for i in range(min(num_planets, len(self.train_spectra))):
            spectrum = self.train_spectra[spectrum_cols].iloc[i]
            # Min-max normalization
            spectrum_norm = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())
            ax2.plot(wavelength_vals, spectrum_norm.values, alpha=0.8,
                    label=f'Planet {self.train_spectra.iloc[i]["planet_id"]}')
        
        ax2.set_xlabel('Wavelength (Âµm)')
        ax2.set_ylabel('Normalized Spectral Density')
        ax2.set_title('ğŸ“Š Normalized Spectra')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Average spectrum and variation
        ax3 = axes[1, 0]
        all_spectra = self.train_spectra[spectrum_cols].values
        mean_spectrum = np.mean(all_spectra, axis=0)
        std_spectrum = np.std(all_spectra, axis=0)
        
        ax3.fill_between(wavelength_vals, 
                        mean_spectrum - std_spectrum,
                        mean_spectrum + std_spectrum,
                        alpha=0.3, color=self.colors['spectrum'], label='Â±1Ïƒ')
        ax3.plot(wavelength_vals, mean_spectrum, 
                color=self.colors['spectrum'], linewidth=2, label='Ortalama')
        
        ax3.set_xlabel('Wavelength (Âµm)')
        ax3.set_ylabel('Spectral Density')
        ax3.set_title('ğŸ“ˆ Mean Spectrum Â± Standard Deviation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. heatmap
        ax4 = axes[1, 1]
        sample_spectra = self.train_spectra[spectrum_cols].iloc[:min(20, len(self.train_spectra))]
        
        im = ax4.imshow(sample_spectra.T, aspect='auto', cmap='viridis')
        ax4.set_xlabel('Planet Index')
        ax4.set_ylabel('Wavelength Index')
        ax4.set_title('Spectral Diversity Map')
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        
        plt.suptitle('ğŸ”¬ SPECTRAL ANALYSIS COMPRASION', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.config.output_dir / 'spectral_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"âœ… Spectral comparison saved")

    def create_summary_report(self):
        """Create summary report"""
        print("\nğŸ“‹ Creating summary report...")
        
        report = f"""
# ğŸš€ ARIEL DATA CHALLENGE 2025 - DATA ANALYSIS REPORT

## ğŸ“Š Dataset Summary
- **Total Planets**: {len(self.train_spectra) if self.train_spectra is not None else 'N/A'}
- **Spectral Points**: {len([col for col in self.train_spectra.columns if col != 'planet_id']) if self.train_spectra is not None else 'N/A'}
- **Wavelength Range**: {f"{self.wavelengths.iloc[0 if len(self.wavelengths) == 1 else 1, 0]:.3f} - {self.wavelengths.iloc[0 if len(self.wavelengths) == 1 else 1, -1]:.3f} Âµm" if self.wavelengths is not None else 'N/A'}

## ğŸ”§ Instrument Information
"""
        
        if self.adc_info is not None:
            for col in self.adc_info.columns:
                if 'gain' in col:
                    inst = col.replace('_adc_gain', '')
                    gain = self.adc_info[col].iloc[0]
                    offset = self.adc_info[col.replace('gain', 'offset')].iloc[0]
                    report += f"- **{inst}**: Gain={gain:.4f}, Offset={offset:.4f}\n"
        
        report += f"""
## ğŸ“ˆ Statistical Summary
"""
        
        if self.train_spectra is not None:
            spectrum_cols = [col for col in self.train_spectra.columns if col != 'planet_id']
            all_spectra = self.train_spectra[spectrum_cols].values
            
            report += f"""
- **Average Spectral Value**: {np.mean(all_spectra):.6f}
- **Standard Deviation**: {np.std(all_spectra):.6f}
- **Minimum Value**: {np.min(all_spectra):.6f}
- **Maximum Value**: {np.max(all_spectra):.6f}

## ğŸ�¯ Visualization Outputs
- Overview Dashboard: `overview_dashboard.png`
- Spectral Comparison: `spectral_comparison.png`
- Telescope Images: `[INSTRUMENT]_images_planet_[ID].png`
- Signal Analysis: `signal_analysis_planet_[ID].png`

## ğŸ“� Notes
This report was generated automatically.
Review individual plot files for detailed analysis.

---

"""
        
        # Save report
        with open(self.config.output_dir / 'analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"âœ… Report saved: {self.config.output_dir / 'analysis_report.md'}")
        print("\n" + "="*50)
        print(report)
        print("="*50)


def main():
    """Main execution function"""
    print("\nğŸ�¨ Starting Ariel Data Visualization...")
    print("=" * 60)
    
    try:
        # Initialize visualizer
        visualizer = ArielDataVisualizer(config)
        
        # 1. Load basic data
        visualizer.load_basic_data()
        
        # 2. Overview dashboard
        visualizer.plot_overview_dashboard()
        
        # 3. Spectral comparison
        visualizer.plot_spectral_comparison(num_planets=8)
        
        # 4. Load sample planet data and visualize
        signals, planet_id = visualizer.load_sample_planet_data(max_frames=50)
        if signals:
            visualizer.plot_telescope_images(signals, planet_id, num_frames=6)
            visualizer.plot_signal_analysis(signals, planet_id)
        
        # 5. Summary report
        visualizer.create_summary_report()
        
        print("\nğŸ�‰ ALL VISUALIZATIONS COMPLETED!")
        print(f"ğŸ“� Outputs: {config.output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"â�Œ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

print("\nğŸ�� Ok! ğŸš€")



import zipfile
import os

# Create ZIP all visualizations
def create_visualization_zip():
    zip_path = '/kaggle/working/ariel_visualizations.zip'
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # add working data
        for root, dirs, files in os.walk('/kaggle/working/visualizations'):
            for file in files:
                file_path = os.path.join(root, file)
                # Use only the file name in the ZIP
                zipf.write(file_path, file)
    
    print(f"âœ… ZIP file as created: {zip_path}")
    return zip_path

# Create ZIP and DOWNLAD
zip_file = create_visualization_zip()

