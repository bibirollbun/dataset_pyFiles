import os
import numpy as np
import pandas as pd
import librosa
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from tqdm.auto import tqdm

def analyze_frequency_range(audio_path, sr=32000):
    """Analyze frequency characteristics of an audio file"""
    try:
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr)
        
        # Compute spectrogram
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Find dominant frequencies
        mean_spectrum = np.mean(S_db, axis=1)
        peak_freq_idx = np.argmax(mean_spectrum)
        peak_freq = freqs[peak_freq_idx]
        
        # Find frequency range containing 90% of energy
        cumsum = np.cumsum(np.exp(mean_spectrum))
        normalized_cumsum = cumsum / cumsum[-1]
        lower_idx = np.where(normalized_cumsum >= 0.05)[0][0]
        upper_idx = np.where(normalized_cumsum >= 0.95)[0][0]
        
        return {
            'peak_freq': peak_freq,
            'lower_freq': freqs[lower_idx],
            'upper_freq': freqs[upper_idx],
            'mean_spectrum': mean_spectrum,
            'freqs': freqs
        }
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def analyze_species_frequencies(audio_dir, species_list=None):
    """Analyze frequency characteristics for each species"""
    results = {}
    
    # If no species list provided, analyze all species
    if species_list is None:
        species_list = [d.name for d in Path(audio_dir).iterdir() if d.is_dir()]
    
    for species in tqdm(species_list, desc="Analyzing species"):
        species_dir = Path(audio_dir) / species
        if not species_dir.exists():
            continue
            
        species_results = []
        for audio_file in species_dir.glob("*.ogg"):
            result = analyze_frequency_range(str(audio_file))
            if result:
                species_results.append(result)
        
        if species_results:
            # Aggregate results
            results[species] = {
                'peak_freq_mean': np.mean([r['peak_freq'] for r in species_results]),
                'peak_freq_std': np.std([r['peak_freq'] for r in species_results]),
                'lower_freq_mean': np.mean([r['lower_freq'] for r in species_results]),
                'upper_freq_mean': np.mean([r['upper_freq'] for r in species_results]),
                'mean_spectrum': np.mean([r['mean_spectrum'] for r in species_results], axis=0),
                'freqs': species_results[0]['freqs']  # Same for all
            }
    
    return results

def plot_species_frequency_ranges(results, output_path):
    """Create visualization of frequency characteristics"""
    # Sort species by peak frequency
    species_by_peak = sorted(results.keys(), 
                           key=lambda x: results[x]['peak_freq_mean'])
    
    # Create subplots
    fig = make_subplots(rows=2, cols=1,
                       subplot_titles=('Frequency Ranges by Species',
                                     'Average Frequency Spectrum'))
    
    # Plot frequency ranges
    fig.add_trace(
        go.Bar(
            name='Frequency Range',
            x=species_by_peak,
            y=[results[s]['upper_freq_mean'] - results[s]['lower_freq_mean'] 
               for s in species_by_peak],
            error_y=dict(
                type='data',
                array=[results[s]['peak_freq_std'] for s in species_by_peak]
            )
        ),
        row=1, col=1
    )
    
    # Plot average spectrum for each species
    for species in species_by_peak:
        fig.add_trace(
            go.Scatter(
                name=species,
                x=results[species]['freqs'],
                y=results[species]['mean_spectrum'],
                mode='lines',
                opacity=0.6
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=1000,
        showlegend=True,
        title_text="Species Frequency Analysis"
    )
    
    fig.write_html(output_path)

def main():
    # Configuration
    audio_dir = "/kaggle/input/birdclef-2025/train_audio"
    
    # List of poorly clustered species from previous analysis
    problem_species = [
        'roahaw', 'strowl1', 'linwoo1', 'blchaw1', 'savhaw1',
        'piepuf1', 'bubcur1', 'grbhaw1', 'creoro1', 'greani1'
    ]
    
    # Analyze frequencies
    print("Analyzing frequency characteristics...")
    results = analyze_species_frequencies(audio_dir, problem_species)
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_species_frequency_ranges(results, "/kaggle/working/frequency_analysis.html")
    
    # Save detailed results
    print("\nSaving detailed results...")
    df_results = pd.DataFrame({
        species: {
            'peak_freq': results[species]['peak_freq_mean'],
            'peak_freq_std': results[species]['peak_freq_std'],
            'lower_freq': results[species]['lower_freq_mean'],
            'upper_freq': results[species]['upper_freq_mean'],
            'bandwidth': (results[species]['upper_freq_mean'] - 
                        results[species]['lower_freq_mean'])
        }
        for species in results
    }).T
    
    df_results.to_csv("/kaggle/working/frequency_analysis.csv")
    
    # Print recommendations
    print("\nRecommended frequency ranges for band-pass filtering:")
    for species in problem_species:
        if species in results:
            lower = results[species]['lower_freq_mean']
            upper = results[species]['upper_freq_mean']
            print(f"{species}: {lower:.1f}Hz - {upper:.1f}Hz")

if __name__ == "__main__":
    main() 


import os
import numpy as np
import pandas as pd
import librosa
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from tqdm.auto import tqdm

def load_and_analyze_audio(audio_path, sr=32000):
    """Load audio file and compute its spectrogram"""
    y, _ = librosa.load(audio_path, sr=sr)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    times = librosa.times_like(S_db)
    freqs = librosa.fft_frequencies(sr=sr)
    return S_db, times, freqs

def create_verification_plot(audio_dir, frequency_ranges_df, output_path):
    """Create a comprehensive visualization to verify frequency ranges"""
    # Get list of species from the frequency ranges DataFrame
    species_list = frequency_ranges_df.index.tolist()
    
    # Create subplots - one row per species
    fig = make_subplots(
        rows=len(species_list), cols=1,
        subplot_titles=[f"{species} (Range: {frequency_ranges_df.loc[species, 'lower_freq']:.1f}Hz - {frequency_ranges_df.loc[species, 'upper_freq']:.1f}Hz)"
                       for species in species_list],
        vertical_spacing=0.04
    )

    # Process each species
    for idx, species in enumerate(species_list, 1):
        species_dir = Path(audio_dir) / species
        
        # Get first audio file for the species
        audio_files = list(species_dir.glob("*.ogg"))
        if not audio_files:
            print(f"Warning: No audio files found for species {species}")
            continue
            
        # Analyze first audio file
        print(f"Processing {species}: {audio_files[0].name}")
        S_db, times, freqs = load_and_analyze_audio(str(audio_files[0]))
        
        # Add spectrogram
        fig.add_trace(
            go.Heatmap(
                z=S_db,
                x=times,
                y=freqs,
                colorscale='Viridis',
                showscale=False
            ),
            row=idx, col=1
        )
        
        # Add horizontal lines for frequency range
        lower_freq = frequency_ranges_df.loc[species, 'lower_freq']
        upper_freq = frequency_ranges_df.loc[species, 'upper_freq']
        
        # Lower frequency line
        fig.add_trace(
            go.Scatter(
                x=[times[0], times[-1]],
                y=[lower_freq, lower_freq],
                mode='lines',
                line=dict(color='red', width=2),
                name=f'Lower freq ({lower_freq:.1f}Hz)',
                showlegend=idx == 1  # Show legend only for first species
            ),
            row=idx, col=1
        )
        
        # Upper frequency line
        fig.add_trace(
            go.Scatter(
                x=[times[0], times[-1]],
                y=[upper_freq, upper_freq],
                mode='lines',
                line=dict(color='yellow', width=2),
                name=f'Upper freq ({upper_freq:.1f}Hz)',
                showlegend=idx == 1  # Show legend only for first species
            ),
            row=idx, col=1
        )
        
        # Update y-axis range to focus on relevant frequencies
        fig.update_yaxes(
            title_text="Frequency (Hz)",
            range=[0, min(upper_freq * 1.5, 16000)],  # Cap at Nyquist frequency
            row=idx, col=1
        )
        
        # Update x-axis
        fig.update_xaxes(
            title_text="Time (s)" if idx == len(species_list) else "",
            row=idx, col=1
        )

    # Update layout
    fig.update_layout(
        title_text="Frequency Range Verification - Spectrograms with Detected Ranges",
        showlegend=True,
        height=300 * len(species_list)  # Set height based on number of species
    )
    
    # Save the plot
    fig.write_html(output_path)
    print(f"\nPlot saved to {output_path}")

def main():
    # Get the project root directory (where the frequency_analysis.csv file is)
    project_root = "/kaggle/working"
    
    # Load the frequency analysis results
    results_path = project_root + "/frequency_analysis.csv"
    # if not results_path.exists():
    #     print(f"Error: Could not find frequency analysis results at {results_path}")
    #     return
        
    df = pd.read_csv(results_path, index_col=0)
    
    # Set paths
    audio_dir = "/kaggle/input/birdclef-2025/train_audio"
    output_path = project_root +"/frequency_verification.html"
    
    print(f"Using audio directory: {audio_dir}")
    print(f"Will save visualization to: {output_path}")
    
    # Create verification plot
    create_verification_plot(audio_dir, df, output_path)

if __name__ == "__main__":
    main() 




