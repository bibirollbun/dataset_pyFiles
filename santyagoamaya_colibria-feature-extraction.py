import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os 
import librosa


train_audio_dir = '/kaggle/input/birdclef-2025/train_audio/'
train_soundscapes_dir = '/kaggle/input/birdclef-2025/train_soundscapes/'

train, taxonomy = pd.read_csv('/kaggle/input/birdclef-2025/train.csv'), pd.read_csv('/kaggle/input/birdclef-2025/taxonomy.csv')
submission =  pd.read_csv('/kaggle/input/birdclef-2025/sample_submission.csv')
train.head()


taxonomy.sample(3)


submission.head()


import pandas as pd
selected_features = ['primary_label','filename','latitude','longitude','scientific_name','common_name']
train = train[selected_features]
# Define your unique labels
labels = {'brtpar1', '1139490', 'compau', 'chbant1', 'yehcar1', 'yecspi2', 'watjac1', 'grasal4', 'grbhaw1', 
          'yebfly1', 'neocor', '81930', 'spbwoo1', '64862', 'grepot1', 'ruther1', 'banana', 'whttro1', 
          '1462711', '42087', '66531', 'soulap1', 'amakin1', '41970', '65373', '714022', 'bafibi1', 'blcant4', 
          'rutjac1', 'plbwoo1', 'anhing', 'yehbla2', '21211', 'recwoo1', 'blbgra1', 'creoro1', 'shtfly1', 
          'amekes', 'blchaw1', '21116', '566513', 'bugtan', 'strcuc1', '1564122', '1462737', 'purgal2', 
          'socfly1', 'gohman1', 'gycwor1', 'bubwre1', 'blhpar1', '65336', 'solsan', '134933', '24292', 
          '42113', 'plukit1', 'savhaw1', 'sobtyr1', 'chfmac1', 'yebsee1', '66016', 'blbwre1', 'mastit1', 
          'smbani', 'whfant1', 'strfly1', 'roahaw', 'rumfly1', '476537', 'butsal1', 'bucmot3', 'colcha1', 
          'bobfly1', '67082', 'rebbla1', 'pavpig2', '1192948', 'whbman1', 'verfly', 'eardov1', 'norscr1', 
          'rinkin1', '67252', 'greibi1', 'greegr', 'cattyr', 'laufal1', 'trokin', 'grekis', 'crebob1', 
          'bubcur1', 'fotfly', 'palhor2', '476538', '24322', 'tropar', 'whwswa1', 'yercac1', '517119', 
          '24272', 'cocher1', 'labter1', 'bicwre1', 'compot1', 'olipic1', 'blcjay1', 'colara1', 'spepar1', 
          'cregua1', 'cargra1', '22976', 'plctan1', '715170', 'leagre', '22973', 'bkmtou1', 'yelori1', 
          'trsowl', 'strher', 'ragmac1', 'yeofly1', '548639', 'tbsfin1', '135045', '65344', 'bkcdon', 
          'stbwoo2', 'piepuf1', '868458', '963335', 'blctit1', 'saffin', 'rtlhum', 'royfly1', '66893', 
          'rutpuf1', 'linwoo1', 'wbwwre1', 'srwswa1', '126247', 'gretin1', 'grnkin', 'littin1', 'secfly1', 
          '41778', '528041', 'bbwduc', 'greani1', 'rubsee1', 'orcpar', 'rosspo1', 'yebela1', '47067', 
          'crcwoo1', '65349', 'snoegr', 'gybmar', 'thbeup1', '66578', 'turvul', 'rugdov', 'baymac', 
          'speowl1', 'cocwoo1', 'cotfly1', 'y00678', '65419', 'bobher1', '52884', '41663', '22333', 
          'piwtyr1', '21038', '787625', 'rufmot1', '65962', 'paltan1', '48124', '555142', '65547', 
          'crbtan1', '1194042', 'ywcpar', 'shghum1', 'cinbec1', 'thlsch3', '1346504', '555086', 'sahpar1', 
          'grysee1', 'blkvul', '523060', 'strowl1', 'whbant1', 'whmtyr1', '65448', 'ampkin1', 'whtdov', 
          'yectyr1', '42007', '46010', 'pirfly1', 'woosto', 'babwar', '50186'}

# Create a dictionary to store DataFrames
dfs = {label: train[train['primary_label'] == label] for label in labels}

# Example: Access one of the DataFrames
print(dfs['brtpar1'].head()) 


%%time
def get_audio_path(df, i, train_audio_dir):
    """Retrieve the audio file path using iloc for correct indexing."""
    return os.path.join(train_audio_dir, df.iloc[i]['filename'])
def extract_numpy_arrays(df):
    numpys = []
    train_audio_dir = '/kaggle/input/birdclef-2025/train_audio/'
    filenames = [get_audio_path(df, i, train_audio_dir) for i in range(0,len(df))]
    for filename in filenames:
        y, sr = librosa.load(filename)
        numpys.append((y,sr))
    return numpys
    
def compute_fft(y, sr):
    sp = np.fft.fft(np.sin(y))
    freq = np.fft.fftfreq(y.shape[-1])
    return sp, freq

numpys = extract_numpy_arrays(dfs['solsan'])
fouriers = [compute_fft(i,j) for i,j in numpys]


import matplotlib.pyplot as plt

# Extract the first audio sample
y, sr = numpys[4]

# Compute FFT for the first sample
sp, freq = compute_fft(y, sr)

# Plot the FFT (real and imaginary parts)
plt.figure(figsize=(10, 5))
plt.plot(freq, sp.real, label="Real Part")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.title("FFT of First Audio Sample")
plt.legend()
plt.show()


def detailed_mel_spectrogram(y, sr):
    # Compute Mel spectrogram with more detailed parameters
    mel_spec = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_mels=128,  # Increase number of mel bands
        fmax=sr/2,   # Maximum frequency
        hop_length=512,  # Adjust hop length for more detailed time resolution
        n_fft=2048   # Increase FFT window for better frequency resolution
    )
    
    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Visualize with more details
    plt.figure(figsize=(15, 6))
    librosa.display.specshow(
        log_mel_spec, 
        sr=sr, 
        x_axis='time', 
        y_axis='mel',
        cmap='viridis'  # Try different colormaps
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Detailed Mel Spectrogram')
    plt.tight_layout()
    plt.show()
    
    return log_mel_spec

detailed_mel_spectrogram(y, sr)


from IPython.display import Audio 
Audio(y, rate=sr)


def extract_advanced_features(y, sr):
    # Spectral Centroid - brightness of sound
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    
    # Spectral Bandwidth - spread of spectrum
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    
    # Spectral Rolloff - frequency below which a certain percentage of total spectral energy lies
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    
    # Zero Crossing Rate - number of times audio signal crosses zero
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    
    # Visualize these features
    plt.figure(figsize=(15, 10))
    plt.subplot(4,1,1)
    plt.plot(spectral_centroids)
    plt.title('Spectral Centroid')
    
    plt.subplot(4,1,2)
    plt.plot(spectral_bandwidth)
    plt.title('Spectral Bandwidth')
    
    plt.subplot(4,1,3)
    plt.plot(spectral_rolloff)
    plt.title('Spectral Rolloff')
    
    plt.subplot(4,1,4)
    plt.plot(zero_crossing_rate)
    plt.title('Zero Crossing Rate')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'spectral_centroids': spectral_centroids,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_rolloff': spectral_rolloff,
        'zero_crossing_rate': zero_crossing_rate
    }

extract_advanced_features(y, sr)


def harmonic_percussive_separation(y, sr):
    # Separate harmonic and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Visualize both components
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2,1,1)
    librosa.display.specshow(
        librosa.amplitude_to_db(
            np.abs(librosa.stft(y_harmonic)), 
            ref=np.max
        ), 
        sr=sr, 
        x_axis='time', 
        y_axis='hz'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Harmonic Component')
    
    plt.subplot(2,1,2)
    librosa.display.specshow(
        librosa.amplitude_to_db(
            np.abs(librosa.stft(y_percussive)), 
            ref=np.max
        ), 
        sr=sr, 
        x_axis='time', 
        y_axis='hz'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Percussive Component')
    
    plt.tight_layout()
    plt.show()
    
    return y_harmonic, y_percussive

y_harmonic, y_percussive = harmonic_percussive_separation(y, sr)


def extract_chroma_features(y, sr):
    # Compute chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    plt.figure(figsize=(15, 5))
    librosa.display.specshow(
        chroma, 
        x_axis='time', 
        y_axis='chroma', 
        cmap='coolwarm'
    )
    plt.colorbar()
    plt.title('Chroma Feature')
    plt.tight_layout()
    plt.show()
    
    return chroma

chroma = extract_chroma_features(y, sr)


Audio(y_percussive, rate=sr)


Audio(y_harmonic, rate=sr)


def extract_low_amplitude_regions(spectrogram, threshold=-20):
    """
    Extract regions of the spectrogram with amplitude below the specified threshold.
    
    Parameters:
    - spectrogram: 2D numpy array of the spectrogram (in dB scale)
    - threshold: dB threshold for extraction (default -20 dB)
    
    Returns:
    - mask: Boolean mask of regions below the threshold
    - extracted_spectrogram: Spectrogram with only low amplitude regions
    """
    # Create a boolean mask for regions below the threshold
    mask = spectrogram > threshold
    
    # Create a copy of the spectrogram with only low amplitude regions
    extracted_spectrogram = np.where(mask, spectrogram, -np.inf)
    
    # Visualization
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Spectrogram')
    librosa.display.specshow(spectrogram, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(1, 2, 2)
    plt.title(f'Spectrogram (Regions < {threshold} dB)')
    librosa.display.specshow(extracted_spectrogram, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.show()
    
    return mask, extracted_spectrogram




# Compute STFT of the harmonic component
stft_harmonic = librosa.stft(y_harmonic)
spectrogram_db = librosa.amplitude_to_db(np.abs(stft_harmonic), ref=np.max)

# Extract low amplitude regions
mask, low_amp_spectrogram = extract_low_amplitude_regions(spectrogram_db, threshold=-40)

# Convert the masked spectrogram back to linear scale
low_amp_spectrogram_linear = librosa.db_to_amplitude(low_amp_spectrogram)

# Ensure the phase information is retained
stft_magnitude = low_amp_spectrogram_linear
stft_phase = np.exp(1j * np.angle(stft_harmonic))

# Reconstruct the audio using the inverse STFT
y_reconstructed = librosa.istft(stft_magnitude * stft_phase)

# Play the reconstructed audio
Audio(data=y_reconstructed, rate=sr)



# Compute STFT of the harmonic component
stft_harmonic = librosa.stft(y_percussive)
spectrogram_db = librosa.amplitude_to_db(np.abs(stft_harmonic), ref=np.max)

# Extract low amplitude regions
mask, low_amp_spectrogram = extract_low_amplitude_regions(spectrogram_db, threshold=-40)

# Convert the masked spectrogram back to linear scale
low_amp_spectrogram_linear = librosa.db_to_amplitude(low_amp_spectrogram)

# Ensure the phase information is retained
stft_magnitude = low_amp_spectrogram_linear
stft_phase = np.exp(1j * np.angle(stft_harmonic))

# Reconstruct the audio using the inverse STFT
y_reconstructed = librosa.istft(stft_magnitude * stft_phase)

# Play the reconstructed audio
Audio(data=y_reconstructed, rate=sr)




