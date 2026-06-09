import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import zipfile
import io


TEST_DATA_DIR = '/kaggle/input/waveform-inversion/test'



DEBUG = True 

DT = 0.001  #time step(sec)
VELOCITY = 2500 # 音速 (m/s)
DISTANCE_DECAY_POWER = 2.0 #n (1.0 to 3.0)
ZIP_FILENAME = "seismic_images.zip" 

GAIN = 1.0 


npy_files = sorted(glob.glob(os.path.join(TEST_DATA_DIR, '*.npy')))
total_files = len(npy_files)

print(f"Found {total_files} NPY files.")


if DEBUG:
    npy_files = npy_files[:10]
    print(f"DEBUG mode enabled: Processing only the first {len(npy_files)} files.")

print(f"Processing files and creating {ZIP_FILENAME}...")

processed_count = 0
error_count = 0

# ZIPファイルを開く (圧縮あり)
try:
    with zipfile.ZipFile(ZIP_FILENAME, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, npy_path in enumerate(npy_files):
            fig = None 
            try:
                file_name = os.path.basename(npy_path)
                file_id = os.path.splitext(file_name)[0]

                seismic_data = np.load(npy_path)

                num_sources, time_steps, num_receivers = seismic_data.shape

                central_source_index = num_sources // 2
                central_source_data = seismic_data[central_source_index]

                # --- AGC (時間減衰補正 + RMS) ---
                epsilon = 1e-8 

                # 1. 時間減衰補正 AGC
                times = np.arange(time_steps) * DT
                gain_time = (times + epsilon)**DISTANCE_DECAY_POWER
                agc_data_time = central_source_data * gain_time[:, np.newaxis]

                # 2. RMS-AGC (時間減衰補正後のデータに適用)
                rms = np.sqrt(np.mean(agc_data_time**2, axis=1))
                gain_rms = GAIN / (rms + epsilon)
                agc_data_final = agc_data_time * gain_rms[:, np.newaxis]


                # プロットの準備 (6.4インチ x 6.4インチ、100 DPIで640x640ピクセル)
                fig, ax = plt.subplots(figsize=(3.2, 3.2))

                im = ax.imshow(agc_data_final, aspect='auto', cmap='gray')

                ax.set_title(f"Seismic Data (Time AGC n={DISTANCE_DECAY_POWER}, RMS AGC G={GAIN}) - ID {file_id}, Central Source")
                ax.set_xlabel("Receivers")
                ax.set_ylabel("Timesteps")

                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label("Amplitude")

                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100)
                buffer.seek(0)

                zip_file.writestr(f"{file_id}.png", buffer.getvalue())
                buffer.close() # バッファを閉じる

                if i < 10:
                    print(f"Displaying plot for {file_id}...")
                    plt.show()
                
                plt.close(fig)
                processed_count += 1

            except Exception as e:
                print(f"Error processing file {npy_path}: {e}")
                error_count += 1
                if fig is not None and plt.fignum_exists(fig.number):
                    plt.close(fig)


except Exception as e:
    print(f"An error occurred during ZIP file creation: {e}")


print(f"\nProcessing finished.")
if DEBUG:
    print("(Ran in DEBUG mode)")
print(f"Successfully processed and added to ZIP: {processed_count} files.")
print(f"Failed to process: {error_count} files.")
if processed_count > 0:
    print(f"Output saved to {ZIP_FILENAME}")
else:
    print(f"{ZIP_FILENAME} was not created as no files were processed successfully.")


npy_files = sorted(glob.glob(os.path.join(TEST_DATA_DIR, '*.npy')))
total_files = len(npy_files)

print(f"Found {total_files} NPY files.")







