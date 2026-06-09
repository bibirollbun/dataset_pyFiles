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
GAIN = 1.0 

ENABLE_FIRST_BREAK_MUTE = True
DT = 0.001

RECEIVER_INTERVAL = 10 
DISTANCE_DECAY_POWER = 1.5 
DISPLAY_SOURCE_INDEX = 1 
MUTE_OFFSET_TIME = 0.08 
PEAK_SEARCH_REC0_START = 0.07 
PEAK_SEARCH_REC0_END = 0.09 
PEAK_SEARCH_REC35_START = 0.20
PEAK_SEARCH_REC35_END = 0.40 
ZIP_FILENAME = "seismic_images_fb_mute.zip"


npy_files = sorted(glob.glob(os.path.join(TEST_DATA_DIR, '*.npy')))
total_files = len(npy_files)

print(f"Found {total_files} NPY files.")


if DEBUG:
    npy_files = npy_files[:10]
    print(f"DEBUG mode enabled: Processing only the first {len(npy_files)} files.")


print(f"Processing files and creating {ZIP_FILENAME}...")

processed_count = 0
error_count = 0


try:
    with zipfile.ZipFile(ZIP_FILENAME, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, npy_path in enumerate(npy_files):
            fig = None 
            velocity_est = None 
            mute_error_occurred = False 

            try:
                file_name = os.path.basename(npy_path)
                file_id = os.path.splitext(file_name)[0]

                seismic_data_original = np.load(npy_path)
                seismic_data = seismic_data_original.copy() 

                num_sources, time_steps, num_receivers = seismic_data.shape
                print(f"Processing {file_id}: shape=({num_sources}, {time_steps}, {num_receivers})")

                if ENABLE_FIRST_BREAK_MUTE:
                    print(f"  Applying FirstBreak Mute for {file_id}...")
                    try:
                        if num_sources == 0:
                             raise ValueError("FirstBreak Mute requires at least one source.")

                        data_src0 = seismic_data[0] # shape: (time_steps, num_receivers)

                        search_start_idx_rec0 = int(PEAK_SEARCH_REC0_START / DT)
                        search_end_idx_rec0 = int(PEAK_SEARCH_REC0_END / DT)
                        if search_end_idx_rec0 > time_steps: search_end_idx_rec0 = time_steps
                        if search_start_idx_rec0 >= search_end_idx_rec0:
                            raise ValueError(f"Invalid time range for receiver 0 peak detection ({PEAK_SEARCH_REC0_START}-{PEAK_SEARCH_REC0_END}s).")
                        if 0 >= num_receivers:
                             raise ValueError("Receiver index 0 is out of bounds.")

                        peak_idx_rec0_rel = np.argmax(np.abs(data_src0[search_start_idx_rec0:search_end_idx_rec0, 0]))
                        peak_idx_rec0 = peak_idx_rec0_rel + search_start_idx_rec0
                        peak_time_rec0 = peak_idx_rec0 * DT
                        print(f"    Peak time at receiver 0 (source 0): {peak_time_rec0:.4f} s (index {peak_idx_rec0})")


                        target_receiver_idx = 35
                        if target_receiver_idx >= num_receivers:
                            raise ValueError(f"Receiver index {target_receiver_idx} is out of bounds (num_receivers={num_receivers}). Cannot estimate velocity.")

                        search_start_idx_rec35 = int(PEAK_SEARCH_REC35_START / DT)
                        search_end_idx_rec35 = int(PEAK_SEARCH_REC35_END / DT)
                        if search_end_idx_rec35 > time_steps: search_end_idx_rec35 = time_steps
                        if search_start_idx_rec35 >= search_end_idx_rec35:
                            raise ValueError(f"Invalid time range for receiver {target_receiver_idx} peak detection ({PEAK_SEARCH_REC35_START}-{PEAK_SEARCH_REC35_END}s).")

                        peak_idx_rec35_rel = np.argmax(np.abs(data_src0[search_start_idx_rec35:search_end_idx_rec35, target_receiver_idx]))
                        peak_idx_rec35 = peak_idx_rec35_rel + search_start_idx_rec35
                        peak_time_rec35 = peak_idx_rec35 * DT
                        print(f"    Peak time at receiver {target_receiver_idx} (source 0): {peak_time_rec35:.4f} s (index {peak_idx_rec35})")


                        delta_t = peak_time_rec35 - peak_time_rec0
                        delta_x = target_receiver_idx * RECEIVER_INTERVAL
                        if delta_t <= 1e-6: 
                            print(f"    Warning: Calculated delta_t is too small ({delta_t:.4f}s). Skipping velocity estimation and mute.")
                            velocity_est = None
                            mute_error_occurred = True
                        else:
                            velocity_est = delta_x / delta_t
                            print(f"    Estimated velocity: {velocity_est:.2f} m/s")

                            print(f"    Applying mute...")
                            
                            if num_sources > 1:
                                source_receiver_scale = (num_receivers - 1) / (num_sources - 1)
                            else:
                                source_receiver_scale = 0 

                            for s in range(num_sources):
                                source_location_approx = s * source_receiver_scale 
                                for r in range(num_receivers):
                                    receiver_location = r 
                                    horizontal_receiver_index_diff = abs(receiver_location - source_location_approx)
                                    horizontal_distance = horizontal_receiver_index_diff * RECEIVER_INTERVAL

                                    estimated_arrival_time = peak_time_rec0 + horizontal_distance / velocity_est

                                    mute_end_time = estimated_arrival_time + MUTE_OFFSET_TIME
                                    mute_end_idx = min(int(mute_end_time / DT), time_steps)

                                    if mute_end_idx > 0:
                                        seismic_data[s, :mute_end_idx, r] = 0.0
                            print(f"    Mute applied.")


                    except Exception as mute_error:
                        print(f"  Error during FirstBreak Mute for file {file_id}: {mute_error}")
                        velocity_est = None 
                        mute_error_occurred = True 
                        seismic_data = seismic_data_original.copy() 
                        print(f"    Reverted to original data for {file_id} due to mute error.")
                else:
                    print(f"  FirstBreak Mute disabled for {file_id}.")



                if DISPLAY_SOURCE_INDEX < 0 or DISPLAY_SOURCE_INDEX >= num_sources:
                    print(f"  Warning: DISPLAY_SOURCE_INDEX ({DISPLAY_SOURCE_INDEX}) is out of range [0, {num_sources-1}]. Using central source index {num_sources // 2} instead.")
                    display_source_idx = num_sources // 2
                else:
                    display_source_idx = DISPLAY_SOURCE_INDEX

                original_display_data = seismic_data_original[display_source_idx]
                muted_display_data = seismic_data[display_source_idx] # ミュートされていなければオリジナルと同じ

                # --- AGC ---
                print(f"  Applying AGC for {file_id} (source {display_source_idx})...")
                epsilon = 1e-8
                times = np.arange(time_steps) * DT
                gain_time = (times + epsilon)**DISTANCE_DECAY_POWER

                # 1. 時間減衰補正 AGC
                agc_data_time_original = original_display_data * gain_time[:, np.newaxis]
                rms_original = np.sqrt(np.mean(agc_data_time_original**2, axis=1))
                gain_rms_original = GAIN / (rms_original + epsilon)
                agc_data_original = agc_data_time_original * gain_rms_original[:, np.newaxis]

                # 2. RMSベース AGC 
                agc_data_time_muted = muted_display_data * gain_time[:, np.newaxis]
                rms_muted = np.sqrt(np.mean(agc_data_time_muted**2, axis=1))
                gain_rms_muted = GAIN / (rms_muted + epsilon)
                agc_data_muted = agc_data_time_muted * gain_rms_muted[:, np.newaxis]

                print(f"  AGC applied.")
                # --- AGC  ---


                fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), sharey=True) 

                im_orig = axes[0].imshow(agc_data_original, aspect='auto', cmap='gray')
                axes[0].set_title("Original (AGC Only)")
                axes[0].set_xlabel("Receivers")
                axes[0].set_ylabel(f"Timesteps (dt={DT}s)")

                im_muted = axes[1].imshow(agc_data_muted, aspect='auto', cmap='gray')
                mute_status_for_title = "Muted"
                if not ENABLE_FIRST_BREAK_MUTE:
                    mute_status_for_title = "Mute Disabled"
                elif mute_error_occurred:
                    mute_status_for_title = "Mute Failed"
                axes[1].set_title(f"{mute_status_for_title} (AGC Applied)")
                axes[1].set_xlabel("Receivers")

                title_lines = [
                    f"Seismic Data Comparison - ID {file_id}, Source Index {display_source_idx}",
                    f"AGC: Time (n={DISTANCE_DECAY_POWER}), RMS (G={GAIN})"
                ]
                if ENABLE_FIRST_BREAK_MUTE:
                    if mute_error_occurred:
                        mute_status = "Failed"
                        title_lines.append(f"FB Mute Status: {mute_status}")
                    elif velocity_est is not None:
                        mute_status = "Enabled"
                        title_lines.append(f"FB Mute Status: {mute_status} (Est. Vel: {velocity_est:.2f} m/s)")
                    else:
                        mute_status = "Enabled (Vel. Est. Skipped)"
                        title_lines.append(f"FB Mute Status: {mute_status}")
                else:
                     mute_status = "Disabled"
                     title_lines.append(f"FB Mute Status: {mute_status}")

                fig.suptitle("\n".join(title_lines), y=1.02) 

                cbar_orig = fig.colorbar(im_orig, ax=axes[0], fraction=0.046, pad=0.04)
                cbar_orig.set_label("Amplitude (AGC Only)")
                cbar_muted = fig.colorbar(im_muted, ax=axes[1], fraction=0.046, pad=0.04)
                cbar_muted.set_label("Amplitude (Muted + AGC)")

                plt.tight_layout(rect=[0, 0.03, 1, 0.98]) 

                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100)
                buffer.seek(0)

                zip_file.writestr(f"{file_id}.png", buffer.getvalue())
                buffer.close() 

                if i < 10:
                    print(f"Displaying plot for {file_id}...")
                    plt.show() 

                plt.close(fig) 
                processed_count += 1
                print(f"  Successfully processed and added {file_id}.png to ZIP.")


            except Exception as e:
                print(f"Error processing file {npy_path}: {e}")
                error_count += 1
                if fig is not None and plt.fignum_exists(fig.number):
                    plt.close(fig)

except Exception as e:
    print(f"An error occurred during ZIP file creation or processing loop: {e}")


# 最終的な結果を表示
print(f"\nProcessing finished.")
if DEBUG:
    print("(Ran in DEBUG mode)")
print(f"Successfully processed and added to ZIP: {processed_count} files.")
print(f"Failed to process: {error_count} files.")
if processed_count > 0:
    print(f"Output saved to {ZIP_FILENAME}")
else:
    print(f"{ZIP_FILENAME} was not created as no files were processed successfully.")




