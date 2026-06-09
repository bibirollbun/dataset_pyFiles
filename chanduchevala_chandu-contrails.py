import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import measure
from skimage.morphology import opening, disk

# --- Configuration & Setup ---
try:
    data_path = Path('/kaggle/input/google-research-identify-contrails-reduce-global-warming')
    test_recs = os.listdir(data_path / 'test')
    print(f"Dataset found at {data_path}. Found {len(test_recs)} test records.")
except FileNotFoundError:
    print("Kaggle dataset not found. Creating dummy data for demonstration.")
    data_path = Path('./dummy_data')
    test_recs = ['1002653297254493116', '1000834164244036115']
    for rec in test_recs:
        os.makedirs(data_path / 'test' / rec, exist_ok=True)
        np.save(data_path / 'test' / rec / 'band_08.npy', np.random.rand(256, 266, 4) * 100)
        np.save(data_path / 'test' / rec / 'band_14.npy', np.random.rand(256, 266, 4) * 100)
        np.save(data_path / 'test' / rec / 'band_15.npy', np.random.rand(256, 266, 4) * 100)
    dummy_df = pd.DataFrame({'record_id': test_recs, 'encoded_pixels': ''})
    dummy_df.to_csv(data_path / 'sample_submission.csv', index=False)
    print("Dummy data created successfully.")


def rle_encode(x, fg_val=1):
    dots = np.where(x.T.flatten() == fg_val)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def list_to_string(x):
    if x:
        s = str(x).replace("[", "").replace("]", "").replace(",", "")
    else:
        s = '-'
    return s


def normalize_band(band):
    min_val = np.min(band)
    max_val = np.max(band)
    if max_val == min_val:
        return band - min_val
    return (band - min_val) / (max_val - min_val)

def create_false_color_composite(rec_path):
    band_15 = np.load(rec_path / 'band_15.npy').sum(axis=2)
    band_14 = np.load(rec_path / 'band_14.npy').sum(axis=2)
    band_08 = np.load(rec_path / 'band_08.npy').sum(axis=2)
    r = normalize_band(band_15)
    g = normalize_band(band_14)
    b = normalize_band(band_08)
    return np.stack([r, g, b], axis=2)


def generate_contrail_mask(rec_path, min_area=20, eccentricity_threshold=0.90, debug=False):
    """
    Generates a contrail mask.
    
    **FIXED**: Relaxed min_area and eccentricity_threshold to be more inclusive.
    """
    band_14 = np.load(rec_path / 'band_14.npy').sum(axis=2)
    band_15 = np.load(rec_path / 'band_15.npy').sum(axis=2)
    feature_map = band_14 - band_15

    threshold = np.percentile(feature_map, 99.5)
    initial_mask = feature_map > threshold

    cleaned_mask = opening(initial_mask, disk(1))
    labels = measure.label(cleaned_mask)
    regions = measure.regionprops(labels)

    final_mask = np.zeros_like(cleaned_mask)
    
    if debug:
        print(f"  - Adaptive threshold value: {threshold:.4f}")
        print(f"  - Pixels in initial mask: {np.sum(initial_mask)}")
        print(f"  - Found {len(regions)} potential regions before filtering.")

    kept_regions = 0
    for region in regions:
        if region.area > min_area and region.eccentricity > eccentricity_threshold:
            for coord in region.coords:
                final_mask[coord[0], coord[1]] = 1
            kept_regions += 1
        elif debug:
            # This part helps you see why regions were discarded
            # print(f"    - Discarded Region: Area={region.area}, Eccentricity={region.eccentricity:.2f}")
            pass

    if debug:
        print(f"  - Kept {kept_regions} regions after filtering.")
        print(f"  - Total pixels in final mask: {np.sum(final_mask)}")

    return final_mask, feature_map, initial_mask


def visualize_detection_process(record_id, false_color_img, feature_map, initial_mask, final_mask):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Advanced Contrail Detection Analysis: Record {record_id}', fontsize=20, y=0.95)
    plt.style.use('dark_background')
    
    axes[0, 0].imshow(false_color_img); axes[0, 0].set_title('A: False-Color Composite', fontsize=14); axes[0, 0].axis('off')
    im = axes[0, 1].imshow(feature_map, cmap='magma'); axes[0, 1].set_title('B: Enhanced Feature Map (Band 14 - 15)', fontsize=14); axes[0, 1].axis('off')
    fig.colorbar(im, ax=axes[0, 1], orientation='vertical', fraction=0.046, pad=0.04)
    axes[1, 0].imshow(initial_mask, cmap='gray'); axes[1, 0].set_title('C: Initial Mask (Adaptive Threshold)', fontsize=14); axes[1, 0].axis('off')
    axes[1, 1].imshow(false_color_img)
    final_mask_overlay = np.ma.masked_where(final_mask == 0, final_mask)
    axes[1, 1].imshow(final_mask_overlay, cmap='cool', alpha=0.7, interpolation='none')
    axes[1, 1].set_title('D: Final Mask (Filtered by Shape)', fontsize=14); axes[1, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


def main():
    submission_path = data_path / 'sample_submission.csv'
    submission = pd.read_csv(submission_path, index_col='record_id')

    # To process all, use: for rec in test_recs:
    for i, rec in enumerate(test_recs[:2]): # Process first 2 records for demo
        print(f"Processing record: {rec}...")
        rec_path = data_path / 'test' / rec

        # Generate mask with debugging info enabled
        final_mask, feature_map, initial_mask = generate_contrail_mask(rec_path, debug=True)
        
        # Only visualize if a contrail was detected to avoid blank plots
        if np.sum(final_mask) > 0:
            false_color_img = create_false_color_composite(rec_path)
            visualize_detection_process(rec, false_color_img, feature_map, initial_mask, final_mask)
        else:
            print(f"  - No contrails detected for record {rec} after filtering.")

        encoded_pixels = list_to_string(rle_encode(final_mask))
        submission.loc[int(rec), 'encoded_pixels'] = encoded_pixels
        print("-" * 30)

    submission.to_csv('submission.csv')
    print("\nSubmission file 'submission_advanced_v2.csv' created successfully.")
    print("Final submission head:")
    print(submission.head())

if __name__ == '__main__':
    main()

