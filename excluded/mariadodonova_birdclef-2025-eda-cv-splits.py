import os
import math
import ast
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import IPython.display as ipd
from scipy import signal
import torchaudio
from sklearn.model_selection import StratifiedKFold

DATA_PATH = "/kaggle/input/birdclef-2025"
SEED = 42
SAMPLE_RATE = 32000
RARE_THRESHOLD = 50
COMMON_TYPE_THRESHOLD = 50
RATING_THRESHOLD = 3.5
METADATA_SAMPLE_COUNT = 2000


def plot_grouped_counts(
    series_list, labels, xlabel, ylabel, title,
    add_labels=True, figsize=(5, 3), rotation=0
):
    # Get union of all category indices (if more than one Series), to align bars
    if len(series_list) != 1:
        categories = sorted(set().union(*[s.index for s in series_list]))
    else:
        categories = series_list[0].index

    # Reindex Series to ensure all have same categories, filling missing values with 0
    series_list = [s.reindex(categories, fill_value=0) for s in series_list]

    # Combine all Series into one DataFrame for grouped bar plotting
    df = pd.concat(series_list, axis=1)
    df.columns = labels

    x = np.arange(len(categories)) # x-tick positions
    width = 0.8 / len(series_list) # bar width

    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Plot each series as a separate group of bars
    for i, col in enumerate(df.columns):
        bar = ax.bar(x + i * width, df[col], width, label=col)

        # Add text labels above bars
        if add_labels:
            for xi, yi in zip(x + i * width, df[col]):
                ax.text(xi, yi + 0.01, str(yi), ha="center", va="bottom", fontsize=9)

    # Center x-ticks under grouped bars and label them
    ax.set_xticks(x + width * (len(series_list) - 1) / 2)
    ax.set_xticklabels(categories, rotation=rotation)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


def plot_histogram(data, xlabel, ylabel, title, figsize=(5, 3)):
    plt.figure(figsize=figsize)
    plt.hist(data, bins=100)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_geography(df, classes=None, classes_col=None, title="Geographic Distribution"):
    plt.figure(figsize=(8, 5))

    if classes is not None and classes_col is not None:
        # Plot each class separately for color-coded map
        for class_name in classes:
            mask = df[classes_col]==class_name
            plt.scatter(
                df.loc[mask, "longitude"],
                df.loc[mask, "latitude"],
                s=5, alpha=1, label=class_name
            )

    else:
        # Plot all points as a single group
        plt.scatter(df["longitude"], df["latitude"], s=5, alpha=1)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def display_similar_audios(df, urls, url_col="url", path_col="filepath", n=2):
    for url in urls:
        print("URL:", url)
        # Load and display audio files from paths
        paths = df.loc[df[url_col] == url, path_col].head(n)

        for path in paths:
            print("Filepath:", path)
            display(ipd.Audio(path))

        print()


def plot_audio_spectrograms(file_paths, class_names):
    """
    Plots the spectrograms and waveforms of given audio files using their native sample rates.
    
    Args:
        file_paths (list of str): List of paths to audio files.
        class_names (list of str): List of corresponding class names.
    """
    num_files = len(file_paths)
    num_cols = 5  # Number of columns
    num_rows = math.ceil(num_files / num_cols) * 2  # Each audio takes 2 rows (spectrogram + waveform)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(13, num_rows * 2))

    if num_rows == 2:
        axs = np.reshape(axs, (num_rows, num_cols))  # Ensure correct indexing for small cases

    for idx, file_path in enumerate(file_paths):
        try:
            # Load audio with native sample rate
            waveform, sample_rate = torchaudio.load(file_path)
            waveform = waveform.numpy().T  # Convert to NumPy and transpose for compatibility

            # Compute spectrogram
            sampleFreqs, segmentTimes, sxx = signal.spectrogram(waveform[:, 0], sample_rate)

            # Determine row and column indices
            i, j = (idx // num_cols) * 2, idx % num_cols  # Spectrogram in row i, waveform in i+1

            # Plot spectrogram
            axs[i][j].pcolormesh((len(segmentTimes) * segmentTimes / segmentTimes[-1]),
                                 sampleFreqs,
                                 10 * np.log10(sxx + 1e-15))
            axs[i][j].set_title(f"{class_names[idx]}", fontsize=10)
            axs[i][j].set_axis_off()

            # Plot waveform
            axs[i + 1][j].plot(waveform)
            axs[i + 1][j].set_axis_off()

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    plt.tight_layout()
    plt.show()

    # Play audio
    for idx, file_path in enumerate(file_paths):
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            waveform = waveform.numpy().T
            print(f"Playing: {class_names[idx]}")
            ipd.display(ipd.Audio(waveform[:, 0], rate=sample_rate))
        except Exception as e:
            print(f"Error playing {file_path}: {e}")


df_taxonomy = pd.read_csv(os.path.join(DATA_PATH, "taxonomy.csv"))
print(df_taxonomy.info())
print()
df_taxonomy.head()


for col in ["primary_label", "inat_taxon_id", "scientific_name", "common_name"]:
    assert df_taxonomy[col].duplicated().sum() == 0, f"duplicated values found in column: '{col}'"


same_names = df_taxonomy.loc[df_taxonomy["scientific_name"] == df_taxonomy["common_name"], "common_name"]
print(f"{len(same_names)} out of {len(df_taxonomy)} scientific names match their common names:")
print(same_names)


plot_grouped_counts(
    [df_taxonomy["class_name"].value_counts()], ["Class"],
    "Taxonomic Class", "Number of samples", "Distribution of Taxonomic Classes"
)


df_train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
print(df_train.info())
print()
df_train.head()


assert df_train["filename"].duplicated().sum() == 0, "duplicate filenames found in 'filename' column"

assert df_train["filename"].map(lambda x: x.split(".")[-1].lower() == "ogg").all(), "some audio files are not in .ogg format"

df_train["filepath"] = df_train["filename"].apply(lambda x: os.path.join(DATA_PATH, "train_audio", x))
assert all(os.path.exists(p) for p in df_train["filepath"]), "some audio file paths do not exist"


# Extract and sort taxonomy from training data
train_taxonomy = (
    df_train[["primary_label", "scientific_name", "common_name"]]
    .drop_duplicates(keep="first")
    .sort_values(by="primary_label")
    .reset_index(drop=True)
)

# Extract and sort full taxonomy reference
full_taxonomy = (
    df_taxonomy[["primary_label", "scientific_name", "common_name"]]
    .sort_values(by="primary_label")
    .reset_index(drop=True)
)

# Check row count matches
assert len(train_taxonomy) == len(full_taxonomy), "mismatch in number of taxonomy entries"

# Check content-wise equality
mismatch_mask = (train_taxonomy != full_taxonomy).any(axis=1)

assert mismatch_mask.sum() == 0, "found mismatched rows between train and full taxonomy"


def parse_list_column(x):
    items = ast.literal_eval(x)
    if len(items) == 1 and items[0] == "":
        items.pop()
    return items

# Parse columns with stringified lists
df_train["type"] = df_train["type"].apply(parse_list_column)
df_train["secondary_labels"] = df_train["secondary_labels"].apply(parse_list_column)

# Combine primary and secondary labels
df_train["labels"] = df_train.apply(
    lambda row: sorted(set([row["primary_label"]] + row["secondary_labels"])),
    axis=1
)

df_train.head()


def get_label_counts(series_of_lists):
    flat_labels = (label for sublist in series_of_lists for label in sublist if label)
    counts = Counter(flat_labels)
    return pd.Series(counts).sort_values(ascending=False)

def show_label_frequencies(df, label2name, labels_col="labels", groupby_col=None, show_rare=True):
    label_counts_per_group = []

    if groupby_col is not None:
        group_names = np.unique(df[groupby_col])

        for group in group_names:
            subset = df[df[groupby_col] == group]
            counts = get_label_counts(subset[labels_col])
            counts = counts.rename(index=label2name)
            label_counts_per_group.append(counts)
    else:
        counts = get_label_counts(df[labels_col])
        counts = counts.rename(index=label2name)
        label_counts_per_group = [counts]
        group_names = ["All Labels"]

    # Plot label frequencies
    plot_grouped_counts(
        label_counts_per_group, group_names,
        "Label", "Count", "Label Frequencies", 
        add_labels=False, figsize=(35, 8), rotation=90
    )

    # Print frequencies and number of rare species if not grouped
    for group_name, counts in zip(group_names, label_counts_per_group):
        print(group_name)
        print(counts)
        print()

    if show_rare and len(label_counts_per_group) == 1:
        rare_count = (counts < RARE_THRESHOLD).sum()
        print(f"Number of rare species (n < {RARE_THRESHOLD}): {rare_count}\n")

label2name = dict(zip(df_taxonomy["primary_label"], df_taxonomy["common_name"]))
show_label_frequencies(df_train, label2name)


df_train["num_labels"] = df_train["labels"].map(len)

plot_grouped_counts(
    [df_train["num_labels"].value_counts()], ["Labels per sample"],
    "Number of labels", "Count", "Distribution of Label Counts per Sample"
)


label_counts = get_label_counts(df_train["labels"]).reset_index()
label_counts.columns = ["primary_label", "counts"]
df_taxonomy = pd.merge(df_taxonomy, label_counts, how="left", on="primary_label")
df_taxonomy


type_counts = get_label_counts(df_train["type"])
most_common_types = type_counts[type_counts > COMMON_TYPE_THRESHOLD]

plot_grouped_counts(
    [most_common_types], ["Type"],
    "Type", "Count", f"Most Common Types (n > {COMMON_TYPE_THRESHOLD})",
    figsize=(5, 5), rotation=90
)

type_counts


url_frequencies = df_train["url"].value_counts()
duplicated_urls = url_frequencies[url_frequencies > 1]

print("Duplicated URLs with their frequencies:")
print(duplicated_urls)
print(f"\nTotal number of duplicated audio clips: {duplicated_urls.sum()}\n")

display_similar_audios(
    df_train, urls=duplicated_urls.index,
    url_col="url", path_col="filepath", n=4
)


df_train[df_train["url"].isin(duplicated_urls.index)].sort_values(by="url")


# Count primary label occurrences before merging
primary_label_counts_before = df_train["primary_label"].value_counts()
rare_species_before = set(primary_label_counts_before[primary_label_counts_before < RARE_THRESHOLD].index)
n_samples_before = len(df_train)

# Merge duplicate samples by URL
labels_by_url = df_train.groupby("url")["labels"].agg(
    lambda label_lists: sorted(set(label for sublist in label_lists for label in sublist if label))
)

# Update DataFrame with merged labels and drop duplicate rows
df_train["labels"] = df_train["url"].map(labels_by_url)
df_train = df_train.drop_duplicates(subset="url")

# Count primary label occurrences after merging
primary_label_counts_after = df_train["primary_label"].value_counts()
rare_species_after = set(primary_label_counts_after[primary_label_counts_after < RARE_THRESHOLD].index)
n_samples_after = len(df_train)

# Summary
print(f"Samples merged: {n_samples_before - n_samples_after}")
print(f"Rare species lost: {rare_species_before - rare_species_after}")
print(f"New 'rare' species: {rare_species_after - rare_species_before}")


df_train = pd.merge(
    df_train, df_taxonomy[["primary_label", "class_name"]],
    how="left", on="primary_label"
)

plot_grouped_counts(
    [df_train["class_name"].value_counts()], ["class"], 
    "Class Name", "Number of samples", "Classes"
)


CLASSES = ["Aves", "Amphibia", "Mammalia", "Insecta"]
labels_per_class = []
for class_name in CLASSES:
    counts = df_train.loc[df_train["class_name"] == class_name, "num_labels"].value_counts()
    labels_per_class.append(counts)

plot_grouped_counts(
    labels_per_class, CLASSES,
    "Number of Labels per Sample", "Number of Samples",
    "Label Count Distribution by Class",
    figsize=(15, 3)
)


print("Ratings Summary Statistics:")
print(df_train["rating"].describe())
print()

rating_counts = df_train["rating"].value_counts().sort_index()
plot_grouped_counts(
    [rating_counts], ["Rating"], 
    "Rating", "Number of Samples", "Distribution of Audio Ratings"
)


# Filter out samples that have a rating
rated = df_train[df_train["rating"] != 0.0]
print("Number of rated samples:", len(rated))

# Get total label frequencies across all data
label_counts = get_label_counts(df_train["labels"]).rename(index=label2name)

# Identify rare species
rare_labels = label_counts[label_counts < RARE_THRESHOLD]

# Count label occurrences among rated samples, split by rating quality
low_rated_counts = rated.loc[rated["rating"] < RATING_THRESHOLD, "common_name"].value_counts()
high_rated_counts = rated.loc[rated["rating"] >= RATING_THRESHOLD, "common_name"].value_counts()

# Filter to only rare species
low_rare = low_rated_counts[low_rated_counts.index.isin(rare_labels.index)]
high_rare = high_rated_counts[high_rated_counts.index.isin(rare_labels.index)]

# Plot counts for low vs high ratings of rare species
plot_grouped_counts(
    [low_rare, high_rare], ["Low Rated", "High Rated"],
    "Species", "Count", "Rare Species with Ratings",
    figsize=(20, 5), rotation=90
)

# Difference between low and high ratings for rare species
low_more_than_high = (low_rated_counts - high_rated_counts)
low_more_than_high = low_more_than_high[low_more_than_high > 0].sort_values(ascending=False)

plot_grouped_counts(
    [low_more_than_high], ["More Low Ratings"],
    "Species", "Count", "Rare Species with More Low Ratings than High",
    figsize=(6, 5), rotation=90
)


# Count class occurrences for species with mostly low and high ratings
low_class_counts = rated.loc[rated["common_name"].isin(low_rated_counts.index), "class_name"].value_counts()
high_class_counts = rated.loc[rated["common_name"].isin(high_rated_counts.index), "class_name"].value_counts()

plot_grouped_counts(
    [low_class_counts, high_class_counts], ["Low Rated", "High Rated"],
    "Class Name", "Number of Samples", "Classes of Rated Species"
)


plot_geography(
    df_train, classes=["Aves", "Mammalia", "Insecta", "Amphibia"],
    classes_col="class_name", title="Geographic Distribution of Classes"
)


def categorize_rating(rating):
    if rating == 0.0:
        return "missing"
    elif rating < RATING_THRESHOLD:
        return "low"
    return "high"

df_train["rating_category"] = df_train["rating"].apply(categorize_rating)

plot_geography(
    df_train,
    classes=["missing", "high", "low"],
    classes_col="rating_category",
    title="Geographic Distribution by Rating Category"
)


# Plot total sample counts by collection source
plot_grouped_counts(
    [df_train["collection"].value_counts()], ["All Classes"],
    "Collection", "Number of Samples", "Audio Samples by Collection"
)

# Plot sample counts by collection for each class
COLLECTIONS = ["XC", "iNat", "CSA"]
class_counts_by_collection = [
    df_train.loc[df_train["collection"] == collection, "class_name"].value_counts()
    for collection in COLLECTIONS
]
plot_grouped_counts(
    class_counts_by_collection, COLLECTIONS,
    "Class", "Number of Samples", "Classes by Collection"
)

# Plot geographic distribution by collection source
plot_geography(
    df_train,
    classes=["XC", "iNat", "CSA"],
    classes_col="collection",
    title="Geographic Distribution by Collection"
)


print("Author distribution:")
print(df_train["author"].value_counts())
print()

print("License distribution:")
print(df_train["license"].value_counts())


def extract_audio_metadata(filepath):
    metadata = torchaudio.info(filepath)
    return pd.Series({
        "sample_rate": metadata.sample_rate,
        "duration": metadata.num_frames / metadata.sample_rate
    })

def show_sampled_meta(df, title):
    # Sample a subset of audio files for metadata extraction
    sampled_filepaths = df["filepath"].sample(n=METADATA_SAMPLE_COUNT, random_state=SEED)
    metadata_df = sampled_filepaths.apply(extract_audio_metadata)
    
    print("Sample Rate Distribution:")
    print(metadata_df["sample_rate"].value_counts())
    
    plot_histogram(
        metadata_df["duration"],
        "Duration (seconds)", "Number of samples", title
    )

    short_audio_count = (metadata_df["duration"] < 5).sum()
    short_audio_percentage = (short_audio_count / METADATA_SAMPLE_COUNT) * 100
    print(f"Percentage of audio clips shorter than 5 seconds: {short_audio_percentage:.2f}%")
    print(f"Min audio duration: {metadata_df['duration'].min()}")

show_sampled_meta(df_train, "Distribution of Train Audio Durations")


SOUNDSCAPE_FOLDER = os.path.join(DATA_PATH, "train_soundscapes")

# Get all file paths within the soundscape folder
soundscape_filepaths = [
    os.path.join(SOUNDSCAPE_FOLDER, filename)
    for filename in os.listdir(SOUNDSCAPE_FOLDER)
    if filename.split(".")[-1].lower() == "ogg"
]

df_soundscapes = pd.DataFrame({"filepath": soundscape_filepaths})
print(df_soundscapes.info())
print()
df_soundscapes.head()


show_sampled_meta(df_soundscapes, "Distribution of Train Soundscapes Audio Durations")


train_file_paths, train_class_names = [], []
for class_name in CLASSES:
    subset = df_train[df_train["class_name"] == class_name].sample(n=2, random_state=SEED)
    subset["names"] = subset.apply(lambda row: f"{row['common_name']} ({row['class_name']})", axis=1)
    train_file_paths.extend(subset["filepath"].tolist())
    train_class_names.extend(subset["names"].tolist())

plot_audio_spectrograms(train_file_paths, train_class_names)


subset = df_soundscapes.sample(n=5, random_state=SEED)
subset["names"] = subset["filepath"].map(lambda x: x.split("/")[-1])

plot_audio_spectrograms(subset["filepath"].tolist(), subset["names"].tolist())


df_train["split"] = "TRAIN"

label_counts = df_train["primary_label"].value_counts()
common_labels = label_counts[label_counts >= RARE_THRESHOLD].index

skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=SEED)

for _, val_indices in skf.split(df_train, df_train["primary_label"]):
    # Use only common primary labels for TEST split
    val_primary_labels = df_train.loc[val_indices, "primary_label"]
    is_common = val_primary_labels.isin(common_labels)
    df_train.loc[val_indices[is_common], "split"] = "TEST"

    break

print(df_train["split"].value_counts())


df_train["fold"] = None

train_mask = df_train["split"] == "TRAIN"
train_data = df_train[train_mask].reset_index()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for fold_id, (_, val_idx) in enumerate(skf.split(train_data, train_data["primary_label"])):
    original_indices = train_data.loc[val_idx, "index"]
    df_train.loc[original_indices, "fold"] = fold_id

df_train["fold"].value_counts()


assert df_train[df_train["split"] != "TRAIN"]["fold"].isnull().all(), "fold assigned outside TRAIN split"
assert df_train[df_train["split"] == "TRAIN"]["fold"].notnull().all(), "missing fold assignments in TRAIN split"


df_train.to_csv("/kaggle/working/birdclef_2025_cv_train.csv", index=False)
df_taxonomy.to_csv("/kaggle/working/birdclef_2025_taxonomy.csv", index=False)


show_label_frequencies(df_train[df_train["split"] == "TEST"], label2name, show_rare=False)


show_label_frequencies(df_train[df_train["fold"].notna()], label2name, groupby_col="fold")


def get_split_groups(df, col, n_folds=5):
    groups, labels = [], []

    # Add test group
    test_counts = df.loc[df["split"] == "TEST", col].value_counts()
    groups.append(test_counts)
    labels.append("Test")

    # Add train folds
    for fold in range(n_folds):
        fold_counts = df.loc[df["fold"] == fold, col].value_counts()
        groups.append(fold_counts)
        labels.append(f"Train Fold {fold}")

    return groups, labels

plot_grouped_counts(
    *get_split_groups(df_train, "num_labels"),
    "Number of Labels", "Count", "Label Count Distribution by Split",
    figsize=(20, 5)
)


plot_grouped_counts(
    *get_split_groups(df_train, "rating"),
    "Rating", "Count", "Ratings Distribution by Split",
    figsize=(20, 5)
)

plot_grouped_counts(
    *get_split_groups(df_train, "rating_category"),
    "Rating Category", "Count", f"Rating Categories with threshold of {RATING_THRESHOLD}",
    figsize=(7, 4)
)


df_train["geo_group"] = None
df_train.loc[df_train["split"] == "TEST", "geo_group"] = "TEST"

for fold in range(5):
    df_train.loc[(df_train["split"] == "TRAIN") & (df_train["fold"] == fold), "geo_group"] = f"Fold {fold}"

plot_geography(
    df_train[df_train["geo_group"].notna()],
    classes=[f"Fold {i}" for i in range(5)] + ["TEST"],
    classes_col="geo_group",
    title="Geographic Distribution by Split and Fold"
)


plot_grouped_counts(
    *get_split_groups(df_train, "class_name"),
    "Class", "Count", "Classes",
    figsize=(9, 3)
)

plot_grouped_counts(
    *get_split_groups(df_train, "collection"),
    "Collection", "Count", "Collections",
    figsize=(9, 3)
)

