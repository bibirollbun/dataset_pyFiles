!pip install opentsne -q


import pandas as pd
import numpy as np

import time

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE as sklearn_TSNE

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from openTSNE import TSNE as openTSNE_TSNE


PATH_TO_TRAIN = "/kaggle/input/playground-series-s5e7/train.csv"

TARGET = "Personality"
df = pd.read_csv(PATH_TO_TRAIN)


df["Personality"].value_counts()


# Separate features and target
X = df.drop(TARGET, axis=1)
y = df[TARGET]

# Column types
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# 1. Numeric imputation (mean)
imputer_num = SimpleImputer(strategy='mean')
X_num = imputer_num.fit_transform(X[numerical_cols])

# 2. Categorical imputation (new 'Missing' category)
X[categorical_cols] = X[categorical_cols].fillna('Missing')
X_cat_raw = X[categorical_cols].astype(str)  # ensure string type for OHE

# Encode categorical
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_cat = ohe.fit_transform(X_cat_raw)

# Scale numeric
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# Combine
X_final = np.hstack([X_num_scaled, X_cat])

# Encode target
if y.dtype == 'O' or y.dtype.name == 'category':
    le = LabelEncoder()
    y_final = le.fit_transform(y)
else:
    y_final = y.values


class_names = ['Extrovert', 'Introvert']
colors = ['blue', 'red']

TRAIN_SIZE = 2000


X_sample, _, y_sample, _ = train_test_split(
    X_final, y_final, 
    train_size=TRAIN_SIZE, 
    stratify=y_final,    # preserves class balance in your subset
    random_state=42
)

tsne = sklearn_TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_sample)

plt.figure(figsize=(7, 7))
for idx, class_value in enumerate(np.unique(y_sample)):
    plt.scatter(
        X_tsne[y_sample == class_value, 0],
        X_tsne[y_sample == class_value, 1],
        c=colors[idx], label=class_names[idx], s=10, alpha=0.7
    )

plt.title(f"t-SNE (n={TRAIN_SIZE})")
plt.xlabel("t-SNE axis 1")
plt.ylabel("t-SNE axis 2")
plt.legend()
plt.tight_layout()
plt.show()


# OPTIONAL - Perform t-SNE with various train_size's and plot time_elapsed vs train_size
def plot_time_vs_size():
    sizes = [50, 500, 1000, 2000, 5000, 10000, 15000]
    times = []
    
    for train_size in sizes:
        # Subsample data
        X_sample, _, y_sample, _ = train_test_split(
            X_final, y_final,
            train_size=train_size,
            stratify=y_final,
            random_state=42
        )
    
        tsne = sklearn_TSNE(n_components=2, random_state=42, perplexity=30)
        start = time.time()
        X_tsne = tsne.fit_transform(X_sample)
        elapsed = time.time() - start
    
        times.append(elapsed)
    
        print(f"Size: {train_size} — t-SNE seconds: {elapsed:.2f}")
    
        plt.figure(figsize=(7, 7))
        for idx, class_value in enumerate(np.unique(y_sample)):
            plt.scatter(
                X_tsne[y_sample == class_value, 0],
                X_tsne[y_sample == class_value, 1],
                c=colors[idx], label=class_names[idx], s=10, alpha=0.7
            )
    
        plt.title(f"t-SNE (n={train_size}) | Time: {elapsed:.1f}s")
        plt.xlabel("t-SNE axis 1")
        plt.ylabel("t-SNE axis 2")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # Final plot of runtime vs train_size
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, times, marker='o', linestyle='-')
    plt.xlabel("Sample size (train_size)")
    plt.ylabel("t-SNE runtime (seconds)")
    plt.title("t-SNE compute time vs sample size")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# plot_time_vs_size()


X_sample, _, y_sample, _ = train_test_split(
    X_final, y_final, 
    train_size=2000, 
    stratify=y_final,    # preserves class balance in your subset
    random_state=42
)

snapshots = []

def collect_embedding(iteration, error, embedding):
    snapshots.append(embedding.copy())

tsne = openTSNE_TSNE(
    n_components=2,
    perplexity=30,
    n_iter=500,
    callbacks=collect_embedding,
    callbacks_every_iters=1,
    initialization="pca",
    random_state=42,
    verbose=True
)

embedding = tsne.fit(X_sample)
snapshots = np.array(snapshots)


num_snapshots = len(snapshots)
steps_to_show = 8  # Change as needed for more/less steps
indices_to_plot = np.linspace(0, num_snapshots-1, steps_to_show, dtype=int)

fig, axes = plt.subplots(len(indices_to_plot), 1, figsize=(5, 3 * len(indices_to_plot)))

for ax, idx in zip(axes, indices_to_plot):
    for i, class_value in enumerate(np.unique(y_sample)):
        mask = y_sample == class_value
        ax.scatter(
            snapshots[idx][mask, 0],
            snapshots[idx][mask, 1],
            c=colors[i],
            s=1,
            label=class_names[i] if idx == 0 else ""
        )
    ax.set_title(f"Step {idx * 10}")
    ax.set_xticks([])
    ax.set_yticks([])
    if idx == 0:
        ax.legend()
plt.tight_layout()
plt.show()


#--- CONFIG ---
skip = 10
frame_indices = np.arange(0, len(snapshots), skip)  # every 10th snapshot
snapshots_anim = snapshots[frame_indices]

#--- PLOT SETUP ---
fig, ax = plt.subplots(figsize=(7, 7))

scatter_objs = [
    ax.scatter([], [], s=12, color=colors[i], label=class_names[i])
    for i in np.unique(y_sample)
]

# Axis limits set to fit all points
all_x = snapshots_anim[:,:,0]
all_y = snapshots_anim[:,:,1]
ax.set_xlim(all_x.min(), all_x.max())
ax.set_ylim(all_y.min(), all_y.max())
ax.set_title("t-SNE Steps: Initialization to Convergence")
ax.legend()

#--- ANIMATION FUNCTION ---
def update(frame):
    current = snapshots_anim[frame]
    for i, class_value in enumerate(np.unique(y_sample)):
        mask = y_sample == class_value
        scatter_objs[i].set_offsets(current[mask])
    ax.set_title(f"t-SNE Step {(frame_indices[frame])}")  # Times 10 if you called callback every 10 iters
    return scatter_objs

ani = FuncAnimation(
    fig, update, frames=len(snapshots_anim),
    blit=False, interval=10, repeat=False
)

# Preview in notebook (optional)
# plt.show()

#--- SAVE AS GIF ---
ani.save("tsne_animation.gif", writer=PillowWriter(fps=12))
print("Saved as tsne_animation.gif")




