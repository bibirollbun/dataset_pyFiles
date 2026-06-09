from PIL import Image
from pathlib import Path
from itertools import *
from functools import *
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

N = 257
NN = N*N
radius = 128

df_image = pd.read_csv('../input/santa-2022/image.csv')

def df_to_image(df):
    side = int(len(df) ** 0.5)  # assumes a square image
    return df_image.set_index(['x', 'y']).to_numpy().reshape(side, side, -1)

image = df_to_image(df_image)


# ==== Config ====
IMG_CSV = '../input/santa-2022/image.csv'   # update if needed
NEIGHBOR_MODE = "8-neighborhood"            # ["8-neighborhood","kNN"]
K = 8                                       # used when NEIGHBOR_MODE="kNN"
RADIUS = 128
SEED = 42

# ==== Utilities ====
import time, sys, numpy as np, pandas as pd
from pathlib import Path
np.random.seed(SEED)

def tic(): return time.perf_counter()
def toc(t0, msg=""): 
    print(f"{msg} {time.perf_counter()-t0:,.2f}s", file=sys.stderr)


df_image = pd.read_csv(IMG_CSV)

def df_to_image(df):
    side = int(len(df) ** 0.5)  # assumes a square image
    return df.set_index(['x', 'y']).to_numpy().reshape(side, side, -1)

image = df_to_image(df_image)


from scipy.sparse import coo_matrix, csr_matrix

def build_graph(image, mode="8-neighborhood", k=8, alpha=0.5):
    side = image.shape[0]
    N = side * side
    
    def h(i, j): return i * side + j
    def unhash(u): return divmod(u, side)

    def pix_cost(i1, j1, i2, j2):
        dcolor = np.abs(image[i1, j1] - image[i2, j2]).sum()
        dspace = np.hypot(i1 - i2, j1 - j2) ** 0.5
        return dcolor + dspace

    rows, cols, vals = [], [], []

    if mode == "8-neighborhood":
        for i in range(side):
            for j in range(side):
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0: 
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < side and 0 <= nj < side:
                            rows.append(h(i, j))
                            cols.append(h(ni, nj))
                            vals.append(pix_cost(i, j, ni, nj))
    else:  # mode == "kNN"
        from sklearn.neighbors import NearestNeighbors
        R, C = np.mgrid[0:side, 0:side]
        X = np.concatenate([
            image.reshape(N, -1),
            alpha * R.reshape(-1, 1),
            alpha * C.reshape(-1, 1)
        ], axis=1)
        nn = NearestNeighbors(n_neighbors=k+1).fit(X)
        dists, idx = nn.kneighbors(X, return_distance=True)
        u = np.repeat(np.arange(N), k)
        v = idx[:, 1:].reshape(-1)
        ui, uj = np.vectorize(unhash)(u)
        vi, vj = np.vectorize(unhash)(v)
        w = [pix_cost(a, b, c, d) for a, b, c, d in zip(ui, uj, vi, vj)]
        rows, cols, vals = u.tolist(), v.tolist(), w

    return csr_matrix(coo_matrix((vals, (rows, cols)), shape=(N, N)))


t0 = tic()
A = build_graph(image, mode=NEIGHBOR_MODE, k=K, alpha=0.5)
toc(t0, "Graph built in")
print(f"Edges: {A.nnz:,} | Nodes: {A.shape[0]:,}")


from scipy.sparse.csgraph import minimum_spanning_tree

t0 = tic()
mst = minimum_spanning_tree(A)  # CSR, upper triangle
toc(t0, "MST computed in")

N = A.shape[0]
total_cost = float(mst.sum())
m_edges = mst.nnz

print(f"Nodes: {N:,}")
print(f"MST edges: {m_edges:,} (expected ≈ {N-1:,})")
print(f"Total MST cost: {total_cost:,.2f}")


import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

side = image.shape[0]

# Use COO for easy (row, col, data)
mst_coo = mst.tocoo()
rows, cols, weights = mst_coo.row, mst_coo.col, mst_coo.data

def unhash(u):
    return divmod(u, side)  # (i, j)

# Normalize RGB for display (handles uint8 or float)
rgb = image[..., :3]
rgb = rgb.astype(float)
rgb_disp = (rgb / (rgb.max() if rgb.max() != 0 else 1)).clip(0, 1)

# Build line segments
segs = []
for u, v in zip(rows, cols):
    i1, j1 = unhash(u)
    i2, j2 = unhash(v)
    segs.append([(j1, i1), (j2, i2)])  # x=j, y=i

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.imshow(rgb_disp, origin='upper')
lc = LineCollection(segs, linewidths=1, alpha=0.7)
lc.set_array(weights)  # color by weight
ax.add_collection(lc)
ax.set_title("MST overlay (edge color = weight)")
ax.set_axis_off()
plt.colorbar(lc, ax=ax, fraction=0.02, pad=0.01)
plt.show()


import json
from pathlib import Path

# Histogram
plt.figure(figsize=(6,4))
plt.hist(weights, bins=50)
plt.title("MST edge weight distribution")
plt.xlabel("Weight"); plt.ylabel("Count")
plt.show()

# Save artifacts for your Kaggle output
out = Path("/kaggle/working")
out.mkdir(parents=True, exist_ok=True)

# Save overlay figure (re-run Cell 7 right before if needed)
fig_path = out / "mst_overlay.png"
plt.figure(figsize=(10,10))
ax = plt.gca()
ax.imshow(rgb_disp, origin='upper')
lc = LineCollection(segs, linewidths=1, alpha=0.7)
lc.set_array(weights)
ax.add_collection(lc)
ax.set_axis_off()
plt.colorbar(lc, ax=ax, fraction=0.02, pad=0.01)
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close()

# Save metrics
with open(out / "metrics.json", "w") as f:
    json.dump({"nodes": int(N),
               "mst_edges": int(m_edges),
               "total_cost": total_cost}, f, indent=2)

print(f"Saved: {fig_path}")
print("Saved: /kaggle/working/metrics.json")


import time, pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree

def run_once(mode, k=None, alpha=0.5):
    t0 = time.perf_counter()
    A = build_graph(image, mode=mode, k=(k or 0), alpha=alpha)
    t1 = time.perf_counter()
    mst = minimum_spanning_tree(A)
    t2 = time.perf_counter()
    return {
        "mode": mode,
        "k": k if k is not None else "-",
        "alpha": alpha,
        "edges": int(A.nnz),
        "build_s": round(t1 - t0, 3),
        "mst_s": round(t2 - t1, 3),
        "total_cost": float(mst.sum())
    }

results = []
for alpha in [0.25, 0.5, 1.0]:
    results.append(run_once("8-neighborhood", None, alpha))
    results.append(run_once("kNN", 8, alpha))
    results.append(run_once("kNN", 16, alpha))

df_results = pd.DataFrame(results).sort_values(["mode","alpha","k"])
df_results


# Cell 10 (safe, self-contained)
def run_pipeline(mode=None, k=None, alpha=0.5, show=True):
    """
    Builds the graph with given params, computes MST, prints stats,
    and optionally shows an overlay plot.

    Requires:
      - image (numpy array)            # from Cell 3
      - build_graph(...)               # from Cell 4
      - tic(), toc()                   # from Cell 2
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from scipy.sparse.csgraph import minimum_spanning_tree

    # Fall back to global tidy params if not provided
    try:
        default_mode = NEIGHBOR_MODE
    except NameError:
        default_mode = "8-neighborhood"
    try:
        default_k = K
    except NameError:
        default_k = 8

    mode = mode or default_mode
    k = default_k if k is None else k

    print(f"Building graph: mode={mode}, k={k}, alpha={alpha}")
    t0 = tic()
    A_local = build_graph(image, mode=mode, k=k, alpha=alpha)
    toc(t0, "Graph built in")
    print(f"Edges: {A_local.nnz:,} | Nodes: {A_local.shape[0]:,}")

    t0 = tic()
    mst_local = minimum_spanning_tree(A_local)
    toc(t0, "MST computed in")

    total_cost_local = float(mst_local.sum())
    m_edges_local = int(mst_local.nnz)
    print(f"MST edges: {m_edges_local:,} | Total cost: {total_cost_local:,.2f}")

    if show:
        ms = mst_local.tocoo()
        side = image.shape[0]

        # Normalize RGB for display
        rgb = image[..., :3].astype(float)
        maxv = rgb.max()
        rgb_disp = (rgb / (maxv if maxv > 0 else 1)).clip(0, 1)

        # Build segments
        def unhash(u): return divmod(u, side)
        segs = []
        for u, v in zip(ms.row, ms.col):
            i1, j1 = unhash(u); i2, j2 = unhash(v)
            segs.append([(j1, i1), (j2, i2)])

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(rgb_disp, origin='upper')
        lc = LineCollection(segs, linewidths=1, alpha=0.7)
        lc.set_array(ms.data)  # color by edge weights
        ax.add_collection(lc)
        ax.set_axis_off()
        plt.colorbar(lc, ax=ax, fraction=0.02, pad=0.01)
        ax.set_title(f"MST overlay ({mode}, k={k}, α={alpha})")
        plt.show()

    # also return in case you want to use them programmatically
    return A_local, mst_local, total_cost_local, m_edges_local

# Example quick run (try plotting off first to sanity-check):
# _A, _mst, _cost, _m = run_pipeline(mode="kNN", k=8, alpha=0.5, show=False)
# run_pipeline(mode="kNN", k=8, alpha=0.5, show=True)


# quick test without plot
_A, _mst, _cost, _m = run_pipeline(show=False)

# with plot and custom params
run_pipeline(mode="kNN", k=8, alpha=0.5, show=True)

