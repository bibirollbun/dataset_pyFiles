%load_ext autoreload
%autoreload 2


import kagglehub


sr3f = kagglehub.package_import('tatamikenn/stanford-rna-3d-folding-utility-packages/versions/3')


import polars as pl

msa = pl.read_parquet("/kaggle/input/stanford-3d-folding-minimal-eda/metadata/msa.parquet")
msa = msa.with_columns(
    pl.col("idx").len().over("target_id").alias("num_samples"),
    pl.col("seq").str.len_chars().alias("seq_len"),
)


import numpy as np
import matplotlib.pyplot as plt


_, axes = plt.subplots(1, 2, figsize=(13, 4))
ax = axes[0]
num_samples = (
    msa.group_by("target_id")
    .agg(num_samples=pl.col("num_samples").first())["num_samples"]
    .to_numpy()
)
mean_samples = num_samples.mean()
ax.axvline(mean_samples, color="red", linestyle="--", label=f"mean: {mean_samples:.0f}")
ax.hist(num_samples, bins=100, color="blue", alpha=0.7, edgecolor="white")
ax.grid()
ax.set(
    xlabel="Number of samples per target",
    ylabel="Count",
    title="Histogram of MSA samples per target",
)
ax.legend()

ax = axes[1]

num_samples = (
    msa.group_by("target_id")
    .agg(num_samples=pl.col("num_samples").first())
    .filter(pl.col("num_samples").lt(15))["num_samples"]
    .to_numpy()
)
uniq, count = np.unique(num_samples, return_counts=True)
ax.bar(uniq, count, color="blue", alpha=0.7, edgecolor="white")
ax.grid()
ax.set(
    xlabel="Number of samples per target",
    ylabel="Count",
    title="Count of targets with < 15 samples",
)

plt.show()


import matplotlib.colors as mcolors
from scipy.special import softmax

plot_pdb = sr3f.plot_pdb
extract_c1_atoms = sr3f.extract_c1_atoms


def calc_seq_chars(target_id):
    """
    æŒ‡å®šã�—ã�Ÿ target_id ã�«å¯¾ã�—ã�¦ã€�ã‚·ãƒ¼ã‚±ãƒ³ã‚¹ã�®æ–‡å­—é…�åˆ—ã‚’è¿”ã�™ã€‚
    è¿”ã‚Šå€¤ã�¯ shape (L, n) ã�¨ã�ªã‚Šã€�å�„åˆ—ã�Œ1ã‚·ãƒ¼ã‚±ãƒ³ã‚¹ã€�å�„è¡Œã�Œä½�ç½®ã‚’ç¤ºã�™ã€‚
    """
    df = msa.filter(pl.col("target_id") == target_id)
    df = df.with_columns(pl.col("seq").str.split("").alias("seqs"))
    seq = np.array(df["seqs"].to_list())
    return seq.T  # è»¢ç½®ã�—ã�¦ã€�è¡Œ: ä½�ç½®, åˆ—: ã‚·ãƒ¼ã‚±ãƒ³ã‚¹


def calc_pairing_score(seq_chars, normalize=True, zero_diag=True, symmetric=True, temperature=0.1):
    """
    å�„ä½�ç½®é–“ã�®mutual information (MI) ã‚’ãƒ™ã‚¯ãƒˆãƒ«åŒ–ã�«ã‚ˆã‚Šè¨ˆç®—ã�—ã�¦ã‚¹ã‚³ã‚¢ã�¨ã�—ã�¦è¿”ã�™ã€‚

    seq_chars: æ–‡å­—é…�åˆ—ã€�shape (L, n)
               å�„è¡Œ: ä½�ç½®, å�„åˆ—: ã‚·ãƒ¼ã‚±ãƒ³ã‚¹
               ã‚®ãƒ£ãƒƒãƒ—ã�¯ '-' ã�¨ã�—ã�¦æ‰±ã�„ã€�MIè¨ˆç®—ã�®éš›ã�«ã�¯é™¤å¤–ã�™ã‚‹ã€‚
    normalize: å�„è¡Œæ–¹å�‘ã�§softmaxã‚’è¨ˆç®—ã�—ã�¦æ­£è¦�åŒ–ã�™ã‚‹å ´å�ˆ True
    zero_diag: è‡ªåˆ†è‡ªèº«ã�¨ã�®ãƒšã‚¢ï¼ˆå¯¾è§’æˆ�åˆ†ï¼‰ã�®ã‚¹ã‚³ã‚¢ã‚’0ã�«ã�™ã‚‹å ´å�ˆ True
    temperature: softmaxè¨ˆç®—æ™‚ã�®æ¸©åº¦ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿ï¼ˆãƒ‡ãƒ•ã‚©ãƒ«ãƒˆã�¯1.0ï¼‰

    å‡ºåŠ›: shape (L, L) ã�®MIã‚¹ã‚³ã‚¢è¡Œåˆ—
    """
    import numpy as np
    from scipy.special import softmax

    # å¯¾è±¡ã�¨ã�™ã‚‹å¡©åŸºã€‚ã‚®ãƒ£ãƒƒãƒ—('-')ã�¯é™¤å¤–ã�™ã‚‹ã€‚
    symbols = np.array(["A", "C", "G", "U"])
    L, n = seq_chars.shape

    # ã‚®ãƒ£ãƒƒãƒ—ã�§ã�ªã�„ã‚·ãƒ¼ã‚±ãƒ³ã‚¹ã�®ãƒ�ã‚¹ã‚¯ (shape: (L, n))
    valid_mask = seq_chars != "-"

    # one-hot encoding: å�„ä½�ç½®ãƒ»ã‚·ãƒ¼ã‚±ãƒ³ã‚¹ãƒ»å¡©åŸºã�«å¯¾ã�—ã�¦ True/False
    # shape: (L, n, 4)
    oh = (seq_chars[:, :, None] == symbols[None, None, :]).astype(np.float64)

    # å�„ä½�ç½®ã�®ãƒ�ãƒ¼ã‚¸ãƒŠãƒ«ã‚«ã‚¦ãƒ³ãƒˆ (æœ‰åŠ¹ã�ªã‚·ãƒ³ãƒœãƒ«ã�®ã‚«ã‚¦ãƒ³ãƒˆ)
    # shape: (L, 4)
    marg_counts = oh.sum(axis=1)

    # å�„ä½�ç½®ã�®æœ‰åŠ¹ã�ªã‚·ãƒ¼ã‚±ãƒ³ã‚¹æ•°
    N = valid_mask.sum(axis=1)

    # ãƒ�ãƒ¼ã‚¸ãƒŠãƒ«ç¢ºç�‡: p_i(a) = count(a) / (æœ‰åŠ¹ã‚·ãƒ¼ã‚±ãƒ³ã‚¹æ•°)
    p = np.where(N[:, None] > 0, marg_counts / N[:, None], 0)

    # å�„ä½�ç½®ãƒšã‚¢ã�§ä¸¡ä½�ç½®ã�¨ã‚‚ã‚®ãƒ£ãƒƒãƒ—ã�§ã�ªã�„ã‚·ãƒ¼ã‚±ãƒ³ã‚¹æ•° (shape: (L, L))
    valid_counts = valid_mask.astype(np.int64) @ valid_mask.astype(np.int64).T

    # å�„ä½�ç½®ãƒšã‚¢ã�®joint counts:
    # joint_counts[i,j,a,b] = âˆ‘â‚– [oh[i,k,a] * oh[j,k,b]]
    # shape: (L, L, 4, 4)
    joint_counts = np.einsum("ika,jkb->ijab", oh, oh)

    # joint probability: p_{ij}(a,b) = joint_counts / (æœ‰åŠ¹ã‚·ãƒ¼ã‚±ãƒ³ã‚¹æ•°)
    joint_probs = np.where(
        valid_counts[:, :, None, None] > 0,
        joint_counts / valid_counts[:, :, None, None],
        0.0,
    )

    # ä½�ç½®ã�”ã�¨ã�®ç‹¬ç«‹åˆ†å¸ƒã�®ç©�: p_i(a)*p_j(b)
    prod = p[:, None, :, None] * p[None, :, None, :]

    # MIè¨ˆç®—: joint_probs * log(joint_probs / (p_i*p_j)) ã‚’å…¨å¡©åŸºãƒšã‚¢ã�«ã�¤ã�„ã�¦è¶³ã�—å�ˆã‚�ã�›ã‚‹
    with np.errstate(divide="ignore", invalid="ignore"):
        mi_terms = np.where(
            joint_probs > 0, joint_probs * np.log(joint_probs / prod), 0.0
        )

    # å�„ä½�ç½®ãƒšã‚¢ (i,j) ã�®MIã�¯å¡©åŸºæ¬¡å…ƒã�§ã�®ç·�å’Œ
    mi_matrix = mi_terms.sum(axis=(-2, -1))

    if zero_diag:
        np.fill_diagonal(mi_matrix, -np.inf)

    if normalize:
        # scipy.special.softmaxã‚’åˆ©ç”¨ã�—ã�¦å�„è¡Œæ–¹å�‘ã�«softmaxæ­£è¦�åŒ– (æ¸©åº¦ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿ä»˜ã��)
        mi_matrix = softmax(mi_matrix / temperature, axis=1)

    if symmetric:
        # å¯¾ç§°åŒ–
        mi_matrix = (mi_matrix + mi_matrix.T) / 2

    return mi_matrix


def plot_pairing_score(sim, threshold=0.5):
    bin_sim = np.where(sim > threshold, 1, 0)
    _, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.imshow(sim, cmap="hot", interpolation="nearest")
    ax.set(
        title="Pairing Score matrix",
        xlabel="Index",
        ylabel="Index",
    )
    plt.colorbar(ax.imshow(sim, cmap="hot", interpolation="nearest"))

    ax = axes[1]
    ax.imshow(bin_sim, cmap="hot", interpolation="nearest")
    ax.set(
        title=f"Pairing Score > {threshold}",
        xlabel="Index",
        ylabel="Index",
    )
    plt.show()


def get_color_lists(color_dict=mcolors.TABLEAU_COLORS):
    color_list = list(color_dict.values())

    return color_list


def search_pairs(pairing_score_mat, threshold=0.5):
    paring_indices = np.argmax(pairing_score_mat, axis=1)
    scores = pairing_score_mat[range(len(pairing_score_mat)), paring_indices]
    candidate_idxs = np.where(scores > threshold)[0]
    return candidate_idxs, scores[candidate_idxs], paring_indices[candidate_idxs]


def plot_pairing_scores(candidate_idxs, pairing_score_mat):
    _, ax = plt.subplots()

    pairs = []
    for idx in candidate_idxs:
        pair_idx = np.argmax(pairing_score_mat[:, idx])
        ax.axvline(
            pair_idx,
            color="gray",
            linestyle="--",
        )
        ax.plot(
            pairing_score_mat[:, idx],
            label=f"index={idx}, pair={pair_idx} (score={pairing_score_mat[pair_idx, idx]:.2f})",
        )
        pairs.append((idx, pair_idx))

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set(
        title="Pairing Score",
        xlabel="Index",
        ylabel="Paring Score",
    )
    plt.show()


def visualize_rna(
    target_id,
    pairs: list[tuple[int, int]],
    scores: list[float],
    width="100%",
    height=600,
    data_dir="/kaggle/input/stanford-3d-folding-minimal-eda/pdb",
):
    pdb_file = f"{data_dir}/{target_id}_1.pdb"

    view = plot_pdb(
        pdb_file, filter_option="C1", per_index=5, width=width, height=height
    )

    c1_atoms = extract_c1_atoms(pdb_file)

    # add bonds
    def add_cylindar(start, end, score):
        view.addCylinder(
            {
                "start": {"x": start["x"], "y": start["y"], "z": start["z"]},
                "end": {"x": end["x"], "y": end["y"], "z": end["z"]},
                "radius": 0.1,
                "color": "cyan",
            }
        )
        mid = (
            (start["x"] + end["x"]) / 2,
            (start["y"] + end["y"]) / 2,
            (start["z"] + end["z"]) / 2,
        )
        view.addLabel(
            f"{score:.2f}",
            {
                "position": {"x": mid[0], "y": mid[1], "z": mid[2]},
                "backgroundColor": "black",
                "backgroundOpacity": 0.3,
                "fontColor": "white",
                "fontSize": 14,
            },
        )

    for (i, j), score in zip(pairs, scores):
        add_cylindar(c1_atoms[i], c1_atoms[j], score)
    view.show()


target_id = "1A51_A"
threshold = 0.3
score_mat = calc_pairing_score(calc_seq_chars(target_id))
plot_pairing_score(score_mat, threshold=threshold)
candidate_idxs, scores, pairing_indices = search_pairs(score_mat, threshold=threshold)
print(f"Pairs with score > {threshold}:")
for i, j, score in zip(candidate_idxs, pairing_indices, scores):
    print(f"p({i}, {j})={score:.2f}")
plot_pairing_scores(candidate_idxs, score_mat)
visualize_rna(target_id, list(zip(candidate_idxs, pairing_indices)), scores)


target_id = "R1190"
threshold = 0.3
score_mat = calc_pairing_score(calc_seq_chars(target_id))
plot_pairing_score(score_mat, threshold=threshold)
candidate_idxs, scores, pairing_indices = search_pairs(score_mat, threshold=threshold)
print(f"Pairs with score > {threshold}:")
for i, j, score in zip(candidate_idxs, pairing_indices, scores):
    print(f"p({i}, {j})={score:.2f}")

visualize_rna(target_id + "_A", list(zip(candidate_idxs, pairing_indices)), scores, height=800)

