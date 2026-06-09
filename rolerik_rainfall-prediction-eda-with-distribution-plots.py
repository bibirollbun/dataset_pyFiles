import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datadir = "/kaggle/input/playground-series-s5e3/"



train = pd.read_csv(datadir + "train.csv", index_col="id")
print("Train size: " + str(train.shape))
train["day2"] = train.index % 365 + 1

test = pd.read_csv(datadir + "test.csv", index_col="id")
print("Test size: " + str(test.shape))
test["day2"] = test.index % 365 + 1

train.head()



train.info()



train.describe()



print("Missing data in train: ", train.isna().sum().sum())
print("Missing data in test: ", test.isna().sum().sum())

test[test.isna().any(axis=1)]
test["winddirection"] = test["winddirection"].ffill()

print("Missing data in test: ", test.isna().sum().sum())



rain = train[train["rainfall"] == 1]
norain = train[train["rainfall"] == 0]
cols = [c for c in train.columns if c not in ["day", "day2", "rainfall"]]
ncols = len(cols)
print("Number of features: ", ncols)



fig, axs = plt.subplots(ncols, 2, figsize=(9, 3*ncols), layout="constrained")

for i in range(ncols):
    axs[i,0].hist(rain[cols[i]], bins=20, histtype="stepfilled", density=True, linewidth=2, edgecolor="blue", facecolor="skyblue", alpha=0.3, label="rain")
    axs[i,0].hist(norain[cols[i]], bins=20, histtype="stepfilled", density=True, linewidth=2, edgecolor="darkorange", facecolor="gold", alpha=0.3, label="no rain")
    axs[i,0].set_xlabel(cols[i])
    axs[i,0].set_ylabel("Density")
    axs[i,0].set_title(cols[i])
    axs[i,0].legend()

    axs[i,1].hist(train[cols[i]], bins=20, histtype="stepfilled", density=True, linewidth=2, edgecolor="green", facecolor="palegreen", alpha=0.3, label="train")
    axs[i,1].hist(test[cols[i]], bins=20, histtype="stepfilled", density=True, linewidth=2, edgecolor="red", facecolor="coral", alpha=0.3, label="test")
    axs[i,1].set_xlabel(cols[i])
    axs[i,1].set_ylabel("Density")
    axs[i,1].set_title(cols[i])
    axs[i,1].legend()



from matplotlib.mlab import GaussianKDE 

def binbar(ax, x, y, bins=10, alpha=0.5, meanline=False, kde=False):
    binedges = np.histogram_bin_edges(x, bins)
    binwidth = binedges[1] - binedges[0]
    bincenters = (binedges[1:] + binedges[:-1]) / 2
    binedges[bins] += 0.01*binwidth
    binidx = np.digitize(x, binedges)
    r = y.groupby(binidx).mean()
    r = r.reindex(range(1, nbins + 1), fill_value=0)
    ymean = y.mean()
    
    ax.bar(bincenters, r, bottom=0, width=binwidth, color="mediumblue", alpha=alpha, label="rain")
    ax.bar(bincenters, 1-r, bottom=r, width=binwidth, color="gold", alpha=alpha, label="no rain")
    if (meanline):
        ax.plot([binedges[0], binedges[-1]], [ymean, ymean], color="r", linestyle="--", alpha=0.8)
    if (kde):
        est = GaussianKDE(x)
        xkde = np.linspace(binedges[0], binedges[-1], bins * 4)
        ykde = est.evaluate(xkde)
        ykde = ykde * ymean / np.max(ykde)
        ax.plot(xkde, ykde, color="r", linestyle="--", alpha=0.8)
    ax.set_xlabel(x.name)
    ax.set_ylabel("Rain")
    ax.set_title(x.name)
    ax.legend()



fig, axs = plt.subplots(5, 2, figsize=(9, 16), layout="constrained")
axs = axs.flatten()
nbins = 10

for i in range(ncols):
    binbar(axs[i], train[cols[i]], train["rainfall"], bins=nbins, alpha=0.4, meanline=True, kde=True)



def runs(binseq):
    binseq = np.array(binseq)
    # Diff sequence, append opposite of first value to force run start at beginning of sequence
    # Diff is expected to be +1 or -1 at start of run, otherwise 0
    binseqdiff = np.diff(binseq, n=1, prepend = [1-binseq[0]])
    runstarts = np.abs(binseqdiff)
    # Use cumsum to give each run a unique id number
    return pd.Series(runstarts.cumsum())

def runlengths(binseq):
    # First value count gives length of each run, second gives distribution of run lengths
    return runs(binseq).value_counts().value_counts().sort_index()
    



rng = np.random.default_rng()
random_rain = rng.choice([0, 1], p=[0.25, 0.75], size=train.shape[0])

rain_runlenghts = runlengths(train["rainfall"])
randomrain_runlenghts = runlengths(random_rain)



fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(rain_runlenghts.index, rain_runlenghts, color="blue", linewidth=2, alpha=0.5, label="rain")
ax.plot(randomrain_runlenghts.index, randomrain_runlenghts, color="green", linewidth=2, linestyle="--", alpha=0.5, label="random")
ax.set_xlim(0, 20)
ax.set_xlabel("Run length")
ax.set_ylabel("Count")
ax.set_title("Run lenghts of rainy/non-rainy days compared to random")
ax.set_yscale("log")
ax.grid(True)
ax.legend()

print()


