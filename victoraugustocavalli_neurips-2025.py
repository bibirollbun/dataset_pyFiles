!pip install gpytorch


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import itertools
from astropy.stats import sigma_clip
import glob
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
import pickle
import time
from numba import njit, prange
import torch, gpytorch
from torch.utils.data import DataLoader, TensorDataset

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


wavelengths = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/wavelengths.csv")
adc_info = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/adc_info.csv")
axis_info = pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/axis_info.parquet")
star_info = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/train_star_info.csv")
train = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/train.csv")
sample = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/sample_submission.csv")
#train_AIRS_0 = pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/train/1010375142/AIRS-CH0_signal_0.parquet")
#train_AIRS_0_dark = pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/train/1010375142/AIRS-CH0_calibration_0/dark.parquet")
#train_AIRS_0_flat = pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/train/1010375142/AIRS-CH0_calibration_0/flat.parquet")
#train_AIRS_0_dead = pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/train/1010375142/AIRS-CH0_calibration_0/dead.parquet")
#train_linear_corr = pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/train/1010375142/AIRS-CH0_calibration_0/linear_corr.parquet")
#test_AIRS_1 = pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/test/1103775/AIRS-CH0_signal_1.parquet")



@njit(parallel=True, fastmath=True)
def poly_correct_numba_unrolled(signals, coeffs):
    """
    Numba-parallel Horner's method, unrolled for degree=6, cache-friendly blocking.
    signals: (t, 32, 282)
    coeffs:  (6, 32, 282) highest degree first
    """
    t, nx, ny = signals.shape
    out = np.empty((t, nx, ny), dtype=np.float64)

    # block the time loop for better cache reuse
    block_size = 64
    for ix in prange(nx):
        for iy in range(ny):
            c0, c1, c2, c3, c4, c5 = coeffs[:, ix, iy]
            for tb in range(0, t, block_size):
                t_end = min(tb + block_size, t)
                for it in range(tb, t_end):
                    s = signals[it, ix, iy]
                    # Horner's method unrolled for degree 6
                    acc = (((((c0 * s) + c1) * s + c2) * s + c3) * s + c4) * s + c5
                    out[it, ix, iy] = acc
    return out

def data_fast_fast_fast(AIRS, AIRS_dark, AIRS_flat, AIRS_dead, linear_corr, adc_info, axis_info):
    # Gain + offset correction
    signals = (AIRS / adc_info["AIRS-CH0_adc_gain"].values[0]) + adc_info["AIRS-CH0_adc_offset"].values[0]
    signals = signals.values.reshape(signals.shape[0], 32, 356)

    L, U = 39, 321
    dt_airs = axis_info['AIRS-CH0-integration_time'].dropna().values
    dt_airs[1::2] += 0.1
    signals = signals[:, :, L:U]

    # Dark/hot/dead mask → np.nan
    dark = AIRS_dark.values[:, L:U]
    hot = sigma_clip(dark, sigma=5, maxiters=5).mask
    dead = AIRS_dead.values[:, L:U]
    mask = np.broadcast_to(hot | dead, signals.shape)
    signals = np.where(mask, signals.mean(axis=(0,1,2)), signals)

    # Polynomial correction (Numba unrolled)
    coeffs = np.flip(linear_corr.values.reshape(6, 32, 356)[:, :, L:U], axis=0)
    signals = poly_correct_numba_unrolled(signals, coeffs)

    # Dark subtraction
    dark_masked = np.where(dead, dark.mean(axis=(0,1)), dark)
    dark_masked = np.broadcast_to(dark_masked, signals.shape)
    signals -= dark_masked * dt_airs[:, None, None]

    # CDS subtraction
    signals = signals[1::2] - signals[::2]

    # Bin sum over 30-frame blocks
    signals = signals.transpose(0, 2, 1)  # (time, 282, 32)
    n_blocks = signals.shape[0] // 30
    cds_binned = signals[:n_blocks * 30].reshape(n_blocks, 30, signals.shape[1], signals.shape[2]).sum(axis=1)

    # Flat-field correction
    flat = AIRS_flat.values[:, L:U].T
    dead_flat = AIRS_dead.values[:, L:U].T
    flat = np.where(dead_flat, flat.mean(axis=(0,1)), flat)
    flat = np.broadcast_to(flat, cds_binned.shape)
    cds_binned /= flat

    return cds_binned


# whether to Load or Process the DATA}
LOAD = 1
# whether to preprocess data fot Submission

SUBMISSION = 0


start_time = time.time()
RANDOM_SEED = 0
N = 1100
if not LOAD:
    names = []
    train_cds_binned = np.zeros((int(N*0.8), 187, 282, 32))
    test_cds_binned = np.zeros((int(N*0.2), 187, 282, 32))
    for i, j in enumerate(glob.glob("/kaggle/input/ariel-data-challenge-2025/train/*")[:N]):
        names.append(j[46:])
        if i < N*0.8:
            print("Train: " + names[i])
            start_time1 = time.time()
            train_AIRS_0 = pd.read_parquet(j + "/AIRS-CH0_signal_0.parquet", engine = "pyarrow", use_threads = True)
            train_AIRS_0_dark = pd.read_parquet(j + "/AIRS-CH0_calibration_0/dark.parquet", engine = "pyarrow", use_threads = True)
            train_AIRS_0_flat = pd.read_parquet(j + "/AIRS-CH0_calibration_0/flat.parquet", engine = "pyarrow", use_threads = True)
            train_AIRS_0_dead = pd.read_parquet(j + "/AIRS-CH0_calibration_0/dead.parquet", engine = "pyarrow", use_threads = True)
            train_linear_corr0 = pd.read_parquet(j + "/AIRS-CH0_calibration_0/linear_corr.parquet", engine = "pyarrow", use_threads = True)
            end_time1 = time.time()
            elapsed_time1 = end_time1 - start_time1
            print(f"Execution time: {elapsed_time1:.4f} seconds")
            train_cds_binned[i] = data_fast_fast_fast(train_AIRS_0, train_AIRS_0_dark, train_AIRS_0_flat, train_AIRS_0_dead, train_linear_corr0, adc_info, axis_info)
            print("(",i,train_cds_binned.shape[1],train_cds_binned.shape[2],train_cds_binned.shape[3],")")
        else:
            print("Test: " + names[i] )
            test_AIRS_0 = pd.read_parquet(j + "/AIRS-CH0_signal_0.parquet", engine = "pyarrow", use_threads = True)
            test_AIRS_0_dark = pd.read_parquet(j + "/AIRS-CH0_calibration_0/dark.parquet", engine = "pyarrow", use_threads = True)
            test_AIRS_0_flat = pd.read_parquet(j + "/AIRS-CH0_calibration_0/flat.parquet", engine = "pyarrow", use_threads = True)
            test_AIRS_0_dead = pd.read_parquet(j + "/AIRS-CH0_calibration_0/dead.parquet", engine = "pyarrow", use_threads = True)
            test_linear_corr0 = pd.read_parquet(j + "/AIRS-CH0_calibration_0/linear_corr.parquet", engine = "pyarrow", use_threads = True)
            test_cds_binned[i-int(N*0.8)] = data_fast_fast_fast(test_AIRS_0, test_AIRS_0_dark, test_AIRS_0_flat, test_AIRS_0_dead, test_linear_corr0, adc_info, axis_info)
            print("(",i,test_cds_binned.shape[1],test_cds_binned.shape[2],test_cds_binned.shape[3],")")
    np.save(os.path.join("/kaggle/working/", 'train.npy'), train_cds_binned)
    np.save(os.path.join("/kaggle/working/", 'test.npy'), test_cds_binned)
    np.save(os.path.join("/kaggle/working/", 'names.npy'), names)
else:
    if SUBMISSION:
        names_test = []
        train_cds_binned = np.load("/kaggle/input/ariel-data-100planets/train.npy")
        test_cds_binned = np.load("/kaggle/input/ariel-data-100planets/test.npy")
        train_cds_binned = np.vstack([train_cds_binned, test_cds_binned])
        del test_cds_binned
        test_planet_list = glob.glob("/kaggle/input/ariel-data-challenge-2025/test/*")
        test_cds_binned = np.zeros((len(test_planet_list), 187, 282, 32))
        names = np.load("/kaggle/input/ariel-data-100planets/names.npy")
        for i, j in enumerate(test_planet_list):
            if (i % 100 == 0) or (i == 0):
                print(i,"/",len(test_planet_list))
            print(os.path.split(j)[1])
            names_test.append(os.path.split(j)[1])
            test_AIRS_0 = pd.read_parquet(j + "/AIRS-CH0_signal_0.parquet", engine = "pyarrow", use_threads = True)
            test_AIRS_0_dark = pd.read_parquet(j + "/AIRS-CH0_calibration_0/dark.parquet", engine = "pyarrow", use_threads = True)
            test_AIRS_0_flat = pd.read_parquet(j + "/AIRS-CH0_calibration_0/flat.parquet", engine = "pyarrow", use_threads = True)
            test_AIRS_0_dead = pd.read_parquet(j + "/AIRS-CH0_calibration_0/dead.parquet", engine = "pyarrow", use_threads = True)
            test_linear_corr0 = pd.read_parquet(j + "/AIRS-CH0_calibration_0/linear_corr.parquet", engine = "pyarrow", use_threads = True)
            test_cds_binned[i] = data_fast_fast_fast(test_AIRS_0, test_AIRS_0_dark, test_AIRS_0_flat, test_AIRS_0_dead, test_linear_corr0, adc_info, axis_info)
    else:
        
        train_cds_binned = np.vstack([np.load("/kaggle/input/ariel-data-1100lanets/train_part_1.npy"), np.load("/kaggle/input/ariel-data-1100lanets/train_part_2.npy"),
                                      np.load("/kaggle/input/ariel-data-1100lanets/train_part_3.npy"), np.load("/kaggle/input/ariel-data-1100lanets/train_part_4.npy")])
        test_cds_binned = np.load("/kaggle/input/ariel-data-1100lanets/test.npy")
        names = np.load("/kaggle/input/ariel-data-1100lanets/names.npy")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")


if SUBMISSION:
    test_list = [int(os.path.split(test_planet_list[i])[1]) for i in range(len(test_planet_list))]
    unsorted_df = pd.DataFrame(test_cds_binned.mean(axis = (1,3)), index = test_list, columns = wavelengths.columns[1:])
    unsorted_df.index.name = 'planet_id'
    sorted_df = unsorted_df.sort_values(by="planet_id")


#plt.plot(wavelengths.iloc[-1][1:].values,train[train["planet_id"]==int(names[-1])].values[0,2:])


#plt.plot(cds_binned.mean(axis=(1,2)), '-', alpha=0.3)


#df = pd.DataFrame(cds_binned.mean(axis=(2)))
#df.columns = wavelengths.iloc[0].values[1:]


planet = 879
nrows = 4
ncols = 8
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10)) # Adjust figsize as needed
axes_flat = axes.flatten()
for i in range(32):
    ax = axes_flat[i]
    ax.plot(diff[planet][i])
    ax.set_title(f'Plot {i+1}') # Add a unique title for each plot
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

# Adjust layout to prevent overlapping titles and labels
plt.tight_layout()

# Display the plots
plt.show()


maxvalue, minvalue = train_cds_binned.transpose(0,3,2,1).max(3), train_cds_binned.transpose(0,3,2,1).min(3)
difftrain = (maxvalue - minvalue)



maxvalue, minvalue = test_cds_binned.transpose(0,3,2,1).max(3), test_cds_binned.transpose(0,3,2,1).min(3)
difftest = (maxvalue - minvalue)


difftrain.shape, difftest.shape


#wavelength = 110
planet = 104
value = []
for wavelength in range(282):
 value.append((np.max(train_cds_binned[planet].mean(2).transpose(1,0)[wavelength]) - np.min(train_cds_binned[planet].mean(2).transpose(1,0)[wavelength])))
plt.subplot(3, 1, 1)
plt.plot(value)
plt.subplot(3, 1, 2)
plt.plot(y[planet])
plt.subplot(3, 1, 3)
plt.plot(train_cds_binned[planet].mean(2).transpose(1,0).mean(1))


# X = train_cds_binned.mean(axis = (1,3))
X = difftrain.mean(1)
# X = pd.DataFrame(X.T)
# X.index = wavelengths.iloc[-1][1:].values

y = []
if SUBMISSION:
    for i in names:
        y.append(train[train["planet_id"]==int(i)].values[0,2:])
else:
    for i in names[:int(N*0.8)]:
        y.append(train[train["planet_id"]==int(i)].values[0,2:])
y = pd.DataFrame(y)
# y.index = wavelengths.iloc[-1][1:].values


if SUBMISSION:
    Xtest = sorted_df.values
if not SUBMISSION:
    # Xtest = test_cds_binned.mean(axis = (1,3))
    Xtest = difftest.mean(1)
    ytest = []
    for i in names[int(N*0.8):]:
        ytest.append(train[train["planet_id"]==int(i)].values[0,2:])
    ytest = pd.DataFrame(ytest).transpose()
    ytest.index = wavelengths.iloc[-1][1:].values
# Xtest = pd.DataFrame(Xtest.T)
# Xtest.index = wavelengths.iloc[-1][1:].values


# X = X.T
# y = y.T
# Xtest = Xtest.T
if SUBMISSION:
    print(X.shape, Xtest.shape, y.shape)
else:
    ytest = ytest.T
    print(X.shape, y.shape, Xtest.shape, ytest.shape)


X = X.reshape(X.shape[0], -1)
Xtest = Xtest.reshape(Xtest.shape[0], -1)
scaler_flux = StandardScaler(with_mean=True, with_std=True)
X_train_scaled = scaler_flux.fit_transform(X)
X_test_scaled  = scaler_flux.transform(Xtest)


n_components = 60   # try 10-50; you can tune this
pca = PCA(n_components=n_components)
Z_train = pca.fit_transform(X_train_scaled)   # (n_train, n_components)
Z_test  = pca.transform(X_test_scaled)        # (n_test,  n_components)


import umap
import hdbscan
def cluster_planets(R, n_neighbors=32, min_dist=0.1, n_components=282, min_cluster_size=32, random_state=42):
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state
    )
    embedding = umap_model.fit_transform(R)
    
    # 3. Apply HDBSCAN for clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean'
    )
    labels = clusterer.fit_predict(embedding)
    
    return labels, embedding


labels, embedding = cluster_planets(X_train_scaled)

print("Cluster counts:", np.unique(labels, return_counts=True))


plt.scatter(embedding[:,0], embedding[:,1], c=labels, cmap='Spectral', s=10)
plt.colorbar(label='Cluster')
plt.title("UMAP + HDBSCAN clustering of planets")
plt.show()


from sklearn.metrics import pairwise_distances
d = pairwise_distances(Z_test, Z_train)  # shape (n_test, 1100)
print("min dist per test -> train:", d.min(axis=1))
print("mean train-train min dist:", pairwise_distances(Z_train, Z_train).min(axis=1).mean())


y_scaler = StandardScaler(with_mean=True, with_std=True)
Y_train_scaled = y_scaler.fit_transform(y)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Xt = torch.from_numpy(Z_train).float().to(device)
Yt = torch.from_numpy(Y_train_scaled).float().to(device)
Xttest = torch.from_numpy(Z_test).float().to(device)

N, d = Xt.shape
P = Yt.shape[1]

train_loader = DataLoader(TensorDataset(Xt, Yt), batch_size=256, shuffle=False)
M = min(128, N)   # inducing points
print(M)
inducing = Xt[torch.randperm(N)[:M]]


class IndependentMTGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, num_tasks, ard_dims):
        # one variational distribution for all tasks (independent across tasks)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        # wrap with multitask independence
        mt_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            variational_strategy, num_tasks=num_tasks
        )
        super().__init__(mt_strategy)

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims, batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

model = IndependentMTGP(inducing, num_tasks=P, ard_dims=d).to(device)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=P).to(device)


model.train(); likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=N)

EPOCHS = 10
for epoch in range(EPOCHS):
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad(set_to_none=True)
        out = model(xb)
        loss = -mll(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"epoch {epoch+1} loss {total_loss:.3f}")


model.eval(); likelihood.eval()
with torch.inference_mode(), gpytorch.settings.fast_pred_var():
    pred = likelihood(model(Xttest))
    mean = pred.mean.reshape(-1, P)        # (M, P)
    sigma = pred.variance.sqrt().reshape(-1, P)

Y_pred = mean.cpu().numpy()
Y_sigma = sigma.cpu().numpy()
Y_pred = y_scaler.inverse_transform(Y_pred)   # (n_test, L)
Y_sigma = Y_sigma * y_scaler.scale_


if not SUBMISSION:
    dfy = pd.DataFrame(Y_pred.T)
    dfy.index = wavelengths.iloc[-1][1:].values
    a = 0
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(dfy.index, dfy.values[:,a], label='Line 1')
    axs[0].set_title('Predited Values')
    axs[1].plot(dfy.index, ytest.T.values[:,a], label='Line 2', linestyle='--')
    axs[1].set_title('Actual Values')


# Plot the second line
#plt.plot(dfy.index, ytest.T.values[:,a], label='Line 2', linestyle='--')


if not SUBMISSION:
    from sklearn.metrics import mean_absolute_error

    # y_true and y_pred should be arrays of the same shape
    mae = mean_absolute_error(ytest.T.values, dfy.values)
    print("MAE:", mae)


# def make_dataframe(pred_array, sigma_pred):
#     df = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/sample_submission.csv")
#     df.drop(0, inplace=True)
#     typed = df["planet_id"].dtype
#     df["planet_id"] = names_test
#     df["planet_id"] = df["planet_id"].astype(typed)
#     for i in range(Y_pred.shape[0]):
#         df.loc[i,df.columns[1:].tolist()] = np.hstack((Y_pred[i][0],Y_pred[i],Y_sigma[i][0],Y_sigma[i])).T
#     return df


def make_dataframe(Y_pred, Y_sigma):
    df = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/sample_submission.csv", index_col="planet_id")
    dg = pd.DataFrame(index=df.index, columns=df.columns)
    for i, j in enumerate(df.index):
        dg.loc[j] = np.hstack((Y_pred[i][0],Y_pred[i],Y_sigma[i][0],Y_sigma[i])).T
    df.reset_index(inplace=True)
    dg.reset_index(inplace=True)
    for col in dg.columns:
        dg[col] = dg[col].astype(df[col].dtype)
    return dg


if SUBMISSION:
    sub = make_dataframe(Y_pred, Y_sigma)
    sub.to_csv('submission.csv',index=False)


#train_cds_binned = np.load("/kaggle/working/train.npy")


#from IPython.display import FileLink
#FileLink(r'names.npy')


#from IPython.display import FileLink
#FileLink(r'test.npy')


#os.remove("/kaggle/working/train.npy")


#np.save(("/kaggle/working/train_part_1.npy"), train_cds_binned[0:220])
#np.save(("/kaggle/working/train_part_2.npy"), train_cds_binned[220:440])
#np.save(("/kaggle/working/train_part_3.npy"), train_cds_binned[440:660])
#np.save(("/kaggle/working/train_part_4.npy"), train_cds_binned[660:880])


#from IPython.display import FileLink
#FileLink(r'train_part_1.npy')


#from IPython.display import FileLink
#FileLink(r'train_part_2.npy')


#from IPython.display import FileLink
#FileLink(r'train_part_3.npy')


#from IPython.display import FileLink
#FileLink(r'train_part_4.npy')




