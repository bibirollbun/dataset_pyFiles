import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import scipy

sys.path.append('/kaggle/input/ariel2025-public/score')
import official_metric

INPUT_DIR = '/kaggle/input/ariel-data-challenge-2025/'


# Label y_true for 1100 training data
train = pd.read_csv(INPUT_DIR + '/train.csv')
y_true = train.iloc[:, 1:].to_numpy()

# Submission file for 1100 training data
submission = pd.read_csv('/kaggle/input/ariel2025-public/score/submission_train.csv')
mu_pred = submission.iloc[:, 1:284].to_numpy()
sigma_pred = submission.iloc[:, 284:].to_numpy()

# Assume planet_id in same order
assert (train['planet_id'] == submission['planet_id']).all()

y_true.shape, mu_pred.shape, sigma_pred.shape


naive_mean = np.mean(y_true)
naive_sigma = np.std(y_true, ddof=1)  # ddof=1 makes negligible difference but I argue that this is statistally correct

print('naive_mean:  %.8e' % naive_mean)
print('naive_sigma: %.8e' % naive_sigma)


# airs_weight=1 is fixed in the official metric, set fgs_weight relative to this airs_weight
fgs_weight = 0.4 / 1.95 * 282
print('fgs_weight: %.8e' % fgs_weight)

s = official_metric.score(train.copy(), submission.copy(), 'planet_id', naive_mean, naive_sigma, fgs_weight=fgs_weight)
# Because score function drop planet_id column, I pass the copies; in case you use them later.

print('Local CV:  %.4f' % s)
print('Public LB: %.3f' % 0.303)


# My Public LB data
sigma_fgs  = np.array([ 1e-3, 0.5e-3,  4e-3,  2e-3,  2e-3,  4e-3])
sigma_airs = np.array([ 1e-3, 0.5e-3,  4e-3,  2e-3,  1e-3,  1e-3])
scores     = np.array([0.303,  0.226, 0.175, 0.255, 0.300, 0.291])


sigma_ideal_fgs = 1e-6
sigma_ideal_airs = 1e-5
sigma_ideal = np.array([sigma_ideal_fgs, ] + [sigma_ideal_airs, ] * 282).reshape(1, 283)

L_ideal = -0.5 * np.log(2 * np.pi * sigma_ideal ** 2)

print('L_ideal: %.8f %.8f' % (L_ideal[0, 0], L_ideal[0, 1]), sigma_ideal.shape)


GLL_mean = scipy.stats.norm.logpdf(y_true,
                                   loc=naive_mean * np.ones_like(y_true),
                                   scale=naive_sigma * np.ones_like(y_true))
GLL_mean.shape


print('L_ref not the constant: %.4f' % GLL_mean.mean())
print('or wavelentgh-dependent mean: %r...' % [float(x) for x in GLL_mean.mean(axis=0)[:4]])


# Data
sigma_fgs  = np.array([ 1e-3, 0.5e-3,  4e-3,  2e-3,  2e-3,  4e-3])
sigma_airs = np.array([ 1e-3, 0.5e-3,  4e-3,  2e-3,  1e-3,  1e-3])
scores     = np.array([0.303,  0.226, 0.175, 0.255, 0.300, 0.291])

# Constant
sigma_perf_fgs = 1e-6
sigma_perf_airs = 1e-5

L_perf_fgs = -0.5 * np.log(2 * np.pi * sigma_perf_fgs ** 2)
L_perf_airs = -0.5 * np.log(2 * np.pi * sigma_perf_airs ** 2)
print('L_perf:', L_perf_fgs, L_perf_airs)


def compute_score(sigma_fgs, sigma_airs, mse_fgs, mse_airs, L_ref):
    """
    Args:
      sigma_fgs (array):  submission data (n, )
      sigma_airs (array): submission data (n, )
      mse_fgs, mse_airs (float): mse = [(mu_pred - mu_true) / 1e-3] ** 2
      w_fgs (float): unnormalized weight for FGS1; w_airs = 1

    Returns:
      scores (array): (n, )
    """
    w_airs = 1.95
    w_fgs = 0.4
    L_fgs  = -0.5 * (np.log(2 * np.pi * sigma_fgs ** 2)  + mse_fgs * (1e-3 / sigma_fgs) ** 2)
    L_airs = -0.5 * (np.log(2 * np.pi * sigma_airs ** 2) + mse_airs * (1e-3 / sigma_airs) ** 2)

    scores = (w_fgs / (L_perf_fgs - L_ref) * (L_fgs - L_ref) + w_airs / (L_perf_airs - L_ref) * (L_airs - L_ref)) / (w_fgs + w_airs)
    return scores


def f_opt(x):
    # Optimize unknown parameters
    mse_fgs, mse_airs, L_ref = x
    scores_theory = compute_score(sigma_fgs, sigma_airs, mse_fgs, mse_airs, L_ref)
    loss = np.mean((scores_theory - scores) ** 2)
    return loss

opt = scipy.optimize.minimize(f_opt, x0=(1, 1, 3))
print('opt parameters', opt.x)
print('rmse', np.sqrt(opt.fun))


mse_fgs, mse_airs, L_ref = opt.x

sigma_smooth = np.linspace(5e-4, 4e-3, 101)
ones = 1e-3 * np.ones_like(sigma_smooth)

scores_smooth  = compute_score(sigma_smooth, sigma_smooth, mse_fgs, mse_airs, L_ref)
scores_smooth2 = compute_score(sigma_smooth,         ones, mse_fgs, mse_airs, L_ref)

plt.title('Scores for fixed $\\mu_\\mathrm{pred}$; $\\mathcal{L}_\\mathrm{ref} = %.4f$' % L_ref)
plt.plot(sigma_smooth, scores_smooth, alpha=0.8, label='$\\sigma_\\mathrm{FGS} = \\sigma_\\mathrm{AIRS} = \\sigma$')
plt.plot(sigma_smooth, scores_smooth2, alpha=0.8, label='$\\sigma_\\mathrm{FGS} = \\sigma; \\, \\sigma_\\mathrm{AIRS}=10^{-3}$')
plt.plot(sigma_fgs, scores, 'x', color='black', label='Local CV scores')
plt.xlabel('$\\sigma$')
plt.ylabel('Score')
plt.legend()
plt.show()


print('Effective constant L_ref:', L_ref)


GLL_pred = scipy.stats.norm.logpdf(y_true, loc=mu_pred, scale=sigma_pred)
GLL_ideal = scipy.stats.norm.logpdf(y_true, loc=y_true, scale=sigma_ideal * np.ones_like(y_true))
GLL_ref = scipy.stats.norm.logpdf(y_true, loc=naive_mean * np.ones_like(y_true), scale=naive_sigma * np.ones_like(y_true))

GLL_pred.shape, GLL_ideal.shape, GLL_ref.shape


# score = a L - b
a = 1 / (GLL_ideal - GLL_ref)
b = GLL_ref / (GLL_ideal - GLL_ref)

a.shape, b.shape


# Plot L_ref histogram
bins = np.linspace(-15, 10, 101)

plt.figure(figsize=(6, 2))
plt.title('L_ref is not a constant')
plt.xlabel('L_ref')
plt.hist(GLL_ref[:, 0], bins, histtype='step', density=True, label='FGS1')
plt.hist(GLL_ref[:, 1:].flatten(), bins, histtype='step', density=True, label='AIRS')
plt.legend(loc=2, frameon=False)
plt.show()

# Plot a, b, where
# score = (L - L_ref) / (L_ideal - L_ref) = a L - b
plt.figure(figsize=(8, 2.5))
plt.suptitle('$\\mathrm{score} = a \\mathcal{L} - b$')
plt.subplot(1, 2, 1)
plt.xlabel('a')
bins = np.linspace(0.03, 0.15, 101)
plt.hist(a[:, 0], bins, label='FGS1', density=True, histtype='step')
plt.hist(a[:, 1:].flatten(), bins, label='AIRS', density=True, histtype='step')
plt.legend(loc=2, frameon=False)

plt.subplot(1, 2, 2)
plt.xlabel('b')
bins = np.linspace(-0.6, 0.6, 101)
plt.hist(b[:, 0], bins, label='FGS1', density=True, histtype='step')
plt.hist(b[:, 1:].flatten(), bins, label='AIRS', density=True, histtype='step')

plt.tight_layout()

plt.show()


print('FGS1')
print('  Mean      a:', np.mean(a[:, 0]))
print('  Effective a:', 1 / (L_ideal[0, 0] - L_ref))
print('')
print('  Mean      b:', np.mean(b[:, 0]))
print('  Effective b:', L_ref / (L_ideal[0, 0] - L_ref))

print('')
print('AIRS')
print('  Mean      a:', np.mean(a[:, 1:]))
print('  Effective a:', 1 / (L_ideal[0, 1] - L_ref))
print('')
print('  Mean      b:', np.mean(b[:, 1:]))
print('  Effective b:', L_ref / (L_ideal[0, 1] - L_ref))




