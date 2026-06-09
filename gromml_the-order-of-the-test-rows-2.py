import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances


new_shuffled_test = pd.read_parquet(
    '/kaggle/input/drw-crypto-market-prediction/test.parquet'
).iloc[:, :250]  # new test data (shuffled)
old_sorted_test = pd.read_parquet(
    '/kaggle/input/the-order-of-the-test-rows/sorted_test.parquet'
).iloc[:, :250]  # old test data (sorted by time)
assert new_shuffled_test.shape == old_sorted_test.shape
assert new_shuffled_test.index.is_monotonic_increasing
assert new_shuffled_test.shape[0] == 538150
assert 10 <= new_shuffled_test.shape[1] < 1000


plt.figure(figsize=(16, 4))
plt.plot(old_sorted_test['X1'].sort_values().to_numpy())
plt.plot(new_shuffled_test['X1'].sort_values().to_numpy())


# for col in tqdm.tqdm(new_shuffled_test.columns):
#     a, b = np.polyfit(
#         x=old_sorted_test[col].sort_values().to_numpy(),
#         y=new_shuffled_test[col].sort_values().to_numpy(),
#         deg=1
#     )
#     old_sorted_test[col] = a * old_sorted_test[col].to_numpy() + b


new_shuffled_test = StandardScaler().fit_transform(new_shuffled_test)
old_sorted_test = StandardScaler().fit_transform(old_sorted_test)


def find_closest_row(distances, debug=False):
    assert len(distances.shape) == 1
    assert distances.shape[0] == 538150
    assert np.all(distances >= 0.0)
    assert debug in [False, True]

    sorted_distances = np.sort(distances)
    diff = np.diff(sorted_distances)
    assert np.all(diff >= 0.0)
    threshold = 1000.0 * np.median(diff[1000:10000])

    if debug:
        fig, axes = plt.subplots(2, 1, figsize=(16, 6))
        axes[0].plot(sorted_distances[:50], marker='o', color='tab:blue')
        axes[0].set_ylabel('distance')
        axes[1].plot(diff[:49], marker='o', color='tab:orange')
        axes[1].axhline(threshold, linestyle='--', color='r')
        axes[1].set_ylabel('distance diff')

    if diff[0] > threshold:
        return np.argmin(distances)

    return -1


# For example, let's take a look at a random row (with index 1755):
find_closest_row(
    pairwise_distances(new_shuffled_test[1755:1756], old_sorted_test)[0],
    debug=True
)


# But not every row has a corresponding match in the old sorted dataset
find_closest_row(
    pairwise_distances(new_shuffled_test[1800:1801], old_sorted_test)[0],
    debug=True
)


batch_size = 2500  # to avoid running out of memory
closest_rows = []  # the index of the closest row in the old test dataset
# all_distances = []  # the distance matrix

for i in tqdm.tqdm(range(0, new_shuffled_test.shape[0], batch_size)):
    window_begin = i
    window_end = min(i + batch_size, new_shuffled_test.shape[0])
    batch_distances = pairwise_distances(
        new_shuffled_test[window_begin:window_end],
        old_sorted_test
    )
    assert len(batch_distances.shape) == 2
    assert batch_distances.shape[0] == window_end - window_begin
    assert batch_distances.shape[1] == old_sorted_test.shape[0]
    for row_distances in batch_distances:
        closest_rows.append(find_closest_row(row_distances))
        # all_distances.append(row_distances)


new_shuffled_test = None
old_sorted_test = None


closest_rows = pd.Series(closest_rows)
# all_distances = pd.DataFrame(all_distances)
# assert all_distances.shape == (len(closest_rows), len(closest_rows))


closest_rows.to_csv('closest_rows.csv')
# all_distances.to_csv('all_distances.csv')


closest_rows




