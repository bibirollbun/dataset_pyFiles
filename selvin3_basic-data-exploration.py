import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = np.load("/kaggle/input/waveform-inversion/train_samples/Style_A/data/data1.npy")
print(data.shape)
x = data[0]   # first sample
print(x.shape)


plt.figure(figsize=(18, 4))

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x[i], aspect="auto", cmap="seismic")
    plt.title(f"Source {i+1}")
    plt.xlabel("Time steps")
    plt.ylabel("Amplitude")

plt.suptitle("Seismic Waveforms: All 5 Sources", fontsize=14)
plt.tight_layout()
plt.show()


receiver_number = 2  # choose receiver index

plt.figure(figsize=(18, 4))

# x[:, :, receiver_number] -> (5,1000).T -> (1000,5)
# X-axis → time
# Each line → one source
plt.plot(x[:, :, receiver_number].T)

plt.xlabel("Time steps")
plt.ylabel("Amplitude")
plt.title(f"Receiver {receiver_number}: Waveforms from All 5 Sources")

plt.tight_layout()
plt.show()


time_id = 300  # try early vs late times

plt.figure(figsize=(6, 4))
plt.imshow(x[:, time_id, :], aspect="auto", cmap="seismic")
plt.colorbar(label="Amplitude")
plt.xlabel("Receivers")
plt.ylabel("Sources")
plt.title(f"Wavefield Snapshot at t={time_id}")
plt.show()


model = np.load("/kaggle/input/waveform-inversion/train_samples/Style_A/model/model1.npy")
print(model.shape)


plt.figure(figsize=(5, 5))
plt.imshow(model[0][0], cmap="viridis")
plt.colorbar(label="Velocity")
plt.title("Velocity Map (Ground Truth)")
plt.axis("off")
plt.show()

