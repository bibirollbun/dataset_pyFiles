import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

#test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv", index_col='id')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
print("Shape of dataset:", df.shape)


# Clean NA data
data = df['BeatsPerMinute'].dropna()

# Calculate mean and standard deviation
mean = np.mean(data)
std_dev = np.std(data)

# Create the histogram
plt.figure(figsize=(10, 6))
sns.histplot(data, bins=30, kde=True, stat='density', color='blue', alpha=0.6)

# Create a range of values for the PDF
x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100)
pdf = norm.pdf(x, mean, std_dev)

# Plot the PDF
plt.plot(x, pdf, 'r', linewidth=2)
plt.title('Histograma y Ajuste de DistribuciÃ³n Normal')
plt.xlabel('Valores')
plt.ylabel('Densidad')
plt.grid()
plt.show()


mean_value = data.mean()

# Calculate RMSE (in Normal distribution)
rmse = np.sqrt(np.mean((data - mean_value) ** 2))
print(f'RMSE: {rmse}')


df_sub['BeatsPerMinute'] = mean_value
df_sub.to_csv('test_predictions_Mean.csv', index=False)
df_sub.head()

