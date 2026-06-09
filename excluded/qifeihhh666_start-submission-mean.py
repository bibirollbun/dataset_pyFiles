import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
print("ok")


df_transfer_plate = pd.read_csv('/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/transfer_plate.csv')

# data: create a DataFrame with Prediction columns
data = {
    'Glucose (g/L)': df_transfer_plate['Glucose (g/L)'], 
    'Sodium Acetate (g/L)': df_transfer_plate['Sodium Acetate (g/L)'],
    'Magnesium Acetate (g/L)': df_transfer_plate['Magnesium Acetate (g/L)']
}
df_train_y = pd.DataFrame(data)

# Calculate the mean of each column
mean_values = df_train_y.mean()

# Print the mean values
print("Mean values of each column:")
print(mean_values)

# Plot the mean values as a bar chart
plt.figure(figsize=(8, 5))
mean_values.plot(kind='bar', color=['skyblue', 'orange', 'lightgreen'])

# Add title and labels
plt.title('Mean Values of Prediction Columns', fontsize=14)
plt.xlabel('Column Name', fontsize=12)
plt.ylabel('Mean Value', fontsize=12)
plt.xticks(rotation=0) 

# Display the mean values on the bar chart
for i, v in enumerate(mean_values):
    plt.text(i, v , f'{v:.2f}', ha='center', fontsize=10)  # Adjust position and format

# Show the plot
plt.tight_layout()
plt.show()


df = pd.read_csv('/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/sample_submission.csv')
df['Glucose']=np.full(96, df_train_y['Glucose (g/L)'].mean())  
df['Sodium Acetate']=np.full(96, df_train_y['Sodium Acetate (g/L)'].mean()) 
df['Magnesium Sulfate']=np.full(96, df_train_y['Magnesium Acetate (g/L)'].mean()) 

df.to_csv('submission.csv',index=False)
print(df.head(5))
print('sucessfully save!')
print("ok")




