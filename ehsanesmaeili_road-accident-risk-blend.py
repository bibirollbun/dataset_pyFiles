import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


sub1 = pd.read_csv('/kaggle/input/predicting-road-accident-risk-vault/submission.csv')
sub2 = pd.read_csv('/kaggle/input/predicting-road-accident-risk-vault/submission (1).csv')
sub3 = pd.read_csv('/kaggle/input/predicting-road-accident-risk-vault/autogluon15.csv')

print("âœ… Loaded submissions:")
print(f"Sub1 shape: {sub1.shape}, Target range: {sub1['accident_risk'].min():.3f} - {sub1['accident_risk'].max():.3f}")
print(f"Sub2 shape: {sub2.shape}, Target range: {sub2['accident_risk'].min():.3f} - {sub2['accident_risk'].max():.3f}")
print(f"Sub3 shape: {sub3.shape}, Target range: {sub3['accident_risk'].min():.3f} - {sub3['accident_risk'].max():.3f}")


oof1 = sub1['accident_risk'].values  # Model 1 predictions
oof2 = sub2['accident_risk'].values  # Model 2 predictions  
oof3 = sub3['accident_risk'].values  # Model 3 predictions

# Stack predictions into matrix
oof_stack = np.column_stack([oof1, oof2, oof3])
print(f"âœ… OOF Stack shape: {oof_stack.shape}")
print(f"âœ… Sample predictions:\n{oof_stack[:5]}")


optimal_weights= [0.5, 0.2, 0.30]  # Manually set weights if desired


# Test predictions for submission
test1 = sub1['accident_risk'].values
test2 = sub2['accident_risk'].values
test3 = sub3['accident_risk'].values

test_stack = np.column_stack([test1, test2, test3])
blended_test = np.dot(test_stack, optimal_weights)

# Create final submission
final_submission = sub1.copy()
final_submission['accident_risk'] = blended_test

# Clip to valid range
final_submission['accident_risk'] = np.clip(final_submission['accident_risk'], 0.001, 0.999)

print("âœ… Blended submission created!")
print(f"Blended range: {final_submission['accident_risk'].min():.3f} - {final_submission['accident_risk'].max():.3f}")


final_submission.head()


# SAVE SUBMISSION
final_submission.to_csv('submission.csv', index=False)
print("ðŸ’¾ Saved: blended_submission.csv")

# QUICK VISUALIZATION
plt.figure(figsize=(12, 4))

# Plot weights
plt.subplot(1, 2, 1)
plt.bar(['Model 1', 'Model 2', 'Model 3'], optimal_weights, color=['red', 'blue', 'green'])
plt.title('Optimal Blend Weights')
plt.ylabel('Weight')
for i, v in enumerate(optimal_weights):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center')

# Plot prediction distributions
plt.subplot(1, 2, 2)
plt.hist(sub1['accident_risk'], bins=50, alpha=0.5, label='Model 1', color='red')
plt.hist(sub2['accident_risk'], bins=50, alpha=0.5, label='Model 2', color='blue')
plt.hist(sub3['accident_risk'], bins=50, alpha=0.5, label='Model 3', color='green')
plt.hist(final_submission['accident_risk'], bins=50, alpha=0.7, label='Blended', color='black', linewidth=2)
plt.title('Prediction Distributions')
plt.legend()

plt.tight_layout()
plt.show()

