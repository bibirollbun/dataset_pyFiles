import pandas as pd
import matplotlib.pyplot as plt
from tabpfn import TabPFNClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.drop(['id', 'rainfall'], axis=1)
y = train['rainfall']
X_test = test.drop('id', axis=1)

model = TabPFNClassifier(device='cuda')
model.fit(X, y)
predictions = model.predict(X_test)

submission = pd.DataFrame({'id': test['id'], 'rainfall': predictions})
submission.to_csv('submission.csv', index=False)


plt.imshow(plt.imread(r'C:\Users\AE\Desktop\Binary Prediction with a Rainfall Dataset\Screenshot 2025-04-27 153456.png'))


from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier

clf = AutoTabPFNClassifier(max_time=120, device="cuda") # 120 seconds tuning time


clf.fit(X, y)
predictions = clf.predict(X_test)

submission = pd.DataFrame({'id': test['id'], 'rainfall': predictions})
submission.to_csv('submission.csv', index=False)


plt.imshow(plt.imread(r'C:\Users\AE\Desktop\Binary Prediction with a Rainfall Dataset\image.png'))

