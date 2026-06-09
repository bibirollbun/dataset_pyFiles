import numpy as np
import pandas as pd

from econml.dml import CausalForestDML


# Load data (must be in same folder as this file, which it will be if you simply unzip the assignment).
Z = np.load('Z.npy')
D = np.load('D.npy')
Y = np.load('Y.npy')

print(Z.shape, D.shape, Y.shape)


model = CausalForestDML(discrete_treatment=True)
model.fit(Y, D, X=Z, W=None)


tau_hat = model.effect(Z)

tau_hat_df = pd.DataFrame({
    'Id': list(range(len(Z))),
    'Predicted': tau_hat,
})


# After you make your predictions, you should submit them on the Kaggle webpage for our competition.
# You may also (and I recommend you do it) send your code to me (at tsdj@sam.sdu.dk).
# Then I can provide feecback if you'd like (so ask away!).

# Below is a small check that your output has the right type and shape
assert isinstance(tau_hat_df, pd.DataFrame)
assert all(tau_hat_df.columns == ['Id', 'Predicted'])
assert len(tau_hat_df) == 10000

# If you pass the checks, the file is saved.
tau_hat_df.to_csv('tau_hat.csv', index=False)

