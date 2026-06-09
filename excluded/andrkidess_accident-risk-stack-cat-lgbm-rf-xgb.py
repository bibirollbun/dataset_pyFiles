import sys
print("Python exe:", sys.executable)

!{sys.executable} -m pip -q install -U scikit-learn==1.7.2 joblib==1.4.2 threadpoolctl==3.5.0 --no-input

import sklearn, joblib, threadpoolctl
print("sklearn:", sklearn.__version__)
print("joblib :", joblib.__version__)




# 1) Paths (til konkurransedata + artefakt-datasettet ditt)
COMP_DIR = "/kaggle/input/playground-series-s5e10"
ART_DIR  = "/kaggle/input/accident-risk-artifacts"

import os, sys, joblib, pandas as pd

print("ART_DIR exists:", os.path.isdir(ART_DIR))
print("Files in ART_DIR:", [os.path.basename(p) for p in os.listdir(ART_DIR)])



# 2) Definer AddFE_RF slik at joblib kan unpickl’e RF-pipelinen
from sklearn.base import BaseEstimator, TransformerMixin

class AddFE_RF(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        s = X["weather"].astype("string")
        self.freq_map_ = s.value_counts(normalize=True).to_dict()
        return self
    def transform(self, X):
        Z = X.copy()
        Z["is_night"]   = (Z["lighting"].astype("string") == "night").astype(int)
        Z["weather__freq"] = Z["weather"].astype("string").map(self.freq_map_).fillna(0.0)
        return Z

# Viktig for unpickling: sørg for at klassen finnes under __main__
import types
main_mod = sys.modules.get("__main__", types.ModuleType("__main__"))
setattr(main_mod, "AddFE_RF", AddFE_RF)
sys.modules["__main__"] = main_mod




# 3) Les test, last stack og predikér
test = pd.read_csv(os.path.join(COMP_DIR, "test.csv"))
X_test = test.drop(columns=["id"])

stack_path = os.path.join(ART_DIR, "stack_model.joblib")
stack = joblib.load(stack_path)

# Prediksjon (pipelines inne i stacken håndterer preprocessing/FE selv)
pred = stack.predict(X_test)
pred = pred.clip(0.0, 1.0)

sub = pd.DataFrame({"id": test["id"], "accident_risk": pred})
out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)
print("✅ Wrote:", out_path)
sub.head()


