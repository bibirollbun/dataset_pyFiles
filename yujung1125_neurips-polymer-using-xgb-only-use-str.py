import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split


train_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")


# Tokenizer (shared for all targets)
charset = sorted(set("".join(train_df['SMILES'])))
char_to_idx = {c: i + 1 for i, c in enumerate(charset)}
vocab_size = len(char_to_idx) + 1

# SMILES to sequence
def smiles_to_seq(smiles):
    return [char_to_idx.get(c, 0) for c in smiles]


def make_model(df, target):
    df = df.dropna(subset=[target])
    X = df['SMILES'].apply(smiles_to_seq)
    X = pad_sequences(X, maxlen=120, padding='post', truncating='post')
    
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor().fit(X_train, y_train)
    return model


models={
    "Tg":make_model(train_df[["SMILES", "Tg"]],'Tg'),
    "FFV":make_model(train_df[["SMILES", "FFV"]],'FFV'),
    "Tc":make_model(train_df[["SMILES", "Tc"]],'Tc'),
    "Density":make_model(train_df[["SMILES", "Density"]],'Density'),
    "Rg":make_model(train_df[["SMILES", "Rg"]],'Rg')
}


# prepare data and predict
test_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")

X_result = test_df['SMILES'].apply(smiles_to_seq)
X_result = pad_sequences(X_result, maxlen=120, padding='post', truncating='post')

Tg_pred = models['Tg'].predict(X_result)
FFV_pred = models['FFV'].predict(X_result)
Tc_pred = models['Tc'].predict(X_result)
Density_pred = models['Density'].predict(X_result)
Rg_pred = models['Rg'].predict(X_result)

# make submission file
test_pd = pd.DataFrame({'id':test_df['id'],'Tg':Tg_pred,'FFV':FFV_pred,'Tc':Tc_pred,'Density':Density_pred,'Rg':Rg_pred})
test_pd.to_csv("submission.csv",index=False)

