import pandas as pd


path = '/kaggle/input/drw-quatro-3/submission__'

df = pd.read_csv(path + 'first_glance_wts(74 13 08 05) 50x50.csv')

df.to_csv("submission__0_12_203.csv", index=False)  # Lb=0.12_203

df


 %run /kaggle/input/drw-quatro-tida/_Tida.py


path = '/kaggle/input/drw-quatro-4/submission__'
                                                     # wts=[74 13 08 05] 50x50  # Lb=0.12_203
FiN = ['0.12_092','0.11_914','0.11_893','0.11_756']  # wts=[77 11 07 05] 50x50  # Lb=0.12_196
FiN = ['0.12_092','0.11_914','0.11_893','0.11_756']  # wts=[70 13 10 07] 50x50  # Lb=0.12_206
FiN = ['0.12_092','0.11_914','0.11_893','0.11_756']  # wts=[70 12 11 07] 50x50  # Lb=0.12_206

df = _Tida ( path, FiN )

df.to_csv('submission.csv', index=False)  # Lb=?

display(df)

