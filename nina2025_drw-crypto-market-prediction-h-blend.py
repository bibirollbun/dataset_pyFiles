import pandas as pd

ds ='/kaggle/input/drw-quatro-4'

%run /kaggle/input/drw-quatro-t/_Tida.py


file_short_names = ['0.12_092','0.11_914','0.11_893','0.11_756']  # Lb=0.12_206
file_short_names = ['0.12_252','0.11_914','0.11_893','0.11_756']  # Lb=0.12_327
file_short_names = ['0.12_307','0.12_280','0.12_252','0.12_154']  # Lb=0.12_407
file_short_names = ['0.12_419','0.12_307','0.12_280','0.12_252']  # Lb=0.12_470


df = _Tida ( ds, file_short_names )

df.to_csv('submission.csv', index=False)

display(df)

