import pandas as pd


path = '/kaggle/input/drw-quatro-2/'

#df1  = pd.read_csv(path+ 'Means_5_files - 07-06-2025.csv' ) # Lb=0.10727
#df2  = pd.read_csv(path+ 'Means_5_files - 11-06-2025.csv' ) # Lb=0.11026
#df3  = pd.read_csv(path+ 'Means_4_files (40-30-20-10).csv') # Lb=0.11427
#df4  = pd.read_csv(path+ 'Means_4_files (25-25-25-25).csv') # Lb=0.11471

p = '14-06-2025--3_Top_public_subm__schema_weights_'

#df5  = pd.read_csv(path+p+          '(34-33-33).csv'      ) # Lb=?
#df6  = pd.read_csv(path+p+          '(40-40-20).csv'      ) # Lb=0.11747    v.17
#df7  = pd.read_csv(path+p+          '(45-45-10).csv'      ) # Lb=0.11754    v.16
#df8  = pd.read_csv(path+p+          '(50-45-05).csv'      ) # Lb=0.11755    v.15
#df9  = pd.read_csv(path+p+          '(54-45-01).csv'      ) # Lb=0.11754    v.14
#df10 = pd.read_csv(path+p+          '(74-21-05).csv'      ) # Lb=0.11758    v.18

p = '15-06-2025--4_Top_public_subm__schema_weights_'

#df11 = pd.read_csv(path+p+       '(25-25-25-25).csv'      ) # Lb=?
#df12 = pd.read_csv(path+p+       '(79-15-00-06).csv'      ) # Lb=0.11759    v.23
#df13 = pd.read_csv(path+p+       '(82-13-00-05).csv'      ) # Lb=0.11759    v.22
#df14 = pd.read_csv(path+p+       '(85-09-04-02).csv'      ) # Lb=0.11754    v.21
#df15 = pd.read_csv(path+p+       '(90-07-02-01).csv'      ) # Lb=0.11756    v.19
#df16 = pd.read_csv(path+p+       '(77-17-004-056).csv'    ) # Lb=0.11758    v.24

p = '16-06-2025--3_Top_public_subm__schema_weights_'

#df17 = pd.read_csv(path+p+       '(34-33-33).csv'         ) # Lb=0.11805
#df18 = pd.read_csv(path+p+       '(995-004-001).csv'      ) # Lb=0.11883
#df19 = pd.read_csv(path+p+       '(80-15-05).csv'         ) # Lb=0.11864
#df20 = pd.read_csv(path+p+       '(90-07-03).csv'         ) # Lb=0.11874
#df21 = pd.read_csv(path+p+       '(95-04-01).csv'         ) # Lb=0.11879

p = '17-06-2025--2_public_subm__schema_weights_'

#df22 = pd.read_csv(path+p+       '(95-05).csv'            ) # Lb=0.11865
#df23 = pd.read_csv(path+p+       '(98-02).csv'            ) # Lb=0.11877
#df24 = pd.read_csv(path+p+       '(99-01).csv'            ) # Lb=0.11880
#df25 = pd.read_csv(path+p+       '(995-005).csv'          ) # Lb=0.11882
#df26 = pd.read_csv(path+p+       '(998-002).csv'          ) # Lb=0.11883

p = '18-06-2025--3_Top+AutoGluon_public_subm__schema_weights_'

#df27 = pd.read_csv(path+p+       '(9930-0040-0010-0020).csv'  ) # Lb=
#df28 = pd.read_csv(path+p+       '(9955-0030-0005-0010).csv'  ) # Lb=
#df29 = pd.read_csv(path+p+       '(9955-0025-0005-0015).csv'  ) # Lb=
#df30 = pd.read_csv(path+p+       '(9970-0010-0010-0010).csv'  ) # Lb=
#df31 = pd.read_csv(path+p+       '(9985-0005-0005-0005).csv'  ) # Lb=

p = '19-06-2025--3_public_subm__schema_weights_'

#df32 = pd.read_csv(path+p+       '(998-0015-0005).csv'        ) # Lb=0.11914
#df33 = pd.read_csv(path+p+       '(998-0015-0005)e.csv'       ) # Lb=0.11914 < df32
#df34 = pd.read_csv(path+p+       '(998-0015-0005)o.csv'       ) # Lb=0.11914 < df32, > df33
#df35 = pd.read_csv(path+p+       '(9974-0021-0004).csv'       ) # Lb=0.11914 > df32
#df37 = pd.read_csv(path+p+       '(9800-0126-0074).csv'       ) # Lb=0.11911

df36 = pd.read_csv(path+p+       '(9921-0074-0005).csv'       ) # Lb=0,11914 > df35 - Best

df = df36
df.to_csv("submission.csv", index=False)
df

