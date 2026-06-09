!unzip /kaggle/input/santander-product-recommendation/train_ver2.csv.zip


import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
# from xgboost import XGBClassifier 
from sklearn.utils.class_weight import compute_class_weight


# load product recommendation database.

df_full = pd.read_csv("/kaggle/working/train_ver2.csv")


df_full.head(5)


# length of the database.

len(df_full['ncodpers'].unique()), len(df_full)


# number of customer_ids with loans.

len(df_full[df_full['ind_pres_fin_ult1'] == 1]['ncodpers'].unique()), len(df_full[df_full['ind_pres_fin_ult1'] == 1]['ncodpers'].unique())/len(df_full['ncodpers'].unique())


# pre-process the dataset.

# (a) combine the following columns: 1. Accounts - Savings + Current + Payroll + Junior + Particular + Mas Particular + Direct Debit + e-account
# 2. Investments - Derivatives Accounts + Funds + Securities 
# 3. Loans - Loans + Mortgage + Home Account 
# 4. Rename Credit Card + Deposits 
df_full['Accounts'] = df_full['ind_ahor_fin_ult1'] | df_full['ind_cco_fin_ult1'] | df_full['ind_cno_fin_ult1'] | df_full['ind_ctju_fin_ult1'] | df_full['ind_ctma_fin_ult1']| df_full['ind_ctop_fin_ult1'] | df_full['ind_ctpp_fin_ult1'] | df_full['ind_ecue_fin_ult1'] | df_full['ind_recibo_ult1']
df_full['Investments'] = df_full['ind_cder_fin_ult1'] | df_full['ind_valo_fin_ult1'] | df_full['ind_fond_fin_ult1']
df_full['Credit Cards'] = df_full['ind_tjcr_fin_ult1']
df_full['Loans'] = df_full['ind_hip_fin_ult1'] | df_full['ind_pres_fin_ult1'] | df_full['ind_viv_fin_ult1']
df_full['Short Term Deposit'] = df_full['ind_deco_fin_ult1']
df_full['Medium Term Deposit'] = df_full['ind_deme_fin_ult1']
df_full['Long Term Deposit'] = df_full['ind_dela_fin_ult1']
df_full['segmento'].replace(to_replace=['02 - PARTICULARES', '03 - UNIVERSITARIO', '01 - TOP'], value=[0, 1, 2], inplace=True)
df_full = df_full[df_full['indfall'] == 'N']
df_full.rename(columns={"ind_empleado":"Employment", "indresi":"Residence", "indext":"Foreigner", "segmento":"Segment", "fecha_dato":"date", "ncodpers":"Customer code", "sexo":"Gender", "renta":"Income", "ind_actividad_cliente":"Active", "ind_plan_fin_ult1":"Pension"}, inplace=True)
df_full.drop(columns=['fecha_alta','pais_residencia','antiguedad','indrel','ult_fec_cli_1t','indrel_1mes','tiprel_1mes','conyuemp','canal_entrada','ind_nuevo',
                     'cod_prov','nomprov','ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1',
                     'ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1',
                     'ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1'
                     ,'tipodom','indfall'], inplace=True)
df_full.head(5)


# make the pivot.

pivot_df = df_full.sort_values('date',ascending=False).groupby("Customer code").agg({'Employment':'last','Gender':'last','age':'last','Residence':'last','Foreigner':'last','Active':'last',
                                                                                     'Pension':'last','Income':'last','Segment':'last','Accounts':'any','Investments':'any',
                                                                                     'Credit Cards':'any','Loans':'any','Short Term Deposit':'any','Medium Term Deposit':'any',
                                                                                     'Long Term Deposit':'any'}).reset_index()
pivot_df[['Pension','Accounts','Investments','Credit Cards','Loans','Short Term Deposit','Medium Term Deposit','Long Term Deposit']] = pivot_df[['Pension','Accounts','Investments','Credit Cards','Loans','Short Term Deposit','Medium Term Deposit','Long Term Deposit']].astype('float32')
pivot_df = pivot_df.fillna(value={'Segment':0.0, 'Income':pivot_df['Income'].median()})
scaler = StandardScaler()
age_scaler = MinMaxScaler()
scaler.fit(pivot_df['Income'].to_numpy().reshape((-1,1)))
age_scaler.fit(pivot_df['age'].to_numpy().reshape((-1,1)))
pivot_df.drop(columns=['Accounts','Investments','Credit Cards','Loans','Short Term Deposit','Medium Term Deposit','Long Term Deposit','Income','Gender']).to_csv("pivot_info.csv")
pivot_df


pivot_save_df = pd.DataFrame(pivot_df, columns=pivot_df.columns)
pivot_save_df = pd.get_dummies(pivot_save_df, columns=['Employment','Gender','Residence','Foreigner','Segment'], dtype='float32')
pivot_save_df['Income'] = scaler.transform(pivot_save_df['Income'].to_numpy().reshape((-1,1)))
pivot_save_df['age'] = age_scaler.transform(pivot_save_df['age'].to_numpy().reshape((-1,1)))
pivot_save_df.to_csv("pivot.csv")


# preprcoessing the full dataframe using value filling from the pivot.

df_full[['Pension','Accounts','Investments','Credit Cards','Loans','Short Term Deposit','Medium Term Deposit','Long Term Deposit']] = df_full[['Pension','Accounts','Investments','Credit Cards','Loans','Short Term Deposit','Medium Term Deposit','Long Term Deposit']].astype('float32')
df_full = df_full.fillna(value={'Segment':0.0, 'Income':pivot_df['Income'].median()})
df_full = pd.get_dummies(df_full, columns=['Employment','Gender','Residence','Foreigner','Segment'], dtype='float32')
df_full['Income'] = scaler.transform(df_full['Income'].to_numpy().reshape((-1,1)))
df_full['age'] = age_scaler.transform(df_full['age'].to_numpy().reshape((-1,1)))
df_full


# training data for the recommendation model. 

Y_df = df_full[['Accounts','Investments','Credit Cards','Loans','Short Term Deposit','Medium Term Deposit','Long Term Deposit']]
X_df = df_full.drop(columns=['Accounts','Investments','Credit Cards','Loans','Short Term Deposit','Medium Term Deposit','Long Term Deposit'])
X_df


# viewing Y_df.

Y_df


# unique user ids - for IntegerLookup vocabulary. 

unique_user_ids = pivot_df['Customer code'].to_list()


# load loans dataset. 

df_loans = pd.read_csv("/kaggle/input/loan-approval-dataset/Loan Dataset.csv")
df_loans 


# unqiue values for loan purpose - 4 categories of loans (Home, Auto, Education and Personal)

df_loans['Loan_Purpose'].unique()


# columns to keep for the model - removing attributes because we can't reliably assume to possess these as part of properiety data. 

df_loans = df_loans.drop(columns=['City/Town','Monthly_Expenses','Existing_Loans','Total_Existing_Loan_Amount','Outstanding_Debt','Loan_History','Loan_Amount_Requested','Loan_Term',
                                 'Interest_Rate','Loan_Type','Co-Applicant','Default_Risk','Applicant_ID'])


# prepare dataset by scaling, one-hot encoding and separate into 4 datasets - one each per purpose of loan.

df_loans = pd.get_dummies(df_loans, columns=['Gender','Marital_Status','Dependents','Education','Employment_Status','Occupation_Type','Residential_Status'], dtype='float32')
df_home_loans = df_loans[df_loans['Loan_Purpose'] == 'Home'].drop(columns="Loan_Purpose")
df_auto_loans = df_loans[df_loans['Loan_Purpose'] == 'Vehicle'].drop(columns="Loan_Purpose")
df_personal_loans = df_loans[df_loans['Loan_Purpose'] == 'Personal'].drop(columns="Loan_Purpose")
df_edu_loans = df_loans[df_loans['Loan_Purpose'] == 'Education'].drop(columns="Loan_Purpose")
df_loans = df_loans.drop(columns='Loan_Purpose')
loans_scaler = MinMaxScaler()
df_loans = pd.DataFrame(loans_scaler.fit_transform(df_loans), columns=df_loans.columns)
df_home_loans = pd.DataFrame(loans_scaler.transform(df_home_loans), columns=df_loans.columns)
df_auto_loans = pd.DataFrame(loans_scaler.transform(df_auto_loans), columns=df_loans.columns)
df_personal_loans = pd.DataFrame(loans_scaler.transform(df_personal_loans), columns=df_loans.columns)
df_edu_loans = pd.DataFrame(loans_scaler.transform(df_edu_loans), columns=df_loans.columns)
df_edu_loans


# loan models.

# cfr_t_data = [df_home_loans, df_auto_loans, df_personal_loans, df_edu_loans]
# classifiers = []
# for cfr_data in cfr_t_data:
#     cfr = XGBClassifier()
#     cfr.fit(cfr_data.drop(columns='Loan_Approval_Status'), cfr_data['Loan_Approval_Status'])
#     print(cfr.score(cfr_data.drop(columns='Loan_Approval_Status'), cfr_data['Loan_Approval_Status']))
#     classifiers.append(cfr)


# for i in range(4):
#     model = classifiers[i]
#     model.save_model('hackathon_xgboost_model_'+ str(i) +'.model')


# recommendation model. 

user_id_input = tf.keras.Input(shape=(1,), name="user_id")
user_features_input = tf.keras.Input(shape=(18,), name="user_features")
user_lookup = tf.keras.layers.IntegerLookup(vocabulary=unique_user_ids, mask_token=None)(user_id_input)
user_embeddings = tf.keras.layers.Embedding(len(unique_user_ids)+1, 64)(user_lookup)
user_embeddings = tf.keras.layers.Reshape((64,))(user_embeddings)
concat_features = tf.keras.layers.concatenate([user_features_input, user_embeddings])
layer_1 = tf.keras.layers.Dense(256, activation="relu")(concat_features)
layer_2 = tf.keras.layers.Dense(512, activation="relu")(layer_1)
layer_3 = tf.keras.layers.Dense(64, activation="relu")(layer_2)
output_layer = tf.keras.layers.Dense(7, activation="sigmoid")(layer_3)
model = tf.keras.Model(inputs=[user_id_input, user_features_input], outputs=output_layer)


model.summary()


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False))


y_integers = []
for arr in Y_df.to_numpy():
    for i in range(len(arr)):
        if arr[i] == 1.0:
            y_integers.append(i)
class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
d_class_weights = dict(enumerate(class_weights))
d_class_weights


model.fit([X_df['Customer code'].to_numpy(), X_df.drop(columns=['Customer code','date']).to_numpy()], Y_df.to_numpy(), epochs=1, class_weight=d_class_weights)


model.save("hackathon_model.keras")


!pip install sdv


pivot_df


df_loans = pd.read_csv("/kaggle/input/loan-approval-dataset/Loan Dataset.csv")
df_loans


df_loans = df_loans.drop(columns=['City/Town','Monthly_Expenses','Existing_Loans','Total_Existing_Loan_Amount','Outstanding_Debt','Loan_History','Loan_Amount_Requested','Loan_Term',
                                 'Interest_Rate','Loan_Type','Co-Applicant','Default_Risk'])
df_loans


import sdv

metadata = sdv.metadata.Metadata.detect_from_dataframe(data=df_loans)
synthesizer = sdv.single_table.GaussianCopulaSynthesizer(metadata)
synthesizer.fit(df_loans)
synthetic_data = synthesizer.sample(num_rows=100)
synthetic_data


synthesizer.get_learned_distributions()


loan_only_pivot_df = pivot_df[pivot_df['Loans'] == 1.0]
loan_only_pivot_df


pivot_df['Gender'] = pivot_df['Gender'].replace(to_replace=['V','H'], value=['Male','Female'])
loan_only_pivot_df = pivot_df
pivot_loan_info_df = pd.DataFrame(columns=df_loans.columns)
print((df_loans['Annual_Income'].min(), df_loans['Annual_Income'].max()))
temp_scaler = MinMaxScaler(feature_range=(df_loans['Annual_Income'].min(), df_loans['Annual_Income'].max()))
loan_only_pivot_df['Income'] = temp_scaler.fit_transform(loan_only_pivot_df['Income'].to_numpy().reshape(-1,1))
loan_only_pivot_df


temp_df = synthesizer.sample(num_rows=len(loan_only_pivot_df))
temp_df['Applicant_ID'] = loan_only_pivot_df['Customer code']
temp_df['Gender'] = loan_only_pivot_df['Gender']
temp_df['Annual_Income'] = loan_only_pivot_df['Income']
temp_df['Age'] = loan_only_pivot_df['age']
temp_df


temp_df.to_csv("pivot_loan_info.csv")
pivot_df.to_csv("pivot.csv")

