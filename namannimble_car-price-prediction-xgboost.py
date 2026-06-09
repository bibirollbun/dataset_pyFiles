import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns   
import os
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import  StandardScaler , MinMaxScaler , LabelEncoder , OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_log_error , mean_absolute_error , mean_absolute_percentage_error , mean_squared_error
from sklearn.model_selection import train_test_split , RandomizedSearchCV , KFold , GridSearchCV

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import  RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import  XGBRegressor

from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer



train = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
test = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')


train.head()


test.head()


print(f"training dataset shape = {train.shape}")
print("="*30)
print(f"test dataset shape = {test.shape}")


print(train.columns)
print("="*60)
print(test.columns)


train.info()


train.isnull().sum()


train.duplicated().sum()


train.describe()


train.columns


train["date"]


train["plate"]



# provided in data 
REGION_CODES = {
    "Republic of Adygea": ["01"],
    "Altai Republic": ["04"],
    "Republic of Bashkortostan": ["02", "102", "702"],
    "Republic of Buryatia": ["03"],
    "Republic of Dagestan": ["05"],
    "Donetsk People's Republic": ["80", "180"],
    "Republic of Ingushetia": ["06"],
    "Kabardino-Balkarian Republic": ["07"],
    "Republic of Kalmykia": ["08"],
    "Karachay-Cherkess Republic": ["09"],
    "Republic of Karelia": ["10"],
    "Komi Republic": ["11"],
    "Republic of Crimea": ["82"],
    "Luhansk People's Republic": ["81", "181"],
    "Republic of Mari El": ["12"],
    "Republic of Mordovia": ["13", "113"],
    "Sakha Republic": ["14"],
    "Republic of North Ossetia": ["15"],
    "Republic of Tatarstan": ["16", "116", "716"],
    "Republic of Tyva (Tuva)": ["17"],
    "Udmurt Republic": ["18"],
    "Republic of Khakassia": ["19"],
    "Chechen Republic": ["20", "95"],
    "Chuvash Republic": ["21", "121"],
    "Altai Krai": ["22", "122"],
    "Zabaykalsky Krai": ["75"],
    "Kamchatka Krai": ["41"],
    "Krasnodar Krai": ["23", "93", "123", "193", "323"],
    "Krasnoyarsk Krai": ["24", "84", "88", "124"],
    "Perm Krai": ["59", "81", "159"],
    "Primorsky Krai": ["25", "125"],
    "Stavropol Krai": ["26", "126"],
    "Khabarovsk Krai": ["27"],
    "Amur Oblast": ["28"],
    "Arkhangelsk Oblast": ["29"],
    "Astrakhan Oblast": ["30", "130"],
    "Belgorod Oblast": ["31"],
    "Bryansk Oblast": ["32"],
    "Vladimir Oblast": ["33"],
    "Volgograd Oblast": ["34", "134"],
    "Vologda Oblast": ["35"],
    "Voronezh Oblast": ["36", "136"],
    "Zaporizhzhia Oblast": ["85", "185"],
    "Ivanovo Oblast": ["37"],
    "Irkutsk Oblast": ["38", "85", "138"],
    "Kaliningrad Oblast": ["39", "91"],
    "Kaluga Oblast": ["40"],
    "Kemerovo Oblast": ["42", "142"],
    "Kirov Oblast": ["43"],
    "Kostroma Oblast": ["44"],
    "Kurgan Oblast": ["45"],
    "Kursk Oblast": ["46"],
    "Leningrad Oblast": ["47", "147"],
    "Lipetsk Oblast": ["48"],
    "Magadan Oblast": ["49"],
    "Moscow Oblast": ["50", "90", "150", "190", "250", "550", "750", "790"],
    "Murmansk Oblast": ["51"],
    "Nizhny Novgorod Oblast": ["52", "152", "252"],
    "Novgorod Oblast": ["53"],
    "Novosibirsk Oblast": ["54", "154", "754"],
    "Omsk Oblast": ["55", "155"],
    "Orenburg Oblast": ["56", "156"],
    "Oryol Oblast": ["57"],
    "Penza Oblast": ["58", "158"],
    "Pskov Oblast": ["60"],
    "Rostov Oblast": ["61", "161", "761"],
    "Ryazan Oblast": ["62"],
    "Samara Oblast": ["63", "163", "763"],
    "Saratov Oblast": ["64", "164"],
    "Sakhalin Oblast": ["65"],
    "Sverdlovsk Oblast": ["66", "96", "196"],
    "Smolensk Oblast": ["67"],
    "Tambov Oblast": ["68"],
    "Tver Oblast": ["69"],
    "Tomsk Oblast": ["70"],
    "Tula Oblast": ["71"],
    "Tyumen Oblast": ["72", "172"],
    "Ulyanovsk Oblast": ["73", "173"],
    "Kherson Oblast": ["84", "184"],
    "Chelyabinsk Oblast": ["74", "174", "774"],
    "Yaroslavl Oblast": ["76"],
    "Moscow": ["77", "97", "99", "177", "197", "199", "777", "797", "799", "977"],
    "Saint Petersburg": ["78", "98", "178", "198"],
    "Sevastopol": ["92"],
    "Jewish Autonomous Oblast": ["79"],
    "Nenets Autonomous Okrug": ["83"],
    "Khanty-Mansi Autonomous Okrug": ["86", "186"],
    "Chukotka Autonomous Okrug": ["87"],
    "Yamalo-Nenets Autonomous Okrug": ["89"],
    "Baikonur": ["94"],
    "Occupational Administration of Kharkiv Oblast": ["188"],
}

# ((letters, numbers range (from, to), region code), is it forbidden to buy (bool), do they have an advantage on the road (bool), level of significance (author's opinion))
GOVERNMENT_CODES = {
    # Moscow
    ("AMP", (0, 999), "97"): ("Government of Russia", 1, 1, 10),
    ("AMP", (0, 999), "77"): ("Partially Government of Russia", 0, 1, 8),
    ("EKX", (0, 999), "77"): ("Partially Federal Protective Service (Federal Protective Service)", 0, 1, 6),
    ("EKX", (0, 999), "97"): ("Partially Federal Protective Service (Federal Protective Service)", 0, 1, 6),
    ("EKX", (0, 999), "99"): ("Partially Federal Protective Service (Federal Protective Service)", 0, 1, 6),
    ("KKX", (0, 999), "77"): ("Partially used on vehicles of Ministry of Security/Federal Counterintelligence Service /Federal Security Service of Russia", 0, 0, 1),
    ("CAC", (500, 999), "77"): ("Former officially 'open' plates of Ministry of Security/Federal Counterintelligence Service /Federal Security Service of Russia", 0, 0, 1),
    ("CAC", (500, 999), "77"): ("Former officially 'open' plates of Ministry of Security/Federal Counterintelligence Service /Federal Security Service of Russia", 0, 0, 1),
    ("AOO", (0, 999), "77"): ("Partially Presidential Administrative Directorate plates", 0, 1, 6),
    ("BOO", (0, 999), "77"): ("Partially Presidential Administrative Directorate plates", 0, 1, 6),
    ("MOO", (0, 999), "77"): ("Partially Presidential Administrative Directorate plates", 0, 1, 6),
    ("COO", (0, 999), "77"): ("Partially Administrative Directorate, Federation Council plates", 0, 1, 6),
    ("AMM", (0, 999), "99"): ("Partially plates of Moscow City Duma deputies, police", 0, 1, 4),
    ("CCC", (0, 999), "77"): ("Partially Central Special Communication, Courier Service, Ministry of Communications", 0, 1, 3),
    ("CCC", (0, 999), "99"): ("Partially Tax Police, Customs, Special Communications", 0, 1, 3),
    ("CCC", (0, 999), "97"): ("Partially Central Special Communication, Courier Service, Ministry of Communications", 0, 1, 3),
    ("KKK", (0, 999), "99"): ("Initially belonged to Courier Service, now used among private individuals", 0, 0, 1),
    ("OOO", (0, 999), "77"): ("Initially intended for Federal Security Service", 0, 0, 1),
    ("KMM", (0, 999), "77"): ("Partially Fire Department plates", 0, 1, 3),
    ("MMP", (300, 320), "77"): ("Partially Federal Security Service plates", 0, 1, 4),
    ("MMP", (0, 299), "77"): ("Partially Government of Russia, Federal Security Service, banks, and private individuals with connections in the traffic police", 0, 1, 2),
    ("MMP", (321, 999), "77"): ("Partially Government of Russia, Federal Security Service, banks, and private individuals with connections in the traffic police", 0, 1, 2),
    ("PMP", (0, 999), "77"): ("Partially Ministry of Justice plates", 0, 1, 3),
    ("AMO", (0, 999), "77"): ("Partially Moscow City Hall plates", 0, 1, 5),
    ("KOO", (0, 999), "77"): ("Partially Constitutional Court plates", 0, 1, 3),
    ("EPE", (0, 999), "77"): ("Partially State Duma plates", 0, 1, 3),
    ("AAA", (0, 999), "77"): ("Partially Administration of the President plates", 0, 1, 6),
    ("KMP", (0, 999), "77"): ("Partially Government of Russia plates", 0, 1, 3),
    ("TMP", (0, 999), "77"): ("Partially Government of Russia plates, as well as private individuals with connections in the traffic police", 0, 1, 2),
    ("YMP", (0, 999), "77"): ("Partially Government of Russia plates, as well as private individuals with connections in the traffic police", 0, 1, 2),
    ("XXX", (0, 999), "77"): ("Private individuals with connections in the traffic police", 0, 1, 2),
    ("YYY", (0, 999), "77"): ("Private individuals with connections in the traffic police", 0, 1, 2),
    ("XKX", (0, 999), "77"): ("Partially Federal Security Service and Federal Protective Service plates", 0, 1, 2),
    ("OMP", (0, 999), "77"): ("Partially Government of Russia, banks, and private individuals with connections in the traffic police", 0, 1, 2),
    ("EEE", (0, 999), "77"): ("Private individuals with connections in the traffic police", 0, 1, 2),

    # Moscow Oblast
    ("AMO", (0, 999), "50"): ("Partially various government agencies (administration, ambulance, traffic police, etc.)", 0, 1, 3),
    ("BMO", (0, 999), "50"): ("Partially various government agencies (administration, ambulance, traffic police, etc.)", 0, 1, 3),
    ("KMO", (0, 999), "50"): ("Partially various government agencies (administration, ambulance, traffic police, etc.)", 0, 1, 3),
    ("CMO", (0, 999), "50"): ("Partially various government agencies (administration, ambulance, traffic police, etc.)", 0, 1, 3),
    ("OMO", (0, 999), "50"): ("Partially various government agencies (administration, ambulance, traffic police, etc.)", 0, 1, 3),
    ("MMO", (0, 999), "50"): ("Partially various government agencies (administration, ambulance, traffic police, etc.)", 0, 1, 3),
    ("TMO", (0, 999), "50"): ("Partially various government agencies (administration, ambulance, traffic police, etc.)", 0, 1, 3),
    ("HMO", (0, 999), "50"): ("Partially various government agencies (administration, ambulance, traffic police, etc.)", 0, 1, 3),
    ("YMO", (0, 999), "50"): ("Partially various government agencies (administration, ambulance, traffic police, etc.)", 0, 1, 3),
    ("XMO", (0, 999), "50"): ("Partially various government agencies (administration, ambulance, traffic police, etc.)", 0, 1, 3),
    ("AMM", (0, 999), "50"): ("Partially plates of the regional administration", 0, 1, 5),
    ("AMM", (0, 999), "90"): ("Partially plates of the regional administration", 0, 1, 5),
    ("MMM", (0, 999), "50"): ("Partially plates of law enforcement in the region (prosecutor's office, EMERCOM, traffic police, etc.)", 0, 1, 5),
    ("MMM", (0, 999), "90"): ("Partially plates of law enforcement in the region (prosecutor's office, EMERCOM, traffic police, etc.)", 0, 1, 5),

    # Saint Petersburg
    ("OBO", (0, 999), "78"): ("Partially Departmental Security Service plates", 0, 1, 4),
    ("OBO", (0, 999), "98"): ("Partially Departmental Security Service plates", 0, 1, 4),
    ("OTT", (0, 999), "78"): ("Partially former traffic police plates (now replaced by 98)", 0, 0, 1),
    ("OTT", (0, 999), "98"): ("Partially traffic police plates", 0, 1, 4),
    ("OMM", (0, 999), "78"): ("Partially city district police plates", 0, 1, 3),
    ("OMM", (0, 999), "98"): ("Partially city district police plates", 0, 1, 3),
    ("OOM", (0, 999), "78"): ("Partially plates of the Main Department of Internal Affairs", 0, 1, 3),
    ("OOM", (0, 999), "98"): ("Partially plates of the Main Department of Internal Affairs", 0, 1, 3),
    ("OKO", (0, 100), "78"): ("Partially former plates of the prosecutor's office and judicial department (now replaced by 98)", 0, 0, 1),
    ("OKO", (0, 100), "98"): ("Partially plates of the prosecutor's office and judicial department", 0, 1, 3),
    ("OKO", (700, 999), "78"): ("Partially former Federal Security Service plates (now replaced by 98)", 0, 0, 1),
    ("OKO", (700, 999), "98"): ("Partially Federal Security Service plates", 0, 1, 3),
    ("OPP", (0, 999), "78"): ("Partially former plates of the Main Department of Internal Affairs (now replaced by 98)", 0, 0, 1),
    ("OPP", (0, 999), "98"): ("Partially plates of the Main Department of Internal Affairs", 0, 1, 3),
    ("OOH", (0, 999), "78"): ("Partially Federal Drug Control Service and Federal Tax Service plates", 0, 1, 3),
    ("OOH", (0, 999), "98"): ("Partially Federal Drug Control Service and Federal Tax Service plates", 0, 1, 3),
    ("OAO", (0, 999), "78"): ("Partially plates of the city and regional administration", 0, 1, 5),
    ("OAO", (0, 999), "98"): ("Partially plates of the city and regional administration", 0, 1, 5),
    ("AAA", (0, 100), "78"): ("Partially plates of the city and regional administration", 0, 1, 6),
    ("AAA", (0, 100), "98"): ("Partially plates of the city and regional administration", 0, 1, 6),
    ("OOO", (0, 899), "78"): ("Commercial plates", 0, 0, 2),
    ("OOO", (0, 899), "98"): ("Commercial plates", 0, 0, 2),
    ("OOO", (900, 999), "78"): ("Partially Federal Protective Service plates", 0, 1, 3),
    ("OOO", (900, 999), "98"): ("Partially Federal Protective Service plates", 0, 1, 3),
    ("OKC", (0, 999), "98"): ("Partially Constitutional Court of the Russian Federation plates", 0, 1, 3),
    ("OOC", (0, 999), "78"): ("Partially plates of heads of enterprises and organizations", 0, 0, 2),
    ("OOC", (0, 999), "98"): ("Partially plates of heads of enterprises and organizations", 0, 0, 2),
    ("MMM", (0, 999), "78"): ("Commercial plates", 0, 0, 2),
    ("MMM", (0, 999), "98"): ("Commercial plates", 0, 0, 2),

    # Altai Republic
    ("XXX", (0, 999), "04"): ("Widespread 'special' plates", 0, 0, 2),
    ("TTT", (0, 999), "04"): ("Rare 'special' plates", 0, 0, 2),
    ("PPP", (0, 999), "04"): ("Partially prosecutor's office of the republic", 0, 1, 3),
    ("PPA", (0, 999), "04"): ("Partially prosecutor's office of the republic", 0, 1, 3),
    ("MPA", (0, 999), "04"): ("Partially Ministry of Internal Affairs of the republic", 0, 1, 3),
    ("OOO", (0, 999), "04"): ("Partially plates of the government of the republic", 0, 1, 5),
    ("HHH", (0, 999), "04"): ("Partially the republic's tax service plates", 0, 1, 3),
    ("CCC", (0, 999), "04"): ("Partially plates belonging to the republic's judges", 0, 1, 3),

    # Republic of Bashkortostan
    ("PKC", (0, 999), "02"): ("Partially State Assembly (Kurultai) plates", 0, 1, 5),
    ("KKC", (0, 999), "02"): ("Partially State Assembly (Kurultai) plates", 0, 1, 5),
    ("OOO", (0, 999), "02"): ("Partially plates of leaders of large enterprises and ministries", 0, 1, 3),
    ("AAA", (0, 999), "02"): ("Partially plates of the republic's government", 0, 1, 5),

    # Republic of Karelia
    ("TTT", (0, 999), "10"): ("Partially government of the republic and Federal Security Service plates", 0, 1, 5),
    ("HHH", (0, 999), "10"): ("Partially plates of city and district administrations of the republic", 0, 1, 4),
    ("MMM", (0, 999), "10"): ("Partially plates of the Ministry of Internal Affairs of the republic", 0, 1, 3),
    ("EMP", (0, 999), "10"): ("Partially plates of the Ministry of Internal Affairs of the republic", 0, 1, 3),
    ("CCC", (0, 999), "10"): ("Partially plates of the prosecutor's office and judges' vehicles", 0, 1, 3),

    # Komi Republic
    ("TTT", (0, 999), "11"): ("Partially government of the republic and Federal Security Service plates", 0, 1, 5),
    ("OOO", (0, 999), "11"): ("Widespread semi-special plates, leaders of large industrial companies", 0, 1, 3),

    # Sakha Republic
    ("PPP", (0, 999), "14"): ("Partially plates of the republic's prosecutor's office", 0, 1, 3),
    ("AAA", (0, 999), "14"): ("Motor pool of the President, Government, Parliament of the republic, as well as heads of state enterprises", 0, 1, 5),

    # Republic of Tatarstan
    ("OAA", (0, 999), "16"): ("Partially plates of heads of district administrations", 0, 1, 5),
    ("OAA", (0, 999), "116"): ("Partially plates of heads of district administrations", 0, 1, 5),
    ("OAA", (0, 999), "716"): ("Partially plates of heads of district administrations", 0, 1, 5),

    # Krasnodar Krai
    ("PPP", (0, 999), "23"): ("Partially plates of the Krai and city administrations", 0, 1, 5),
    ("HHH", (0, 999), "23"): ("Partially plates of the tax authorities", 0, 1, 3),
    ("OOO", (0, 999), "23"): ("Partially plates of the Krai and city administrations", 0, 1, 5),
    ("KKK", (0, 999), "23"): ("Partially plates of the Krai administration", 0, 1, 5),

    # Krasnoyarsk Krai
    ("KPK", (0, 999), "24"): ("Partially plates of the Krai administration", 0, 1, 5),
    ("OOO", (0, 999), "24"): ("Partially Federal Security Service plates of the Krai", 0, 1, 3),
    ("MKK", (0, 999), "24"): ("Partially former plates of the Ministry of Internal Affairs of the Krai", 0, 0, 1),

    # Primorsky Krai
    ("BOO", (0, 999), "25"): ("Partially military plates", 0, 1, 3),
    ("BOO", (0, 999), "125"): ("Partially city services plates in Vladivostok and districts", 0, 1, 2),
    ("AAA", (0, 999), "25"): ("Issued first in Vladivostok", 0, 0, 2),
    ("AAA", (0, 999), "125"): ("One of the most 'special' series, prosecutor's office", 1, 1, 5),
    ("HHH", (0, 999), "25"): ("Partially plates of the administration and vehicles of City Duma deputies", 0, 1, 3),
    ("MMM", (0, 999), "25"): ("Partially plates of the deputies of the Krai Legislative Assembly", 0, 1, 3),
    ("CCC", (0, 999), "25"): ("Partially plates of the Krai administration", 0, 1, 5),
    ("XXX", (0, 999), "25"): ("Partially plates of the prosecutor's office and the Department of Internal Affairs", 0, 1, 2),
    ("OOO", (0, 999), "25"): ("Partially former plates of the Krai administration (during Governor Evgeny Nazdratenko)", 0, 0, 1),
    ("TTT", (0, 999), "25"): ("Partially former plates of the Vladivostok administration and federal agencies in the Krai (during Mayor Yuri Kopylov)", 0, 0, 1),
    ("MBK", (0, 999), "25"): ("Partially plates for employees of the Department of Internal Affairs", 0, 1, 3),
    ("MBK", (0, 999), "125"): ("Partially plates for employees of the Department of Internal Affairs", 0, 1, 3),
    ("MOO", (0, 999), "25"): ("Partially plates for Krai agencies of the Department of Internal Affairs, EMERCOM, firefighters, etc.", 0, 1, 2),
    ("MOO", (0, 999), "125"): ("Partially plates for Krai agencies of the Department of Internal Affairs, EMERCOM, firefighters, etc.", 0, 1, 2),
    ("HOO", (0, 999), "25"): ("Partially plates of the Department of Internal Affairs, traffic police in the southeastern region of the Krai (Nakhodka)", 0, 1, 3),
    ("HOO", (0, 999), "125"): ("Partially plates of the Department of Internal Affairs, traffic police in the southeastern region of the Krai (Nakhodka)", 0, 1, 3),
    ("YOO", (0, 999), "25"): ("Partially plates of the Department of Internal Affairs, traffic police in the central region of the Krai (Ussuriysk)", 0, 1, 3),
    ("YOO", (0, 999), "125"): ("Partially plates of the Department of Internal Affairs, traffic police in the central region of the Krai (Ussuriysk)", 0, 1, 3),
    ("COO", (0, 999), "25"): ("Partially plates of the Department of Internal Affairs, traffic police in the northern region of the Krai (Spassk-Dalny)", 0, 1, 3),
    ("COO", (0, 999), "125"): ("Partially plates of the Department of Internal Affairs, traffic police in the northern region of the Krai (Spassk-Dalny)", 0, 1, 3),

    # Vologda Oblast
    ("AAA", (0, 999), "35"): ("Partially plates of the regional government and Vologda city administration", 0, 1, 5),

    # Volgograd Oblast
    ("AAM", (0, 999), "34"): ("Partially plates of the Oblast Duma", 0, 1, 3),
    ("PAA", (0, 999), "34"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("AAA", (0, 999), "34"): ("Partially plates of the Oblast Prosecutor's Office", 0, 1, 3),
    ("ACK", (0, 999), "34"): ("Partially plates of the Investigative Committee, Main Department of Internal Affairs", 0, 1, 3),
    ("YYY", (0, 999), "34"): ("Partially Federal Security Service plates", 0, 1, 3),
    ("AAK", (0, 999), "34"): ("Partially plates of the Federal Bailiff Service, Ministry of Justice, and Judicial Department", 0, 1, 3),

    # Voronezh Oblast
    ("Ğ�Ğ�Ğ�", (0, 999), "36"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("BOA", (0, 999), "36"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("MMM", (0, 999), "36"): ("Partially plates of the Oblast Prosecutor's Office", 0, 1, 3),

    # Kaliningrad Oblast
    ("AAK", (0, 999), "39"): ("Partially plates of the Oblast Administration, Federal Security Service, and Prosecutor's Office", 0, 1, 5),
    ("KKK", (0, 999), "39"): ("Partially plates of the Oblast Administration, Federal Security Service, and Prosecutor's Office", 0, 1, 5),
    ("PPP", (0, 999), "39"): ("Partially former plates of the Oblast Administration (during Governor Boos)", 0, 0, 1),

    # Kaluga Oblast
    ("OOO", (0, 999), "40"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("TTT", (0, 999), "40"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("PPP", (0, 999), "40"): ("Partially plates of the Oblast Prosecutor's Office", 0, 1, 3),

    # Kurgan Oblast
    ("OOO", (0, 999), "45"): ("Partially former plates of the Oblast Administration", 0, 0, 1),
    ("TTT", (0, 999), "45"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("OKO", (0, 999), "45"): ("Partially plates of the Oblast Prosecutor's Office", 0, 1, 3),

    # Novosibirsk Oblast
    ("AAA", (0, 199), "54"): ("Plates for Presidential Plenipotentiaries", 1, 1, 7),
    ("AAA", (200, 999), "54"): ("'Special' plates", 0, 1, 4),
    ("HHH", (0, 999), "54"): ("Partially plates of the Novosibirsk mayor's office, Oblast Administration, and Oblast Council", 0, 1, 5),
    ("ACK", (0, 999), "54"): ("Partially Federal Security Service plates of the Oblast", 0, 1, 3),
    ("AHO", (0, 999), "54"): ("Partially former plates of the Oblast Administration", 0, 0, 1),
    ("AAO", (0, 999), "54"): ("Partially plates of various government agencies, including district administrations of Novosibirsk", 0, 1, 3),
    ("PPP", (0, 999), "54"): ("'Morozov' plates (introduced by former head of traffic police Pyotr Morozov)", 0, 1, 2),
    ("MOP", (0, 999), "54"): ("'Morozov' plates (introduced by former head of traffic police Pyotr Morozov)", 0, 1, 2),

    # Oryol Oblast
    ("AAA", (0, 999), "57"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("AOO", (0, 999), "57"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("OAO", (0, 999), "57"): ("Partially plates of directors of public joint-stock companies", 0, 1, 2),

    # Rostov Oblast
    ("APO", (0, 999), "61"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("AAA", (0, 999), "61"): ("Partially plates of district heads of Rostov-on-Don, mayors of Oblast cities", 0, 1, 5),
    ("APY", (0, 999), "61"): ("Partially plates of the Rostov-on-Don administration", 0, 1, 5),
    ("KKK", (0, 999), "61"): ("Partially former plates for Presidential Plenipotentiaries (during Viktor Kazantsev)", 0, 0, 1),
    ("HHH", (0, 999), "61"): ("Partially plates of the Oblast Prosecutor's Office", 0, 1, 3),
    ("MMM", (0, 999), "61"): ("Partially plates of the Oblast Department of Internal Affairs", 0, 1, 3),
    ("OOO", (0, 999), "61"): ("Partially plates of the Oblast Legislative Assembly", 0, 1, 4),
    ("BBK", (0, 999), "61"): ("Partially plates of insurance companies in Rostov-on-Don", 0, 1, 1),

    # Saratov Oblast
    ("AAA", (0, 999), "164"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("PPP", (0, 999), "164"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("XXX", (0, 999), "64"): ("Partially plates of the Oblast courts", 0, 1, 3),
    ("MMM", (0, 999), "64"): ("Partially plates of the Oblast Prosecutor's Office", 0, 1, 3),
    ("OAA", (0, 999), "64"): ("Partially Federal Security Service plates of the Oblast", 0, 1, 3),

    # Tomsk Oblast
    ("ATO", (0, 999), "70"): ("Partially plates of the Oblast Administration", 0, 1, 5),

    # Tyumen Oblast
    ("ATO", (0, 999), "72"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("PTO", (0, 999), "72"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("MTO", (0, 999), "72"): ("Partially plates of the Oblast Prosecutor's Office", 0, 1, 3),
    ("HTO", (0, 999), "72"): ("Partially plates of the Tax Service", 0, 1, 3),
    ("CTO", (0, 999), "72"): ("Partially plates of the Oblast courts", 0, 1, 3),
    ("YTO", (0, 999), "72"): ("Partially plates of the bailiff service", 0, 1, 3),
    ("BAA", (0, 999), "72"): ("Partially plates of the Oblast Ministry of Internal Affairs", 0, 1, 3),
    ("KKK", (0, 999), "72"): ("'Gangster' plates", 0, 1, 1),

    # Arkhangelsk Oblast
    ("TTT", (0, 999), "29"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("PPP", (0, 999), "29"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("MAO", (0, 999), "29"): ("Partially plates of the Oblast Ministry of Internal Affairs", 0, 1, 3),

    # Ryazan Oblast
    ("APO", (0, 999), "62"): ("Partially plates of the Oblast Administration", 0, 1, 5),

    # Samara Oblast
    ("PAA", (0, 999), "63"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("AAP", (0, 999), "63"): ("Partially plates of the Oblast Administration", 0, 1, 5),
}



def get_region_code(plate):
    region_code = plate[6:]
    for region , code in REGION_CODES.items():
        if region_code in code:
            return region
    return "UNKNOWN REGION"


get_region_code("A123BC36")  # Example usage, should return "Voronezh Oblast"


def preprocessing_data(train):
    train["date"] = pd.to_datetime(train["date"] , errors='coerce')
    train["year"] = train["date"].dt.year.astype("int64")
    train["month"] = train["date"].dt.month.astype("int64")
    train["day"] = train["date"].dt.day.astype("int64")
    train["day_of_week"] =train["date"].dt.dayofweek.astype("int64")
    train["week_of_year"] = train["date"].dt.isocalendar().week
    train["is_weekend"] = train["day_of_week"].isin([5, 6]).astype('int64')  
    train["is_start_of_month"] = (train["day"] == 1).astype('int64')
    train["plate_number"] = train["plate"].apply(lambda x : x[1:4]).astype("str")
    train["plate_series"] = train["plate"].apply(lambda x : x[0] + x[4:6]).astype("str")
    train['region'] = train['plate'].str.extract(r"(\d{2,3})$")[0].astype(str) # consider last 2-3 digits
    train = train.sort_values(by=['region', 'date'])
    train['price'] = np.log1p(train['price'])
    train['is_palindrome_letters'] = train['plate_series'].apply(lambda x: x == x[::-1])
    train['is_palindrome_numbers'] = train['plate_number'].apply(lambda x: x == x[::-1])
    train['unique_letters_count'] = train['plate_series'].apply(lambda x: len(set(x)))
    train['unique_numbers_count'] = train['plate_number'].apply(lambda x: len(set(x)))
    train["has_repeating_digits"] = train["plate_number"].apply(lambda x: len(set(x)) < len(x))
    digits = train['plate_number'].apply(lambda s: list(map(int, s)))
    train[['d1', 'd2', 'd3']] = pd.DataFrame(digits.tolist(), index=train.index)
    train['sum_numbers'] = train['d1'] + train['d2'] + train['d3']
    train['product_numbers'] = train['d1'] * train['d2'] * train['d3']
    train = train.drop(columns=['id', 'plate', 'date', 'd1', 'd2', 'd3', 'sum_numbers'])
    
    return train


def feature_engineering(df):
    le = LabelEncoder()
    col = ['region', 'is_palindrome_letters', 'is_palindrome_numbers', 'has_repeating_digits', 'plate_number', 'plate_series']
    df = df.apply(lambda x: le.fit_transform(x) if x.name in col else x, axis=0)
    return df

df = preprocessing_data(train)
df = feature_engineering(df)

df


df.info()


param_grids = {
    'DecisionTree': {
        'model__max_depth': [3, 5, 10, None],
        'model__min_samples_split': [2, 5, 10],
    },
    'RandomForest': {
        'model__n_estimators': [100, 200],
        'model__max_depth': [5, 10, None],
        'model__min_samples_split': [2, 5],
    },
    'CatBoost': {
        'model__iterations': [100, 200],
        'model__depth': [4, 6, 8],
        'model__learning_rate': [0.01, 0.1],
    },
    'XGBoost': {
        'model__n_estimators': [100, 200],
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.1],
    }
}



pipelines = {
    'DecisionTree': Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', DecisionTreeRegressor())
    ]),
    'RandomForest': Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', RandomForestRegressor())
    ]),
    'CatBoost': Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', CatBoostRegressor())
    ]),
    'XGBoost': Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', XGBRegressor())
    ])
}



# SMAPE function
def smape_(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0  # avoid division by zero
    return np.mean(diff) * 100

# Sklearn scorer for RandomizedSearchCV
smape_scorer = make_scorer(smape_, greater_is_better=False)


df.columns


X = df.drop(columns=['price'])
y = df['price'] # very large values, so we use log transformation


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


results = []
for name in pipelines:
    print(f"\nğŸ”� Training {name} with RandomizedSearchCV...")
    
    search = RandomizedSearchCV(
        pipelines[name],
        param_distributions=param_grids[name],
        n_iter=5,
        scoring=smape_scorer,  # use scorer object here
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    y_pred = search.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    smape_score = smape_(y_test, y_pred)  # call the actual function here
    
    print(f"âœ… Best Params for {name}:")
    print(search.best_params_)
    
    print(f"ğŸ“‰ MSE: {mse:.4f}")
    print(f"ğŸ“‰ MAE: {mae:.4f}")
    print(f"ğŸ“‰ SMAPE: {smape_score:.2f}%")
    
    results.append({
        'Model': name,
        'Best Params': search.best_params_,
        'MSE': mse,
        'MAE': mae,
        'SMAPE': smape_score
    })


for result in results:
    print(f"\nModel: {result['Model']}")
    print(f"Best Params: {result['Best Params']}")
    print(f"MSE: {result['MSE']:.4f}")
    print(f"MAE: {result['MAE']:.4f}")
    smape_val = result.get('SMAPE')
    if smape_val is not None:
        print(f"SMAPE: {smape_val:.2f}%")
    else:
        print("SMAPE: -")


test.head()


test.info()


df_test = preprocessing_data(test)
df_test = df_test.drop(columns=['price'])
df_test = feature_engineering(df_test)


model = search.best_estimator_
model 



predictions = model.predict(df_test)
predictions


submission = pd.DataFrame({
    'id': test['id'],
    'price': np.expm1(predictions)  # reversing the log transformation
})

submission.head()


submission.to_csv('submission.csv', index=False)




