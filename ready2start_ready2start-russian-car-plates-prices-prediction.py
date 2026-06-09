# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# V1. RandomForestRegressor 단순 적용
# V2. REGION CODES, GOVERNMENT CODES 데이터 활용
# V3. xgboost 적용
import xgboost


# MAPE (Mean Absolute Percentage Error, 평균 절대 백분율 오류)
# 실제 값이 0인 경우 계산이 불가능함.
# 실제 값이 0에 가까운 경우 값이 지나치게 높아질 수 있음.


# SMAPE (Symmetric Mean Absolute Percentage Error)
# 실제 값이 0인 경우에도 계산이 가능함.
# 실제 값이 0에 가까운 경우에도 200% 이하를 유지함.


# SMAPE 계산 함수

def smape(actual, forecast):
    denominator = np.abs(actual) + np.abs(forecast)  # 분모
    diff = np.abs(actual - forecast) / denominator
    diff[denominator == 0] = 0.0  # 분모가 0인 경우 0으로 리리
    
    return 200 * np.mean(diff)


submission = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv')
train_data = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
test_data = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')


train_data.head(3)


test_data.head(3)


print(train_data.shape, test_data.shape)


train_data.info()


# id 컬럼 제거
train_data.drop(columns=['id'], axis=1, inplace=True)


# registration plate = 1 letter + 3 digits + 2 letters + region code
# 3 letters / 3 digits / region code로 분리하기

train_data['letters'] = train_data['plate'].str.slice(0, 1) + train_data['plate'].str.slice(4, 6)
train_data['digits'] = train_data['plate'].str.slice(1, 4)
train_data['region_code'] = train_data['plate'].str.slice(6, 9)

train_data.drop(columns=['plate'], axis=1, inplace=True)

train_data.head(3)


train_data.info()


# date 컬럼 datetime형으로 변환
train_data['date'] = pd.to_datetime(train_data['date'])
print(train_data.dtypes)

# date 컬럼에서 년도, 월, 일, 시간, 요일 데이터 추출
# V3. date timestamp 데이터 추가
train_data['timestamp'] = train_data['date'].astype('int64')
train_data['year'] = train_data['date'].dt.year
train_data['month'] = train_data['date'].dt.month
train_data['day'] = train_data['date'].dt.day
train_data['hour'] = train_data['date'].dt.hour
train_data['dayofweek'] = train_data['date'].dt.dayofweek

train_data.drop(columns=['date'], axis=1, inplace=True)

train_data.head(3)


train_data.info()


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
    ("ААА", (0, 999), "36"): ("Partially plates of the Oblast Administration", 0, 1, 5),
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


# V2 추가
# REGION CODES에서 region name 컬럼 추출

# REGION CODE => NAME 딕셔너리 생성
region_code_to_name = dict()

for name, codes in REGION_CODES.items():
    for code in codes:
        region_code_to_name[code] = name

train_data['region_name'] = train_data['region_code'].map(lambda x: region_code_to_name.get(str(x)))
train_data.head(3)


# V2 추가
# GOVERNMENT_CODES에서 구매금지여부, 도로어드벤티지여부, 중요도 컬럼 추출

# GOVERNMENT CODE별 정보 정리
government_code_info = dict()

for key, value in GOVERNMENT_CODES.items():
    letter, min_num, max_num, region_code = key[0], key[1][0], key[1][1], key[2]
    is_forbidden, has_advantage, significance = value[1], value[2], value[3]

    if letter in government_code_info:
        government_code_info[letter].append([region_code, min_num, max_num, is_forbidden, has_advantage, significance])
    else:
        government_code_info[letter] = [ [region_code, min_num, max_num, is_forbidden, has_advantage, significance] ]

def get_values(info_data):

    # GOVERNMENT CODE 정보에 해당 글자가 없는 경우
    if letter not in government_code_info:
        return [0, 1, 5]  # 임의로 중간값으로 넣어줌.

    for region_code, min_num, max_num, is_forbidden, has_advantage, significance in government_code_info[letter]:
        if (region_code == info_data['region_code']) & (min_num <= int(info_data['digits']) <= max_num):
            return [is_forbidden, has_advantage, significance]
        
    return [0, 1, 5]  # 임의로 중간값으로 넣어줌.


# is_forbidden, has_advantage, significance 컬럼 추출
train_data['is_forbidden'] = 0
train_data['has_advantage'] = 1
train_data['significance'] = 5

for idx in range(len(train_data)):
    is_forbidden, has_advantage, significance = get_values(train_data.iloc[idx])

    train_data.loc[idx, 'is_forbidden'] = is_forbidden
    train_data.loc[idx, 'has_advantage'] = has_advantage
    train_data.loc[idx, 'significance'] = significance

train_data.head(5)


train_data.info()


from sklearn.preprocessing import LabelEncoder

# Label Encoding
# letters, digits, region_code
letters_encoder = LabelEncoder()
train_data['letters'] = letters_encoder.fit_transform(train_data['letters'])

digits_encoder = LabelEncoder()
train_data['digits'] = digits_encoder.fit_transform(train_data['digits'])

region_code_encoder = LabelEncoder()
train_data['region_code'] = region_code_encoder.fit_transform(train_data['region_code'])

# V2 추가
# region_name 레이블 인코딩
region_name_encoder = LabelEncoder()
train_data['region_name'] = region_name_encoder.fit_transform(train_data['region_name'])

train_data.head(3)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# X, Y 분리
y = train_data['price']
x = train_data.drop(columns=['price'], axis=1)

x.head(3)


# 테스트 데이터, 검증용 데이터 분리
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.2, random_state=0)


train_x.head(3)


from sklearn.model_selection import GridSearchCV


# V2 수정
reg = RandomForestRegressor(
    n_estimators = 1000,
    n_jobs = -1,
    random_state = 0,
)


# V3 : XGBRegressor 모델 적용
# reg = xgboost.XGBRegressor(
#     n_jobs = -1,
#     random_state = 0,
#     n_estimators = 35,   # 하이퍼파라미터 튜닝의 결과로 얻은 값
# );

reg.fit(train_x, train_y)

# 하이퍼파라미터 튜닝 결과
# max_depth : None / n_estimators : 35

# parameters = {
#     'n_estimators': [10, 30, 50, 100, 300, 500],
#     'max_depth': [5, 10, 20, None]
# }

# parameters = {
#     'n_estimators': [15, 20, 25, 30, 35, 40],
# }

# clf = GridSearchCV(
#     reg, 
#     parameters,
#     scoring = 'neg_mean_absolute_percentage_error',
# )

# clf.fit(train_x, train_y)


print('train data SMAPE :', smape(train_y, reg.predict(train_x)))
print('valid data SMAPE :', smape(valid_y, reg.predict(valid_x)))

# print(clf.best_params_)
# print('train data SMAPE :', smape(train_y, clf.predict(train_x)))
# print('valid data SMAPE :', smape(valid_y, clf.predict(valid_x)))

# Try 1
# {'max_depth': None, 'n_estimators': 30}
# train data SMAPE : 69.55926870516602
# valid data SMAPE : 71.30929799373986

# Try 2
# {'n_estimators': 35}
# train data SMAPE : 69.75746999329847
# valid data SMAPE : 71.60075877158367


# 테스트 데이터 전처리
test_data = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')

# registration plate = 1 letter + 3 digits + 2 letters + region code
# 3 letters / 3 digits / region code로 분리하기
test_data['letters'] = test_data['plate'].str.slice(0, 1) + test_data['plate'].str.slice(4, 6)
test_data['digits'] = test_data['plate'].str.slice(1, 4)
test_data['region_code'] = test_data['plate'].str.slice(6, 9)

# date 컬럼 datetime형으로 변환
test_data['date'] = pd.to_datetime(test_data['date'])

# date 컬럼에서 년도, 월, 일, 시간, 요일 데이터 추출
# V3. date timestamp 데이터 추가
test_data['timestamp'] = test_data['date'].astype('int64')
test_data['year'] = test_data['date'].dt.year
test_data['month'] = test_data['date'].dt.month
test_data['day'] = test_data['date'].dt.day
test_data['hour'] = test_data['date'].dt.hour
test_data['dayofweek'] = test_data['date'].dt.dayofweek

test_data.drop(columns=['price', 'id', 'plate', 'date'], axis=1, inplace=True)

test_data.head(3)


# V2 추가
# REGION CODES에서 region name 컬럼 추출
test_data['region_name'] = test_data['region_code'].map(lambda x: region_code_to_name.get(str(x)))
test_data.head(3)

# V2 추가
# is_forbidden, has_advantage, significance 컬럼 추출
test_data['is_forbidden'] = 0
test_data['has_advantage'] = 1
test_data['significance'] = 5

for idx in range(len(test_data)):
    is_forbidden, has_advantage, significance = get_values(test_data.iloc[idx])

    test_data.loc[idx, 'is_forbidden'] = is_forbidden
    test_data.loc[idx, 'has_advantage'] = has_advantage
    test_data.loc[idx, 'significance'] = significance

test_data.head(3)


# Label Encoding
# letters, digits, region_code
test_data['letters'] = letters_encoder.transform(test_data['letters'])
test_data['digits'] = digits_encoder.transform(test_data['digits'])
test_data['region_code'] = region_code_encoder.transform(test_data['region_code'])

# V2 추가 - region_name
test_data['region_name'] = region_name_encoder.transform(test_data['region_name'])

test_data.head(3)


prediction = reg.predict(test_data)
submission['price'] = prediction


submission.to_csv('submission.csv', index=False)




