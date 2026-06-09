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


import csv, datetime, os, random, zipfile
from collections import defaultdict
import numpy as np
import pandas as pd
import xgboost as xgb

cat_cols = [
    'ind_empleado','sexo','ind_nuevo','indrel','indrel_1mes','tiprel_1mes',
    'indresi','indext','indfall','ind_actividad_cliente','segmento',
    'pais_residencia','canal_entrada'
]

target_cols = [
    'ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1',
    'ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',
    'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1',
    'ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1',
    'ind_nom_pens_ult1','ind_recibo_ult1'
]

def extract_data_if_needed(zip_files, input_dirs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for zip_name in zip_files:
        extracted_name = zip_name.rsplit('.zip', 1)[0]
        extracted_path = os.path.join(output_dir, extracted_name)
        if os.path.exists(extracted_path):
            continue
        zip_path = None
        for base in input_dirs:
            candidate = os.path.join(base, zip_name)
            if os.path.exists(candidate):
                zip_path = candidate
                break
        if not zip_path:
            raise FileNotFoundError(f"Не найден архив {zip_name} ни в одном из путей: {input_dirs}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

def detect_base():
    for root, _, files in os.walk("/kaggle/input"):
        if "train_ver2.csv.zip" in files:
            return root + "/"
    return "/kaggle/input/"

def build_maps_and_renta(path):
    months = {'2015-04-28','2015-05-28','2015-06-28'}
    maps = {c: {'__UNK__': 0} for c in cat_cols}
    rents = defaultdict(list)
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['fecha_dato'] not in months:
                continue
            for c in cat_cols:
                v = row[c].strip()
                if v not in ('', 'NA') and v not in maps[c]:
                    maps[c][v] = len(maps[c])
            r = row['renta'].strip()
            prov = row['nomprov'].strip() or '__UNK__'
            if r not in ('', 'NA'):
                try: rents[prov].append(float(r))
                except ValueError: pass
    prov_median = {p: float(np.median(v)) for p, v in rents.items() if v}
    global_median = float(np.median([x for v in rents.values() for x in v])) if rents else 100000.0
    return maps, prov_median, global_median

def getTarget(row):
    return [0 if row[c].strip() in ('', 'NA') else int(float(row[c])) for c in target_cols]

def getIndex(row, col, maps):
    v = row[col].strip()
    return maps[col].get(v, 0) if v not in ('', 'NA') else 0

def getAge(row):
    age = row['age'].strip()
    if age in ('', 'NA'): age = 40.0
    else: age = min(max(float(age), 18.0), 100.0)
    return round((age - 18.0) / (100.0 - 18.0), 4)

def getCustSeniority(row):
    v = row['antiguedad'].strip()
    v = 0.0 if v in ('', 'NA') else min(max(float(v), 0.0), 256.0)
    return round(v / 256.0, 4)

def getRent(row, prov_median, global_median):
    r = row['renta'].strip()
    if r in ('', 'NA'):
        prov = row['nomprov'].strip() or '__UNK__'
        r_val = prov_median.get(prov, global_median)
    else:
        r_val = min(max(float(r), 0.0), 1_500_000.0)
    return round(r_val / 1_500_000.0, 6)

def getMonth(row): return int(row['fecha_dato'].split('-')[1])

def getjoinMonth(row):
    raw = row['fecha_alta'].strip()
    return int(random.choice(range(1,13))) if raw in ('', 'NA') else int(raw.split('-')[1])

def processDataMK(path, cust_dict, lag_cust_dict, maps, prov_median, global_median):
    x_vars_list, y_vars_list = [], []
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['fecha_dato'] not in ['2015-04-28', '2015-05-28', '2015-06-28', '2016-04-28', '2016-05-28', '2016-06-28']:
                continue
            cid = int(row['ncodpers'])
            if row['fecha_dato'] in ['2015-04-28', '2016-04-28']:
                lag_cust_dict[cid] = getTarget(row); continue
            if row['fecha_dato'] in ['2015-05-28', '2016-05-28']:
                cust_dict[cid] = getTarget(row); continue
            x_vars = [getIndex(row, c, maps) for c in cat_cols]
            x_vars += [getAge(row), getMonth(row), getjoinMonth(row), getCustSeniority(row), getRent(row, prov_median, global_median)]
            if row['fecha_dato'] == '2016-06-28':
                prev = cust_dict.get(cid, [0]*22)
                lag  = lag_cust_dict.get(cid, [0]*22)
                x_vars_list.append(x_vars + prev + lag)
            elif row['fecha_dato'] == '2015-06-28':
                prev = cust_dict.get(cid, [0]*22)
                lag  = lag_cust_dict.get(cid, [0]*22)
                curr = getTarget(row)
                new_products = [max(c - p, 0) for c, p in zip(curr, prev)]
                if sum(new_products) == 0: continue
                for idx, prod in enumerate(new_products):
                    if prod > 0:
                        x_vars_list.append(x_vars + prev + lag)
                        y_vars_list.append(idx)
    return x_vars_list, y_vars_list, cust_dict, lag_cust_dict

def runXGB(train_X, train_y, seed_val=0):
    param = {
        'objective': 'multi:softprob',
        'eta': 0.09,
        'max_depth': 6,
        'num_class': 22,
        'eval_metric': 'mlogloss',
        'min_child_weight': 12,
        'subsample': 0.85,
        'colsample_bytree': 0.9,
        'seed': seed_val
    }
    num_rounds = 70
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    return xgb.train(list(param.items()), xgtrain, num_rounds)

if __name__ == "__main__":
    start_time = datetime.datetime.now()

    data_path = "/kaggle/working/"
    base = detect_base()  # попытается найти каталог с train_ver2.csv.zip
    search_dirs = [base, "/kaggle/input/", data_path, "."]

    zip_files = ["train_ver2.csv.zip", "test_ver2.csv.zip", "sample_submission.csv.zip"]
    extract_data_if_needed(zip_files, search_dirs, data_path)

    train_csv = os.path.join(data_path, "train_ver2.csv")
    test_csv  = os.path.join(data_path, "test_ver2.csv")

    maps, prov_median, global_median = build_maps_and_renta(train_csv)

    print('Starting file processing')
    train_X_list, train_y_list, cust_dict, lag_cust_dict = processDataMK(
        train_csv, {}, {}, maps, prov_median, global_median
    )
    print('Finished file processing')

    train_X = np.array(train_X_list); train_y = np.array(train_y_list)
    del train_X_list, train_y_list
    print(train_X.shape, train_y.shape)
    print(datetime.datetime.now() - start_time)

    test_X_list, _, cust_dict, lag_cust_dict = processDataMK(
        test_csv, cust_dict, lag_cust_dict, maps, prov_median, global_median
    )
    test_X = np.array(test_X_list); del test_X_list
    print(test_X.shape)
    print(datetime.datetime.now() - start_time)

    print("Building model..")
    model = runXGB(train_X, train_y, seed_val=0)
    del train_X, train_y

    print("Predicting..")
    preds = model.predict(xgb.DMatrix(test_X))
    del test_X
    print(datetime.datetime.now() - start_time)

    print("Getting the top products..")
    test_id = np.array(pd.read_csv(test_csv, usecols=['ncodpers'])['ncodpers'])
    new_products = [np.maximum(preds[i] - cust_dict.get(idx, [0]*22), 0) for i, idx in enumerate(test_id)]
    top_idx = np.fliplr(np.argsort(new_products, axis=1))[:, :7]
    final_preds = [" ".join(np.array(target_cols)[row]) for row in top_idx]
    pd.DataFrame({'ncodpers': test_id, 'added_products': final_preds}).to_csv('actualxgb_sub.csv', index=False)
    print(datetime.datetime.now() - start_time)

