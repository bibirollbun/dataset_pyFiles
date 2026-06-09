import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
import re

# --- 1. Data Loading and Cleaning ---
def clean_ExtendedQuantity(o):
    if not isinstance(o, str): return np.nan
    o = o.replace(',', '').strip()
    if '\n' in o: return 3.0
    if o.endswith('-'): o = o[:-1]
    num_part = re.match(r'[\d.]+', o)
    return float(num_part.group(0)) if num_part else np.nan

LABEL_TO_CODE_MAP = {
    'LADDER_SPLICE': 412780, 'UPS_BATT_KWH': 366537, 'AISLE_DOOR': 986688, 'CONTAIN_ROOF': 445067,
    'COPPER_CAT6A_3FT': 332139, 'CHILLER_TON': 938549, 'PDU_RARITAN_PX3': 729660, 'BUSWAY_800A_4M': 771069,
    'FIBER_MTP12_10M': 777509, 'PDU_APC_BASIC_30A': 550635, 'HORIZ_MANAGER_2U': 366568, 'COPPER_CAT6A_7FT': 474703,
    'RAISED_FLOOR_SF': 321260, 'PP_FIBER_24LC': 282660, 'PERFORATED_TILE': 622673, 'RACK_OPEN_45U': 469739,
    'CAGE_NUT_KIT': 854862, 'CRAC_20T': 404968, 'DCIM_LICENSE': 950300, 'FIBER_MTP12_20M': 578517,
    'LADDER_10FT': 931178, 'PDU_APC_SW_30A': 490566, 'VERT_MANAGER_0U': 571642, 'BLANKING_PANELS': 254798,
    'COPPER_CAT6A_14FT': 461822, 'RACK_CPI_45U': 422984, 'RDHX_35KW': 419111, 'RACK_APC_42U': 419476,
    'PP_CAT6_48': 173049, 'SWITCHGEAR_SEC': 900121, 'CLEAN_AGENT_CYL': 287969, 'GENSET_1MW': 312279,
    'FIBER_MTP24_30M': 938404, 'INROW_30KW': 937143, 'PP_CAT6_24': 131611, 'PUMP_CHW_CW': 218694,
    'VFD_DRIVE': 477545, 'BATTERY_SET': 639485, 'PANELBOARD_LV': 118404, 'AHU_CENTRAL': 930791,
    'DUCTWORK_FT': 698245, 'COOLING_TOWER': 554994, 'PIPING_CHW_CW_FT': 589590, 'TRANSFORMER_MV_LV': 834041,
    'MV_SWITCHGEAR_LINEUP': 526171, 'UPS_SYSTEM_MODULE': 266351
}

# Dummy training data (as before).
# In your real notebook, this will be loaded from train.csv
train = pd.DataFrame({
    'id': range(13420, 13430), 'PROJECT_TYPE': ['Commercial', 'Data Center'] * 5,
    'ItemDescription': ['LADDER_SPLICE cable', 'UPS_BATT_KWH system', 'AISLE_DOOR handle', 'CONTAIN_ROOF panel', 'PDU_APC_BASIC_30A unit', 'RACK_APC_42U server', 'LADDER_SPLICE bracket', 'UPS_BATT_KWH battery', 'AISLE_DOOR sensor', 'CONTAIN_ROOF sealant'],
    'MasterItemNo': ['LADDER_SPLICE', 'UPS_BATT_KWH', 'AISLE_DOOR', 'CONTAIN_ROOF', 'PDU_APC_BASIC_30A', 'RACK_APC_42U', 'LADDER_SPLICE', 'UPS_BATT_KWH', 'AISLE_DOOR', 'CONTAIN_ROOF'],
    'ExtendedQuantity': ['10', '20.5', '30,000', '40-', '50', '60 EA', '70', '80', '90', '100'],
    'QtyShipped': ['5', '15.5', '25', '35', '45', '55', '65', '75', '85', '95'],
    'CORE_MARKET': ['East', 'West'] * 5, 'STATE': ['CA', 'TX'] * 5
})

# --- THE FIX ---
# Manually created a LARGER dummy test set to prove the code scales.
# This now has 5 rows.
print("Using a larger 5-row test set for demonstration...")
test = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'PROJECT_TYPE': ['Data Center', 'Commercial', 'Data Center', 'Commercial', 'Data Center'],
    'ItemDescription': ['PDU_RARITAN_PX3 power strip', 'HORIZ_MANAGER_2U cable organizer', 'RACK_APC_42U server rack', 'LADDER_10FT section', 'GENSET_1MW generator'],
    'ExtendedQuantity': ['500,000', '60,000 EA', '75', '12 EA', '1'],
    'CORE_MARKET': ['Central', 'East', 'West', 'South', 'Central'],
    'STATE': ['IL', 'NY', 'CA', 'TX', 'IL']
})

# Apply cleaning
for df in [train, test]:
    df['ExtendedQuantity'] = df['ExtendedQuantity'].apply(clean_ExtendedQuantity)
    df['ItemDescription'] = df['ItemDescription'].str.replace('\n', ' ').str.strip()
    for col in ['CORE_MARKET', 'STATE']: df[col] = df[col].fillna('missing')
train['QtyShipped'] = train['QtyShipped'].apply(clean_ExtendedQuantity)

# --- 2. Feature Engineering & Preprocessing ---
features = ['ExtendedQuantity', 'CORE_MARKET', 'STATE', 'ItemDescription']
categorical_features = ['CORE_MARKET', 'STATE']
numerical_features = ['ExtendedQuantity']
text_feature = 'ItemDescription'

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('text', TfidfVectorizer(ngram_range=(1, 2), max_features=5000), text_feature)
    ],
    remainder='drop'
)

# --- 3. Training the Two Specialist Models ---
train_main = train[train['PROJECT_TYPE'] != 'Data Center'].copy()
train_dc = train[train['PROJECT_TYPE'] == 'Data Center'].copy()

model_main_c = make_pipeline(preprocessor, RandomForestClassifier(random_state=42))
model_main_r = make_pipeline(preprocessor, RandomForestRegressor(random_state=42))
model_dc_c = make_pipeline(preprocessor, RandomForestClassifier(random_state=42))
model_dc_r = make_pipeline(preprocessor, RandomForestRegressor(random_state=42))

print("Training models...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model_main_c.fit(train_main[features], train_main['MasterItemNo'])
    model_main_r.fit(train_main.dropna(subset=['QtyShipped'])[features], train_main.dropna(subset=['QtyShipped'])['QtyShipped'])
    model_dc_c.fit(train_dc[features], train_dc['MasterItemNo'])
    model_dc_r.fit(train_dc.dropna(subset=['QtyShipped'])[features], train_dc.dropna(subset=['QtyShipped'])['QtyShipped'])

# --- 4. Prediction on the Full Test Set ---
print("Generating predictions...")
pred_c = np.empty(len(test), dtype=object)
pred_r = np.empty(len(test))

dc_indices = test['PROJECT_TYPE'] == 'Data Center'
main_indices = ~dc_indices

if dc_indices.any():
    pred_c[dc_indices] = model_dc_c.predict(test.loc[dc_indices, features])
    pred_r[dc_indices] = model_dc_r.predict(test.loc[dc_indices, features])
if main_indices.any():
    pred_c[main_indices] = model_main_c.predict(test.loc[main_indices, features])
    pred_r[main_indices] = model_main_r.predict(test.loc[main_indices, features])

# --- 5. Creating and Saving the Final Submission File ---
final_pred_c_series = pd.Series(pred_c)
final_pred_c_mapped = final_pred_c_series.map(LABEL_TO_CODE_MAP).fillna(final_pred_c_series)
submission = pd.DataFrame({
    'id': test.id,
    'MasterItemNo': final_pred_c_mapped.astype(int),
    'QtyShipped': pred_r.round(2)
})

# Save the DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)

print('\n✅ Submission file "submission.csv" created successfully!\n')
print(f"--- Submission File Preview (now with {len(submission)} rows) ---")
print(submission.head())


