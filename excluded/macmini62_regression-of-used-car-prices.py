import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
plt.rc(
  'axes',
	labelpad=10,
  labelweight='semibold',
  labelcolor="#1B1B1B",
  titlepad=12,
  titlesize=14,
  titleweight='bold',
  titlelocation='center'
)

import re
from collections import Counter


def dct(df: pd.DataFrame) -> pd.DataFrame:

	# Rename the incorrectlly defined car brands
	def rename(x):
		brands = list(x.values)
		brands_counter = Counter(brands)
		most_common_item, _ = brands_counter.most_common(1)[0]
		
		return most_common_item
		
	df.loc[:, 'brand'] = df.groupby(['model'])['brand'].transform(lambda x: rename(x))

	# Update the null fuel type values with values from the engines descriptions
	fn = df.loc[(df['fuel_type'].isna()) | (df['fuel_type'] == '–')]

	PC = ['Petrol', 'Gas', 'Gasoline', 'Litre', 'L']
	DC = ['Diesel', 'Litre', 'L']
	EC = ['Electric', 'Motor', 'Battery']

	def check_fuel_type(engine):
		y = engine.split()
		
		if 'Hybrid' in y:
			return 'Hybrid'
		if 'Flex' in y:
			return 'E85 Flex Fuel'
		
		# If the engine description contains amphere designation, classify as Electric
		z = re.search(r'\b\d+(\.\d+)?Ah\b', y[0])
		if z:
			return 'Electric'
		
		for i in y:
			if i in PC:
				return 'Gasoline'
			elif i in DC:
				return 'Diesel'
			elif i in EC:
				return 'Electric'

	fn.loc[:, 'fuel_type'] = fn['engine'].apply(lambda x: check_fuel_type(x))
	df.update(fn)

	def fill1(x):
		y = list(x.mode())
		if len(y) > 0:
			return x.fillna(y[0])
		
	# Update the remaining fuel_type values with values from other records of the same vehicle.
	df.loc[:, 'fuel_type'] = df.groupby(['brand', 'model', 'model_year'])['fuel_type'].transform(lambda x: fill1(x))

	# Update any other record with a null fuel_type value
	df.loc[:, 'fuel_type'] = df['fuel_type'].fillna('–')

	# Fill all the accidents null values as none reported
	df.loc[:, 'accident'] = df['accident'].fillna('None reported')

	# Fill all the clean title null values as none reported
	df.loc[:, 'clean_title'] = df['clean_title'].fillna('Not recorded')

	# Create additional HP and CC variables gotten from the engines description
	def check_HP(engine):
		y = engine.split()[0].split('HP')
		if len(y) == 2:
			return float(y[0])
		
	def convert(cap):
		return float(cap.group(0).split('L')[0])*1000
		
		
	def check_CC(engine):
		y = engine.split()
		if len(y) > 1:
			if 'Liter' in y:
				return float(y[0]) * 1000
			else:
				z = re.search(r'\b\d+(\.\d+)?L\b', y[0])
				if z:
					return convert(z)
				else:
					zz = re.search(r'\b\d+(\.\d+)?L\b', y[1])
					if zz:
						return convert(zz)

	df.loc[:, 'horse_power'] = df['engine'].apply(lambda x: check_HP(x))
	df.loc[:, 'cc'] = df['engine'].apply(lambda x: check_CC(x))

	# Update the remaining horse_power and cc values with values from other records of the same vehicle.
	df.loc[:, 'horse_power'] = df.groupby(['brand', 'model', 'model_year'])['horse_power'].transform(lambda x: fill1(x))
	df.loc[:, 'cc'] = df.groupby(['brand', 'model', 'model_year'])['cc'].transform(lambda x: fill1(x))

	# Update the remaining null values
	df['cc'] = df['cc'].astype('float')
	df['horse_power'] = df['horse_power'].astype('float')

	#  Update the engines of same vehicle with different records

	def update(x):
		engines = list(x.values)
		engines_counter = Counter(engines)
		most_common_item, _ = engines_counter.most_common(1)[0]
		if most_common_item == '–' and len(engines_counter.keys()) > 1:
			most_common_item, _ = engines_counter.most_common(2)[1]
		
		for i in engines:
			if i == '–':
				return most_common_item
			else:
				return i

	df.loc[:, 'engine'] = df.groupby(['brand', 'model', 'model_year'])['engine'].transform(lambda x: update(x))

	# Handle transmission inconsistent values
	def check_gears(x):
		y = re.search(r'\b\d+\b', x)
		if y:
			return y.group(0)
		
	df.loc[:, 'gears'] = df['transmission'].apply(lambda x: check_gears(x))

	# AUTO = ['Automatic', 'A/T', 'AT']
	# MANUAL = ['M/T', 'Manual', 'Mt']
	# DUAL = ['w/Dual', 'At/Mt']
	# CVT = ['CVT', 'Variable', 'CVT-F', 'F']

	# def transmission(x):
	#   y = x.split()
	#   for i in y:
	#     if i in AUTO:
	#       return 'Automatic'
	#     elif i in MANUAL:
	#       return 'Manual'
	#     elif i in DUAL:
	#       return 'Dual'
	#     elif i in CVT:
	#       return 'CVT'
			
	#   return x
		
	# df.loc[:, 'transmission'] = df['transmission'].apply(lambda x: transmission(x))

	# Handle missing gear values
	def fill2(x):
		y = list(x.mode())
		if len(y) > 0:
			return x.fillna(y[0])
		elif len(y) > 1:
			return x.fillna(y[1])
		
	df.loc[:, 'gears'] = df.groupby(['brand', 'model', 'model_year'])['gears'].transform(lambda x: fill2(x))
	df['gears'] = df['gears'].astype('float')

	return df


train_df = dct(pd.read_csv('/kaggle/input/playground-series-s4e9/train.csv'))
test_df = dct(pd.read_csv('/kaggle/input/playground-series-s4e9/test.csv'))


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder as le

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score


def ft(df: pd.DataFrame) -> pd.DataFrame:
	df = df.drop(columns='id')
	for col in df.columns:
		df[col] = le().fit_transform(df[col])
	
	return df


plt.figure(figsize=(24, 10))
sns.heatmap(ft(train_df).corr(method='spearman'), vmax=1, vmin=-1, cmap='Blues', linewidths=2, annot=True)

plt.show()


df_copy = ft(train_df).copy()

y = df_copy.pop('price')
X = df_copy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)


regression_models = {
  'Decision Tree: ': DecisionTreeRegressor(criterion='squared_error', splitter='random', max_features='sqrt', random_state=42, ccp_alpha=1),
  'Linear Regression: ': LinearRegression(fit_intercept=True, copy_X=True, n_jobs=10, positive=False),
  'Linear SVR: ': LinearSVR(epsilon=0, C=1, loss='squared_epsilon_insensitive', intercept_scaling=1, random_state=42, max_iter=2000)
}

def evaluate_model(m, X_train, y_train, X_test, y_test) -> dict[str, float]:
    y_pred = m.predict(X_test)
    
    scores = {
      'MAE': mean_absolute_error(y_test, y_pred),
      'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
      'R2': r2_score(y_test, y_pred),
      'MAPE': mean_absolute_percentage_error(y_test, y_pred)
    }
    cv_rmse = np.sqrt(-cross_val_score(m, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    scores['CV_RMSE_mean'] = cv_rmse.mean()
    scores['CV_RMSE_std'] = cv_rmse.std()
    
    return scores

models = dict()

for name, m in regression_models.items():
    model = m.fit(X_train, y_train)
    print(name, evaluate_model(model, X_train, y_train, X_test, y_test))
    
    models.update({name: model})


model_prediction_results = pd.DataFrame(columns=[model for model in models])

for model in models:
    m = models.get(model)
    if m is not None:
        predicts = m.predict(ft(test_df))
        model_prediction_results[model] = np.array(predicts.squeeze())
    else:
        print(f"Warning: Model '{model}' is None and will be skipped.")

model_prediction_results


# Use the Linear Regression because it performes better than the rest of the models.

results = pd.DataFrame({
  'id': test_df['id'].values,
  'price': model_prediction_results['Linear Regression: '].values
})

results.loc[:, 'price'] = results['price'].round(3).astype('float')
results.to_csv('submission.csv', index=False)

results

