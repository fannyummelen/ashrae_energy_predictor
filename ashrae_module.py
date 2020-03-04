###############################################################
# functions specifically for the ASHRAE Great Energy Predictor#	
###############################################################

###############################################################
# import python essentials and custom modules				  #
###############################################################

import loading_and_saving as ls
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime
import math
import joblib
from scipy.stats import iqr
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import time


###############################################################
# clean train data 					  						  #
###############################################################

def clean_train(df_train, df_weather, df_building, save = True):
	print('Removing outliers in training data.')
	df = join_data(df_train, df_building, df_weather)
	df = remove_unrepresentative_data(df)
	df = outlier_resolver(df, save = save)
	return df


###############################################################
# buildings metadata				  						  #
###############################################################

def clean_buildings(df_building, new_imputer_value = False, save = True):
	print('Resolving NaN values in building meta data.')
	# Replacing NaN values in year_built column with the median for the relevant meter combination.
	built_year_imputer(df_building)
	# It is easier to have na numerical values for building uses instead of strings.
	building_use_cat, building_use_label = df_building.primary_use.factorize()
	df_building.primary_use = building_use_cat
	# Resolve NaN values in the floor count and calculate the volume of the building as a feature.
	df_building['volume'] = volume(df_building)
	# save cleaned data
	if save == True:
		ls.save_csv_data(df_building, 'building_prepped.csv', reuse = True)
	return df_building

# which combination of meters is present in each building?
def create_meter_combinations(df_train, df_building, save = True):
	print('Creating a csv file with all building IDs and the present meters.')
	df_train = df_train.merge(df_building, on ='building_id', how='left')
	meter_comb = df_train.groupby(['building_id']).meter.unique().reset_index()
	meter_comb.rename(columns = {'meter': 'meter_group'}, inplace = True)
	for i in range(len(meter_comb)):
		meter_comb.loc[i, 'meter_group'] = 'group_' + str(sorted(meter_comb.meter_group[i]))
	if save == True:
		ls.save_csv_data(meter_comb, 'meter_combinations.csv', reuse = True)

# calculate mean building year for each meter group
def build_year_median_calc(df_train, df_building, save = True):
	print('Calculating median building year per meter combination, and saving to csv file.')
	df_meter_combos = ls.read_csv_data('meter_combinations.csv')
	df_train = df_train.merge(df_building, on ='building_id', how='left')
	df_train = df_train.merge(df_meter_combos, on ='building_id', how='left')
	df_train.dropna(subset=['year_built'], inplace = True)
	df_temp = df_train.groupby(['meter_group']).year_built.median().reset_index()
	if save == True:
		ls.save_csv_data(df_temp, 'meter_combos_and_median_reading.csv', reuse = True)

# inset missing year built values based on meter group
def built_year_imputer(df):
	df_imputer = ls.read_csv_data('meter_combos_and_median_reading.csv')
	df_meter_combos = ls.read_csv_data('meter_combinations.csv')
	counter = 0
	for i, row in df_imputer.iterrows():
		year = row.year_built
		rowsaffected = df_meter_combos[df_meter_combos.meter_group == row.meter_group].building_id
		masker = df.building_id.isin(rowsaffected)
		df['year_built'].mask(masker, df['year_built'].fillna(year), inplace = True)

# calculate the building volume
def volume(df):
	median_floor = df.floor_count.median()
	median_area = df.square_feet.median()
	df.floor_count.fillna(median_floor, inplace = True)
	return (df.square_feet *df.floor_count)


###############################################################
# weather data 						 						  #
###############################################################

def clean_weather(df, save = True, input = 'train'):
	print('Resolving NaN values in weather data and creating time attributes.')
	# solve NaN issues for all weather attributes
	df_temp = df.loc[df.air_temperature.isnull()]
	df.loc[df.air_temperature.isnull()] = weather_single_nan_imputer(df_temp, df.loc[df.timestamp.isin(df_temp.timestamp)], attrib = 'air_temperature')

	df_temp = df.loc[df.dew_temperature.isnull()]
	df.loc[df.dew_temperature.isnull()] = weather_single_nan_imputer(df_temp, df.loc[df.timestamp.isin(df_temp.timestamp)], attrib = 'dew_temperature')

	df_temp = df.loc[df.precip_depth_1_hr.isnull()]
	df.loc[df.precip_depth_1_hr.isnull()] = weather_single_nan_imputer(df_temp, df.loc[df.timestamp.isin(df_temp.timestamp)], attrib = 'precip_depth_1_hr')

	df_temp = df.loc[df.cloud_coverage.isnull()]
	df.loc[df.cloud_coverage.isnull()] = weather_single_nan_imputer(df_temp, df.loc[df.timestamp.isin(df_temp.timestamp)], attrib = 'cloud_coverage')

	df_temp = df.loc[df.wind_speed.isnull()]
	df.loc[df.wind_speed.isnull()] = weather_single_nan_imputer(df_temp, df.loc[df.timestamp.isin(df_temp.timestamp)], attrib = 'wind_speed')

	df_temp = df.loc[df.sea_level_pressure.isnull()]
	df.loc[df.sea_level_pressure.isnull()] = weather_single_nan_imputer(df_temp, df.loc[df.timestamp.isin(df_temp.timestamp)], attrib = 'sea_level_pressure')

	# resolve NaNs created by missing timestamps when merging the weather df to the train or test df
	if input == 'train':
		df_missing = ls.read_csv_data('train_nan.csv')
	if input == 'test':
		df_missing = ls.read_csv_data('test_nan.csv')
	df_missing = weather_nan_imputer(df_missing, df)
	df = pd.concat([df, df_missing])

	# for NaNs where there is also no data from neighbouring sites
	df.air_temperature.fillna(method='ffill', inplace = True)
	df.dew_temperature.fillna(method='ffill', inplace = True)
	df.wind_speed.fillna(method='ffill', inplace = True)
	df.precip_depth_1_hr.fillna(method='ffill', inplace = True)
	df.sea_level_pressure.fillna(method='ffill', inplace = True)
	df.cloud_coverage.fillna(method='ffill', inplace = True)

	df.air_temperature.fillna(method='bfill', inplace = True)
	df.dew_temperature.fillna(method='bfill', inplace = True)
	df.wind_speed.fillna(method='bfill', inplace = True)
	df.precip_depth_1_hr.fillna(method='bfill', inplace = True)
	df.sea_level_pressure.fillna(method='bfill', inplace = True)
	df.cloud_coverage.fillna(method='bfill', inplace = True)

	# add custom attributes
	df['time'] = pd.to_datetime(df['timestamp'])
	df['hour'] = df['time'][:].apply(lambda row: row.hour)
	df['month'] = df['time'][:].apply(lambda row: row.month)
	df['hour'] = df.apply(lambda x: remove_time_difference(x.hour, x.site_id), axis = 1)
	df['work'] = df.apply(lambda x: add_work_time(x), axis = 1)

	RH_calc(df)

	df = df.drop(['wind_direction'], axis = 1)

	# save cleaned data
	if (save == True) & (input == 'train'):
		ls.save_csv_data(df, 'weather_train_prepped.csv', reuse = True)
		return df
	if (save == True) & (input == 'test'):
		ls.save_csv_data(df, 'weather_test_prepped.csv', reuse = True)

# fill in nan values based on similar sites
def weather_nan_imputer(df_nan, df_total):
	site_group_USA_west = [1, 5, 12]
	site_group_USA_east = [3, 6, 7, 11, 14, 15]
	site_group_Europe1 = [2, 4, 10]
	site_group_Europe2 = [9, 13]
	missing_attribs = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']
	for index, row in df_nan.iterrows():
		for attrib in missing_attribs:
			if row.site_id in site_group_USA_west:
				group = site_group_USA_west
			elif row.site_id in site_group_Europe1:
				group = site_group_Europe1
			elif row.site_id in site_group_Europe2:
				group = site_group_Europe2
			else:
				group = site_group_USA_east
			if df_total[attrib][(df_total.site_id.isin(group)) & (df_total.timestamp == row.timestamp)].isnull().all():
				pass
			else:
				attrib_mean = np.nanmean(df_total[attrib][(df_total.site_id.isin(group)) & (df_total.timestamp == row.timestamp)])
				df_nan.loc[index, attrib] = attrib_mean
	return df_nan

# fill in nan values based on similar sites
def weather_single_nan_imputer(df_nan, df_total, attrib = 'sea_level_pressure'):
	site_group_USA_west = [1, 5, 12]
	site_group_USA_east = [3, 6, 7, 11, 14, 15]
	site_group_Europe1 = [2, 4, 10]
	site_group_Europe2 = [9, 13]
	for index, row in df_nan.iterrows():
		if row.site_id in site_group_USA_west:
			group = site_group_USA_west
		elif row.site_id in site_group_Europe1:
			group = site_group_Europe1
		elif row.site_id in site_group_Europe2:
			group = site_group_Europe2
		else:
			group = site_group_USA_east
		if 	df_total[attrib].loc[(df_total.site_id.isin(group)) & (df_total.timestamp == row.timestamp)].isnull().all():
			pass
		else:
			attrib_mean = np.nanmean(df_total[attrib].loc[(df_total.site_id.isin(group)) & (df_total.timestamp == row.timestamp)])
			df_nan.loc[index, attrib] = attrib_mean
	return df_nan

# remove the time difference between the east and west coast
def remove_time_difference(h, siteid):
	if siteid in [1, 5, 12]:
		if h <=20:
			return h + 3
		else:
			return h -21
	elif siteid in [2, 4, 10]:
		if h >= 3:
			return h - 3
		else:
			return h + 21
	elif siteid in [9, 13]:
		if h >= 2:
			return h - 2
		else:
			return h + 22		
	else:
		return h

# add a column with working day or free day to df_weather
def add_work_time(df):
	if df.time.weekday() in [5, 6]:
		return 0 # this represents a free day
	else:
		return 1 # this represents a working day

# calculate relative humidity
def RH_calc(df):
	TD = df.dew_temperature
	T = df.air_temperature
	df['RH']  = 100*(np.exp((17.625*TD)/(243.04+TD))/np.exp((17.625*T)/(243.04+T)))
	df = df.drop(['dew_temperature'], axis=1)


###############################################################
# create supporting csv files							  	  #  								  
###############################################################

def means_and_medians_creator(df_train, df_building, df_weather, save = True):
	df = join_data(df_train, df_building, df_weather)
	df['time'] = pd.to_datetime(df['timestamp'])
	df['month'] = df['time'][:].apply(lambda row: row.month)
	remove_unrepresentative_data(df)
	print('Creating csv file with mean and std of meter readings.')
	add_mean_reading(df, save = save)
	print('Creating csv file with median of meter readings each month.')
	add_median_reading_month(df, save = save)
	print('Creating csv file with median of meter readings based on non-zero readings only.')
	add_median_non_zero(df, save = save)

# create a csv file with the mean and std of the meter reading grouped by meter and building id
def add_mean_reading(df, save = True):
	mean_meter_reading = df.groupby(['meter', 'building_id']).meter_reading.mean().reset_index()
	mean_meter_reading.rename(columns={'meter_reading': 'mean_meter_reading_building'}, inplace = True)
	std_meter_reading = df.groupby(['meter', 'building_id']).meter_reading.std().reset_index()
	std_meter_reading.rename(columns={'meter_reading': 'std_meter_reading_building'}, inplace = True)
	if save == True:
		ls.save_csv_data(mean_meter_reading, 'mean_meter_reading.csv', reuse = True)
		ls.save_csv_data(std_meter_reading, 'std_meter_reading.csv', reuse = True)

# create a csv file with the median meter reading grouped by meter and building id based on only non-zero readings.
def add_median_non_zero(df, save = True):
	df['median_non_zero'] = 0
	df_temp2 = df[df.meter_reading > 1]
	for building in df_temp2.building_id.unique():
		df_temp1 = df_temp2[df_temp2.building_id == building]
		for meter in df_temp1.meter.unique():
			df_temp = df_temp1.meter_reading[df_temp1.meter == meter]
			df.loc[(df.building_id == building) & (df.meter == meter), 'median_non_zero'] = np.median(df_temp)
	df_save = df.groupby(['building_id', 'meter']).median_non_zero.mean().reset_index()
	if save == True:
		ls.save_csv_data(df_save, 'median_meter_reading_non_zero.csv', reuse = True)

# create a csv file with the median meter and inter quartile range for every meter for every month
def add_median_reading_month(df, save = True):
	median_meter_reading = df.groupby(['meter', 'building_id', 'month'])
	df_temp = median_meter_reading["meter_reading"].agg([np.median, iqr]).reset_index()
	if save == True:
		ls.save_csv_data(df_temp, 'median_meter_reading_month.csv', reuse = True)	

# create a csv file with which buildings have airconditioning
def add_airco(df, save = True):
	print('Creating a csv file with which buildings have airconditioning.')
	df['airco'] = 0
	df['summer'] = 0
	df['time'] = pd.to_datetime(df['timestamp'])
	df['month'] = df['time'][:].apply(lambda row: row.month)
	df.loc[(df.month > 5) & (df.month < 9), 'summer'] = 1
	df.loc[((df.month < 4)|(df.month > 10)), 'summer'] = -1
	df_temp = df.groupby(['summer', 'meter', 'building_id']).meter_reading.median().reset_index()
	df_temp2 = df_temp.loc[(df_temp.summer == 1) & (df_temp.meter == 0)]
	df_temp3 = df_temp.loc[(df_temp.summer == -1) & (df_temp.meter == 0)]
	df_temp2 = df_temp2.merge(df_temp3, on ='building_id', how='inner')
	df_airco_buildings = df_temp2.building_id.loc[df_temp2.meter_reading_x > df_temp2.meter_reading_y * 1.5]
	df.loc[df.building_id.isin(list(df_airco_buildings)), 'airco'] = 1
	df_save = df.groupby(['building_id']).airco.mean().reset_index()
	if save == True:
		ls.save_csv_data(df_save, 'airco.csv', reuse = True)	

# create a csv file with how much percent of the time meter readings are non-zero for each building
def add_meter_usage(df_train, df_building, df_weather, save = True):
	print('Creating a csv file with how much percent of the time meter readings are non-zero for each building.')
	df = join_data(df_train, df_building, df_weather)
	df['meter_usage'] = 0
	for building in df.building_id.unique():
		df_temp1 = df[df.building_id == building]
		for meter in df_temp1.meter.unique():
			df_temp = df_temp1[df_temp1.meter == meter]
			df.loc[(df.building_id == building) & (df.meter == meter), 'meter_usage'] = 100.0 * len(df_temp[df_temp.meter_reading > 1]) / len(df_temp)
	df_save = df.groupby(['building_id', 'meter']).meter_usage.mean().reset_index()
	if save == True:
		ls.save_csv_data(df_save, 'meter_usage.csv', reuse = True)	

# find timestamps missing in weather data
# WARNING: function should be applied to cleaned dataframes!
def find_weather_nans(df_train, df_building, df_weather, input = 'train'):
	df = join_data(df_train, df_building, df_weather)
	# here it does not matter which weather attribute is taken, these NaNs are created by missing timestamps
	nan_index = df.index[df['air_temperature'].apply(np.isnan)]
	df_nan = df.ix[nan_index]
	df_nan = df_nan.groupby(['site_id', 'timestamp']).air_temperature.mean().reset_index()
	df_export = pd.concat([df_nan['site_id'], df_nan['timestamp'],df_nan['air_temperature'], df_nan['air_temperature'], df_nan['air_temperature'], df_nan['air_temperature'], df_nan['air_temperature'], df_nan['air_temperature'],df_nan['air_temperature']], axis=1, keys=['site_id', 'timestamp', 
		'air_temperature', 'cloud_coverage','dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed'])
	if input == 'train':	
		ls.save_csv_data(df_export, 'train_nan.csv', reuse = True)	
	if input == 'test':
		ls.save_csv_data(df_export, 'test_nan.csv', reuse = True)	


###############################################################
# join dataframes 											  #
###############################################################
# the actual joining
def join_data(df_train, df_building, df_weather):
	df_train = df_train.merge(df_building, on ='building_id', how='left')
	df_train = df_train.merge(df_weather, on =['site_id', 'timestamp'], how='left')
	return df_train

def add_features(df, input = 'train'):
	# Resolve NaN values
	if input == 'train':
		df.dropna(inplace = True)
	# df= outlier_resolver(df)
	# Add extra features
	df_mean_reading = ls.read_csv_data('mean_meter_reading.csv')
	print('Adding mean meter reading per building.')
	df = df.merge(df_mean_reading, on = ['building_id', 'meter'], how = 'left')
	# df_median_reading_site = ls.read_csv_data('median_meter_reading_site.csv')
	# print('Adding median meter reading per site.')
	# df = df.merge(df_median_reading_site, on = ['site_id', 'meter'], how = 'left')
	df_airco = ls.read_csv_data('airco.csv')
	print('Adding airco feature.')
	df = df.merge(df_airco, on = ['building_id'], how = 'left')

	df_meter_usage = ls.read_csv_data('meter_usage.csv')
	print('Adding meter usage feature.')
	df = df.merge(df_meter_usage, on = ['building_id', 'meter'], how = 'left')

	# df_median = ls.read_csv_data('median_meter_reading_non_zero.csv')
	# print('Adding non-zero median feature.')
	# df = df.merge(df_median, on = ['building_id', 'meter'], how = 'left')

	df_std = ls.read_csv_data('std_meter_reading.csv')
	print('Adding meter reading std feature.')
	df = df.merge(df_std, on = ['building_id', 'meter'], how = 'left')

	print("Adding feature that summarizes the influence of the weather.")
	df = model_weather_influence(df, input = 'predict')
	return df

def remove_unrepresentative_data(df):
	site0_unrepresentative_idx = df.query('(site_id == 0) and (meter == 0) and (month < 6)').index
	df.drop(index = site0_unrepresentative_idx, inplace = True)
	site7_unrepresentative_idx = df.query('(building_id == 799 or building_id == 802 or building_id == 803) and month < 10').index
	df.drop(index = site7_unrepresentative_idx, inplace = True)
	building1232_unrepresentative_idx = df.query('(building_id == 1232) and (meter == 1) and (month > 5) and (month < 9)').index
	df.drop(index = building1232_unrepresentative_idx, inplace = True)
	return df

def outlier_resolver(df, save = True):
	df_median_reading = ls.read_csv_data('median_meter_reading_month.csv')
	meter_factor = [[1.7, 0.6], [2.5, 0.8], [3.0, 1.0], [1.5, 1.0]]
	todrop = []
	for building in df_median_reading.building_id.unique():
		df_temp1 = df[df.building_id == building]
		for meter in df_median_reading.meter[df_median_reading.building_id == building].unique():
			if meter == 1 | meter == 2 | meter ==3:
				continue
			df_temp2 = df_temp1[df_temp1.meter == meter]
			for month in df_median_reading.month[(df_median_reading.building_id == building) & (df_median_reading.meter == meter)].unique():
				df_temp3 = df_temp2[df_temp2.month == month]
				median_reading = df_median_reading['median'][(df_median_reading.building_id == building) & (df_median_reading.meter == meter) & (df_median_reading.month == month)]
				interq_range = df_median_reading['iqr'][(df_median_reading.building_id == building) & (df_median_reading.meter == meter) & (df_median_reading.month == month)]
				max_reading = int(median_reading + meter_factor[meter][0] * interq_range)
				min_reading = int(median_reading - meter_factor[meter][1] * interq_range)
				rowsaffected = df_temp3[(df_temp3.meter_reading < min_reading) | (df_temp3.meter_reading > max_reading)].index
				todrop.append(rowsaffected)
	todrop = np.concatenate(todrop)
	df.drop(todrop, inplace = True)
	df_export = pd.concat([df['building_id'], df['meter'], df['timestamp'], df['meter_reading']], axis=1)
	if save == True:
		ls.save_csv_data(df_export, 'train_prepped.csv', reuse = True)
	return df_export


def scaling_and_d_reduction(df, percent = 0.98, scale = True, pca = True, input = 'val'):
	if scale == True:
		if input == 'train':
			print("Scaling training data.")
			scaler = StandardScaler()
			scaler.fit_transform(df)
			ls.save_model(scaler, 'scaler.pkl')
		elif input == 'val':
			scaler = ls.load_model('scaler.pkl')
			scaler.transform(df)
		elif input == 'plot':
			scaler = ls.load_model('scaler.pkl')
			scaler.transform(df.drop(['timestamp' ,'time'], axis = 1))
		else:
			print('The argument "input" should be equal to "train" or "val".')
			print('Printed from "scaling_and_d_reduction" in "ashrae_module".')
			return exit()
	if pca == True:
		print('Number of attributes before PCA:' + str(len(df.columns)))
		if input == 'train':
			pca = PCA(n_components = percent)
			df = pca.fit_transform(df)
			ls.save_model(pca, 'pca.pkl')
		elif input == 'val':
			pca = ls.load_model('pca.pkl')
			df = pca.transform(df)
		print(df)
		print('Number of attributes after PCA:' + str(df.shape))
	return df

def model_weather_influence(df, input = 'train'):
	print('Training a small model to summarize the influence of the weather.')
	filename = 'weather_tree_reg.sav'
	df_mean_reading = ls.read_csv_data('mean_meter_reading.csv')
	if input == 'train':
		df['time'] = pd.to_datetime(df['timestamp'])
		df['day'] = df['time'][:].apply(lambda row: row.strftime('%j'))
		df = df.merge(df_mean_reading, on = ['building_id', 'meter'], how = 'left')
		df['relative_reading'] = df['meter_reading'] / df['mean_meter_reading_building']
		df_temp1 = pd.concat([df['relative_reading'], df['site_id'], df['meter'], df['day'], df['air_temperature'], df['RH'], df['wind_speed'], df['sea_level_pressure'], df['precip_depth_1_hr'], df['cloud_coverage']], axis=1)
		df_temp = df_temp1.groupby(['site_id', 'meter', 'day']).agg(np.mean).reset_index()
		target = df_temp['relative_reading']
		df_temp = df_temp.drop(['relative_reading', 'day'], axis = 1)
		weather_tree_reg = DecisionTreeRegressor()
		weather_tree_reg.fit(df_temp, target)
		joblib.dump(weather_tree_reg, filename)
	elif input == 'predict':
		weather_tree_reg = joblib.load(filename)
		df_fit = pd.concat([df['site_id'], df['meter'], df['air_temperature'], df['RH'], df['wind_speed'], df['sea_level_pressure'], df['precip_depth_1_hr'], df['cloud_coverage']], axis=1)
		df['reading_weather_based'] = weather_tree_reg.predict(df_fit)
		ls.save_csv_data(df_fit.head(), 'for_columns_weather.csv', reuse = True)
		return df
	else:
		print('Argument "input" should be equal to "train" or "val".')


###############################################################
# other														  #
###############################################################

def roundup(x):
    return int(math.ceil(x / 10.0)) * 10


###############################################################
# work in progress											  #
###############################################################

# for each primary use, calculate the ratio between the meter readings on work days and weekends
# def add_primary_use_work(df, save = True):
# 	df_mean_reading = ls.read_csv_data('mean_meter_reading.csv')
# 	df = df.merge(df_mean_reading, on = ['building_id', 'meter'], how = 'left')
# 	df['meter_reading_ratio'] = df['meter_reading'] / df['mean_meter_reading_building']
# 	df_temp1 = df.groupby(['primary_use', 'meter', 'work']).meter_reading_ratio.mean().reset_index()
# 	df_temp2 = df.groupby(['primary_use', 'meter']).meter_reading_ratio.mean().reset_index()
# 	df_temp2["usage_work_ratio"] = 0
# 	for i in df_temp1.primary_use.unique():
# 		df_temp3 = df_temp1.loc[df_temp1.primary_use == i]
# 		for j in df_temp3.meter.unique():
# 			print(df_temp1.meter_reading_ratio[(df_temp1.meter == j) & (df_temp1.primary_use == i) & (df_temp1.work == 0)] )
# 			print(df_temp2.loc[(df_temp2.meter == j) & (df_temp2.primary_use == i), 'usage_work_ratio'])
# 			#df_temp2.loc[(df_temp2.meter == j) & (df_temp2.primary_use == i), 'usage_work_ratio'] = df_temp1.meter_reading_ratio[(df_temp1.meter == j) & (df_temp1.primary_use == i) & (df_temp1.work == 0)] / df_temp1.meter_reading_ratio[(df_temp1.meter == j) & (df_temp1.primary_use == i) & (df_temp1.work == 1)]
# 			df_temp2.loc[(df_temp2.meter == j) & (df_temp2.primary_use == i), 'usage_work_ratio'] = df_temp1.meter_reading_ratio[(df_temp1.meter == j) & (df_temp1.primary_use == i) & (df_temp1.work == 0)] 
# 	print(df_temp2)
# 	if save == True:
# 		ls.save_csv_data(df_temp2, 'primary_use_work.csv', reuse = True)	





