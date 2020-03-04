###############################################################
#														      #
# ASHREA great energy predictor III           				  # 
# predictions on the test data 								  #
# by Fanny Ummelen, 2019-11-05								  #
#															  #	
###############################################################

###############################################################
# import python essentials, model, and load csv files		  #
###############################################################

import loading_and_saving as ls
import ashrae_module as ashrae

import pandas as pd
import numpy as np 
from sklearn.tree import DecisionTreeRegressor
import joblib
import json


###############################################################
# configuration												  #
###############################################################

# load configuration file
with open('config.json') as f:
	config = json.load(f)

# load the trained model for each meter
loaded_model = []
for meter in range(4):
	filename = config.get('train_model').get('model_name') +'_meter_' + str(meter) + '.sav'
	loaded_model.append(joblib.load(filename))

# load the csv files for the test set
df_building = ls.read_csv_data('building_prepped.csv')
df_test = ls.read_csv_data('test.csv')
# df_weather = ls.read_csv_data('weather_test.csv')
df_weather = ls.read_csv_data('weather_test_prepped.csv')
# df_submission = ls.read_csv_data('sample_submission.csv')

# split the test data into two:
split_test = round((len(df_test)/2))

###############################################################
# create csv with missing weather timestamps				  #
###############################################################

# df_test_2018 = df_test.iloc[split_test:,:]
# del df_test
# df_weather.air_temperature.fillna(method='ffill', inplace = True)
# print(df_weather.isna().sum())
# ashrae.find_weather_nans(df_test_2018, df_building, df_weather, input = 'test')

# print('Cleaning weather data.')
# ashrae.clean_weather(df_weather, save = True, input = 'test')
# exit()

###############################################################
# 2017 pipeline												  #
###############################################################

df_test_2017 = df_test.iloc[:split_test, :]
del df_test

print('Merging dataframes 2017.')
df_total_2017 = ashrae.join_data(df_test_2017, df_building, df_weather)
print('Adding custom features.')
df_total_2017 = ashrae.add_features(df_total_2017)
features = config.get('train_model').get('features')
features.append('row_id')
features.remove('meter_reading')
features2017 = df_total_2017[features]
print('Scaling 2017 data.')
X_test_reduced_2017 = ashrae.scaling_and_d_reduction(features2017.drop(['row_id', 'meter'], axis = 1), percent = 0.98, input = 'val', pca = config.get('train_model').get('pca'))
X_test_reduced_2017['row_id'] = df_total_2017.row_id

print('Making predictions for 2017.')
X_test_reduced_2017['predictions'] = 0

for meter in range(4):
	X_test_reduced_2017.loc[df_total_2017.meter == meter, 'predictions'] = loaded_model[meter].predict(X_test_reduced_2017.loc[df_total_2017.meter == meter].drop(['predictions', 'row_id'], axis = 1))

X_test_reduced_2017.predictions = np.expm1(X_test_reduced_2017.predictions)
print('Removing negative predictions.')
X_test_reduced_2017.loc[X_test_reduced_2017.predictions < 0, 'predictions'] = 0
print('Putting 2017 results in submission file.')
submission2017 = X_test_reduced_2017[['row_id', 'predictions']]

del df_total_2017, X_test_reduced_2017, features2017

###############################################################
# 2018 pipeline												  #
###############################################################

df_test = ls.read_csv_data('test.csv')
df_test_2018 = df_test.iloc[split_test:,:]
del df_test

print('Merging dataframes 2018.')
df_total_2018 = ashrae.join_data(df_test_2018, df_building, df_weather)
print('Adding custom features.')
df_total_2018 = ashrae.add_features(df_total_2018)
features2018 = df_total_2018[features]
print('Scaling 2018 data.')
X_test_reduced_2018 = ashrae.scaling_and_d_reduction(features2018.drop(['row_id', 'meter'], axis = 1), percent = 0.98, input = 'val', pca = config.get('train_model').get('pca'))
X_test_reduced_2018['row_id'] = df_total_2018.row_id

print('Making predictions.')
X_test_reduced_2018['predictions'] = 0

for meter in range(4):
	X_test_reduced_2018.loc[df_total_2018.meter == meter, 'predictions'] = loaded_model[meter].predict(X_test_reduced_2018.loc[df_total_2018.meter == meter].drop(['predictions', 'row_id'], axis = 1))

X_test_reduced_2018.predictions = np.expm1(X_test_reduced_2018.predictions)
print('Removing negative predictions.')
X_test_reduced_2018.loc[X_test_reduced_2018.predictions < 0, 'predictions'] = 0
print('Putting 2018 results in submission file.')
submission2018 = X_test_reduced_2018[['row_id', 'predictions']]

print('Saving results to submission file.')
df_submission = pd.concat([submission2017, submission2018])
df_submission.rename(columns={'predictions': 'meter_reading'}, inplace = True)
df_submission.to_csv('energy_prediction.csv', index = False)
print('Done!')