###############################################################
#														      #
# ASHREA great energy predictor III           				  # 
# by Fanny Ummelen, 2019-11-05								  #
#															  #	
###############################################################


###############################################################
# import python essentials and custom modules				  #
###############################################################

import pandas as pd
import numpy as np 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import json

import loading_and_saving as ls
import ashrae_module as ashrae
import ashrae_visualizations as visualizations

###############################################################
# configuration												  #
###############################################################

# load configuration file
with open('config.json') as f:
	config = json.load(f)

# define the model and hyperparameters
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(min_samples_leaf=100)

# from sklearn.neighbors import KNeighborsRegressor
# model = KNeighborsRegressor(n_neighbors = 5, weights = "distance")

# from sklearn.linear_model import SGDRegressor
# model = SGDRegressor()


###############################################################
# data cleaning												  #
###############################################################

if config.get('clean_data_from_scratch').get('enable'):
	# load original csv files into panda dataframes
	print('Loading original data files.')
	df_building = ls.read_csv_data('building_metadata.csv')
	df_train = ls.read_csv_data('train.csv')
	df_weather = ls.read_csv_data('weather_train.csv')


	# calculate some additional features and save them to csv files to save time for future runs of this script
	if config.get('clean_data_from_scratch').get('options').get('make_supporting_csvs'):
		# create a csv file with all building IDs and the present meters
		ashrae.create_meter_combinations(df_train, df_building)
		# calculate median building year per meter combination, and saving to csv file
		ashrae.build_year_median_calc(df_train, df_building)
		# create a csv file with the mean and std of the meter reading grouped by meter and building id
		# create a csv file with the median meter and inter quartile range for every meter for every month
		# create a csv file with the median meter reading grouped by meter and building id based on only non-zero readings
		ashrae.means_and_medians_creator(df_train, df_building, df_weather)
		# create a csv file with who much percent of the time meter readings are non-zero for each building
		ashrae.add_meter_usage(df_train, df_building, df_weather, save = True)
		# find timestamps missing in weather data
		ashrae.find_weather_nans(df_train, df_building, df_weather, input = 'train')
		# determine which of the buildings have airconditioning
		df_total = ashrae.join_data(df_train, df_building, df_weather)
		ashrae.add_airco(df_total, save = True)

	# resolve NaN values in the building data and save the cleaned data to csv files
	df_building = ashrae.clean_buildings(df_building, save = True)
	# resolve NaN values in the weather data and add time attributes
	df_weather = ashrae.clean_weather(df_weather, save = True, input = 'train')
	# remove outliers in the meter readings
	df_train = ashrae.clean_train(df_train, df_weather, df_building, save = True)

	# calculate additional features that require joined, processed data, and save them to csv files
	if config.get('clean_data_from_scratch').get('options').get('make_supporting_csvs'):
		df_total = ashrae.join_data(df_train, df_building, df_weather)
		# build a model to combine the influence of the individual weather attributes on the meter reading
		ashrae.model_weather_influence(df_total, input = 'train')

###############################################################
# join data frames and add custom features 					  #
###############################################################

# to save time, we can directly load the cleaned data and comment out the previous section.
print('Loading csv files with cleaned data.')
df_building = ls.read_csv_data('building_prepped.csv')
df_train = ls.read_csv_data('train_prepped.csv')
df_weather = ls.read_csv_data('weather_train_prepped.csv')

# join data frames
df_total = ashrae.join_data(df_train, df_building, df_weather)

# add features from separate csv files to the total data frame
df_total = ashrae.add_features(df_total)


###############################################################
# data visualization 										  #
###############################################################

if config.get('make_visualizations').get('enable'):
	if config.get('make_visualizations').get('type').get('meter_reading_boxplots').get('enable'):
		visualizations.meter_reading_boxplots(df_train)
	if config.get('make_visualizations').get('type').get('violin_plots').get('enable'):
		visualizations.violins_per_site(df_weather, property = config.get('make_visualizations').get('plot_property'))
	if config.get('make_visualizations').get('type').get('weather_heatmap').get('enable'):
		visualizations.cor_heatmap(df_weather, property = config.get('make_visualizations').get('plot_property'))
	if config.get('make_visualizations').get('type').get('temperature_evolution_during_day').get('enable'):
		visualizations.time_plots_temperature(df_weather)
	if config.get('make_visualizations').get('type').get('plot_meter_reading_vs_time').get('enable'):
		visualizations.time_plot_meter_reading(df_total, building_id = config.get('make_visualizations').get('building_id'), meter = config.get('make_visualizations').get('meter'), start_time = config.get('make_visualizations').get('start_time'), end_time = config.get('make_visualizations').get('end_time'))
	if config.get('make_visualizations').get('type').get('time_plots_per_meter_and_site').get('enable'):
		visualizations.time_plots_per_meter_and_site(df_total)
	if config.get('make_visualizations').get('type').get('plot_raw_and_cleaned_meter_readings').get('enable'):
		df_train = ls.read_csv_data('train.csv')
		visualizations.raw_vs_cleaned(df_train, df_total, building_id = config.get('make_visualizations').get('building_id'), meter = config.get('make_visualizations').get('meter'), start_time = config.get('make_visualizations').get('start_time'), end_time = config.get('make_visualizations').get('end_time'))
	if config.get('make_visualizations').get('type').get('plot_primary_use_per_site').get('enable'):
		visualizations.primary_use_per_site(df_building)


###############################################################
# prepare data for model training	       					  #
###############################################################

if config.get('train_model').get('enable'):
	# select the features that will be used to train the model and split the data in a training and validation set
	df_total_features = df_total[config.get('train_model').get('features')]

	X_train, X_val = train_test_split(df_total_features, test_size = config.get('train_model').get('relative_size_validation_set'))
	
	# split the data into target and other features:
	y_train = X_train.meter_reading
	y_val = X_val.meter_reading
	X_train = X_train.drop(['meter_reading'], axis = 1)
	X_val = X_val.drop(['meter_reading'], axis = 1)

	# save the names of all features to a csv file (because the names will be removed in the next step, and we need them later to evaluate the feature importances)
	ls.save_csv_data(X_train.drop(['meter'], axis = 1).head(), 'for_columns.csv', reuse = True)

	# For some ML techniques it is required that the data is scaled and that the dimensionality is not too high.
	X_train_reduced = ashrae.scaling_and_d_reduction(X_train.drop('meter', axis = 1), percent = 0.98, input = 'train', pca = config.get('train_model').get('pca'))
	X_val_reduced = ashrae.scaling_and_d_reduction(X_val.drop('meter', axis = 1), percent = 0.98, input = 'val', pca = config.get('train_model').get('pca'))


###############################################################
# model training 		       								  #
###############################################################

if config.get('train_model').get('enable'):
	for meter in range(4):
		print('Training model for meter ' + str(meter) +'.')
		X_train_per_meter = X_train_reduced.loc[X_train.meter == meter]
		X_val_per_meter = X_val_reduced.loc[X_val.meter == meter]
		y_train_per_meter = y_train.loc[X_train.meter == meter]
		y_val_per_meter = y_val.loc[X_val.meter == meter]
		y_train_per_meter = np.log(y_train_per_meter + 1)
		model.fit(X_train_per_meter, y_train_per_meter)
		y_val_predict_log = model.predict(X_val_per_meter)
		y_val_predict = np.expm1(y_val_predict_log)
		y_val_predict[y_val_predict < 0] = 0
		print('RMSLE for validation set for meter ' + str(meter) + ':')
		print(np.sqrt(mean_squared_log_error(y_val_predict, y_val_per_meter)))
		filename = config.get('train_model').get('model_name') +'_meter_' + str(meter) + '.sav'
		joblib.dump(model, filename)


###############################################################
# model evaluation 					       					  #
###############################################################

if config.get('train_model').get('enable'):
	if config.get('train_model').get('check_feature_importances'):
	# check feature importance of decission tree model
		df_temp = ls.read_csv_data('for_columns.csv')
		temp_list = list(df_temp.columns) 

		for i in range(4):
			filename = model_type + '_meter_' + str(i) + '.sav'
			loaded_model = joblib.load(filename)
			for name, score in zip(temp_list, loaded_model.feature_importances_):
				print(name, score)

	if config.get('train_model').get('make_learning_curves'):
		X_train_per_meter = X_train_reduced.loc[X_train.meter == config.get('train_model').get('meter_learning_curves')]
		X_val_per_meter = X_val_reduced.loc[X_val.meter == config.get('train_model').get('meter_learning_curves')]
		y_train_per_meter = y_train.loc[X_train.meter == config.get('train_model').get('meter_learning_curves')]
		y_val_per_meter = y_val.loc[X_val.meter == config.get('train_model').get('meter_learning_curves')]
		
		visualizations.plot_learning_curves(model, X_train_per_meter, X_val_per_meter, y_train_per_meter, y_val_per_meter)


