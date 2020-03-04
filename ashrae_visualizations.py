###############################################################
# visualization functions for the ASHRAE project			  #	
###############################################################

import loading_and_saving as ls
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_log_error
import math
# from datetime import datetime
# import math
# import joblib


###############################################################
# Visualizations											  #
###############################################################

def plot_style():
	sns.set_context(context = 'talk')
	plt.style.use(['dark_background'])
	palette = 'Set1'
	color_list = sns.color_palette('Set1', n_colors=9) + sns.color_palette('Set3', n_colors=12)
	return palette, color_list

def time_plots_per_meter_and_site(df):
	palette, color_list = plot_style()
	# because we plot the data on a log scale, zero values need to be removed
	df[df.meter_reading.round() == 0] = 0.1
	meter_type_string = [' Electricity', ' Chilled Water', ' Steam', ' Hot Water']
	for investigated_site in range(16):
		for meter_type in range(4):
			df_temp = df[(df.meter == meter_type) & (df.site_id == investigated_site)]
			df_temp2 = df_temp.groupby(['month', 'building_id']).meter_reading.mean().reset_index()
			fig = plt.figure(figsize = (10, 8))
			j=0
			for i in df_temp2.building_id.unique():
				plt.plot(df_temp2[df_temp2.building_id == i].month, df_temp2[df_temp2.building_id == i].meter_reading)
				j += 1
				if j > 10:
					break
			plt.xlabel('Month')
			plt.yscale('log')
			plt.xticks(rotation = 90)
			plt.ylabel('Meter Reading')
			plt.title('Site ' + str(investigated_site) + meter_type_string[meter_type])
			plt.legend(df_temp.building_id.unique(), bbox_to_anchor=(1.025,1.025), loc="upper left")
			plt.subplots_adjust(left = 0.2, right = 0.8, bottom = 0.25)
			ls.save_figure('meter' + str(meter_type) + '_site'+ str(investigated_site) +'.png')
			plt.show()
	df[df.meter_reading == 0.1] = 0

def meter_reading_boxplots(df):
	palette, color_list = plot_style()
	g = sns.catplot(x="meter_reading", row="meter",
	                kind="box", orient="h", height=1.5, aspect=4,
	                data=df,
	                color = color_list[1])
	g.set(xscale="log");
	plt.xlabel('Meter Reading')
	ls.save_figure('meter_reading_boxplots.png')
	plt.show()

def time_plots_temperature(df_weather):
	palette, color_list = plot_style()
	site_id_label = [str(i) for i in range(df_weather.site_id.nunique())]
	fig = plt.figure(figsize = (10, 8))
	for i in range(df_weather.site_id.nunique()):
		df_temp = df_weather[df_weather.site_id == i]
		plt.plot(df_temp.groupby(['hour']).air_temperature.mean(), color = color_list[i])
	plt.xlabel('Time in hour')
	plt.ylabel('Air Temperature')
	plt.title('Time Evolution of Temperature at Different Sites')
	plt.legend(site_id_label)
	plt.legend(site_id_label, bbox_to_anchor=(1.025,1.025), loc="upper left", ncol=1)
	plt.subplots_adjust(left = 0.1, right = 0.7, bottom = 0.25)
	ls.save_figure('temp_evolution_per_site_id.png')
	plt.show()

def time_plot_meter_reading(df, building_id = 0, meter = 0, start_time = '2016-01-01', end_time = '2016-12-31'):
	df['time'] = pd.to_datetime(df['timestamp'])
	start_time = datetime.strptime(start_time, '%Y-%m-%d')
	end_time = datetime.strptime(end_time, '%Y-%m-%d')
	df = df[(df.time >= start_time) & (df.time < end_time)]

	palette, color_list = plot_style()
	fig = plt.figure(figsize = (8, 8))
	df_temp = df[(df.building_id == building_id) & (df.meter == meter)]
	plt.plot(df_temp.time, df_temp.meter_reading)
	plt.xlabel('Time')
	plt.xticks(rotation = 90)
	plt.ylabel('Meter Reading')
	plt.subplots_adjust(left = 0.1, right = 0.7,bottom = 0.25)
	plt.title('Building: '+ str(building_id) + ' Meter: ' + str(meter))
	ls.save_figure('Building'+ str(building_id) + '_Meter' + str(meter) + '.png')
	plt.show()

def raw_vs_cleaned(df_raw, df_cleaned, building_id = 0, meter = 0, start_time = '2016-01-01', end_time = '2016-12-31'):
	df_raw = df_raw[(df_raw.building_id == building_id) & (df_raw.meter == meter)]
	df_raw['time'] = pd.to_datetime(df_raw['timestamp'])
	df_cleaned = df_cleaned[(df_cleaned.building_id == building_id) & (df_cleaned.meter == meter)]
	df_cleaned['time'] = pd.to_datetime(df_cleaned['timestamp'])
	start_time = datetime.strptime(start_time, '%Y-%m-%d')
	end_time = datetime.strptime(end_time, '%Y-%m-%d')
	df_raw = df_raw[(df_raw.time >= start_time) & (df_raw.time < end_time)]
	df_cleaned = df_cleaned[(df_cleaned.time >= start_time) & (df_cleaned.time < end_time)]
	
	palette, color_list = plot_style()
	fig = plt.figure(figsize = (8, 8))
	plt.plot(df_raw.time, df_raw.meter_reading, color = color_list[0])
	plt.plot(df_cleaned.time, df_cleaned.meter_reading, '--',color = color_list[1])
	plt.xlabel('Time')
	plt.xticks(rotation = 90)
	plt.ylabel('Meter Reading')
	plt.legend(['raw', 'cleaned'])
	plt.subplots_adjust(left = 0.15,bottom = 0.25)
	plt.title('Building: ' + str(building_id) + ' Meter: ' + str(meter) + ' actual reading vs prediction.')
	ls.save_figure('Building' + str(building_id) + '_Meter' + str(meter) + 'actual_and_predicted.png')
	plt.show()	

# plot distributions per site
def violins_per_site(df, property = 'year_built'):
	palette, color_list = plot_style()
	fig = plt.figure(figsize = ([20, 5]))
	sns.violinplot(data = df.dropna(subset=[property]),
	              x = 'site_id',
	              y = property,
	              palette = palette)
	plt.subplots_adjust(left=0.2, bottom=0.2)
	plt.xlabel('Site ID')
	plt.ylabel(property)
	ls.save_figure(str(property) + '_per_site_id.png')
	plt.show()

# also check the building use distribution per site
def primary_use_per_site(df_building):
	palette, color_list = plot_style()
	fig, ax = plt.subplots(figsize = ([20, 8]))
	building_temp = [0] * 16
	bottom_temp = [0] * 16
	ind = list(range(16))
	for j in range(len(df_building.primary_use.unique())):
		for i in range(16):
			building_temp[i] = len(df_building[(df_building.primary_use == j) & (df_building.site_id == i)])
		plt.bar(ind, building_temp, bottom=bottom_temp, color = color_list[j])
		bottom_temp = np.add(bottom_temp, building_temp)
	plt.subplots_adjust(right=0.7)
	plt.ylabel('Counts')
	plt.xlabel('Site ID')
	ax.set_xticks(ind)
	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	# Put a legend to the right of the current axis
	ax.legend(df_building.primary_use.unique(), loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 14}, ncol = 2)
	ls.save_figure('primary_use_per_site.png')
	plt.show()

# calculate the correlation matrix
def cor_heatmap(df, property = 'air_temperature'):
	palette, color_list = plot_style()
	df_temp = df.pivot(index='timestamp', columns='site_id', values=property)
	corr = df_temp.corr()
	color_list_heat = sns.color_palette('Blues_r')
	sns.heatmap(corr, 
	        xticklabels=corr.columns,
	        yticklabels=corr.columns,
	        cmap = color_list_heat)
	plt.title(str(property) + '_heatmap')
	ls.save_figure(str(property) + '_heatmap.png')
	plt.show()

def plot_learning_curves(model, X_train, X_val, y_train, y_val):
	palette, color_list = plot_style()
	train_errors, val_errors = [], []
	y_train_log = np.log(y_train + 1)
	print("Calculating learning curves.")
	for m in range (1000, len(X_train), math.floor(len(X_train) / 10)):
		model.fit(X_train[:m], y_train_log[:m])
		y_train_log_predict = model.predict(X_train[:m])
		y_val_log_predict = model.predict(X_val)
		y_train_predict = np.expm1(y_train_log_predict)
		y_val_predict = np.expm1(y_val_log_predict)
		train_errors.append(mean_squared_log_error(y_train[:m], y_train_predict))
		val_errors.append(mean_squared_log_error(y_val, y_val_predict))
	palette, color_list = plot_style()
	fig = plt.figure(figsize = (8, 8))
	plt.plot(np.sqrt(train_errors), color = color_list[0], label = 'train')
	plt.plot(np.sqrt(val_errors), color = color_list[1], label = 'val')
	plt.ylabel('RMSLE')
	plt.xlabel('training set size (a.u.)')
	plt.legend(['training set', 'validation set'])
	ls.save_figure('learning_curve.png')
	plt.show()
