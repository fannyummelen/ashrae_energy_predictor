# check feature importance of forest
import joblib
import loading_and_saving as ls
import numpy as np 

filename = 'tree_reg.sav'
loaded_model = joblib.load(filename)
df_temp = ls.read_csv_data('for_columns.csv')
temp_list = list(df_temp.columns) 

for i in range(4):
	filename = 'tree'+str(i)+'_reg.sav'
	loaded_model = joblib.load(filename)
	for name, score in zip(temp_list, loaded_model.feature_importances_):
		print(name, score)


# filename = 'weather_tree_reg.sav'
# loaded_model = joblib.load(filename)
# df_temp = ls.read_csv_data('for_columns_weather.csv')
# temp_list = list(df_temp.columns) 
# for name, score in zip(temp_list, loaded_model.feature_importances_):
# 	print(name, score)