import os
import pandas as pd
from matplotlib import pyplot as plt
import joblib

main_path =  os.path.dirname(os.path.realpath(__file__))
input_folder = 'csv_input'
output_folder_csv = 'csv_output'
output_folder_png = 'png_output'
model_folder = 'models'

def read_csv_data(filename):
	csv_path = os.path.join(main_path, input_folder, filename)
	if not os.path.isfile(csv_path):
		print('There is no file by the name "'+ str(filename) +'" in directory "' + str(os.path.join(main_path, input_folder)) +'".')
		print('Printed from "read_csv_data" in "loading_and_saving".')
		return exit()
	else:
		return pd.read_csv(csv_path)

def save_csv_data(data, filename, reuse = False):
	if reuse == True:
		directory = input_folder
	else:
		directory = output_folder_csv
	csv_path = os.path.join(main_path, directory) 
	if not os.path.isdir(csv_path):
		os.mkdir(csv_path)
		print('The directory to which you try to save your data does not exits yet, hence it is created.')
		print('Printed from "save_csv_data" in "loading_and_saving".')	
	return data.to_csv(os.path.join(csv_path, filename), index = False)

def save_figure(filename):
	csv_path = os.path.join(main_path, output_folder_png) 
	if not os.path.isdir(csv_path):
		os.mkdir(csv_path)
		print('The directory to which you try to save your figure does not exits yet, hence it is created.')
		print('Printed from "save_figure" in "loading_and_saving".')
	return plt.savefig(os.path.join(csv_path, filename), transparent=True)

def save_model(model, filename):
	csv_path = os.path.join(main_path, model_folder)
	if not os.path.isdir(csv_path):
		os.mkdir(csv_path)
		print('The directory to which you try to save your model does not exits yet, hence it is created.')
		print('Printed from "save_model" in "loading_and_saving".')
	return joblib.dump(model, os.path.join(csv_path, filename)) 

def load_model(filename):
	csv_path = os.path.join(main_path, model_folder, filename)
	if not os.path.isfile(csv_path):
		print('There is no file by the name "'+ str(filename) +'" in directory "' + str(os.path.join(main_path, model_folder)) +'".')
		print('Printed from "load_model" in "loading_and_saving".')
	return joblib.load(os.path.join(csv_path))


