import pandas as pd
import numpy as np
import os
import csv 
import sys
import random
sys.path.append("resources/")
from cv_tools import generate_validation_data

def run_base_script(worker_file, categories_file): 
	name = "bin\\get-another-label.bat --categories " + str(categories_file) + " --input " + str(worker_file)
	print(name)
	os.system(name)

def run_complete_scrip(worker_file, categories_file, golden_file, eval_file): 
	name = "bin\\get-another-label.bat --gold " + str(golden_file) + " --eval " + str(eval_file) + " --categories " + str(categories_file) + " --input " + str(worker_file)
	print(name)
	os.system(name)

def process_cv_predictions(worker_data_file, cv_predictions_file, username = "Computer-Vision-Model"): 
	cv_pred = []
	with open(cv_predictions_file) as csvfile: 
		csvreader = csv.reader(csvfile, delimiter=',') 
		for row in csvreader: 
			pano_id = row[0]
			sv_x = row[1]
			sv_y = row[2]
			name = str(pano_id) + "," + str(sv_x) + "," + str(sv_y)
			cv_label = row[3]
			user_label = row[4]
			end_label = user_label
			if cv_label != user_label: 
				end_label = "Not-" + user_label
			line = username + "\t" + name + "\t" + end_label + "\n"
			cv_pred.append(line)
	with open(worker_data_file, 'a+') as workerfile:
		workerfile.writelines(cv_pred)
	return worker_data_file 

def create_file_for_cv(input_file, output_file, verbose = False):
	timestamp = "2019-04-16 2:48:37AM"
	rows = []
	with open(input_file, "r") as file: 
		lines = file.readlines()
		if verbose: 
			print(str(len(lines)) + " items inside the input file")
		for line in lines:
			line = line.strip()
			row = line.split("\t")
			pano_id, sv_x, sv_y = row[1].split(",")
			label_type = row[2]
			if "Not" in label_type:
				label_type = label_type[4:]
			rows.append([timestamp, pano_id, sv_x, sv_y, label_type])
	if verbose:
		print("Numbers of items written to output file is " + str(len(rows)))
	with open(output_file, 'w+', newline='') as csvfile:
		writer = csv.writer(csvfile)
		for row in rows: 
			writer.writerow(row)

valid = ["CurbRamp", "NoCurbRamp", "SurfaceProblem", "Obstacle"]
def create_validation_from_csv(csv_filename, output_filename, verbose = False): 
	if not os.path.exists(csv_filename): 
		print("Could not find specified csv file: " + str(csv_filename))
		return 
	user_validations = pd.read_csv(user_validations, skiprows = 1)
	user_validations.columns = ['user_id', 'time_stamp', 'label_id', 'pano_id', 'sv_x', 'sv_y', 'agree_user', 'label_type_user']
	user_validations = user_validations[user_validations['agree_user'] != "unclear"]
	user_validations = user_validations[user_validations['pano_id'] != "#NAME?"]
	user_validations.set_index(['pano_id', 'sv_x', 'sv_y'], inplace=True)
	if verbose: 
		print("Shape of user validations dataframe is " + str(user_validations.shape))
	with open(output_filename, "w+") as file:
		for indexes, row in user_validations.iterrows():
			pano_id, sv_x, sv_y = indexes 
			name = str(pano_id) + "," + str(sv_x) + "," + str(sv_y)
			user_id = row[0]
			agree_user = str(row[3]) == "agree"
			label_type = row[4]
			if label_type not in valid:
				continue
			if not agree_user: 
				label_type = "Not-" + str(label_type)
			lister = [user_id.strip(),name.strip(),label_type.strip()]
			line = '\t'.join(lister) + "\n"
			file.write(line)
	return output_filename

def make_ground_dataset(ground_truth_file, user_validations, output_filename, golden_file, eval_file, factor = 0.75, verbose = False): 
	ground_truth = pd.read_csv(ground_truth_file, skiprows = 1)
	ground_truth.columns = ['user_id', 'pano_id', 'sv_x', 'sv_y', 'label_type_ground', 'agree_ground']
	ground = list(ground_truth['user_id'].unique())
	if verbose:
		print("Shape of ground truth dataset is " + str(ground_truth.shape))
		print("Number of unique users for ground truth is " + str(len(ground)))
	ground_truth.set_index(['pano_id', 'sv_x', 'sv_y'], inplace=True)
	del ground_truth['user_id']
	user_validations = pd.read_csv(user_validations, skiprows = 1)
	user_validations.columns = ['user_id', 'time_stamp', 'label_id', 'pano_id', 'sv_x', 'sv_y', 'agree_user', 'label_type_user']
	user_validations = user_validations[user_validations['agree_user'] != "unclear"]
	user_validations = user_validations[user_validations['pano_id'] != "#NAME?"]
	user_validations = user_validations[~user_validations['user_id'].isin(ground)]
	user_validations.set_index(['pano_id', 'sv_x', 'sv_y'], inplace=True)
	intersection = user_validations.join(ground_truth, how='inner')
	if verbose: 
		print("Shape of intersection dataframe is " + str(intersection.shape))
		print("Shape of user validation data_frame is " + str(user_validations.shape)) 
	lines = []
	for indexes, row in intersection.iterrows(): 
		pano_id, sv_x, sv_y = indexes 
		name = str(pano_id) + "," + str(sv_x) + "," + str(sv_y)
		label_type = row[5]
		if label_type not in valid: 
			continue
		up_ground = str(row[6]) == "False"
		if up_ground:
			label_type = "Not-"  + str(label_type)
		line = name + "\t" + label_type + "\n"
		lines.append(line)
	random.shuffle(lines)
	if verbose: 
		print("Number of ground truth items are " + str(len(lines)))
	index = int(factor * len(lines))
	initial = lines[:index]
	later = lines[index:]
	with open(golden_file, "w+") as file:
		file.writelines(initial)
	with open(eval_file, "w+") as file:
		file.writelines(later)
	with open(output_filename, "w+") as file:
		for indexes, row in user_validations.iterrows():
			pano_id, sv_x, sv_y = indexes 
			name = str(pano_id) + "," + str(sv_x) + "," + str(sv_y)
			user_id = row[0]
			agree_user = str(row[3]) == "agree"
			label_type = row[4]
			if label_type not in valid:
				continue
			if not agree_user: 
				label_type = "Not-" + str(label_type)
			line = str(user_id) + "\t" + str(name) + "\t" + str(label_type) + "\n"
			file.write(line)
	return output_filename
	

def run_get_another_label(input_file, path_to_panos, categories_file, ground_file = None, computer_vision = False, verbose = False): 
	ground_truth = ground_file != None 
	output_file = "workers-validations.txt"
	golden_file = "initial.txt"
	eval_file = "later.txt"
	if ground_truth:
		make_ground_dataset(ground_file, input_file, output_file, golden_file, eval_file, verbose = verbose)
	else: 
		create_validation_from_csv(input_file, output_file, verbose = verbose)
	if computer_vision: 
		path_to_summary = "single"
		file_for_cv = "validation-data-for-cv.csv"
		summary_file = os.path.join(path_to_summary, "summary.csv")	
		create_file_for_cv(output_file, file_for_cv, verbose= verbose)	
		summary_file = generate_validation_data(file_for_cv, path_to_panos, path_to_summary, verbose = verbose)
		if verbose: 
			print("Location of summary file " + str(summary_file))
		process_cv_predictions(output_file, summary_file) 
	if ground_truth:
		run_complete_scrip(output_file, categories_file,golden_file, eval_file)
	else:
		run_base_script(output_file, categories_file)