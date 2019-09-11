# Written by Devesh Sarda, Galen Weld, and Kavi Dey. June - August 2019

import sys
import utils
from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np
import threading
import shutil
from collections import defaultdict
from queue import Queue
import os
import csv 
import time

try:
	from xml.etree import cElementTree as ET
except ImportError as e:
	from xml.etree import ElementTree as ET

from point import Point as Point
from pano_feats import Pano as Pano
from pano_feats import Feat as Feat
from clustering import non_max_sup

import torchvision
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from TwoFileFolder import TwoFileFolder
from resnet_extended1 import extended_resnet18 
import pandas as pd
import numpy as np
import math

#0 - pano_id, 1 - sv_x, 2 - sv_y, 3- NoCurbRamp, 
pytorch_label_from_int = ["NoCurbRamp", "Null", "Obstacle", "CurbRamp", "SurfaceProblem"]
sys.path.append("../")

def predict_from_crops(dir_containing_crops, model_path,verbose=False):
	''' use the TwoFileFolder dataloader to load images and feed them
		through the model
		returns a dict mapping pano_ids to dicts of {coord: prediction lists}
	'''
	data_transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
	if verbose:
		print("Building dataset and loading model...")
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	image_dataset = TwoFileFolder(dir_containing_crops, meta_to_tensor_version=2, transform=data_transform)
	dataloader    = torch.utils.data.DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=4)

	len_ex_feats = image_dataset.len_ex_feats
	dataset_size = len(image_dataset)

	panos = image_dataset.classes

	if verbose:
		print("Using dataloader that supplies {} extra features.".format(len_ex_feats))
		print("")
		print("Finished loading data. Got crops from {} panos.".format(len(panos)))
	model_ft = extended_resnet18(len_ex_feats=len_ex_feats)
	try:
		model_ft.load_state_dict(torch.load(model_path))
	except RuntimeError as e:
		model_ft.load_state_dict(torch.load(model_path, map_location='cpu'))
	model_ft = model_ft.to( device )
	optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

	model_ft.eval()

	paths_out = []
	pred_out  = []

	if verbose:
		print("Computing predictions....")
	for inputs, labels, paths in dataloader:
		inputs = inputs.to(device)
		labels = labels.to(device)
		# zero the parameter gradients
		optimizer_ft.zero_grad()
		with torch.set_grad_enabled(False):
			outputs = model_ft(inputs)
			_, preds = torch.max(outputs, 1)

			paths_out += list(paths)
			pred_out  += list(outputs.tolist())

	predictions = defaultdict(dict)
	for i in range(len(paths_out)):
		path  = paths_out[i]
		preds = pred_out[i]

		_, filename = os.path.split(path)
		filebase, _ = os.path.splitext(filename)
		pano_id, coords = filebase.split('crop')

		#print pano_id, coords, preds
		predictions[pano_id][coords] = preds

	return predictions

def write_predictions_to_file(predictions_dict, root_path, pred_file_name, verbose=False, save_id = True):
	path = os.path.join(root_path,pred_file_name)
	if os.path.exists(path):
		os.remove(path)
	with open(path, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			for pano_id in predictions_dict.keys():
				predictions = predictions_dict[pano_id]
				count = 0
				for coords, prediction in predictions.items():
					if type(prediction) != list:
						prediction = list(prediction)
					x,y = coords.split(',')
					if(save_id == True):
						row = [pano_id] + [x,y] + prediction
					else:
						row = [x,y] + prediction
					writer.writerow(row)
					count += 1
			if verbose:
				print("\tWrote {} predictions to {}.".format(count + 1, path))
	return path

def sliding_window(pano, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, stride=100, bottom_space=1600, side_space=300, cor_thresh=70):
	''' take in a pano and produce a set of feats, ready for writing to a file
		labels assigned if the crop is within cor_thresh of a true label
		
		try cor_thresh = stride/sqrt(2)
	'''
	x, y = side_space, 0
	while(y > - (GSV_IMAGE_HEIGHT/2 - bottom_space)):
		while(x < GSV_IMAGE_WIDTH - side_space):
			# do things in one row
			
			# check if there's any features near this x,y point
			p = Point(x,y)
			
			label = 8 # for null
			for feat in pano.all_feats():
				if p.dist( feat.point() ) <= cor_thresh:
					if label == 8:
						label = feat.label_type
					else:
						if label != feat.label_type:
							#print "Found conflicting labels, skipping."
							continue
			row = [pano.pano_id, x, y, label, pano.photog_heading, None,None,None]
			yield Feat(row)
			
			x += stride
		y -= stride # jump down a row
		x = side_space

def make_sliding_window_crops(pano_id, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, path_to_gsv_scrapes, num_threads=4, verbose=False):
	''' make a set of crops for a single pano based on a sliding window'''
	num_crops = 0
	num_fail  = 0

	error_panos =  set()

	crop_queue = Queue(0)

	pano_root = os.path.join(path_to_gsv_scrapes,pano_id[:2],pano_id)

	pano_img_path   = pano_root + ".jpg"
	pano_xml_path   = pano_root + ".xml"
	pano_depth_path = pano_root + ".depth.txt"
	pano_yaw_deg = 0
	
	pano_img = Image.open(pano_img_path)
	depth = None

	path_to_depth = os.path.join(path_to_gsv_scrapes, pano_id[:2], pano_id + ".depth.txt")
	with open(path_to_depth, 'rb') as f:
		depth = np.loadtxt(f)

	try:
		pano_yaw_deg = utils.extract_panoyawdeg(pano_xml_path)
	except Exception as e:
		print("Error extracting Pano Yaw Deg:")
		print(e)

	pano = Pano()
	pano.photog_heading = None

	n_crops = 0
	for feat in sliding_window(pano, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT): # ignoring labels here
		sv_x, sv_y = feat.sv_image_x, feat.sv_image_y
		if verbose:
			print("added crop ({},{}) to queue".format(sv_x, sv_y))
		output_filebase = os.path.join('temp','crops',pano_id+'_crop{},{}'.format(sv_x, sv_y))
		crop_queue.put([pano_id, sv_x, sv_y, pano_yaw_deg, output_filebase])
		n_crops += 1

	quit_cropping = False
	def crop_threader(c_queue,depth_data,pano_img_copy,name,quit,verbose):
		while not c_queue.empty() and not quit:
			try:
				crop_info = c_queue.get()
				if verbose:
					print("cropping around ({},{})".format(crop_info[1],crop_info[2]))
					#print "crop_info: ", crop_info
				utils.make_single_crop(pano_img_copy ,GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, depth_data, crop_info[0], crop_info[1], crop_info[2], crop_info[3], crop_info[4])
				c_queue.task_done()
			except Exception as e:
				print("\t cropping failed")
				print(e)
				quit = True
				return

	c_threads = []
	for i in range(0, num_threads):
		c_threads.append(threading.Thread(target=crop_threader, args=(crop_queue,depth,pano_img.copy(),i,quit_cropping,verbose, )))
		c_threads[i].start()

	try:
		while not crop_queue.empty():
			time.sleep(0.1)
	except KeyboardInterrupt:
		print("canceling...")
		quit_cropping = True
		while not crop_queue.empty():
			try:
				crop_queue.get(False)
			except Empty:
				continue
			crop_queue.task_done()

	for i in range(0, num_threads):
		c_threads[i].join()
	#if verbose:
	#   print "Finished. {} crops succeeded, {} failed.".format(num_crops, num_fail) ### UPDATE WITH LOCKS AND MULTITHREADING
	#   print "Failed to find XML for {} panos:".format(len(error_panos)) ############## UPDATE WITH LOCKS AND MULTITHREADING

def read_predictions_from_file(path):
	predictions = defaultdict(list)

	with open(path, 'r') as csvfile:
		reader = csv.reader(csvfile)

		for row in reader:
			x, y = row[0], row[1]
			prediction = list(map(float, row[2:]))
			# let this work for processed predictions, as well
			if len(prediction) == 1:
				try:
					prediction = int(prediction[0])
				except ValueError:
					continue

			predictions["{},{}".format(x,y)] = prediction
	return predictions


def annotate(img, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, pano_yaw_deg, coords, label, color, show_coords=True, box=None):
	""" takes in an image object and labels it at specified coords
		translates streetview coords to pixel coords
		if given a box, marks that box around the label
	"""
	sv_x, sv_y = coords
	x = ((float(pano_yaw_deg) / 360) * GSV_IMAGE_WIDTH + sv_x) % GSV_IMAGE_WIDTH
	y = GSV_IMAGE_HEIGHT / 2 - sv_y

	if show_coords: label = "{},{} {}".format(sv_x, sv_y, label)

	# radius for dot
	r = 20
	draw = ImageDraw.Draw(img)
	draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
	if box is not None:
		half_box = box/2
		p1 = (x-half_box, y-half_box)
		p2 = (x+half_box, y+half_box)
		draw.rectangle([p1,p2], outline=color)

	font  = ImageFont.truetype("roboto.ttf", 60, encoding="unic")
	draw.text((x+r+10, y), label, fill=color, font=font)

def show_predictions_on_image(pano_root, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, correct, out_img, show_coords=True, show_box=False, verbose=False):
	''' annotates an image with with predictions. 
		each of the arguments in (correct, incorrect, predicted_gt_pts, missed_gt_points is
		a dict of string coordinates and labels (output from scoring.score)
		leave predicted and missed as default to skip ground truth
		show_coords will plot the coords, and show box will plot the bounding box. '''
	pano_img_path   = pano_root + ".jpg"
	pano_xml_path   = pano_root + ".xml"
	pano_depth_path = pano_root + ".depth.txt"
	pano_yaw_deg    = utils.extract_panoyawdeg(pano_xml_path)

	img = Image.open(pano_img_path)

	depth = None
	with open(pano_depth_path, 'rb') as f:
		depth = np.loadtxt(f)

	# convert from pytorch encoding to str
	for d in correct:
		int_label = correct[d]
		correct[d] =  pytorch_label_from_int[int_label]

	def annotate_batch(predictions, color, verbose=False):
		count = 0
		if verbose:
			print("predictions: ", predictions)
		coords, prediction = predictions[0],predictions[1]
		sv_x, sv_y = map(float, coords.split(','))

		if show_box:
			x = ((float(pano_yaw_deg) / 360) * GSV_IMAGE_WIDTH + sv_x) % GSV_IMAGE_WIDTH
			y = GSV_IMAGE_HEIGHT / 2 - sv_y
			try:
				box = utils.predict_crop_size(x, y, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, depth)
			except:
				print("Couldn't get crop size for ({},{})... skipping box".format(x,y))
				box = None
		else: box = None

		label = str(prediction)
		
		if verbose:
			print("Found a {} at ({},{})".format(label, sv_x, sv_y))
		annotate(img, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, pano_yaw_deg, (sv_x, sv_y), label, color, show_coords, box)
		count += 1
		return count

	# gt colors
	cor_color  = ImageColor.getrgb('palegreen')

	true = 0
	pred = 0
	for d in correct:
		marked = annotate_batch([d, correct[d]], cor_color,verbose=verbose)

	img.save(out_img)
	if verbose:
		print("Marked {} predicted and {} true labels on {}.".format(pred, true, out_img))

	return

def pred_pano_labels(pano_id, path_to_gsv_scrapes, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, model_dir, num_threads=4, save_labeled_pano=True, verbose=False):
	''' takes a panorama id and returns a dict of the filtered predictions'''
	path_to_folder = os.path.join(path_to_gsv_scrapes,pano_id[:2],pano_id)
	path_to_xml = path_to_folder + ".xml"
	(GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT) = utils.extract_width_and_height(path_to_xml)
	now = time.time()
	temp = os.path.join('temp','crops')
	if not os.path.exists(temp):
		os.makedirs(temp)
	if not os.path.exists('viz'):
		os.makedirs('viz')
	utils.clear_dir(temp)
	make_sliding_window_crops(pano_id, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, path_to_gsv_scrapes, num_threads=num_threads, verbose=verbose)
	
	model_name = utils.get_model_name()
	model_path = os.path.join(model_dir, model_name+'.pt')

	preds = predict_from_crops("temp", model_path,verbose=verbose)
	preds_loc = write_predictions_for_every_pano(path_to_gsv_scrapes, preds, verbose=verbose)
	if(len(preds_loc) == 0): 
		return None
	pred = read_predictions_from_file(preds_loc)
	pred_dict = non_max_sup(pred, radius=150, clip_val=4.5, ignore_ind=1, verbose=verbose)

	if save_labeled_pano:
		pano_root = os.path.join(path_to_gsv_scrapes,pano_id[:2],pano_id)
		out_img = os.path.join("viz",pano_id+"_viz.jpg")
		show_predictions_on_image(pano_root, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, pred_dict, out_img, show_coords=False, show_box=True, verbose=verbose)

	utils.clear_dir(temp)
	if verbose:
		print("{} took {} seconds".format(pano_id, time.time()-now))
	return pred_dict

def single_crops(crop_dir,path_dir,model_dir, verbose=False):
	model_name = utils.get_model_name()
	model_path = os.path.join(model_dir, model_name+'.pt')
	preds = predict_from_crops(crop_dir,model_path,verbose=verbose)
	#print(preds)
	#utils.clear_dir("temp/crops")
	preds_loc = write_predictions_to_file(preds,path_dir,"completelabels.csv", verbose=verbose)

def convertfromhorizontopixel(sv_image_x, sv_image_y, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT,PanoYawDeg):
	im_width = GSV_IMAGE_WIDTH
	im_height = GSV_IMAGE_HEIGHT
	x = ((float(PanoYawDeg) / 360) * im_width + sv_image_x) % im_width
	y = im_height / 2 - sv_image_y
	return (x, y)

'''
	Method to get width and height of image from its metadata
	Takes in the path of the xml file of the image to get dimensions of 
'''
def get_data(path_to_metadata_xml):
	pano = {}
	pano_xml = open(path_to_metadata_xml, 'rb')
	tree = ET.parse(pano_xml)
	root = tree.getroot()
	for child in root:
		if(child.tag == 'data_properties' or child.tag == 'projection_properties'):
			pano[child.tag] = child.attrib
	return [pano['data_properties']['image_width'],
			pano['data_properties']['image_height'],
			str(180 - float(pano['projection_properties']['pano_yaw_deg']))]

def get_width_and_height(path_to_metadata_xml):
	pano = {}
	pano_xml = open(path_to_metadata_xml, 'rb')
	tree = ET.parse(pano_xml)
	root = tree.getroot()
	for child in root:
		if child.tag == 'data_properties':
			pano[child.tag] = child.attrib
	return (int(pano['data_properties']['image_width']) , int(pano['data_properties']['image_height']))
'''
	Runs the entire program. 
	Takes in prediction array (['11900.0,-1000.0', 'Obstruction']) and a pano_id
	'''
def make_crop(predictions, pano, path_to_panos): 
	complete_path = os.path.join('single','crops')
	if not os.path.exists(complete_path):
		os.makedirs(complete_path)
	path_to_pano = os.path.join(path_to_panos,pano[:2],pano)
	image =  path_to_pano + ".jpg"
	im = None
	if os.path.exists(image):
		try:
			im = Image.open(image)
		except Exception as ex:
			template = "An exception of type " + str(type(ex).__name__) + " occured importing image: " + str(image)
			print(template)
	else: 
		#print("The following image doesn't exist: " + str(image))
		return 0
	xml = path_to_pano + ".xml"
	depth_name = path_to_pano + ".depth.txt"
	#Reading in the depth data and Image
	depth = None
	with open(depth_name, 'rb') as f:
		depth = np.loadtxt(f)
	#Reading in the image
	#Getting width, height and yaw of image
	data = get_data(xml)
	width = int(data[0])
	height = int(data[1])
	yawdeg = float(data[2])
	m = 0
	if not os.path.exists(complete_path):
		os.makedirs(complete_path)
	for prediction in predictions:
		prediction = prediction.strip()
		output = os.path.join(complete_path,pano + "_crop" + str(prediction))
		coord = prediction.split(',')
		imagex = float(coord[0])
		imagey = float(coord[1])
		if im != None:
			utils.make_single_crop(im,width,height,depth,pano, imagex, imagey, yawdeg, output)
			m += 1
	return m

crops_total = 0

def thread_crop_image(queue, path_to_panos):
	global crops_total
	while not queue.empty():
		row = queue.get()
		pano_id = row.pop(0)
		crops_total += make_crop(row, pano_id, path_to_panos)
		t = "\rCrops made so far is " + str(crops_total)
		sys.stdout.write(t)
		sys.stdout.flush()

def make_crop_threading(dict_image, path_to_panos, verbose, num_threads): 
	q = Queue()
	count = 0
	for pano_id, coords in dict_image.items():
		complete = [pano_id] + coords
		q.put(complete)
		count += 1
	threads = []
	for i in range(num_threads):
		t = threading.Thread(target = thread_crop_image, args=(q, path_to_panos, ))
		threads.append(t)
		t.start()
	for proc in threads:
		proc.join()

path_to_completelabels = os.path.join("single", "completelabels.csv")
def get_results(verbose):
	if(len(os.listdir(os.path.join("single", "crops"))) > 0):
		if os.path.exists(path_to_completelabels):
			os.remove(path_to_completelabels)
		if(verbose):
			print("Delted a old compeltelabels file")
		single_crops("single","single", "models", verbose=True)
	elif(verbose):
		print("No new crops to run CV")


def read_complete_file(ignore_null):
	rows = []
	now = time.time()
	if os.path.exists(path_to_completelabels):
		with open(path_to_completelabels, 'r') as csvfile: 
			csvreader = csv.reader(csvfile) 
			for row in csvreader: 
				if(len(row) == 0):
					continue
				max = 3
				value = []
				for i in range(0, len(row)):
					if(i < 3):
						value.append(row[i])
					elif (ignore_null and i == 4): 
						continue
					elif(row[i] > row[max]):
						max = i
				value.append(pytorch_label_from_int[max - 3])
				if (max < 0 or max >= len(row)):
					print(str(max) + " < " + str(len(row)))
				percentage = round(float(row[max]), 2)
				value.append(percentage) 
				rows.append(value)
	return rows

#Stores the pano_id, the x and y coordinates, the prediction and the confidence level
def exact_labels(ignore_null):
	rows = read_complete_file(ignore_null)
	dict = {}
	for row in rows:
		file_name = row[0][:len(row[0]) - 1]
		x = row[1]
		y = row[2]
		label = row[3]
		percentage = float(row[4])
		key = file_name + "," + str(x) + "," + str(y)
		result = label + "," + str(percentage)
		dict[key] = result
	return dict
'''
0 - timestamp
1 - label_id
2 - pano_id
3 - image_x
4 - image_y
5 - validation
6 - label_type
'''

def read_complete(): 
	df = pd.read_csv(path_to_completelabels, header = None)
	df.columns = ["pano_id", "sv_x", "sv_y", "NoCurbRamp", "Null", "Obstacle", "CurbRamp", "SurfaceProblem"]
	df.set_index(['pano_id', 'sv_x', 'sv_y'], inplace=True)
	return df 

def get_sigmoid(x): 
	return 1.0/(1.0 + math.exp(0.5 * (float(x) - 2.5)))

def get_score(cv_label,cv_confidence, user_label, user_confidence): 
	if user_label != cv_label: 
		return 0.5 + 0.5 * (get_sigmoid(user_confidence) ** 2)
	else:
		return get_sigmoid(cv_confidence)

user_data = {}

# 0 - pano_id, 1 - x, 2 - y, 3 - CVlabel, 4 - userlabel, 5 - confidence 
def write_summary_file(rows_dict, labels_list , add_to_summary, path_to_summary):
	global user_data
	new_lables = [] 
	raw_values = None
	if os.path.exists(path_to_completelabels): 
		raw_values = read_complete()
	title = ["label_id", "pano_id", "sv_x", "sv_y", "cv_label", "cv_confidence", "user_label", "user_label_confidence", "priority_score"]
	name_of_summaryfile = os.path.join(path_to_summary,"summary.csv")
	if os.path.exists(name_of_summaryfile):
		os.remove(name_of_summaryfile)
	with open(name_of_summaryfile, 'w+', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(title)
			for first, value in add_to_summary.items():
				pano_id, x, y = first.split(",")
				x = int(float(x))
				y = int(float(y)) 
				values = value[2:]
				cvlabel = value[0]
				confidence = value[1]
				complete = pano_id + "," + str(float(x)) + "," + str(float(y))
				for label_id, userlabel in user_data[complete]:
					value = float(values[pytorch_label_from_int.index(userlabel)])
					score = get_score(cvlabel, confidence, userlabel, value)
					row = [label_id, pano_id, x, y, cvlabel, confidence, userlabel, value, score]
					writer.writerow(row)
			for labelrow in labels_list:
				pano_id = labelrow[0]
				x = float(labelrow[1])
				y = float(labelrow[2])
				complete = pano_id + "," + str(x) + "," + str(y)
				if(complete in rows_dict and complete not in add_to_summary):
					first_label = labelrow[3]
					index = (pano_id + "_", x, y)
					values = raw_values.loc[index, :]
					cv_respone = rows_dict[complete]
					label,confidence  = cv_respone.split(",")
					for label_id, user_label in user_data[complete]:
						user_confidence = round(float(values[user_label]), 2)
						score = get_score(label, confidence, user_label, user_confidence)
						value = [label_id, pano_id, int(x), int(y)] + [label, confidence] 
						display = value.copy() + [user_label, user_confidence, score]
						writer.writerow(display)
						raw_stuff = list(values)
						rounded_stuff = [round(float(item), 2) for item in raw_stuff] 
						value = value[1:] + rounded_stuff			
						new_lables.append(value)			
	return new_lables

def generate_image_date(dict_valid, existing_labels, verbose): 
	dict_image = {}
	for key in dict_valid: 
		pano_id, x, y = key.split(",")
		complete = pano_id + "," + str(float(x)) + "," + str(float(y))
		if complete in existing_labels:
			continue
		test = x + "," + y
		if pano_id in dict_image:
			dict_image[pano_id].append(test)
		else:
			dict_image[pano_id] = [test]
	total = 0
	for pano_id, points in dict_image.items():
		total += len(points)
	if(verbose):
		print("Number of new labels to consider: " + str(total))
	return dict_image

#Dict_row: pano,x,y : label
def generate_labelrows(dict_row):
	labelrow = []
	for complete_name,label in dict_row.items():
		pano_id, x, y = complete_name.split(",")
		label = [pano_id, x, y, label]
		labelrow.append(label)
	return labelrow

#Time stamp, Pano_Id, X, Y, label type
def read_validation_data(path, date_after, existing_labels, add_to_summary, number_agree, verbose):
	global user_data
	dict = {}
	updated ={}
	totalcount = 0
	if os.path.exists(path):
		with open(path) as csvfile: 
			csvreader = csv.reader(csvfile, delimiter=',') 
			next(csvreader)
			for row in csvreader:
				time_stamp = row[0]
				date, time = time_stamp.split(" ")
				if(date > date_after):
					totalcount += 1
					label_id = row[1]
					pano_id = row[2]
					x = row[3]
					y = row[4]
					complete = pano_id + "," + str(float(x)) + "," + str(float(y))
					label = row[5]
					if not(complete in dict):
						dict[complete] = [complete in existing_labels, label] 
						user_data[complete] =[(label_id, label)]
					else:
						dict[complete].append(label)
						user_data[complete].append((label_id, label))
			counter = 0
			for key,predictions in dict.items():
				count = [0, 0, 0, 0, 0]
				counter += 1
				add = predictions.pop(0)
				for pred in predictions:
					count[pytorch_label_from_int.index(pred)] += 1
				maxval = np.argmax(count)
				if(count[maxval] < number_agree):
					continue
				labeltype = pytorch_label_from_int[maxval]
				if(add): 
					row = existing_labels[key]
					add_to_summary[key] = row
				else:
					updated[key] =  labeltype
	else:
		print("Could not find input file at " + path)
	if(verbose):
		print("Already have results for " + str(len(add_to_summary)) + " items")
		print("Need to get results for: " + str(len(updated)))
	return updated

def labels_already_made(path_to_panos):
	dict = {}
	sub = [x[0] for x in os.walk(path_to_panos)]
	sub.pop(0)
	for file in sub: 
		name_of_existing_file = os.path.join(file,"already.csv")
		if(os.path.exists(name_of_existing_file)):
			with open(name_of_existing_file) as csvfile:
				csvreader = csv.reader(csvfile, delimiter=',') 
				for row in csvreader:
					pano_id = row[0]
					x = row[1]
					y = row[2]
					potential = pano_id + "," + str(float(x)) + "," + str(float(y))
					values = row[3:]
					dict[potential] = values
	return dict
# 0 - pano_id, 1 - x, 2 - y, 3 - CVlabel, 4 - confidence 
def update_labels_already_made(new_lables,path_to_panos):
	writer = None
	for row in new_lables:
		pano_id = row[0]
		complete = os.path.join(path_to_panos,pano_id[:2],"already.csv")
		with open(complete, 'a+', newline='') as csvfile:
			writer = csv.writer(csvfile)
			if(writer != None):
				writer.writerow(row)

def generate_data(input_data, date_after,path_to_panos, ignore_null, number_agree, path_to_summary, verbose, num_threads):
	crops = os.path.join("single","crops")
	if not os.path.exists(crops):
		os.makedirs(crops)
	utils.clear_dir(crops)
	existing_labels = labels_already_made(path_to_panos)
	add_to_summary = {}
	dict_valid = read_validation_data(input_data, date_after, existing_labels, add_to_summary, number_agree, verbose)
	dict_image = generate_image_date(dict_valid, existing_labels, verbose)
	make_crop_threading(dict_image, path_to_panos, verbose, num_threads)
	get_results(verbose)
	rows_dict = exact_labels(ignore_null)
	labels_list = generate_labelrows(dict_valid)
	new_labels = write_summary_file(rows_dict, labels_list, add_to_summary, path_to_summary)
	if(verbose):
		print("Number of new labels is " + str(len(new_labels)))
	update_labels_already_made(new_labels,path_to_panos)
	utils.clear_dir(crops)
	if os.path.exists(path_to_completelabels):
		os.remove(path_to_completelabels)

def generate_validation_data(input_data,path_to_panos,path_to_summary, number_agree = 1,num_threads = 4, date_after = "2018-06-28", verbose = False):
	if not os.path.isdir(path_to_panos):
		print("There is no such directory for path to panos")
		return "Couldn't generate a summary file"
	first = time.time()
	ignore_null = True 
	generate_data(input_data, date_after, path_to_panos, ignore_null, number_agree, path_to_summary, verbose, num_threads)
	second = time.time()
	if(verbose):
		print("Program took: " + str((second - first)/60.0) + " minutes")
	return (os.path.join(path_to_summary,"summary.csv"))


def batch_save_pano_labels(pano_ids, path_to_gsv_scrapes, model_dir, num_threads=4, verbose=False):
	''' takes a panorama id and returns a dict of the filtered predictions'''
	start = time.time()
	crops = os.path.join('temp','crops')
	if not os.path.exists(crops):
		os.makedirs(crops)
	if not os.path.exists('viz'):
		os.makedirs('viz')
	utils.clear_dir(crops)

	for pano_id in pano_ids:
		now = time.time()

		pano_root = os.path.join(path_to_gsv_scrapes,pano_id[:2],pano_id)
		path_to_xml = pano_root + ".xml"
		(GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT) = utils.extract_width_and_height(path_to_xml)
		make_sliding_window_crops(pano_id, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, path_to_gsv_scrapes, num_threads=num_threads, verbose=verbose)
		
		model_name = "20ep_sw_re18_2ff2"
		model_path = os.path.join(model_dir, model_name+'.pt')

		preds = predict_from_crops("temp", model_path,verbose=verbose)
		preds_loc = write_predictions_for_every_pano(preds, path_to_gsv_scrapes+pano_id[:2], "labels.csv", verbose=verbose)
		utils.clear_dir(crops)
		print("{} took {} seconds".format(pano_id, time.time()-now))
	print("total time: {} seconds".format(time.time()-start))

def get_pano_labels(pano_id, path_to_gsv_scrapes, b_box=None, radius=150, clip_val=4.5, save_labeled_pano=False, verbose=False):
	pano_root = os.path.join(path_to_gsv_scrapes,pano_id[:2],pano_id)
	(GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT) = utils.extract_width_and_height(pano_root+".xml")

	pred = read_predictions_from_file(pano_root+"_labels.csv")
	pred_dict = non_max_sup(pred, radius=radius, clip_val=clip_val, ignore_ind=1, verbose=verbose)

	if b_box != None:
		for loc in pred_dict:
			if not utils.inside_b_box(loc,b_box):
				pred_dict.pop(loc)

	if save_labeled_pano:
		pano_root = os.path.join(path_to_gsv_scrapes,pano_id[:2],pano_id)
		out_img = os.path.join("viz",pano_id+"_viz.jpg")
		show_predictions_on_image(pano_root, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, pred_dict, out_img, show_coords=False, show_box=True, verbose=verbose)
	return pred_dict
