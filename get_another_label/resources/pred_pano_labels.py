from collections import defaultdict
import sys
import os
import csv
import shutil

from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np

import threading
import time
from queue import Queue

import utils

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

pytorch_label_from_int = ('Missing Cut', "Null", 'Obstruction', "Curb Cut", "Sfc Problem")


############################################


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
		print "Building dataset and loading model..."
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

def write_predictions_to_file(predictions_dict, root_path, pred_file_name, verbose=False):
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
					row = [pano_id] + [x,y] + prediction
					writer.writerow(row)
					count += 1
			if verbose:
				print("\tWrote {} predictions to {}.".format(count + 1, path))
	return path

def write_predictions_for_every_pano(root, predictions_dict, verbose=False):
	recent = ""
	for pano_id in predictions_dict.keys():
		sub = pano_id[:2]
		folder = os.path.join(root,sub)
		if not os.path.exists(folder):
			print("Couldn't find: " + folder)
			continue
		file = "labels_" + pano_id[:len(pano_id) - 1] + ".csv"
		path = os.path.join(root,sub,file)
		if os.path.exists(path):
			os.remove(path)
		predictions = predictions_dict[pano_id]
		with open(path, 'w+', newline='') as csvfile:
			writer = csv.writer(csvfile)
			count = 0
			for coords, prediction in predictions.items():
				if type(prediction) != list:
					prediction = list(prediction)
				x,y = coords.split(',')
				row = [x,y] + prediction
				writer.writerow(row)
				count += 1
			if verbose:
				print("\tWrote {} predictions to {}.".format(count + 1, path))
		recent = path
	return str(recent) 

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
		print "Error extracting Pano Yaw Deg:"
		print e

	pano = Pano()
	pano.photog_heading = None

	n_crops = 0
	for feat in sliding_window(pano, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT): # ignoring labels here
		sv_x, sv_y = feat.sv_image_x, feat.sv_image_y
		if verbose:
			print "added crop ({},{}) to queue".format(sv_x, sv_y)
		output_filebase = os.path.join('temp','crops',pano_id+'_crop{},{}'.format(sv_x, sv_y))
		crop_queue.put([pano_id, sv_x, sv_y, pano_yaw_deg, output_filebase])
		n_crops += 1

	quit_cropping = False
	def crop_threader(c_queue,depth_data,pano_img_copy,name,quit,verbose):
		while not c_queue.empty() and not quit:
			try:
				crop_info = c_queue.get()
				if verbose:
					print "cropping around ({},{})".format(crop_info[1],crop_info[2])
					#print "crop_info: ", crop_info
				utils.make_single_crop(pano_img_copy ,GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, depth_data, crop_info[0], crop_info[1], crop_info[2], crop_info[3], crop_info[4])
				c_queue.task_done()
			except Exception as e:
				print "\t cropping failed"
				print e
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
		print "canceling..."
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
			prediction = map(float, row[2:])

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
			print "predictions: ", predictions
		coords, prediction = predictions[0],predictions[1]
		(sv_x, sv_y) = coords

		if show_box:
			x = ((float(pano_yaw_deg) / 360) * GSV_IMAGE_WIDTH + sv_x) % GSV_IMAGE_WIDTH
			y = GSV_IMAGE_HEIGHT / 2 - sv_y
			try:
				box = utils.predict_crop_size(x, y, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, depth)
			except:
				print "Couldn't get crop size for ({},{})... skipping box".format(x,y)
				box = None
		else: box = None

		label = str(prediction)
		
		if verbose:
			print "Found a {} at ({},{})".format(label, sv_x, sv_y)
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
		print "Marked {} predicted and {} true labels on {}.".format(pred, true, out_img)

	return


def pred_pano_labels(pano_id, path_to_gsv_scrapes, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, model_dir, num_threads=4, save_labeled_pano=True, verbose=False):
	''' takes a panorama id and returns a dict of the filtered predictions'''
	path_to_folder = path_to_gsv_scrapes + pano_id[:2] + "/" + pano_id
	path_to_xml = path_to_folder + ".xml"
	(GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT) = utils.extract_width_and_height(path_to_xml)
	now = time.time()
	if not os.path.exists('temp/crops'):
		os.makedirs('temp/crops')
	if not os.path.exists('viz'):
		os.makedirs('viz')
	utils.clear_dir("temp/crops")
	make_sliding_window_crops(pano_id, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, path_to_gsv_scrapes, num_threads=num_threads, verbose=verbose)
	
	model_name = "seattle_pt_20ep_re18_2ff2"
	model_path = os.path.join(model_dir, model_name+'.pt')

	preds = predict_from_crops("temp/", model_path,verbose=verbose)
	preds_loc = write_predictions_for_every_pano(path_to_gsv_scrapes, preds, verbose=verbose)
	if(len(preds_loc) == 0): 
		return None
	pred = read_predictions_from_file(preds_loc)
	pred_dict = non_max_sup(pred, radius=150, clip_val=4.5, ignore_ind=1, verbose=verbose)

	if save_labeled_pano:
		pano_root = os.path.join(path_to_gsv_scrapes,pano_id[:2],pano_id)
		out_img = os.path.join("viz",pano_id+"_viz.jpg")
		show_predictions_on_image(pano_root, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, pred_dict, out_img, show_coords=False, show_box=True, verbose=verbose)
	utils.clear_dir("temp/crops")
	print("{} took {} seconds".format(pano_id, time.time()-now))
	return pred_dict

def single_crops(crop_dir,path_dir,model_dir, verbose=False):
	model_name = utils.get_model_name()
	model_path = os.path.join(model_dir, model_name+'.pt')
	preds = predict_from_crops(crop_dir,model_path,verbose=verbose)
	preds_loc = write_predictions_to_file(preds,path_dir,"completelabels.csv", verbose=verbose)

def batch_save_pano_labels(pano_ids, path_to_gsv_scrapes, model_dir, num_threads=4, verbose=False):
	''' takes a panorama id and returns a dict of the filtered predictions'''
	start = time.time()
	if not os.path.exists('temp/crops'):
		os.makedirs('temp/crops')
	if not os.path.exists('viz'):
		os.makedirs('viz')
	utils.clear_dir('temp/crops')

	for pano_id in pano_ids:
		now = time.time()

		pano_root = os.path.join(path_to_gsv_scrapes,pano_id[:2],pano_id)
		path_to_xml = pano_root + ".xml"
		(GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT) = utils.extract_width_and_height(path_to_xml)
		make_sliding_window_crops(pano_id, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, path_to_gsv_scrapes, num_threads=num_threads, verbose=verbose)
		
		model_name = "20ep_sw_re18_2ff2"
		model_path = os.path.join(model_dir, model_name+'.pt')

		preds = predict_from_crops("temp/", model_path,verbose=verbose)
		preds_loc = write_predictions_to_file(preds, path_to_gsv_scrapes+pano_id[:2], "labels.csv", verbose=verbose)
		utils.clear_dir('temp/crops')
		print "{} took {} seconds".format(pano_id, time.time()-now)
	print "total time: {} seconds".format(time.time()-start)

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

if __name__ == '__main__':
	print(pred_pano_labels("1a1UlhadSS_3dNtc5oI10Q", "../panos/", 13312, 6656, "../models/", num_threads=4, save_labeled_pano=True, verbose=True))
	print(pred_pano_labels("4s6C3NR6YRvHCYKMM_00QQ", "../panos/", 16384, 8192, "../models/", num_threads=4, save_labeled_pano=True, verbose=True))