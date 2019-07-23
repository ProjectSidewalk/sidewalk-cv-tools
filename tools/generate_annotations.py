import sys

sys.path.append("resources/")

import utils
import numpy as np
import shutil

try:
	from queue import Queue
except ImportError:
	from Queue import Queue

try:
	from xml.etree import cElementTree as ET
except ImportError as e:
	from xml.etree import ElementTree as ET

import os
import csv 
import time

pytorch_label_from_int = ["NoCurbRamp", "Null", "Obstacle", "CurbRamp", "SurfaceProblem"]

def get_data(path_to_metadata_xml):
	pano = {}
	pano_xml = open(path_to_metadata_xml, 'rb')
	tree = ET.parse(pano_xml)
	root = tree.getroot()
	for child in root:
		if(child.tag == 'data_properties' or child.tag == 'projection_properties'):
			pano[child.tag] = child.attrib
	return [pano['data_properties']['image_width'], pano['data_properties']['image_height'], pano['projection_properties']['pano_yaw_deg']]

def get_bounding_box(path_to_panos, pano_id, sv_x, sv_y):
	box = None

	depth_path = os.path.join(path_to_panos,pano_id[:2], pano_id + ".depth.txt")
	pano_xml_path = os.path.join(path_to_panos,pano_id[:2], pano_id + ".xml")
	if (not os.path.exists(depth_path) or not os.path.exists(pano_xml_path)):
		return None

	depth = None

	with open(pano_depth_path, 'rb') as f:
	   depth = np.loadtxt(f)

	GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT,pano_yaw_deg = get_data(pano_xml_path)
	x = ((float(pano_yaw_deg) / 360) * GSV_IMAGE_WIDTH + sv_x) % GSV_IMAGE_WIDTH
	y = GSV_IMAGE_HEIGHT / 2 - sv_y

	try:
		box = utils.predict_crop_size(x, y, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, depth)
	except:
		print("Couldn't get crop size for ({},{})... skipping box".format(x,y))
	
	if (box == None):
		return None

	(x1,y1) = (x - box/2, y - box/2)
	(x2,y2) = (x + box/2, y + box/2)

	return (x1,y1,x2,y2)

def read_validation_data(path):
	dict = {}
	updated ={}
	if os.path.exists(path):
		with open(path) as csvfile: 
			csvreader = csv.reader(csvfile, delimiter=',') 
			next(csvreader)
			for row in csvreader:
				pano_id = row[1]
				x = row[2]
				y = row[3]
				complete = pano_id + "," + str(x) + "," + str(y)
				label = row[4]
				if not (complete in dict):
					dict[complete] = [label]
				else:
					dict[complete].append(label)
			counter = 0
			for key,predictions in dict.items():
				count = [0, 0, 0, 0, 0]
				counter += 1
				for pred in predictions:
					count[pytorch_label_from_int.index(pred)] += 1
				maxval = np.argmax(count)
				labeltype = pytorch_label_from_int[maxval]
				updated[key] =  labeltype
	else:
		print("Can't find the file at " + str(path))
	return updated

def write_output(filename, rows): 
	with open(filename, 'w+', newline='') as csvfile:
		writer = csv.writer(csvfile)
		for row in rows:
			writer.writerow(row)

def make_annotation_file(path_to_panos, path_to_file, final_file):
	print(os.path.exists("resources"))
	print(os.listdir("resources"))
	dict = read_validation_data(path_to_file)
	rows = []
	for key,pred in dict.items(): 
		pano_id,x,y = key.split(",")
		(x1,y1,x2,y2) = get_bounding_box(path_to_panos, pano_id, x, y)
		name = pano_id + ".jpg"
		row = [name, x1, y1, x2, y2,pred]
	write_output(final_file, rows)
