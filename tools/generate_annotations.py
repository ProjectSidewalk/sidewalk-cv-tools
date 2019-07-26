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
import math
import random 

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
	pano_depth_path = os.path.join(path_to_panos,pano_id[:2], pano_id + ".depth.txt")
	pano_xml_path = os.path.join(path_to_panos,pano_id[:2], pano_id + ".xml")

	if not os.path.exists(pano_depth_path):
		print("Couldn't find " + pano_depth_path)
		return None

	if not os.path.exists(pano_xml_path):
		print("Couldn't find " + pano_xml_path)
		return None

	depth = None

	with open(pano_depth_path, 'rb') as f:
		depth = np.loadtxt(f)

	GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, pano_yaw_deg = get_data(pano_xml_path)
	GSV_IMAGE_WIDTH = int(GSV_IMAGE_WIDTH)
	GSV_IMAGE_HEIGHT = int(GSV_IMAGE_HEIGHT)
	pano_yaw_deg = float(pano_yaw_deg)
	x = ((float(pano_yaw_deg) / 360) * GSV_IMAGE_WIDTH + sv_x) % GSV_IMAGE_WIDTH
	y = GSV_IMAGE_HEIGHT / 2 - sv_y

	if(math.isnan(x) or math.isnan(y)):
		print(str(GSV_IMAGE_WIDTH) + " " + str(GSV_IMAGE_HEIGHT) + " " + str(pano_yaw_deg) + " " + str(x) + " " + str(y))

	try:
		box = utils.predict_crop_size(x, y, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, depth)
	except:
		print("Couldn't get crop size for ({},{})... skipping box".format(x,y))
	
	if (box == None or math.isnan(box)):
		return None

	(x1,y1) = (x - box/2, y - box/2)
	(x2,y2) = (x + box/2, y + box/2)

	return (x1,y1,x2,y2)


#0 - Timestamp, 1 - Label-ID, 2 - pano_id, 
#3 - sv_x, 4 - sv_y, 5 - agree, 6 - label
{2: "NoCurbRamp", 8 : "Null", 3: "Obstacle",1 : "CurbRamp", 4: "SurfaceProblem"}
def read_validation_data(path, path_to_panos):
	dict = {}
	updated ={}
	counter = 0
	if os.path.exists(path):
		with open(path) as csvfile: 
			csvreader = csv.reader(csvfile, delimiter=',')
			next(csvreader) 
			for row in csvreader:
				pano_id = row[0]
				counter += 1
				x = row[1]
				y = row[2]
				#if not (row[5] == "agree"):
					#continue
				complete = pano_id + "," + str(x) + "," + str(y)
				label = values[int(row[3])]
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
	print("Count is " + str(counter))
	return updated

def write_output(filename, rows): 
	if(os.path.exists(filename)):
		os.remove(filename)
	with open(filename, 'w+', newline='') as csvfile:
		writer = csv.writer(csvfile)
		for row in rows:
			writer.writerow(row)
			
base = "/content/drive/My Drive/Colab/Faster_RCNN_for_Open_Images_Dataset_Keras/Storage/"
#row = [name, x1, y1, x2, y2, pred, x, y]
def write_to_file(name, rows,final_file):
	fixed = []
	for line in rows:
		line = line[:len(line) - 1]
		split = line.split(",")
		split[0] = base + name + "/" + split[0] + ".jpg"
		for i in range(1,5):
			split[i] = str(int(float(split[i])))
		joined = ""
		for part in split:
			joined += part + ","
		joined = joined[:len(joined) - 1] + "\n"
		fixed.append(joined)
	output = name + "_" + final_file
	if(os.path.exists(output)):
		os.remove(output)
	file = open(output,'w+')
	file.writelines(fixed)
	file.close()
	
def split_data(factor, final_file):
	file = open(final_file,"r")
	lines = file.readlines()
	number = int(factor * len(lines))
	random.shuffle(lines)
	train = lines[:number]
	test = lines[number:]
	file.close()
	write_to_file("test", test, final_file)
	write_to_file("train", train, final_file)

def make_annotation_file(path_to_panos, path_to_file, final_file):
	if not os.path.exists(path_to_panos):
		print("Could not path_to_panos")
		return
	dict = read_validation_data(path_to_file, path_to_panos)
	rows = []
	print("Dict is " + str(len(dict)))
	for key,pred in dict.items(): 
		pano_id,x,y = key.split(",")
		values = get_bounding_box(path_to_panos, pano_id, int(x), int(y))
		if values == None:
			continue
		(x1,y1,x2,y2) = values
		name = pano_id + ".jpg"
		row = [name, x1, y1, x2, y2, pred, x, y]
		rows.append(row)
	print("Number of annotations is " + str(len(rows)))
	write_output(final_file, rows)
	split_data(0.8, final_file)
