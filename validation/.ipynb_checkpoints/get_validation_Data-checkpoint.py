import sys
import utils
import pred_pano_labels as pred
from PIL import Image
import numpy as np
import multiprocessing

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

''' Method to convert x, y horizon coordinates to (x,y) coordinates on the image 
	Takes in x, y horizon coordinates, image width and height and yaw degreee of panorama
'''
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
	return [pano['data_properties']['image_width'], pano['data_properties']['image_height'], pano['projection_properties']['pano_yaw_deg']]

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
	complete_path = "single/crops"
	unique = []
	for i, pred in enumerate(predictions):
		crop_image = complete_path +"/" + pano + "_crop" + str(pred) + ".jpg"
		if not os.path.exists(crop_image):
			unique.append(i)
	if(len(unique) == 0):
		return 0
	if not os.path.exists(complete_path):
		os.makedirs(complete_path)
	#Gets name of all the necessary files 
	path_to_pano = path_to_panos + pano[:2] + "/" + pano
	image =  path_to_pano + ".jpg"
	im = None
	if os.path.exists(image):
		im = Image.open(image)
	else: 
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
	for i in unique:
		prediction = predictions[i].strip()
		output = complete_path +"/" + pano + "_crop" + str(prediction) 
		coord = prediction.split(',')
		imagex = float(coord[0])
		imagey = float(coord[1])
		if im != None:
			utils.make_single_crop(im,width,height,depth,pano, imagex, imagey, yawdeg, output)
			m += 1
	return m


def make_crop_threading(dict_image, path_to_panos, num_threads = 4): 
	start = time.time()
	crops_q = [[] for i in range(num_threads)]
	count = 0
	for pano_id, coords in dict_image.items():
		complete = [pano_id] + coords
		crops_q[count % num_threads].append(complete)
		count += 1
		print("There are " + str(count) + " items in line")

	def thread_crop_image(items, path_to_panos):
		count = 0
		for row in items: 
			pano_id = row.pop(0)
			if(count % 500 == 0):
				print(row)
			make_crop(row, pano_id, path_to_panos)

	threads = []
	for i in range(num_threads):
		t = multiprocessing.Process(target = thread_crop_image, args=(crops_q[i], path_to_panos, ))
		threads.append(t)
		t.start()

	for proc in processes:
		proc.join()

	end = time.time()
	print("The program took " + str(end - start))
			
path_to_completelabels = "single/completelabels.csv" 
def get_results():
	if os.path.exists(path_to_completelabels):
		os.remove(path_to_completelabels)
		print("Delted a old compeltelabels file")
	#return
	normal_path = "single"
	pred.single_crops(normal_path + "/",normal_path, "models/", verbose=True)

def read_complete_file(ignore_null):
	rows = []
	now = time.time()
	if os.path.exists(path_to_completelabels):
		with open(path_to_completelabels, 'r') as csvfile: 
			csvreader = csv.reader(csvfile) 
			for row in csvreader: 
				if(len(row) == 0):
					print("Empty row")
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
				percentage = round(100.0 * (float(row[max]) + 10.0)/20.0)
				value.append(percentage)
				#print(str(row[max]) + " -> " + str(percentage))
				#print(value) 
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
		percentage = int(float(row[4]))
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
def read_validation_data(path, date_after):
	dict = {}
	if os.path.exists(path):
		with open(path) as csvfile: 
			csvreader = csv.reader(csvfile, delimiter=',') 
			next(csvreader)
			for row in csvreader:
				time_stamp = row[0]
				#print(row)
				date, time = time_stamp.split(" ")
				agree = (row[5] == "agree")
				if(date > date_after and agree):
					pano_id = row[2]
					x = row[3]
					y = row[4]
					complete = pano_id + "," + str(float(x)) + "," + str(float(y))
					if not complete in dict:
						label = row[6]
						dict[complete] = label
	return dict

# 0 - pano_id, 1 - x, 2 - y, 3 - CVlabel, 4 - userlabel, 5 - confidence 
def write_summary_file(rows_dict, labels_list, ignore_null):
	name_of_summaryfile = "single/summary" + str(ignore_null) + ".csv"
	if os.path.exists(name_of_summaryfile):
		os.remove(name_of_summaryfile)
	with open(name_of_summaryfile, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for labelrow in labels_list:
	            pano_id = labelrow[0]
	            x = labelrow[1]
	            y = labelrow[2]
	            complete = pano_id + "," + str(float(x)) + "," + str(float(y))
	            if complete in rows_dict:
		            cv_respone = rows_dict[complete]
		            label,confidence  = cv_respone.split(",")
		            value = [pano_id, x, y] + [label, labelrow[3]] + [confidence]
		            writer.writerow(value)
	return name_of_summaryfile


def generate_image_date(dict_valid): 
	dict_image = {}
	for key in dict_valid: 
		pano_id, x, y = key.split(",")
		test = x + "," + y
		if pano_id in dict_image:
			dict_image[pano_id].append(test)
		else:
			dict_image[pano_id] = [test]
	return dict_image


#Dict_row: pano,x,y : label
def generate_labelrows(dict_row):
	labelrow = []
	for complete_name,label in dict_row.items():
		pano_id, x, y = complete_name.split(",")
		label = [pano_id, x, y, label]
		labelrow.append(label)
	return labelrow

def generate_data(input_data, date_after,path_to_panos, ignore_null):
	dict_valid = read_validation_data(input_data, date_after)
	dict_image = generate_image_date(dict_valid)
	count = 0
	print("Going to create new crops")
	make_crop_threading(dict_image, path_to_panos)
	'''
	for pano_id, coords in dict_image.items():
		#print("Currently at " + str(count) + " crops")
		count += make_crop(coords, pano_id, path_to_panos)
	print(str(count) + " new crops were created")
	'''
	get_results()
	rows_dict = exact_labels(ignore_null)
	labels_list = generate_labelrows(dict_valid)
	write_summary_file(rows_dict, labels_list, ignore_null)

def generate_results_data(input_data, date_after,path_to_panos):
	if not os.path.isdir(path_to_panos):
		print("There is no such directory for path to panos")
		return False
	'''
	ignore_null = False
	start = time.time()
	generate_data(input_data, date_after, path_to_panos, ignore_null)
	print("Not ignoring took: " + str(first - start))
	'''
	first = time.time()
	ignore_null = True 
	generate_data(input_data, date_after, path_to_panos, ignore_null)
	second = time.time()
	print("Ignorning took: " + str(second - first))
	return True
'''
def test_threading(path_to_panos):
	input_data = "validations-seattle.csv"
	date_after = "2018-06-28"
	dict_valid = read_validation_data(input_data, date_after)
	dict_image = generate_image_date(dict_valid)
	make_crop_threading(dict_image, path_to_panos)
'''

if __name__ == "__main__":
	path_to_panos = "panos/new_seattle_panos/"
	date_after = "2018-06-28"
	if os.path.isdir(path_to_panos):
		generate_results_data("validations-seattle.csv", date_after, path_to_panos)
	else:
		print("There is no such directory")
	#test_threading(path_to_panos)