import sys
import utils
import pred_pano_labels as pred
from PIL import Image
import numpy as np
import threading
import shutil
try:
    from queue import Queue
except ImportError:
    from Queue import Queue
import shutil
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
	complete_path = "single\\crops"
	
	if not os.path.exists(complete_path):
		os.makedirs(complete_path)
	unique = []
	for i, pred in enumerate(predictions):
		crop_image = complete_path +"\\" + pano + "_crop" + str(pred) + ".jpg"
		if not os.path.exists(crop_image):
			unique.append(i)
	#print("Length of unique is " + str(len(unique)))
	if(len(unique) == 0):
		print("All data for " + pano_id + " already exists")
		return 0
	#Gets name of all the necessary files 
	path_to_pano = path_to_panos + pano[:2] + "\\" + pano
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
	for i in unique:
		prediction = predictions[i].strip()
		output = complete_path +"\\" + pano + "_crop" + str(prediction) 
		coord = prediction.split(',')
		imagex = float(coord[0])
		imagey = float(coord[1])
		if im != None:
			utils.make_single_crop(im,width,height,depth,pano, imagex, imagey, yawdeg, output)
			m += 1
	return m

def thread_crop_image(items, path_to_panos, queue):
	count = 0
	for row in items: 
		pano_id = row.pop(0)
		count += make_crop(row, pano_id, path_to_panos)
	queue.put(count)

def make_crop_threading(dict_image, path_to_panos, num_threads = 4): 
	utils.clear_dir("single/crops")
	q = Queue()
	crops_q = [[] for i in range(num_threads)]
	count = 0
	for pano_id, coords in dict_image.items():
		complete = [pano_id] + coords
		crops_q[count % num_threads].append(complete)
		count += 1
	threads = []
	for i in range(num_threads):
		t = threading.Thread(target = thread_crop_image, args=(crops_q[i], path_to_panos,q, ))
		threads.append(t)
		t.start()
	for proc in threads:
		proc.join()
	total = 0 
	while not q.empty():
		total += q.get()
	print("Created " + str(total) + " new crops")

path_to_completelabels = "single/completelabels.csv" 
def get_results():
	if os.path.exists(path_to_completelabels):
		os.remove(path_to_completelabels)
		print("Delted a old compeltelabels file")
	#return
	if(len(os.listdir("single\\crops")) > 0):
		pred.single_crops("single\\","single", "models/", verbose=True)
	else:
		print("No new crops to run CV")

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


# 0 - pano_id, 1 - x, 2 - y, 3 - CVlabel, 4 - userlabel, 5 - confidence 
def write_summary_file(rows_dict, labels_list, ignore_null, existing_labels):
	new_lables = []
	name_of_summaryfile = "single/summary" + str(ignore_null) + ".csv"
	if os.path.exists(name_of_summaryfile):
		os.remove(name_of_summaryfile)
	with open(name_of_summaryfile, 'w+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for first, second in existing_labels.items():
            	pano_id, x, y = first.split(",")
            	cvlabel, userlabel, confidence = second.split(",")
            	x = int(float(x))
            	y = int(float(x))
            	row = [pano_id, x, y, cvlabel, userlabel, confidence]
            	writer.writerow(row)
            labelrowcount = 0
            for labelrow in labels_list:
	            pano_id = labelrow[0]
	            x = labelrow[1]
	            y = labelrow[2]
	            complete = pano_id + "," + str(float(x)) + "," + str(float(y))
	            if(complete in rows_dict and complete not in existing_labels):
		            cv_respone = rows_dict[complete]
		            label,confidence  = cv_respone.split(",")
		            value = [pano_id, x, y] + [label, labelrow[3]] + [confidence]
		            new_lables.append(value)
		            writer.writerow(value)
		            labelrowcount += 1
            print("Label row count is " + str(labelrowcount))
	return new_lables


def generate_image_date(dict_valid, existing_labels): 
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
	print("Number of points in dict_image is " + str(total))
	return dict_image

#Dict_row: pano,x,y : label
def generate_labelrows(dict_row):
	labelrow = []
	for complete_name,label in dict_row.items():
		pano_id, x, y = complete_name.split(",")
		label = [pano_id, x, y, label]
		labelrow.append(label)
	return labelrow

def read_validation_data(path, date_after, existing_labels, add_to_summary):
	dict = {}
	totalcount = 0
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
					totalcount += 1
					pano_id = row[2]
					x = row[3]
					y = row[4]
					complete = pano_id + "," + str(float(x)) + "," + str(float(y))
					if complete in existing_labels:
						add_to_summary[complete] = existing_labels[complete]
						continue
					if not (complete in dict):
						label = row[6]
						dict[complete] = label
	print("Already have results for " + str(len(add_to_summary)) + " items")
	return dict

name_of_existing_file = "single/labelsalreadycalculated.csv"
def labels_already_made():
	dict = {}
	if(os.path.exists(name_of_existing_file)):
		with open(name_of_existing_file) as csvfile:
			csvreader = csv.reader(csvfile, delimiter=',') 
			for row in csvreader:
				pano_id = row[0]
				x = row[1]
				y = row[2]
				cvlabel = row[3]
				userlabel = row[4]
				confidence = row[5]
				potential = pano_id + "," + str(float(x)) + "," + str(float(y))
				value = cvlabel + "," + userlabel + "," + confidence
				dict[potential] = value
	return dict

def update_labels_already_made(new_lables):
	with open(name_of_existing_file, 'a+', newline='') as csvfile:
		writer = csv.writer(csvfile)
		for row in new_lables:
			writer.writerow(row)

def generate_data(input_data, date_after,path_to_panos, ignore_null):
	existing_labels = labels_already_made()
	add_to_summary = {}
	dict_valid = read_validation_data(input_data, date_after, existing_labels, add_to_summary)
	print("Length of dict_valid " + str(len(dict_valid)))
	dict_image = generate_image_date(dict_valid, existing_labels)
	make_crop_threading(dict_image, path_to_panos)
	get_results()
	rows_dict = exact_labels(ignore_null)
	print("Length of rows_dict is " + str(len(rows_dict)))
	labels_list = generate_labelrows(dict_valid)
	print("Length of labels_list is " + str(len(labels_list)))
	new_labels = write_summary_file(rows_dict, labels_list, ignore_null, add_to_summary)
	print("Number of new labels is " + str(len(new_labels)))
	update_labels_already_made(new_labels)

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

def test_threading():
	input_data = "validations-seattle.csv"
	path_to_panos = "D:\\SeattleImages\\validation\\panos\\new_seattle_panos\\"
	date_after = "2018-06-28"
	dict_valid = read_validation_data(input_data, date_after)
	dict_image = generate_image_date(dict_valid)
	make_crop_threading(dict_image, path_to_panos)

if __name__ == "__main__":
	path_to_panos = "panos/new_seattle_pano s/"
	date_after = "2018-06-28"
	if os.path.isdir(path_to_panos):
		generate_results_data("validations-seattle.csv", date_after, path_to_panos)
	else:
		print("There is no such directory")
	#test_threading(path_to_panos)