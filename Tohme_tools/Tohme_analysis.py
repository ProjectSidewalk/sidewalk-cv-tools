import os 
import random
import sys 
sys.path.append("resources")
import cv_tools as tools 
import utils 
try:
	from xml.etree import cElementTree as ET
except ImportError as e:
	from xml.etree import ElementTree as ET
import cv2 
from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np 
import json
import csv 

def convert_sv_to_pixel(sv_x, sv_y, xml_value, image_width, image_height):
	pano_yaw_degree = 360 - (xml_value + 180) % 360
	x = int(round(((float(pano_yaw_degree) / 360) * image_width + sv_x) % image_width))
	y =  int(round(image_height / 2 - sv_y))
	return (x, y)

def convert_pixel_to_sv(x, y, xml_value, image_width, image_height): 
	sv_x = round(x - float(xml_value)/360 * image_width)
	sv_y = round(image_height/2 - y)
	return (sv_x, sv_y)

class TohmeObject:

	def __init__(self, pano_id, base_path):
		self.pano_id = pano_id
		self.base_path = base_path
		self.points = {}
	
	def add_point(self, sv_x, sv_y, label_id, label_type): 
		if not label_id in self.points:
			self.points[label_id] = [label_type, [], []]
		self.points[label_id][1].append(sv_x)
		self.points[label_id][2].append(sv_y)
	
	def is_same(self, pano_id, label_id):
		return (self.pano_id == pano_id)
	
	def get_centriods(self): 
		centers = []
		for item in self.points.values():
			label_type = item[0]
			x_center = float(sum(item[1]))/len(item[1])
			y_center = float(sum(item[2]))/len(item[2])
			centers.append((label_type, x_center, y_center))
		return centers
	
	def get_bounding_boxes(self, xml_value, image_width, image_height, factor = 1.0): 
		areas = []
		for i, item in enumerate(self.points.values()): 
			label_type = item[0]
			st_x = item[1]
			st_y = item[2]
			x_coords = []
			y_coords = []
			for sv_x, sv_y in zip(st_x, st_y):
				x, y = convert_sv_to_pixel(sv_x, sv_y, xml_value, image_width, image_height)
				x_coords.append(x)
				y_coords.append(y)
			x1 = min(x_coords)
			y1 = min(y_coords)
			x2 = max(x_coords)
			y2 = max(y_coords)
			x_c = (x1 + x2)/2 
			y_c = (y1 + y2)/2 
			size = factor * max(x2 - x1, y2 - y1)/2
			top_left = int(round(x_c - size)), int(round(y_c - size))
			bottom_right = int(round(x_c + size)), int(round(y_c + size))
			part = (label_type, top_left, bottom_right)
			if part not in areas:
				areas.append(part)
		return areas 

	def contains_point(self, sv_x, sv_y, label_id):
		return (label_id in self.points and sv_x in self[label_id][1] and sv_y in self[label_id][2])

def get_data(path_to_metadata_xml):
	pano = {}
	pano_xml = open(path_to_metadata_xml, 'rb')
	tree = ET.parse(pano_xml)
	root = tree.getroot()
	for child in root:
		if(child.tag == 'data_properties' or child.tag == 'projection_properties'):
			pano[child.tag] = child.attrib
	if(len(pano) == 0):
		return None 
	return (int(pano['data_properties']['image_width']), int(pano['data_properties']['image_height']), float(pano['projection_properties']['pano_yaw_deg']))


def convert_label_to_color(label_type):
	if(str(label_type) == '1'): 
		return (255, 0, 0)
	return (0, 255, 0)

def read_in_file(file_name, path_to_panos): 
	dict = {}
	with open(file_name) as csvfile: 
		csvreader = csv.reader(csvfile, delimiter=',') 
		next(csvreader)
		for row in csvreader:
			if(len(row) == 0):
				continue
			label_id = row[0]
			pano_id = row[1]
			base_path = os.path.join(path_to_panos, pano_id)
			path_to_image = os.path.join(base_path, "images", "pano.jpg")
			if not os.path.exists(path_to_image):
				continue
			label_type = row[2]
			sv_x = int(row[3])
			sv_y = int(row[4])
			if not pano_id in dict:
				dict[pano_id] = TohmeObject(pano_id, base_path)
			dict[pano_id].add_point(sv_x, sv_y, label_id, label_type)
	locations = list(dict.values())
	return locations

def write_json_file(pano_id, x1, x2, y1, y2, xml_value, image_width, image_height): 
	crop_size = x2 - x1
	center_x = int(x1 + x2)/2
	center_y = (y1 + y2)/2
	sv_x, sv_y = convert_pixel_to_sv(center_x, center_y, xml_value, image_width, image_height)
	meta = {'crop size' : crop_size,
			'sv_x'      : sv_x,
			'sv_y'      : sv_y,
			'crop_x'    : center_x,
			'crop_y'    : center_y,
			'pano yaw'  : xml_value,
			'pano id'   : pano_id
		   }
	sv_x = str(sv_x)
	sv_y = str(sv_y)
	meta_filename = pano_id + "_crop" + sv_x + "," + sv_y + ".json"
	file_path = os.path.join("single", "crops", meta_filename)
	with open(file_path, 'w+') as metafile:
		json.dump(meta, metafile)
	return (sv_x, sv_y)

pytorch_label_from_int = ["NoCurbRamp", "Null", "Obstacle", "CurbRamp", "SurfaceProblem"]
def read_cv_prediction(file_name): 
	dict = {}
	with open(file_name) as csvfile: 
		csvreader = csv.reader(csvfile, delimiter=',')
		for row in csvreader: 
			pano_id = str(row.pop(0))
			sv_x = str(row.pop(0))
			sv_y = str(row.pop(0))
			confidence = max(row)
			label_name = pytorch_label_from_int[row.index(confidence)]
			meta_filename = pano_id + "crop" + sv_x + "," + sv_y + ".json"
			file_path = os.path.join("single", "crops", meta_filename)
			if not os.path.exists(file_path):
				continue
			with open(file_path, 'r') as jsonfile: 
				data = json.load(jsonfile)
			center_x = data['crop_x']
			center_y = data['crop_y']
			key = str(int(float(center_x))) + "," + str(int(float(center_y)))
			value = [pano_id, sv_x, sv_y, label_name, confidence]
			dict[key] = value
	return dict

def write_summary_file(results, name_of_summaryfile): 
	with open(name_of_summaryfile, 'w+', newline='') as csvfile:
		writer = csv.writer(csvfile)
		for value in results.values(): 
			writer.writerow(value)

def compare_cv_to_ground_truth(file_name, path_to_panos, verbose, summary_file_name): 
	path_to_sum = os.path.join("single", "completelabels.csv")
	results = read_cv_prediction(path_to_sum)
	locations = read_in_file(file_name, path_to_panos)
	total = 0
	for item in locations:
		path_to_xml = os.path.join(item.base_path,"meta.xml")
		pred = get_data(path_to_xml)
		if(pred == None):
			continue
		image_width, image_height, xml_value = pred
		image = cv2.imread(os.path.join(item.base_path, "images", "pano.jpg"))
		bboxes = item.get_bounding_boxes(xml_value, image_width, image_height)
		for group in bboxes: 
			label_type, top_left, bottom_right = group
			color = convert_label_to_color(label_type)
			label_name = "CurbRamp"
			if(label_type == '2'):
				label_name = "NoCurbRamp"
			x1,y1 = top_left
			x2, y2 = bottom_right
			center_x = int((x1 + x2)/2)
			center_y = int((y1 + y2)/2)
			name = str(center_x) + "," + str(center_y)
			if name in results: 
				results[name].append(label_name)
				total += 1
	if verbose: 
		print("Total number of points that were present was " + str(total))
	write_summary_file(results, summary_file_name)

def show_display(file_name, path_to_panos, verbose):
	locations = read_in_file(file_name, path_to_panos)
	if verbose:
		print("Number of panoramas found is " + str(len(locations)))
	acceptable = 0
	total = 0
	for item in locations:
		path_to_xml = os.path.join(item.base_path,"meta.xml")
		pred = get_data(path_to_xml)
		if(pred == None):
			continue
		total += 1
		image_width, image_height, xml_value = pred
		image = cv2.imread(os.path.join(item.base_path, "images", "pano.jpg"))
		bboxes = item.get_bounding_boxes(xml_value, image_width, image_height)
		for group in bboxes: 
			label_type, top_left, bottom_right = group
			color = convert_label_to_color(label_type)
			x1,y1 = top_left
			x2, y2 = bottom_right
			cropped = image[y1:y2, x1: x2]
			sv_x, sv_y = write_json_file(item.pano_id,x1,x2,y1,y2,xml_value,image_width, image_height)
			file_name = item.pano_id + "_crop" + sv_x + "," + sv_y + ".jpg"
			file_path = os.path.join("single", "crops", file_name) 
			cv2.imwrite(file_path,cropped) 

def run_analysis(file_name,path_to_panos, summary_file_name, verbose = False): 
	crops = os.path.join("single", "crops")
	utils.clear_dir(crops)
	show_display(file_name, path_to_panos, verbose)
	tools.get_results(verbose)
	compare_cv_to_ground_truth(file_name, path_to_panos, verbose, summary_file_name)
	utils.clear_dir(crops)


