import sys
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

def get_data(path_to_metadata_xml):
	pano = {}
	pano_xml = open(path_to_metadata_xml, 'rb')
	tree = ET.parse(pano_xml)
	root = tree.getroot()
	for child in root:
		if(child.tag == 'data_properties' or child.tag == 'projection_properties'):
			pano[child.tag] = child.attrib
	return [pano['data_properties']['image_width'], pano['data_properties']['image_height'], pano['projection_properties']['pano_yaw_deg']]

def get_bounding_box(path_to_panos, pano_id, sv_x, sv_y, depth):
	box = None

	depth_path = os.path.join(path_to_panos,pano_id[:2], pano_id + ".depth.txt")
	pano_xml_path = os.path.join(path_to_panos,pano_id[:2], pano_id + ".xml")
	if (!os.path.exists(depth_path) or !os.path.exists(pano_xml_path)):
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

	x1 = (1.0 * x1)/GSV_IMAGE_WIDTH
	x2 = (1.0 * x2)/GSV_IMAGE_WIDTH
	y1 = (1.0 * y1)/GSV_IMAGE_HEIGHT
	y2 = (1.0 * y2)/GSV_IMAGE_HEIGHT

	return (x1.y1,x2,y2)


