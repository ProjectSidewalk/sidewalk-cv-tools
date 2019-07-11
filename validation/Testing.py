import sys
import utils
import pred_pano_labels as pred
from PIL import Image
import numpy as np
try:
	from xml.etree import cElementTree as ET
except ImportError as e:
	from xml.etree import ElementTree as ET
import os
import csv 
import time

pytorch_label_from_int = ["Missing Cut", "Null", "Obstruction", "Curb Cut", "Sfc Problem"]


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
def make_crop(predictions, pano): 
	complete_path = "single/crops"
	if not os.path.exists(complete_path):
		os.makedirs(complete_path)
	#Gets name of all the necessary files 
	path_to_file = "panos/" + pano[:2] + "/" + pano
	image =  path_to_file + ".jpg"
	xml = path_to_file + ".xml"
	depth_name = path_to_file + ".depth.txt"
	#Reading in the depth data and Image
	depth = None
	with open(depth_name, 'rb') as f:
	    depth = np.loadtxt(f)
	#Reading in the image 
	im = None
	if os.path.exists(image):
		im = Image.open(image)
	else: 
		print("Can't find image")
	#Getting width, height and yaw of image
	data = get_data(xml)
	width = int(data[0])
	height = int(data[1])
	yawdeg = float(data[2])
	for prediction in predictions:
		prediction = prediction.strip()
		coord = prediction.split(',')
		output = complete_path +"/" + pano + "_crop" + str(prediction)
		imagex = float(coord[0])
		imagey = float(coord[1])
		if im != None:
			utils.make_single_crop(im,width,height,depth,pano, imagex, imagey, yawdeg, output)
			
path = "single/completelabels.csv" 
def get_results():
	if os.path.exists(path):
		os.remove(path)
	normal_path = "single"
	pred.single_crops(normal_path + "/",normal_path, "models/", verbose=True)

def read_complete_file():
	rows = []
	now = time.time()
	if os.path.exists(path):
		with open(path, 'r') as csvfile: 
			csvreader = csv.reader(csvfile) 
			for row in csvreader: 
				max = 3
				value = []
				for i in range(0, len(row)):
					if(i < 3):
						value.append(row[i])
					elif(row[i] > row[max]):
						max = i
				value.append(pytorch_label_from_int[max - 3])
				percentage = round(100.0 * (float(row[max]) + 10.0)/20.0)
				value.append(percentage)
				#print(str(row[max]) + " -> " + str(percentage))
				#print(value) 
				rows.append(value)
	took = time.time() - now 
	print(took)
	return rows

#Stores the pano_id, the x and y coordinates, the prediction and the confidence level
def exact_labels():
	rows = read_complete_file()
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

def write_summary_file(rows_dict, labels_list):
	path = "single/summary.csv"
	if os.path.exists(path):
        os.remove(path)
	with open(path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for labelrow in labels_list:
	            pano_id = labelrow[0]
	            x = labelrow[1]
	            y = labelrow[2]
	            complete = pano_id + "," + x + "," + y
	            cv_respone = rows_dict[complete]
	            label,confidence  = cv_respone.split(",")
	            value = [label] + [label == labelrow[3]] + [confidence]
	            writer.writerow(value)
	return path
'''
def import_model(model_name, verbose): 
	model_name = "20ep_sw_re18_2ff2"
    model_path = os.path.join(model_dir, model_name+'.pt')
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    if verbose:
        print "Building dataset and loading model..."
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 

 def write_predictions_to_file(predictions_dict, root_path, pred_file_name, verbose=False):
    path = os.path.join(root_path,pred_file_name)
    with open(path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for pano_id in predictions_dict.keys():
                predictions = predictions_dict[pano_id]
                count = 0
                for coords, prediction in predictions.iteritems():
                    if type(prediction) != list:
                        # this way we can write int labels too
                        prediction = [prediction]
                        x,y = coords.split(',')
                        row = [pano_id] + [x,y] + prediction
                        writer.writerow(row)
                        count += 1
            if verbose:
                print "\tWrote {} predictions to {}.".format(count, path)
    return path
    ''' 
def generate_test_labelrows():
	labelrow = []
	First = "4s6C3NR6YRvHCYKMM_00QQ"
	Prediction1 = {'11900.0,-1000.0': 'Obstruction'}
	for coord, label in Prediction1.items():
		x,y = coord.split(",")
		label = [First, x, y, label]
		labelrow.append(label)
	Second = "1a1UlhadSS_3dNtc5oI10Q"
	Prediction2 = {'700.0,-600.0': 'Missing Cut', '9900.0,-1000.0': 'Curb Cut', '1200.0,-700.0': 'Curb Cut', '1700.0,-500.0': 'Obstruction'}
	for coord, label in Prediction2.items():
		x,y = coord.split(",")
		label = [Second, x, y, label]
		labelrow.append(label)
	return labelrow

if __name__ == "__main__":
	#Run the entire program
	prediction = ['700.0,-600.0','9900.0,-1000.0','1200.0,-700.0','1700.0,-500.0']
	pano = "1a1UlhadSS_3dNtc5oI10Q"
	make_crop(prediction, pano)
	prediction = ['11900.0,-1000.0']
	pano = "4s6C3NR6YRvHCYKMM_00QQ"
	make_crop(prediction, pano)
	get_results()
	rows_dict = exact_labels()
	print(rows_dict)
	labels_list = generate_test_labelrows()
	write_summary_file(rows_dict, labels_list)
	#utils.clear_dir("single/crops")


#
'''
Prediction: {'11900.0,-1000.0': 'Obstruction'}
'''
#Second: 1a1UlhadSS_3dNtc5oI10Q
'''
Prediction: 
{'700.0,-600.0': 'Missing Cut', '9900.0,-1000.0': 'Curb Cut', '1200.0,-700.0': 'Curb Cut', '1700.0,-500.0': 'Obstruction'}
'''
