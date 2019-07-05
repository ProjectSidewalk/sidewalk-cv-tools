# sidewalk-cv-tools
Set of functions for using computer vision to label panoramas and validate human-labeled streetscape items.

## Setup
###Anaconda Installation

Download Anaconda for your system [here](https://www.anaconda.com/distribution/)

Follow [these](https://docs.anaconda.com/anaconda/user-guide/tasks/switch-environment/) instructions to create a Python 2.7 environment

After entering the environment, install the requiremnts.txt using this command ```conda install --file requirements.txt```

PyTorch is not included in that file because the installation is system specific. Follow the instructions [here](https://pytorch.org/get-started/locally/) to install it. Make sure to select CUDA for faster Prediction.

##Labeling
The function used to label a given panorama is ```pred_pano_labels``` in the file labeling/pred_pano_labels.py.

####Arguments:
- pano_id: The id of the pano that the prediction should be run on. It should be a 22 character string like: ```1a1UlhadSS_3dNtc5oI10Q```
- path\_to\_gsv\_scrapes: The path to the root folder where the panoramas are stored. The folder structure should look like this:
	```
	├── \[first 2 characters of pano_id\]
	│   ├── \[pano_id\].jpg
	│   ├── \[pano_id\].txt
	│   ├── \[pano_id\].xml
	```
	or in practice:
	```
	├── 1a
	│   ├── 1a1UlhadSS_3dNtc5oI10Q.jpg
	│   ├── 1a1UlhadSS_3dNtc5oI10Q.txt
	│   ├── 1a1UlhadSS_3dNtc5oI10Q.xml
	```
- num_threads (optional): This controls the number of CPU threads used in multithreaded workloads. The default is 4. For optimal speed, set this to the number of *logical* cores your cpu has
- GSV_IMAGE_WIDTH: The width of the panorama image in pixels
- GSV_IMAGE_HEIGHT: The height of the panorama image in pixels
- model_dir: The root directory that the model is stored in
- save_labeled_pano (optional): This controls whether the function saves a panorama with the labels drawn on it to ```temp/viz``` for debugging. The default is ```python False```.
- verbose (optional): This enables/disables debugging printouts. The default is ```python False```

####Returns:
- A dict that stores the XY coordinate of each predicted label and the corresponding string label. Ex:
		```{'700.0,-600.0': 'Missing Cut', '9900.0,-1000.0': 'Curb Cut', '1200.0,-700.0': 'Curb Cut', '1700.0,-500.0': 'Obstruction'}```
		This can be converted to XY pixel coordinats on the panorama using this formula:
			First convert the dict key which is a string to two numbers: ```sv_x, sv_y = map(float, coords.split(','))```
			```
			x = ((float(pano_yaw_deg) / 360) * GSV_IMAGE_WIDTH + sv_x) % GSV_IMAGE_WIDTH
    		y = GSV_IMAGE_HEIGHT / 2 - sv_y
    		```
    		Where ```GSV_IMAGE_WIDTH``` and ```GSV_IMAGE_HEIGHT``` are the width and height of the panorama, and ```pano_yaw_deg``` is the rotation of the panorama from true north.

####Examples:
There are two examples in ```labeling/samples```. One uses a lower resolution DC or Newberg image, while the other uses a higher resolution Seattle image.

If you want to write your own code that uses ```pred_pano_labels```, and you need to save the labeled pano to disk, make sure to copy the ```roboto.tff``` file into the same directory as the program calling the function to avoid errors.

##Validation
_Needs to be coded_