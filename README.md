# Sidewalk-CV-Tools
Set of functions for using computer vision to label panoramas and validate human-labeled streetscape items.

## Setup
## Anaconda Installation

Download Anaconda for your system [here](https://www.anaconda.com/distribution/)

Follow [these](https://docs.anaconda.com/anaconda/user-guide/tasks/switch-environment/) instructions to create a Python 2.7 environment

After entering the environment, install the requiremnts.txt using this command ```conda install --file requirements.txt```

PyTorch is not included in that file because the installation is system specific. Follow the instructions [here](https://pytorch.org/get-started/locally/) to install it. Make sure to select CUDA for faster Prediction.

## Labeling
There are 3 functions used in labeling. All three are in the file ```resources/cv_tools.py```

### pred\_pano\_labels
An all in one funtion that predicts the labels for a panorama and returns them.

**Arguments:**
- *pano\_id:* The id of the pano that the prediction should be run on. It should be a 22 character string like: ```1a1UlhadSS_3dNtc5oI10Q```
- *path\_to\_gsv\_scrapes:* The path to the root folder where the panoramas are stored. The folder structure should look like this:
	```
	[pano-root-dir]
	├── [first 2 characters of pano_id]
	│   ├── [pano_id].jpg
	│   ├── [pano_id].txt
	│   ├── [pano_id].xml
	```
	or in practice:
	```
	panos
	├── 1a
	│   ├── 1a1UlhadSS_3dNtc5oI10Q.jpg
	│   ├── 1a1UlhadSS_3dNtc5oI10Q.txt
	│   ├── 1a1UlhadSS_3dNtc5oI10Q.xml
	```
- *num\_threads (optional):* This controls the number of CPU threads used in multithreaded workloads. The default is 4. For optimal speed, set this to the number of *logical* cores your cpu has
- *model\_dir:* The root directory that the model is stored in
- *save\_labeled\_pano (optional):* This controls whether the function saves a panorama with the labels drawn on it to ```temp/viz``` for debugging. The default is ```False```.
- *verbose (optional):* This enables/disables debugging printouts. The default is ```False```

**Returns:**
- A tuple that stores the XY coordinate of each predicted label and the corresponding string label. Ex:
		```{(700.0,-600.0): 'Missing Cut', (9900.0,-1000.0): 'Curb Cut', (1200.0,-700.0): 'Curb Cut', '1700.0,-500.0': (Obstruction)}```
		This can be converted to XY pixel coordinats on the panorama using this formula:
			```
			x = ((float(pano_yaw_deg) / 360) * GSV_IMAGE_WIDTH + sv_x) % GSV_IMAGE_WIDTH
    		y = GSV_IMAGE_HEIGHT / 2 - sv_y
    		```
    		Where ```GSV_IMAGE_WIDTH``` and ```GSV_IMAGE_HEIGHT``` are the width and height of the panorama, and ```pano_yaw_deg``` is the rotation of the panorama from true north. Each of those variables are in the ```.xml``` file that came with the panorama.

**Examples:**
```full_pipeline_sample.py``` in ```tools/samples/``` runs the full pipeline on the panoramas ```1a1UlhadSS_3dNtc5oI10Q```, and ```4s6C3NR6YRvHCYKMM_00QQ```. The first image is a lower resolution DC/Newberg image, while the second is a higher resolution Seattle image.


### batch\_save\_pano\_labels
A function that takes a list of panoramas, predicts the labels for each panorama and saves the *raw, unclustered* labels to disk.

**Arguments:**
- *pano\_ids:* A list of pano ids that the prediction should be run on. Each pano id should be a 22 character string like: ```1a1UlhadSS_3dNtc5oI10Q```
- *path\_to\_gsv\_scrapes:* The path to the root folder where the panoramas are stored. The folder structure should look like this:
	```
	[pano-root-dir]
	├── [first 2 characters of pano_id]
	│   ├── [pano_id].jpg
	│   ├── [pano_id].txt
	│   ├── [pano_id].xml
	```
	or in practice:
	```
	panos
	├── 1a
	│   ├── 1a1UlhadSS_3dNtc5oI10Q.jpg
	│   ├── 1a1UlhadSS_3dNtc5oI10Q.txt
	│   ├── 1a1UlhadSS_3dNtc5oI10Q.xml
	```
- *num\_threads (optional):* This controls the number of CPU threads used in multithreaded workloads. The default is 4. For optimal speed, set this to the number of *logical* cores your cpu has
- *model\_dir:* The root directory that the model is stored in
- *save\_labeled\_pano (optional):* This controls whether the function saves a panorama with the labels drawn on it to ```temp/viz``` for debugging. The default is ```False```.
- *verbose (optional):* This enables/disables debugging printouts. The default is ```False```

**Examples:**
```save_labels_sample.py``` in ```tools/samples/``` predicts and saves labels for the panoramas ```4s6C3NR6YRvHCYKMM_00QQ```, and ```M7Vsy3VP0X-9LAmkpsOz0w```. The first image is a lower resolution DC/Newberg image, while the second is a higher resolution Seattle image.


### get\_pano\_labels
A function that takes a pano id, and returns the filtered predictions.

**Arguments:**
- *pano\_ids:* A list of pano ids that the prediction should be run on. Each pano id should be a 22 character string like: ```1a1UlhadSS_3dNtc5oI10Q```
- *path\_to\_gsv\_scrapes:* The path to the root folder where the panoramas are stored. The folder structure should look like this:
	```
	[pano-root-dir]
	├── [first 2 characters of pano_id]
	│   ├── [pano_id].jpg
	│   ├── [pano_id].txt
	│   ├── [pano_id].xml
	│   ├── [pano_id]_labels.csv
	```
	or in practice:
	```
	panos
	├── 1a
	│   ├── 1a1UlhadSS_3dNtc5oI10Q.jpg
	│   ├── 1a1UlhadSS_3dNtc5oI10Q.txt
	│   ├── 1a1UlhadSS_3dNtc5oI10Q.xml
	│   ├── 1a1UlhadSS_3dNtc5oI10Q_labels.csv
	```
- *b\_box (optional):* This specifies the bounding box which the labels must be within in order to be returned.
- *radius (optional):* This specifies the clustering radius. If this is left blank, no bounding box will be used to cull predictions
- *clip\_val (optional):* This specifies the clip value; the value which a predicion must be above for it not to be ignored.
- *num\_threads (optional):* This controls the number of CPU threads used in multithreaded workloads. The default is 4. For optimal speed, set this to the number of *logical* cores your cpu has
- *model\_dir:* The root directory that the model is stored in
- *save\_labeled_pano (optional):* This controls whether the function saves a panorama with the labels drawn on it to ```temp/viz``` for debugging. The default is ```False```.
- *verbose (optional):* This enables/disables debugging printouts. The default is ```False```

**Returns:**
- A tuple that stores the XY coordinate of each predicted label and the corresponding string label. Ex:
		```{(700.0,-600.0): 'Missing Cut', (9900.0,-1000.0): 'Curb Cut', (1200.0,-700.0): 'Curb Cut', '1700.0,-500.0': (Obstruction)}```
		This can be converted to XY pixel coordinats on the panorama using this formula:
			```
			x = ((float(pano_yaw_deg) / 360) * GSV_IMAGE_WIDTH + sv_x) % GSV_IMAGE_WIDTH
    		y = GSV_IMAGE_HEIGHT / 2 - sv_y
    		```
    		Where ```GSV_IMAGE_WIDTH``` and ```GSV_IMAGE_HEIGHT``` are the width and height of the panorama, and ```pano_yaw_deg``` is the rotation of the panorama from true north. Each of those variables are in the ```.xml``` file that came with the panorama.

**Examples:**
```save_labels_sample.py``` in ```tools/samples/``` reads the predicted labels for the panoramas ```4s6C3NR6YRvHCYKMM_00QQ```, and ```M7Vsy3VP0X-9LAmkpsOz0w```. It runs Non-Maximum Supression to reduce false positives and returns the filtered result.

**NOTE FOR DISPLAYING LABELS ON PANORAMAS:**
If you want to write your own code that uses ```pred_pano_labels```, and you need to save the labeled pano to disk, make sure to copy the ```roboto.tff``` file into the same directory as the program calling the function to avoid errors.

## Validation
There is one main function for performing validation and it is in ```resources/cv_tools.py```
### generate\_validation\_data
A function that takes in a CSV file containing user validations and generates a resulting csv file comparing the CV prediction to the user validations 

**Arguments:**
- *input\_data:* The path to the csv file that has the user labels for which validations need to be run. Please ensure that each row has the following format: ```Timestamp, label_id, pano_id, SV_X, SV_Y, label_type```. Each row can have additional information after these columns but these columns need to be present for the function to row. Please ensure that the Timestamp has the following format: ```YYYY-MM-DD HR:MM:SS(AM OR PM)``` to get valid results. The label types must be among the following to get proper results from the CV model: ``` NoCurbRamp, Obstacle, CurbRamp, SurfaceProblem ```. An example row would look like this: ```2019-04-16 2:48:37AM, 3215, treyXG0iNYWO-B5UX1nRbw,3912,-276,CurbRamp```. The pano_id should be a string containing exactly 22 characters like the one shown in the example and please ensure that the streetview coordinates are integers and not floats or doubles. Please also make sure to include column headers for otherwise the first item in the csv file will be ignored from getting the results. 
- *path\_to\_gsv\_scrapes:* The path to the root folder where the panoramas are stored. The folder structure should look like this:
	```
	[pano-root-dir]
	├── [first 2 characters of pano_id]
	│   ├── [pano_id].jpg
	│   ├── [pano_id].txt
	│   ├── [pano_id].xml
	│   ├── [pano_id]_labels.csv
	```
	or in practice:
	```
	panos
	├── 1a
	│   ├── 1a1UlhadSS_3dNtc5oI10Q.jpg
	│   ├── 1a1UlhadSS_3dNtc5oI10Q.txt
	│   ├── 1a1UlhadSS_3dNtc5oI10Q.xml
	│   ├── 1a1UlhadSS_3dNtc5oI10Q_labels.csv
	```
- *path\_to\_summary:* The complete path to the folder where the summary.csv file which contains information about the CV's prediction is saved. Please see the returns section about more information about the summary file. 
- *number\_agree (optional):* This is the minimum number of labels that must be present in the input file for each unique location (which can basicially be though as pano_id + sv_x + sv_y) before the CV's get predictions on the label. The default is ```1``` so that every label in the input file is analyzed. 
- *num\_threads (optional):* This is the number of threads to run while making the crops necessary for performing validations. This value will change based on the hardware specifications of each device but the default value is ```4```. The num threads for your device should be equal to the number of logicial processors you have present which can be found at Task Manager -> Peformance -> CPU to get the most optimial performance
- *date\_after (optional):* If a value is specified then the program only considers labels who time stamp is after the date_after value. Otherwise, if not value is specified then program considers all the values in the input_file. 
- *verbose (optional):* This enables/disables debugging printouts. The default is ```False```

**Returns:**
The path to a csv file called ```summary.csv``` in the folder specificed in the path\_to\_summary argument. Each row of the ouput file will have the following format ```label_id, pano_id, SV_X, SV_Y, CVLabel, CVLabel_Confidence, UserLabel, UserLabel_Confidence, Priorirty Score``` The user label confidence is the confidence value that the cv gave for the user label type while the CV Label Confidence is the confidence value for the label type that the CV think the object is most likely to be and may often overlap with the user label type. The confidence values are the raw confidence values (can be positive or negative) rounded to 2 decimal places. An example row looks like ```9,	4s6C3NR6YRvHCYKMM_00QQ,3400,-400,CurbRamp,0.95,Obstacle,0.77```

 The SV_X and the SV_Y be converted to XY pixel coordinats on the panorama using this formula:
			```
			x = ((float(pano_yaw_deg) / 360) * GSV_IMAGE_WIDTH + sv_x) % GSV_IMAGE_WIDTH
    		y = GSV_IMAGE_HEIGHT / 2 - sv_y
    		```
    		Where ```GSV_IMAGE_WIDTH``` and ```GSV_IMAGE_HEIGHT``` are the width and height of the panorama, and ```pano_yaw_deg``` is the rotation of the panorama from true north. Each of those variables are in the ```.xml``` file that came with the panorama. 

The priority score is calculated by using the CVLabel, CVLabel_Confidence, UserLabel, UserLabel_Confidence values in such a way that labels on which the CV system disagrees with the user label are given high priority, lower values for cv label confidence and user label confidence are given higher priority. Additionally, if the cv system disagrees with the user label then labels with higher differences between cv label confidences and user label confidence are given more priority because a high value indiciates that the CV is sure of its prediction and thus it disagreeing with the user is more striking. 

**Examples:**
```validations_seattle_runner.py``` in ```tools/samples/``` generates a summary csv file for the labels-seattle.csv in the tools folder. The validations seattle contains correctly formatted input data for user labels since April 16
