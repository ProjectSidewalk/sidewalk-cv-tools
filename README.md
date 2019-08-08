# Tohme-Tools 

Branch to run computer vision model on tohme dataset 

## Setup
## Anaconda Installation

Download Anaconda for your system [here](https://www.anaconda.com/distribution/)

Follow [these](https://docs.anaconda.com/anaconda/user-guide/tasks/switch-environment/) instructions to create a Python 2.7 environment

After entering the environment, install the requiremnts.txt using this command ```conda install --file requirements.txt```

PyTorch is not included in that file because the installation is system specific. Follow the instructions [here](https://pytorch.org/get-started/locally/) to install it. Make sure to select CUDA for faster Prediction.

## Run_analysis
The primary method is the run_analysis method in the Tohme_analysis.py file 

**Arguments:**
- *input\_data:* The path to the csv file that has the user labels for which validations need to be run. Please ensure that each row has the following format: ```Label_id, pano_id, Label_type, SV_X, SV_Y```. Each row can have additional information after these columns but these columns need to be present for the function to row. Please not that the label type is 1 for a Curb Ramp and 2 for a No Curb Ramp. An example row would look like this: ```474,5umV8SPGE1jidFGstzcQDA,1,1550,-1391```. The tohme_research_kotaro contains all the labels placed by researcher kotaro and can be used as an reference for the input csv file structure 
- *path_to_panos:* The path to the root folder where the panoramas are stored. The folder structure should look like this:
	```
	[pano-root-dir]
	├── [pano_id]
	│   ├── images
		 ├── pano.jpg
	│   ├── [pano_id].depth.txt
	│   ├── meta.xml
	```
	or in practice:
	```
	Panos
	├── 5J5Fm8t9Azuo1nA1_WpsGw
	│   ├── 1a1UlhadSS_3dNtc5oI10Q.jpg
	│   ├── images
		 ├── pano.jpg
	│   ├── 5J5Fm8t9Azuo1nA1_WpsGw.depth.txt
	│   ├── meta.xml
	```
- *path\_to\_summary:* The path to where the summary files which includes the results of the CV and comparsion with user needs to be saved needs to be saved. 

**Returns:**
- . The path where the summary file was created and each row of the summary file has the following format:  
	```Pano_id,SV_X, SV_Y, CV_Label,CV_confidence,Ground_Truth_Label```. An example row would look like: ```5J5Fm8t9Azuo1nA1_WpsGw,-3142,-486,CurbRamp,2.92,CurbRamp```. Please note that the confidence value is not a percentage and that the CV_Label could be label types other than Curb Ramp and No Curb Ramp 

