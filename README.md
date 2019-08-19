# Sidewalk-CV-Tools
Set of functions for using computer vision to label panoramas and validate human-labeled streetscape items.

## Setup
## Anaconda Installation

Download Anaconda for your system [here](https://www.anaconda.com/distribution/)

Follow [these](https://docs.anaconda.com/anaconda/user-guide/tasks/switch-environment/) instructions to create a Python 2.7 environment

After entering the environment, install the requiremnts.txt using this command ```conda install --file requirements.txt```

PyTorch is not included in that file because the installation is system specific. Follow the instructions [here](https://pytorch.org/get-started/locally/) to install it. Make sure to select CUDA for faster Prediction.


### get\_user\_quality\_data
Contains the method run_get_another_label which is used to populate the results folder with the data regarding worker quality and validation predictions 

**Arguments:**
- *input\_file:* It is the CSV file that contains the user validations that needs to be analyzed and for whom results need to be determined. Each row should have the format ```user_id, time_stamp, label_id, pano_id, sv_x, sv_y, agree_user, label_type_user``` and a example row may look this: ```2a783c12-395c-4cdc-b954-b5a225771360,17:52.4,43729,YDHfcGyL-LbQfWwcn9JMug,12532,-331,disagree,SurfaceProblem```. Please make sure to include headers for each column as that is what the program relies on to determine these fields 

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
- *categories\_file:* A file contains the categories of all the label types that is going to be appear with the user labels. There is a set of 4 label types with each set being like (Curb Ramp, Not-Curb Ramp). A working file is attached is present in the folder and just needs to be specified. 
- *path\_to\_ground\_file (optional):*  It is the CSV file that contains golden (ground truth) values against which user performace can be better evaluated but the default value is set to None. Each row should have the form: ```user_id,pano_id,sv_x,sv_y,label_type, agree``` and a example row looks like: ```87833d72-b357-4e2c-81cd-23f58ff04c59,s_MaNEBOE3Jj37L-1F2mLw,6420,-1683,SurfaceProblem,TRUE```. Please make sure to include headers for each of the columns and make sure to have user_ids as otherwise ground truth users will be compared against themselves. 
- *computer\_vision (optional):* The default value is False but passing in True will pass in the Computer Vision system's prediction on the input_file validations as those of a user named CV 
- *verbose (optional):* This enables/disables debugging printouts. The default is ```False```

**Returns:**
- Files of user performance and predicitons of the actual validation type for each of the unique label locations in the results folder. Please see https://github.com/ipeirotis/Get-Another-Label/wiki/Output-Files for complete information about each of the files and data points 
