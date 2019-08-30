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
	```
	or in practice:
	```
	panos
	├── 1a
	│   ├── 1a1UlhadSS_3dNtc5oI10Q.jpg
	│   ├── 1a1UlhadSS_3dNtc5oI10Q.txt
	│   ├── 1a1UlhadSS_3dNtc5oI10Q.xml
	```
- *path\_to\_summary:* The complete path to the folder where the summary.csv file which contains information about the CV's prediction is saved. Please see the returns section about more information about the summary file. 
- *number\_agree (optional):* This is the minimum number of labels that must be present in the input file for each unique location (which can basicially be though as pano_id + sv_x + sv_y) before the CV's get predictions on the label. The default is ```1``` so that every label in the input file is analyzed. 
- *num\_threads (optional):* This is the number of threads to run while making the crops necessary for performing validations. This value will change based on the hardware specifications of each device but the default value is ```4```. The num threads for your device should be equal to the number of logicial processors you have present which can be found at Task Manager -> Peformance -> CPU to get the most optimial performance
- *date\_after (optional):* If a value is specified then the program only considers labels who time stamp is after the date_after value. Otherwise, if not value is specified then program considers all the values in the input_file. 
- *verbose (optional):* This enables/disables debugging printouts. The default is ```False```

**Returns:**
The path to a csv file called ```summary.csv``` in the folder specificed in the path\_to\_summary argument. Each row of the ouput file will have the following format ```label_id,CVLabel, CVLabel_Confidence, UserLabel, UserLabel_Confidence, Priorirty Score, No Curb Ramp Confidence, Null Confidence, Obstacle Confidence, Curb Ramp confidence, Surface Problem confidence``` The user label confidence is the confidence value that the cv gave for the user label type while the CV Label Confidence is the confidence value for the label type that the CV think the object is most likely to be and may often overlap with the user label type. The confidence values are the raw confidence values (can be positive or negative) rounded to 2 decimal places. An example row looks like ```9,CurbRamp,0.95,Obstacle,0.77,0.7541790510209603,-0.86,0.82,0.77,0.95,-2.6```

 The SV_X and the SV_Y be converted to XY pixel coordinats on the panorama using this formula:
			```
			x = ((float(pano_yaw_deg) / 360) * GSV_IMAGE_WIDTH + sv_x) % GSV_IMAGE_WIDTH
    		y = GSV_IMAGE_HEIGHT / 2 - sv_y
    		```
    		Where ```GSV_IMAGE_WIDTH``` and ```GSV_IMAGE_HEIGHT``` are the width and height of the panorama, and ```pano_yaw_deg``` is the rotation of the panorama from true north. Each of those variables are in the ```.xml``` file that came with the panorama. 

The priority score is calculated by using the CVLabel, CVLabel_Confidence, UserLabel, UserLabel_Confidence values in such a way that labels on which the CV system disagrees with the user label are given high priority, lower values for cv label confidence and user label confidence are given higher priority. Additionally, if the cv system disagrees with the user label then labels with higher differences between cv label confidences and user label confidence are given more priority because a high value indiciates that the CV is sure of its prediction and thus it disagreeing with the user is more striking. 