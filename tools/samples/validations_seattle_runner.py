import sys
sys.path.append("../")
sys.path.append("resources\\")
import cv_tools as data
path_to_panos = "D:\\SeattleImages\\validation\\panos\\new_seattle_panos\\"
date_after = "2018-06-28"
path_to_summary = "single\\"
input_file = "validations-seattle.csv"
print(data.generate_results_data(input_file,path_to_panos, path_to_summary, number_agree = 5))