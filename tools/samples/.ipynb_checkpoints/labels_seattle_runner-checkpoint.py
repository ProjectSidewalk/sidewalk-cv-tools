import sys
sys.path.append("../")
sys.path.append("resources/")
import cv_tools as data

def main():
    path_to_panos = "panos/"
    date_after = "2018-06-28"
    path_to_summary = "single"
    input_file = "labels-testing.csv"
    print(data.generate_validation_data(input_file,path_to_panos, path_to_summary))