import sys
sys.path.append("../")
sys.path.append("resources/")
import cv_tools as data

def main():
    path_to_panos = "D:\\SeattleImages\\validation\\panos\\new_seattle_panos"
    date_after = "2018-06-28"
    path_to_summary = "viz"
    input_file = "Quality_Inferance_for_cv.csv"
    print(data.generate_validation_data(input_file,path_to_panos, path_to_summary,verbose = True))

main()