# Written by Kavi Dey August 2019

import sys
sys.path.append("../")
from cv_tools import generate_validation_data

def main(): 
    print(generate_validation_data("labels-testing.csv","../panos/", "sample"))

if __name__ == "__main__":
    main()