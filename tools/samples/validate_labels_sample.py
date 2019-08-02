import sys
sys.path.append("../")
from cv_tools import generate_validation_data

if __name__ == "__main__":
    print(generate_validation_data("labels-testing.csv","../panos/", "sample"))
