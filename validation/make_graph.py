from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from datetime import date
from scipy import stats
import get_validation_Data as data

today = str(date.today())

class Range: 

    def __init__(self, lower_bound, upper_bound):
        self.lower = lower_bound
        self.upper  = upper_bound
        self.labels = ["NoCurbRamp", "Null", "Obstacle", "CurbRamp", "SurfaceProblem"]
        features = len(self.labels)
        self.total = [0 for i in range(features)]
        self.accurate = [0 for i in range(features)]

    def inrange(self, value):
        return (value > self.lower and value <= self.upper)

    def addtodata(self, label, match):
        val = self.labels.index(label)
        self.total[val] += 1
        if(match):
            self.accurate[val] += 1
    
    def toString(self):
        return (str(self.total) + "\n" + str(self.accurate))
    
    def getvalfromlabel(self, label):
        con = self.labels.index(label)
        return (self.accurate[con], self.total[con])
    
    def getaverage(self):
        return (self.lower + self.upper)/2


def create_ranges():
    ranges = []
    current = 0.0
    while(current < 100.0):
        copy = Range(current, min(current + cover, 100.0))
        current += cover 
        ranges.append(copy)
    return ranges

def make_scatter(label, x, freq, bar_titles, ignore_null): 
    fig, ax = plt.subplots()
    if(len(x) == 0): 
        ax.set_title(label)
        plt.show()
        return
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, freq)
    ax.scatter(x, freq, color ='r', s = 25, marker = "o")
    xmin, xmax, ymin, ymax = plt.axis()
    lower = x[0] - cover
    upper = x[len(x) - 1] + cover
    ax.set(xlim=(0,100), ylim=(0,110))
    ax.set_xlabel('CV Confidence Level')
    ax.set_ylabel('User - Model Agreement Percentage')
    text = " Including Null"
    if(ignore_null):
        text = " Not Including Null"
    #print(text)
    label += text
    ax.set_title(label)
    ax.plot(x, freq)
    R_ValueText = "R Value: " + str(round(r_value,3))
    ax.annotate(R_ValueText, xy = (5, 100))
    '''
    xfac = 0.02 * (upper - lower)
    yfac = 2.5
    for i,cv in enumerate(x):
        ax.annotate(str(bar_titles[i]), xy=(cv - xfac, freq[i] + yfac))
    '''
    name = label + "_" + today + "_" + str(len(x)) + "_" + str(cover) + ".jpg"
    location = "C:\\Users\\deves\\OneDrive\\Pictures\\SideWalk\\" + name
    if os.path.exists(location):
        os.remove(location)
    plt.savefig(location) 
    plt.show()

def import_from_file(ignore_null, ranges):
    path = "single\\summary" + str(ignore_null) + ".csv"
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile) 
        for row in csvreader:
            cvlabel = row[3]
            userlabel = row[4]
            accurate = (cvlabel == userlabel)
            confidence = float(row[5])
            for val in ranges: 
                if val.inrange(confidence):
                    val.addtodata(cvlabel, accurate)
                

def make_graphs(ignore_null, ranges):
    import_from_file(ignore_null, ranges)
    for label in ranges[0].labels:
        if(label == "Null"):
            continue
        x = []
        y = []
        bar_titles = []
        for copy in ranges:
            (num, total) = copy.getvalfromlabel(label)
            if total != 0:
                x.append(copy.getaverage())
                percentage = (100.0 * num)/total
                y.append(percentage)
                bar_titles.append(total)
        make_scatter(label, x, y, bar_titles, ignore_null)

def make_both_graphs():
    ranges = create_ranges()
    ignore_null = True
    make_graphs(ignore_null, ranges)
    '''
    ranges = create_ranges()
    ignore_null = False
    make_graphs(ignore_null, ranges)
    '''

def make_validation_graphs(input_file, path_to_panos, date_after, numofboxes = 20):
    global cover
    cover = 100.0/numofboxes
    next = data.generate_results_data(input_file, date_after, path_to_panos)
    if(next):
        make_both_graphs()

if __name__ == "__main__":
    path_to_panos = "D:\\SeattleImages\\validation\\panos\\new_seattle_panos\\"
    date_after = "2018-06-28"
    input_file = "validations-seattle.csv"
    make_validation_graphs(input_file, path_to_panos, date_after)