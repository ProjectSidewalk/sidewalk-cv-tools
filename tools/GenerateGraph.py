import matplotlib.pyplot as plt
import csv 

class Range: 

	def __init__(self, lower_bound, upper_bound):
		self.lower = lower_bound
		self.upper  = upper_bound
		self.labels = ["Missing Cut", "Null", "Obstruction", "Curb Cut", "Sfc Problem"]
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


numberofboxes = 20
cover = 100.0/numberofboxes
ranges = []
current = 0.0
while(current < 100.0): 
	
