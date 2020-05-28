import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

data_files=[]
for root, dirs, files in os.walk("."):
    for filename in files:
    	if ".pkl" and "False" in filename:    # easily choose which files u want to compare
        	data_files.append(filename)


fig, ax = plt.subplots()

ax.set(xlabel='Communication Rounds', 
	    ylabel='Test Accuracies',
        title='Experiments')
ax.hlines(93,0,50, colors='r',linestyle = 'dashed', label='Target Accuracy')


plot_data=dict()
for filename in data_files:
	pickle_in = open(filename,'rb')
	data = pickle.load(pickle_in)
	plot_data.update({filename:data['test_accuracies']}) 

	
		
for exp_name, data_set in plot_data.items():
	if "quantize_float16" in exp_name:
		ax.plot(data_set,label = exp_name, linestyle ='--')

	else:
		ax.plot(data_set,label = exp_name, linestyle = '-')

ax.legend(loc = 'lower right')
ax.grid()
plt.show()

