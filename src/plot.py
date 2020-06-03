import os
import pickle

import matplotlib.pyplot as plt

from definitions import ROOT_DIR

data_files = []
dir_path = os.path.join(ROOT_DIR, "outputs")
for filename in os.listdir(dir_path):
    if ".pkl" and "non_iid_mix_True" in filename:  # easily choose which files u want to compare
        data_files.append(os.path.join(dir_path, filename))

fig, ax = plt.subplots()

ax.set(xlabel='Communication Rounds',
       ylabel='Test Accuracies',
       title='Experiments')
ax.hlines(93, 0, 50, colors='r', linestyle='dashed', label='Target Accuracy')

plot_data = dict()
for filename in data_files:
    pickle_in = open(filename, 'rb')
    data = pickle.load(pickle_in)
    plot_data.update({filename: data['test_accuracies']})

for exp_name, data_set in plot_data.items():
    if "quantize_float16" in exp_name:
        ax.plot(data_set, label=exp_name, linestyle='--')

    else:
        ax.plot(data_set, label=exp_name, linestyle='-')

ax.legend(loc='lower right')
ax.grid()
fig.tight_layout()
plt.show()
