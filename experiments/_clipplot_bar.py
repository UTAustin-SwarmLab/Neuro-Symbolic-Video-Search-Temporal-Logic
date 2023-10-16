import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('/opt/Neuro-Symbolic-Video-Frame-Search/experiments/res_clipGprop1.pkl', 'rb') as file:
    res_dict = pickle.load(file)

# iterate through the dictionary 
category = 'accuracy'
a ,b, c = [],[], []


for key, value in res_dict.items():
    a.append(key)
    b.append(value[category][0])
    c.append(value[category][1])


# set bar width
# set y limits
plt.ylim(0.7, 1)
plt.bar(a, b, width=0.001)
# plt.errorbar(a, b, yerr=c, fmt="o", color="r")

 
# save plot in this directory
plt.savefig('/opt/Neuro-Symbolic-Video-Frame-Search/experiments/boxplot.png')
