import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools

'''
To get the information about importance of features (variables), please type in the command line: 
        python plot_sorted_weights.py file_with_weights.txt
'''
#Load the weights from txt file.
mat = np.loadtxt('weights.txt') 

#Compute sum of weights of the last layer for each feature and its averages.
all_sums = np.sum(np.absolute(mat), axis=1)
all_avg = [all_sums[i]/3 for i in range(0, all_sums.shape[0])]

#Sort the avearges in the decreasing order.
ind = np.array(all_avg).argsort()[::-1]
all_avg = sorted(all_avg, reverse=True)

#Add labels to original features.
labels_old = ['ra', 'dec', 'psfMag_u', 'psfMagErr_u', 'psfMag_g', 'psfMagErr_g', 'psfMag_r', 'psfMagErr_r', 'psfMag_i', 'psfMagErr_i', 'psfMag_z', 'psfMagErr_z', 'modelMag_u', 'modelMagErr_u', 'modelMag_g', 'modelMagErr_g', 'modelMag_r', 'modelMagErr_r', 'modelMag_i', 'modelMagErr_i', 'modelMag_z', 'modelMagErr_z', 'petroRad_u', 'petroRadErr_u', 'petroRad_g', 'petroRadErr_g', 'petroRad_r', 'petroRadErr_r', 'petroRad_i', 'petroRadErr_i', 'petroRad_z', 'petroRadErr_z', 'q_u', 'qErr_u', 'q_g', 'qErr_g', 'q_r', 'qErr_r', 'q_i', 'qErr_i', 'q_z', 'qErr_z', 'u_u', 'uErr_u', 'u_g', 'uErr_g', 'u_r', 'uErr_r', 'u_i', 'uErr_i', 'u_z', 'uErr_z', 'mE1_u', 'mE1_g', 'mE1_r', 'mE1_i', 'mE1_z', 'mE2_u', 'mE2_g', 'mE2_r', 'mE2_i', 'mE2_z']
labels = [labels_old[i] for i in ind]
print(labels)
print(ind)

fig = plt.figure()
x = [i for i in range(0, all_sums.shape[0])]
plt.bar(x, all_avg, 0.45, color='r')
plt.xticks(x, labels, rotation='vertical')
plt.subplots_adjust(bottom=0.65)
plt.tick_params(axis='x', labelsize=8)
plt.show()
fig.savefig('weights.jpeg', dpi=120, bbox_inches='tight')
