import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import sys

'''
To get the confusion matrix, please type in the command line: 
        python confusion_matrix.py file_with_true_labels.txt file_with_predicted_labels.txt
'''

#Confusion matrix.
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    np.set_printoptions(precision=2)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, float("{0:.3f}".format(cm[i, j])),
                 horizontalalignment="center",
                 color="white" if float("{0:.3f}".format(cm[i, j])) > float("{0:.3f}".format(thresh)) else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Read saved true and predicted labels.
y_truth = np.loadtxt(sys.argv[1]) 
y_pred = np.loadtxt(sys.argv[2])

# Compute confusion matrix.
cmat = confusion_matrix(y_truth, y_pred)
print(cmat)

# Name of the classes in data.
class_names = ['Other', 'Star', 'Galaxy']  

# Plot normalized confusion matrix
fig = plt.figure()
plot_confusion_matrix(cmat, classes=class_names, normalize=True, title='Normalized confusion matrix')
plt.show()
fig.savefig('confusion_matrix.jpeg', dpi=120, bbox_inches='tight')
