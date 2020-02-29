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
def plot_confusion_matrix(y_truth, y_pred,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute confusion matrix.
    # Name of the classes in data.
    classes = ['Other', 'Star', 'Galaxy']  
    cmat = confusion_matrix(y_truth, y_pred)
    plt.imshow(cmat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    np.set_printoptions(precision=2)

    if normalize:
        cmat = cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cmat)

    thresh = cmat.max() / 2.
    for i, j in itertools.product(range(cmat.shape[0]), range(cmat.shape[1])):
        plt.text(j, i, float("{0:.3f}".format(cmat[i, j])),
                 horizontalalignment="center",
                 color="white" if float("{0:.3f}".format(cmat[i, j])) > float("{0:.3f}".format(thresh)) else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Plot normalized confusion matrix
    fig = plt.figure()
    plt.show()
    fig.savefig('confusion_matrix.jpeg', dpi=120, bbox_inches='tight')
