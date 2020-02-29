import numpy as np
from preprocessing.astronomical_prepare import getData
from models.nn_tf import nn_tf_model
from postprocessing.confusion_matrix import plot_confusion_matrix

# run astronomical analysis
prepared_data = getData(test=False)
returned_data = nn_tf_model(prepared_data)  # 
test_labels = prepared_data[-1]
test_accuracy, predicted_labels, train_history = returned_data

# plot the confusion matrix
# Read saved true and predicted labels.
y_truth = test_labels.flatten()
y_pred = predicted_labels.argmax(axis=1)
print(y_truth, y_pred)

plot_confusion_matrix(y_truth, y_pred)