import numpy as np
from preprocessing.astronomical_prepare import getData
from models.nn_tf import nn_tf_model


# run astronomical analysis
prepared_data = getData()
returned_data = nn_tf_model(prepared_data)