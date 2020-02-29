import numpy as np
import tensorflow as tf

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
import matplotlib.pyplot as plt


def nn_tf_model(input_data):
  train_data, train_labels, validation_data, validation_labels, test_data, test_labels = input_data
  num_features = train_data.shape[-1]
  model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(num_features,)),
    tf.keras.layers.Dense(62, activation=None),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(62, activation=None),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(62, activation=None),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation=None),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_features)
  ])
  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

  # print model summary
  model.summary()

  # train 
  batch_size=400
  history = model.fit(train_data, train_labels, epochs=8, verbose=1, shuffle=True, batch_size=batch_size,
    callbacks=[tfdocs.modeling.EpochDots()], validation_data=(validation_data, validation_labels))

  # test
  __, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
  print('\nTest accuracy:', test_acc)
  
  probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
  predictions = probability_model.predict(test_data)
  return (test_acc, predictions, history)
