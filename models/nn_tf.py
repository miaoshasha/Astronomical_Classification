import numpy as np
import tensorflow as tf
from tensorflow import keras


def nn_tf_model(input_data):
  train_data, train_labels, validation_data, validation_labels, test_data, test_labels = input_data
  model = keras.Sequential([
    keras.layers.Flatten(input_shape=(num_features,)),
    keras.layers.Dense(62, activation=None),
    keras.layers.LayerNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dense(62, activation=None),
    keras.layers.LayerNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dense(62, activation=None),
    keras.layers.LayerNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation=None),
    keras.layers.LayerNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_features)
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
  test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
  print('\nTest accuracy:', test_acc)
  
  probability_model = tf.keras.Sequential([model, 
                                         keras.layers.Softmax()])
  predictions = probability_model.predict(test_images)
  return (test_acc, predictions, history)
