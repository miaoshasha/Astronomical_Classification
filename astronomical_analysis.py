import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, RobustScaler

# import data
#data_path = "https://drive.google.com/open?id=0B6cw6JxDC2-5UVZyc3VjS2daVzg"
data_path = "/Users/yishasun/Documents/mls/trainingData.csv"
df = pd.read_csv(data_path, delimiter=",", header=0)
# shuffle
df = df.sample(frac=1).reset_index(drop=True)
# pop out 'type' because it's sdss classification
df.pop('type')
desc = df.describe()
print(desc)
# split for train, val and test data : balanced sampling for each class
num_classes = int(np.max(df.loc[:, "truth"]))+1
features = [x for x in df.columns if x not in ["truth"]]
num_features = len(features)
print("We have {} features, {} classes, {} data entries".format(num_features, num_classes, df.shape[0]))


train_fraction = 0.8
validation_fraction = 0.1
test_fraction = 0.1
print("Train, validation and test split: {} : {} : {}".format(
  train_fraction, validation_fraction, test_fraction))

groups = df.groupby("truth")
countdf = pd.DataFrame(groups.count())[features[0]].reset_index().rename(columns={features[0]:'frequency'})
df = df.merge(countdf, on="truth", how="left")

#training
groups = df.groupby("truth")
train_df = pd.DataFrame(groups.apply(lambda x: x.iloc[:int(x['frequency'].iloc[0]*train_fraction),:])).reset_index(drop=True)
train_df = train_df.sample(frac=1).reset_index(drop=True)
train_size = train_df.shape[0]

# validation
validation_df = pd.DataFrame(groups.apply(lambda x: x.iloc[int(x['frequency'].iloc[0]*train_fraction):int(x['frequency'].iloc[0]*float(train_fraction+validation_fraction)),:])).reset_index(drop=True)
validation_size = validation_df.shape[0]

# test 
test_df = pd.DataFrame(groups.apply(lambda x : x.iloc[int(x['frequency'].iloc[0]*(train_fraction+validation_fraction)):,:])).reset_index(drop=True)
test_size = test_df.shape[0]
print("Train size: {}, validation size: {}, test size: {}".format(train_size, validation_size, test_size))

# remove outliers 
clean_data = True
if clean_data:
  #compute lower quantile to cut out the ridiculous -9999.0:
  groups = train_df[features+['truth']].groupby("truth")
  loquantiledf = pd.DataFrame(groups.apply(lambda x: x[features].quantile(0.2))).reset_index()    
  loquantiledf.rename(columns={x:x+'_loq' for x in features},inplace=True)
  train_df = train_df.merge(loquantiledf, on='truth', how='left')
  validation_df = validation_df.merge(loquantiledf, on='truth', how='left')
  test_df = test_df.merge(loquantiledf, on='truth', how='left')
  test_ = test_df[features[0]] < test_df[features[0]+'_loq']
  
  for feature in features:
    train_df.loc[train_df[feature] < train_df[feature+'_loq'], feature] = train_df[feature+'_loq']
    validation_df.loc[validation_df[feature] < validation_df[feature+'_loq'], feature] = validation_df[feature+'_loq']
    test_df.loc[test_df[feature] < test_df[feature+'_loq'], feature] = test_df[feature+'_loq']

# normalize
scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
scaler.fit(train_df.loc[:, features])   # get mean and std
train_data = scaler.transform(train_df.loc[:,features].values)  # apply whitening
train_labels = np.expand_dims(train_df.loc[:,'truth'].values, axis=1).astype(np.int32)

validation_data = scaler.transform(validation_df.loc[:,features].values)  # apply whitening
validation_labels = np.expand_dims(validation_df.loc[:,'truth'].values, axis=1).astype(np.int32)

# normalize the test data as well
test_images = scaler.transform(test_df.loc[:,features].values)
test_labels = np.expand_dims(test_df.loc[:,"truth"].values, axis=1).astype(np.int32)

# build model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(num_features,)),
    keras.layers.Dense(62, activation=None, kernel_regularizer=None, bias_regularizer=None),
    keras.layers.LayerNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(62, activation=None, kernel_regularizer=None, bias_regularizer=None),
    keras.layers.LayerNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_features)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train 
model.fit(train_data, train_labels, epochs=10)

# validation
validation_loss, validation_acc = model.evaluate(validation_data,  validation_labels, verbose=2)
print('\nTest accuracy:', validation_acc)

# test
#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#print('\nTest accuracy:', test_acc)


# plot training his and follow-up analysis