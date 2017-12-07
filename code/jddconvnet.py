import os,time
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import layers
from keras import models
from keras import regularizers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras

start_time = time.time()
def loadData(source_file, target_file, pre_file):
	source_data = pd.read_csv(source_file).replace([np.inf, -np.inf], 9999).fillna(0)
	target_data = pd.read_csv(target_file).fillna(0)
	pred_data = pd.read_csv(pre_file).replace([np.inf, -np.inf], 9999).fillna(0)


	source_data_array = np.log1p(np.array(source_data)).reshape(90993,4,10,1)
	target__data_array = np.array(target_data).reshape(90993, 1, 1, 1)
	pred_data_array = np.log1p(np.array(pred_data)).reshape(90993, 4, 10, 1)
	return source_data_array, target__data_array, pred_data_array


source_data, target_data, pred_data = loadData('../data/sourceData_8-10.csv',
												'../data/targetData_11.csv',
												'../data/preData_9-11.csv')

# training_features, testing_features, training_target, testing_target = train_test_split(source_data, target_data, test_size=0.2, random_state=42)
print("source shape is:", source_data.shape)
print("target shape is: ", target_data.shape)
print("pred shape is:", pred_data.shape)

# construct CNN model
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(4,10,1)))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(64,(1,3), activation='relu',kernel_regularizer=regularizers.l2(l=0.001)))
model.add(layers.MaxPool2D(1,2))
model.add(layers.Dense(1,activation='relu'))
print("summary is:", model.summary())


# training_features = training_features.reshape((72794, 3, 9, 1))
# testing_features = testing_features.reshape((18199,3,9,1))


model.compile(optimizer='adam', loss='mse')

callbacks = [
keras.callbacks.TensorBoard(
	# Log files will be written at this location
	log_dir='../log',
	# We will record activation histograms every 1 epoch
	histogram_freq=1,
	)
]


model.fit(source_data, target_data, epochs=500, batch_size=64,validation_split=0.1,verbose=1)
# test_loss= model.evaluate(testing_features, testing_target,batch_size=128)
# print("test_loss is: ", test_loss)

# prediction
pre = model.predict(pred_data)
print("pre result done.")
pre_pd = pd.DataFrame(np.array(pre).reshape(90993,1))
pre_pd[pre_pd <1] = 0
pre_pd.to_csv('../result/prediction_CNN.csv')
print("submission time is: ", round(((time.time() - start_time) / 60), 2))

