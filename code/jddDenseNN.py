import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import matplotlib.pyplot as plt
from keras import regularizers
import pandas as pd
import numpy as np
from keras import layers
from keras import models
from sklearn.model_selection import train_test_split


def loadData(source_file, target_file, pre_file):
	source_data = pd.read_csv(source_file).replace([np.inf, -np.inf], 9999).fillna(0)
	target_data = pd.read_csv(target_file).fillna(0)
	pred_data = pd.read_csv(pre_file).replace([np.inf, -np.inf], 9999).fillna(0)

	source_data_array = np.log1p(np.array(source_data)).reshape(90993, 15)
	target__data_array = np.array(target_data).reshape(90993, 1)
	pred_data_array = np.log1p(np.array(pred_data)).reshape(90993, 15)
	return source_data_array, target__data_array, pred_data_array


source_data, target_data, pred_data = loadData('../data/userInfoSum_8-10.csv',
												'../data/targetData_11.csv',
												'../data/userInfoSum_9-11.csv')

training_features, testing_features, training_target, testing_target = train_test_split(source_data, target_data, test_size=0.2, random_state=42)
print("source shape is:", source_data.shape)
print("target shape is: ", target_data.shape)
print("pred shape is:", pred_data.shape)

# load Densely NN model
original_model = models.Sequential()
original_model.add(layers.Dense(128, kernel_regularizer= regularizers.l2(0.001),
								activation='relu', input_shape=(15, )))
original_model.add(layers.Dropout(0.4))
original_model.add(layers.Dense(128, kernel_regularizer= regularizers.l2(0.001),
								activation='relu'))
original_model.add(layers.Dropout(0.3))
original_model.add(layers.Dense(64,kernel_regularizer= regularizers.l2(0.001), activation='relu'))
original_model.add(layers.Dropout(0.2))
original_model.add(layers.Dense(32, activation='relu'))
original_model.add(layers.Dense(1, activation='sigmoid'))

# compile
original_model.compile(optimizer='adam', loss='mse')

# fit
original_hist = original_model.fit(training_features, training_target, epochs=200, verbose=0,
									batch_size=32, validation_data=(testing_features, testing_target))

# plot
epochs = range(0, 200)
original_val_loss = original_hist.history['loss']
plt.plot(epochs, original_val_loss, 'b',label='Original model')
plt.xlabel('Epochs')
plt.ylabel('training loss')
plt.legend()
plt.show()

