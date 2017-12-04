import pandas as pd
from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
import numpy as np
from keras.utils.test_utils import keras_test
import time
# from keras.utils import plot_model


start_time = time.time()
def loadData(source_file, target_file, pre_file):
	source_data = pd.read_csv(source_file).replace([np.inf, -np.inf], 99999).fillna(0)
	target_data = pd.read_csv(target_file).replace([np.inf, -np.inf], 99999).fillna(0)
	pred_data = pd.read_csv(pre_file).replace([np.inf, -np.inf], 99999).fillna(0)
	
	source_data_array = np.array(source_data).reshape(90993,3,9)
	target__data_array = np.array(target_data).reshape(90993,1,1)
	pred_data_array = np.array(pred_data).reshape(90993,3,9)
	return source_data_array, target__data_array, pred_data_array

source_data, target_data,pred_data = loadData('../data/monthData/sourceData_8-10.csv',\
	'../data/monthData/targetData_11.csv','../data/monthData/preData_9-11.csv')
print("source shape is:", source_data.shape)
print("target shape is: ",target_data.shape)
print("pred shape is:", pred_data.shape)


k = 5
num_val_samples = len(source_data) // k
num_epochs = 200
all_scores = []
for i in range(k):
	print('processing fold #', i)
	# Prepare the validation data: data from partition # k
	val_data = source_data[i * num_val_samples: (i + 1) * num_val_samples]
	val_targets = target_data[i * num_val_samples: (i + 1) * num_val_samples]

	# Prepare the training data: data from all other partitions
	partial_train_data = np.concatenate([source_data[:i * num_val_samples], source_data[(i + 1) * num_val_samples:]],axis=0)
	partial_train_targets = np.concatenate([target_data[:i * num_val_samples],target_data[(i + 1) * num_val_samples:]],axis=0)
	
	# Build the Keras model (already compiled)
	model = SimpleSeq2Seq(input_dim=9,input_length=3, hidden_dim=10, output_length=1, output_dim=1,dropout=0.3, depth=3)
	# model = Seq2Seq(input_dim=9, input_length=3,hidden_dim=10, output_length=1, output_dim=1, depth=4)
	# model = AttentionSeq2Seq(input_dim=9, input_length=3, hidden_dim=10, output_length=1, output_dim=1, depth=3)
	model.compile(loss='mse', optimizer='adam')

	# Train the model (in silent mode, verbose=0)
	model.fit(partial_train_data, partial_train_targets,epochs=num_epochs, batch_size=128, verbose=1)

	# Evaluate the model on the validation data
	val_mse = model.evaluate(val_data, val_targets, verbose=1)
	print("val_mse is:", val_mse)
	all_scores.append(val_mse)

print("mean of mse is:", np.mean(all_scores))

pre = model.predict(pred_data)
print("pre result done.")
pre_pd = pd.DataFrame(np.array(pre).reshape(90993,1))
pre_pd[pre_pd <1] = 0
pre_pd.to_csv('../result/prediction_LSTM.csv')
print("submission time is: ", round(((time.time() - start_time) / 60), 2))

