import pandas as pd
from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
import numpy as np
from keras.utils.test_utils import keras_test
import time


start_time = time.time()
def loadData(source_file, target_file, pre_file):
	source_data = pd.read_csv(source_file)
	target_data = pd.read_csv(target_file)
	pred_data = pd.read_csv(pre_file)
	source_data_array = np.array(source_data).reshape(90993,1,21)
	target__data_array = np.array(target_data).reshape(90993,1,1)
	pred_data_array = np.array(pred_data).reshape(90993,1,21)
	return source_data_array, target__data_array, pred_data_array

source_data, target_data,pred_data = loadData('../data/monthData/sourceData_8-10.csv',\
	'../data/monthData/targetData_11.csv','../data/monthData/preData_9-11.csv')
print("source shape is:", source_data.shape)
print("target shape is: ",target_data.shape)
print("pred shape is:", pred_data.shape)


# model = SimpleSeq2Seq(input_dim=21, hidden_dim=10, output_length=1, output_dim=1)
# model.compile(loss='mse', optimizer='rmsprop')


model = AttentionSeq2Seq(input_dim=21, input_length=1, hidden_dim=64, output_length=1, output_dim=1, depth=4)
model.compile(loss='mse', optimizer='adam')


model.fit(source_data, target_data, nb_epoch=2000, verbose=1)
pre = model.predict(pred_data)
print("pre result done.")
pd.DataFrame(np.array(pre).reshape(90993,1)).to_csv('../result/prediction_LSTM.csv')
print("submission time is: ", round(((time.time() - start_time) / 60), 2))

