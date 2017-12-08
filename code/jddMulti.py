import os,time
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import Model
from keras import layers
from keras import Input
from keras import regularizers
import pandas as pd
import numpy as np

start_time = time.time()
# load data
def loadData(source_file, sum_file, target_file, pre_file):
    source_data = pd.read_csv(source_file).replace([np.inf, -np.inf], 9999).fillna(0)
    sum_data = pd.read_csv(sum_file).replace([np.inf, -np.inf], 9999).fillna(0)
    target_data = pd.read_csv(target_file).fillna(0)
    pred_data = pd.read_csv(pre_file).replace([np.inf, -np.inf], 9999).fillna(0)

    source_data_array = np.log1p(np.array(source_data)).reshape(90993,4,10,1)
    sum_data_array = np.log1p(np.array(sum_data)).reshape(90993,1,1,9)
    target__data_array = np.array(target_data).reshape(90993, 1, 1, 1)
    pred_data_array = np.log1p(np.array(pred_data)).reshape(90993, 4, 10, 1)
    return source_data_array, sum_data_array, target__data_array, pred_data_array


source_data, sum_data, target_data, pred_data = loadData('../data/sourceData_8-10.csv',
                                                '../data/userInfoSum_8-10.csv',
												'../data/targetData_11.csv',
												'../data/preData_9-11.csv')

print("source shape is:", source_data.shape)
print("sum_data shape is:", sum_data.shape)
print("target shape is: ", target_data.shape)
print("pred shape is:", pred_data.shape)

# ConvNet
month_input = Input(shape=(4,10, 1), dtype='float32',name='month_input')
month_input1 = layers.Conv2D(32,(3,3), activation='relu')(month_input)
month_input2 = layers.MaxPool2D(2,2)(month_input1)
month_input3 = layers.Dropout(0.2)(month_input2)
month_input4 = layers.Conv2D(64,(1,3),activation='relu',\
            kernel_regularizer=regularizers.l2(l=0.001))(month_input3)

month_input5 = layers.MaxPool2D(1,2)(month_input4)

# DenseNet
sum_input = Input(shape=(1,1,9), dtype='float32', name='sum_input')
sum_input1 = layers.Dense(128,activation='relu')(sum_input)
sum_input2 = layers.Dropout(0.5)(sum_input1)
sum_input3 = layers.Dense(64, activation='relu')(sum_input2)
sum_input4 = layers.Dropout(0.2)(sum_input3)
sum_input5 = layers.Dense(32, activation='relu',kernel_regularizer=regularizers.l2(l=0.001))(sum_input4)

# Concat
concatenated = layers.concatenate([month_input5, sum_input5], axis=-1)
answer = layers.Dense(1, activation='relu')(concatenated)

model = Model([month_input,sum_input], answer)
model.compile(optimizer='adam',loss='mse')
print("model summary is:",model.summary())

# fit
model.fit({'month_input':source_data,'sum_input':sum_data},
            target_data,
            validation_split=0.1,
            epochs=1000,
            batch_size=64,
            verbose=1
            )

# prediction
pre_sum_data = pd.read_csv('../data/userInfoSum_9-11.csv').replace([np.inf, -np.inf], 9999).fillna(0)
pre_sum_data_array = np.log1p(np.array(pre_sum_data)).reshape(90993,1,1,9)
pre = model.predict({'month_input':pred_data,'sum_input':pre_sum_data_array})
print("pre result done.")
pre_pd = pd.DataFrame(np.array(pre).reshape(90993,1))
pre_pd[pre_pd <1] = 0
pre_pd.to_csv('../result/prediction_Multi.csv')
print("submission time is: ", round(((time.time() - start_time) / 60), 2))
