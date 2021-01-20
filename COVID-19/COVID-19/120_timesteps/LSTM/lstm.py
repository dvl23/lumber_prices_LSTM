import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import TimeSeriesSplit

np.random.seed(199)
tf.random.set_seed(199)

# hyperparameters
timeframe = 120
epochs = 2000 # 10

forecast = pd.read_excel('COVID_recession_training.xlsx') #Importing data with pandas

#===============================================TRAINING==================================================================
forecast_processed = forecast.iloc[:,4:5].values #Grabbing only the price column
forecast_processed = np.flip( forecast_processed, axis=None )
n_train = forecast_processed.shape[0]

#SAVING DATA TO CSV
training_set = pd.DataFrame(forecast_processed).to_csv('training_set.csv')

#=====================================================VALIDATION==========================================================


#stock_price = forecast.iloc[:, 4:5].values 
stock_price = forecast_processed[:]
plt.plot(stock_price)
plt.xlabel('Time')
plt.ylabel('Random Length Stock Price (train)')
plt.show()

scaler = MinMaxScaler(feature_range = (0,1))

forecast_scaled = scaler.fit_transform(forecast_processed)

#================================================PROCESSING TRAINING =====================================================
X_train = [] #creating list
y_train = [] #creating list
for i in range(timeframe, n_train):
    X_train.append(forecast_scaled[i-timeframe:i, 0]) #X_train should contain the stock value for the past 60 days
    y_train.append(forecast_scaled[i, 0]) #Y-train should contain the stock value at the 61st day
X_train, y_train = np.array(X_train), np.array(y_train)#convert to numpy array
print( 'X_train.shape',X_train.shape )
print( 'y_train.shape',y_train.shape )

X_train = np.reshape( X_train, (X_train.shape[0], X_train.shape[1], 1) )

print("X_train shape is:", X_train.shape)
print("Y_shape is:", y_train.shape)

#=============================================TRYING TO SPLIT THE DATA==================================================

tss = TimeSeriesSplit(n_splits = 5)
z = 0
for train_index, val_index in tss.split(X_train):
	print(len(train_index))
	print(len(val_index))

	print("TRAIN:", train_index, "TEST:", val_index)

	X, X_val = X_train[train_index], X_train[val_index]
	y, y_val = y_train[train_index], y_train[val_index]




	regressor = Sequential()

	regressor.add(LSTM(units = 50, recurrent_activation = 'sigmoid', return_sequences = True, input_shape = (X_train.shape[1], 1)))

	regressor.add(LSTM(units = 50, recurrent_activation = 'sigmoid', return_sequences = True))

	regressor.add(LSTM(units = 50, recurrent_activation = 'sigmoid', return_sequences = True))

	regressor.add(LSTM(units = 50, recurrent_activation = 'sigmoid', return_sequences = True))

	regressor.add(LSTM(units = 50, recurrent_activation = 'sigmoid'))

	regressor.add(Dense(units = 1))

	#plot_model(regressor, to_file='120_timesteps.png', show_shapes = True, show_layer_names = True)

	regressor.summary()

	callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience = 80),
	            tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 150),
	            tf.keras.callbacks.CSVLogger('COVID_recession_120TS_log'+str(z)+'.csv', append=True)]


	opt = tf.keras.optimizers.Adam(learning_rate=0.001)
	regressor.compile(optimizer = opt, loss = 'mean_squared_error', metrics = [tf.keras.metrics.RootMeanSquaredError(),
	                                                                             tf.keras.metrics.MeanAbsoluteError()])
	regressor.fit(X, y, validation_data = (X_val, y_val), epochs = epochs, batch_size = 512, callbacks=callbacks)

#==========================================================================================
#Testing the model
#==========================================================================================

	#dataset_test = pd.read_excel('test.xlsx') #Imporing test data set
	dataset_test = pd.read_excel('COVID_recession_testing.xlsx') #Imporing test data set
	#print('dataset_test',dataset_test.head()) #reading five first rows
	print(dataset_test.shape)

	test_processed = dataset_test.iloc[:,4:5].values #Grabbing only the price column
	print('test_processed',type(test_processed),test_processed.shape)
	#print(test_processed)
	test_processed = np.flip( test_processed, axis=None )
	#print(test_processed)
	n_test = test_processed.shape[0]

	#real_stock_price = dataset_test.iloc[:, 4:5].values 
	real_stock_price = test_processed[:]
	plt.plot(real_stock_price)
	plt.xlabel('Time')
	plt.ylabel('Random Length Stock Price (test)')
	plt.show()

	dataset_total = pd.concat((forecast['Close'], dataset_test['Close']), axis = 0)
	inputs = dataset_total[len(dataset_total) - len(dataset_test) - timeframe:].values
	print('dataset_total', dataset_total.shape )

	inputs = forecast_processed
	inputs = np.append( forecast_processed, test_processed )
	print('inputs', inputs.shape )

	inputs = inputs.reshape(-1,1)
	inputs = scaler.transform(inputs)

	X_test = []
	Y_test = []
	#for i in range(timeframe, timeframe+n_test):
	#    X_test.append(inputs[i-timeframe:i, 0])
	#    Y_test.append(inputs[i,0])
	for i in range(n_test):
		j = n_train + i 
		X_test.append( inputs[j-timeframe:j,0] )
		Y_test.append( inputs[j,0] )
	X_test = np.array(X_test)
	X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
	Y_test = np.array(Y_test)
	predicted_stock_price = regressor.predict(X_test)
	predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

	predicted = pd.DataFrame(predicted_stock_price).to_csv('predictions'+str(z)+'.csv')


	plt.plot(real_stock_price, color = 'black', label = 'Random Length Price')
	plt.plot(predicted_stock_price, color = 'green', label = 'Pred. Random Length Price')
	plt.title('Random Length Stock Price Prediction - Great recession')
	plt.xlabel('Time')
	plt.ylabel('Random Length Stock Price (US$)')
	plt.legend()
	plt.show()


	score = regressor.evaluate(X_test, Y_test, verbose= 0)

	print("timeframe is:", timeframe )
	print("MSE testing is:", round(score[0],5))
	print("RMSE testing is:", round(score[1],5))
	print("MAE testing is:", round(score[2],5))

	f = open('metrics'+str(z)+'.txt', "w")
	f.write('MSE = '+ str(score[0]) + '\n')
	f.write('RMSE = '+str(score[1]) + '\n')
	f.write('MAE = '+str(score[2]) +'\n')
	f.close()

	z = z+1