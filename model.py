#!/opt/conda/bin/python

'''Custom Keras model class to do an end-end evaluation. Takes test and train data inputs
and generates features -> preprocesses data -> trains model -> evaluates on test data'''

import numpy as np
import pandas as pd
import talib as ta
import copy
import cPickle as pickle
from calculations import *

from sklearn import linear_model, model_selection
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler, MinMaxScaler, LabelEncoder

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from talib import abstract

class Model(object):
	"""docstring for ClassName"""
	def __init__(self, data, options):
		# Constructor
		self.options = options
		self.raw_data = self.getData(data)
		self.y = self.binaryClassification()
		self.x = self.features()
		self.x, self.y = self.prepareData()
		self.feature_list = [col for col in self.x if col not in self.raw_data.columns]
		
		self.model = None
		self.scaler = None
		self.predictions = None
		self.px = None
		self.score = None
		
	def getData(self, data):
		# load pricing data
		self.raw_data = pd.read_csv(data)
		self.raw_data = self.raw_data.set_index("DATETIME")
		'''self.raw_data = np.log(self.raw_data)
								print self.raw_data.tail()'''
		return self.raw_data

	def features(self):
		# Generate features from input data
		
		x = copy.deepcopy(self.raw_data)

		# placeholder features
		x['x1'] = x['CLOSE'].pct_change()
		x['x2'] = x['CLOSE'].pct_change(5)
		

		return x

	def binaryClassification(self):

		period = self.options['time_period']
		prices = copy.deepcopy(self.raw_data)
		if self.options['classification_method'] == 'on_close':
			prices['y'] = np.zeros(prices['CLOSE'].shape)
			prices['NEXT'] = prices['CLOSE'].shift(-period)
			prices['Diff'] = prices['NEXT'] - prices['CLOSE']

			prices['y'].loc[prices['Diff'] > 0] = 1
			prices['y'].loc[prices['Diff'] < 0] = 0
			prices['y'].loc[prices['Diff'] == 0] = 2
			return prices['y']
		

	def prepareData(self):

		x = copy.deepcopy(self.x)
		y = copy.deepcopy(self.y)
		temp = pd.concat([x, y], axis = 1)
		temp = temp.replace([np.inf, -np.inf], np.nan)
		temp = temp.dropna()
		temp = temp.loc[temp['y'] != 2]
		# Filter

		'''try:
			temp.index = pd.to_datetime(temp.index, format = "%d/%m/%Y %H:%M")
		except:
			temp.index = pd.to_datetime(temp.index, format = "%Y-%m-%d %H:%M:%S")

		temp = temp.between_time(dt.time(self.options['hour_start'],00), dt.time(self.options['hour_end'],00))'''

		x_prepared = temp.drop('y',1)
		y_prepared = temp[['y']]

		assert(len(x_prepared) == len(y_prepared))
		assert(x_prepared.index == y_prepared.index).all()

		return x_prepared, y_prepared

	def train_model(self):
		# place split & train here
		# execute in main
		# allows child of class without training 
		# for forward test
		self.X_train, self.X_test, self.Y_train, self.Y_test = self.split()

		if self.options['scale'] == True:
			self.X_train, self.X_test = self.scale(self.X_train, self.X_test)

		print self.X_test[1000]
		self.model = self.train()
		self.score = self.evaluate()

	def split(self):

		price_info_cols = list(self.raw_data.columns.values)

		x = self.x.drop(price_info_cols,1).as_matrix()
		y = self.y.values.ravel()
		assert(len(x) == len(y))

		#x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, train_size = self.options['split'], random_state = 42)

		p = 0.8
		ix_train = int(p*len(x))
		x_train = x[0:ix_train]
		x_test = x[ix_train:]
		y_train = y[0:ix_train]
		y_test = y[ix_train:]

		return x_train, x_test, y_train, y_test

	def scale(self, x_train, x_test):

		self.scaler = MinMaxScaler(feature_range = [0,1])
		x_train = self.scaler.fit_transform(x_train)
		x_test = self.scaler.transform(x_test)

		return x_train, x_test


	def train(self):

		# train model

		# format data for model input
		ls_x = self.X_train
		ls_y = self.Y_train
		assert not np.any(np.isnan(ls_x) | np.isinf(ls_x))

		print self.feature_list
		feature_count = len(self.feature_list)
		x_test = self.X_test

		ls_x = np.reshape(ls_x, [ls_x.shape[0], 1, ls_x.shape[1]])
		x_test = np.reshape(self.X_test, [self.X_test.shape[0], 1, self.X_test.shape[1]])

		print ls_x.shape

		# Model architecture
		clf = Sequential()
		clf.add(LSTM(64, input_shape=ls_x.shape, return_sequences=True))
		clf.add(Activation('tanh'))
		clf.add(LSTM(32, return_sequences=True))
		clf.add(Activation('tanh'))
		clf.add(Dense(1))
		clf.add(Activation('sigmoid'))

		clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		clf.fit(ls_x, ls_y, batch_size=128, validation_data=(x_test, self.Y_test), epochs=5)

		return clf

	def predict_y(self):

		assert self.model is not None
		try:
			x_test = np.reshape(self.X_test, [self.X_test.shape[0], 1, self.X_test.shape[1]])
			y_predictions = self.model.predict(x_test)
		except:
			y_predictions = self.model.predict(self.X_test)

		y_predictions = np.round(y_predictions)

		return y_predictions

	def predict_px(self):

		assert self.model is not None
		try:
			x_test = np.reshape(self.X_test, [self.X_test.shape[0], 1, self.X_test.shape[1]])
			px = self.model.predict_proba(x_test)
		except:
			px = self.model.predict_proba(self.X_test)

		return px

	def evaluate(self):
		# evaluate model
		assert self.model is not None
		self.predictions = self.predict_y()
		self.px = self.predict_px()
		print self.predictions
		score = accuracy_score(self.Y_test, self.predictions)
		return score

	def export(self):
		# export model
		with open('model.pkl', 'wb') as model:
			pickle.dump(self.model, model)
		# export scaler
		if self.scaler is not None:
			with open('scaler.pkl', 'wb') as scaler:
				pickle.dump(self.scaler, scaler)

	def save_context(self):

		if self.predictions is None:
			self.predictions = self.predict_y()
			self.px = self.predict_px()

		assert(len(self.x)==len(self.predictions))
		df_predictions = pd.DataFrame(zip(self.predictions,self.px[:,1]), index=self.x.index, columns=['y_predict','px'])
		context = pd.concat([self.x, self.y, df_predictions], axis=1)
		context['correct'] = np.zeros(context['CLOSE'].shape)
		context['correct'].loc[context['y_predict']==context['y']]=1
		context.to_csv('context.csv')

	def forwardTest(self, data):
		child = Model(data, self.options)
		price_info_cols = list(child.raw_data.columns.values)
		x = child.x.drop(price_info_cols,1).as_matrix()
		if self.options['scale'] == True:
			assert self.scaler is not None
			x = self.scaler.transform(x)
		y = child.y.values.ravel()
		try:
			px = self.model.predict_proba(x)
		except:
			x = np.reshape(x, [x.shape[0], 1, x.shape[1]])
			px = self.model.predict_proba(x)

		child.score = accuracy_score(y, np.round(px))
		print child.score
		# Result formatting for backtest
		try:
			df_result = pd.DataFrame(zip(y,px[:,1]), index=child.x.index)
		except:
			print y.shape
			print px.shape
			px = [float(x) for x in px]
			print min(px)
			print max(px)
			df_result = pd.DataFrame(zip(y,px), index=child.x.index)
		df_result.to_csv('forward_test.csv')

def main():

	options = {'time_period': 5,
				'split': 0.9,
				'classification_method': 'on_close',
				'scale': True,
				'hour_start': 0,
				'hour_end': 23}

	my_model = Model('test_data.csv', options)
	#print my_model.x.tail(10)
	my_model.x.to_csv('feature_vector.csv')
	my_model.train_model()
	print my_model.score
	my_model.forwardTest('train_data.csv')

if __name__ == "__main__":

  print("Running")

  try:

    main()

  except KeyboardInterrupt:

    print('Interupted...Exiting...')
