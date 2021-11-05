from tensorflow.python.platform import flags
import numpy as np
import pandas as pd
from os import path
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras import optimizers
from keras.models import model_from_json
from sklearn.metrics import r2_score

flags.DEFINE_integer('batch_size', 64, 'set the batch size')
flags.DEFINE_integer('num_feature', 60, 'set the size of input feature')
flags.DEFINE_integer('num_epochs', 1, 'set the number of epochs')
flags.DEFINE_string('mode', 'train', 'set whether to train or test')
flags.DEFINE_boolean('keep', False, 'set whether to restore a model, when test mode, keep should be set to True')
flags.DEFINE_string('datadir', '', 'set the data root directory')
flags.DEFINE_string('labeldir', '', 'set the label root directory')
flags.DEFINE_string('logdir', '', 'set the log directory')
flags.DEFINE_string('filename', '', 'set the test filename')


FLAGS = flags.FLAGS

batch_size = FLAGS.batch_size
num_feature = FLAGS.num_feature
num_epochs = FLAGS.num_epochs
datadir = FLAGS.datadir
labeldir = FLAGS.labeldir
logdir = FLAGS.logdir
filename = FLAGS.filename
keep = FLAGS.keep
mode = FLAGS.mode

if keep == True:
	# load model
	json_file = open(path.join(logdir, "model.json"), 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(path.join(logdir ,"model.h5"))
	print("Loaded model from disk")

	if mode == 'test':
		#open test file
		testfile = []
		with open(filename) as infile:
			for line in infile:
				testfile.append(np.fromstring(line, dtype = int, sep = ","))
			testfile = np.array(testfile)

		#predict
		predictions = loaded_model.predict(testfile)
		print('The recommended price is ' + str(predictions[0][0]))


	if mode == 'train':
		dataset = []
		with open(datadir) as infile:
			for line in infile:
				dataset.append(np.fromstring(line, dtype = int, sep = ","))
			dataset = np.array(dataset)
		f = open(labeldir, 'r')
		label_list = f.readlines()
		f.close()
		label = np.array(label_list)

		dataset_train, dataset_test, label_train, label_test = train_test_split(dataset, label, test_size = 0.2)
		adam = optimizers.Adam(lr = 0.01, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
		loaded_model.compile(loss = 'mean_squared_logarithmic_error', optimizer = adam, metrics = ['accuracy'])
		loaded_model.fit(dataset_train, label_train, epochs = num_epochs, batch_size = batch_size)

		scores = loaded_model.evaluate(dataset_test, label_test)
		print("Validation loss: %.2f" %scores[0])
		print(r2_score(label_test.astype(np.float64), loaded_model.predict(dataset_test).astype(np.float64)))
		#print("%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))

		#save model
		model_json = loaded_model.to_json()
		with open(path.join(logdir ,"model.json"), "w") as json_file:
		    json_file.write(model_json)
		loaded_model.save_weights(path.join(logdir ,"model.h5"))
		print("Saved model to disk")	


if keep == False:
	#readfile and convert to numpy array
	dataset = []
	with open(datadir) as infile:
		for line in infile:
			dataset.append(np.fromstring(line, dtype = int, sep = ","))
		dataset = np.array(dataset)
	f = open(labeldir, 'r')
	label_list = f.readlines()
	f.close()
	label = np.array(label_list)
	dataset_train, dataset_test, label_train, label_test = train_test_split(dataset, label, test_size = 0.2)

	model = Sequential()
	model.add(Dense(32*16, input_dim = 32*32*3, activation='relu'))
	model.add(Dense(16*16, activation = 'relu'))
	model.add(Dense(16*8, activation = 'relu'))
	model.add(Dense(64, activation = 'relu'))
	model.add(Dense(32, activation = 'relu'))
	model.add(Dense(32, activation = 'relu'))
	model.add(Dense(16, activation = 'relu'))
	model.add(Dense(16, activation = 'relu'))
	model.add(Dense(8, activation = 'relu'))
	model.add(Dense(4, activation = 'relu'))
	model.add(Dense(1, activation = 'relu'))
	adam = optimizers.Adam(lr = 0.01, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
	model.compile(loss = 'mean_squared_logarithmic_error', optimizer = adam, metrics = ['accuracy'])
	model.fit(dataset_train, label_train, epochs = num_epochs, batch_size = batch_size)

	scores = model.evaluate(dataset_test, label_test)
	print("Validation loss: %.2f" %scores[0])
	print(r2_score(label_test.astype(np.float64), model.predict(dataset_test).astype(np.float64)))
	#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	#save model
	model_json = model.to_json()
	with open(path.join(logdir ,"model.json"), "w") as json_file:
	    json_file.write(model_json)
	model.save_weights(path.join(logdir ,"model.h5"))
	print("Saved model to disk")