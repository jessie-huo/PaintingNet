from tensorflow.python.platform import flags
import numpy as np
import pandas as pd
from os import path
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K
from keras import optimizers
from keras.models import model_from_json
from sklearn.metrics import r2_score
from keras.callbacks import TensorBoard

flags.DEFINE_integer('batch_size', 16, 'set the batch size')
flags.DEFINE_integer('num_feature', 60, 'set the size of input feature')
flags.DEFINE_integer('num_epochs', 50, 'set the number of epochs')
flags.DEFINE_string('mode', 'test', 'set whether to train or test')
flags.DEFINE_boolean('keep', True, 'set whether to restore a model, when test mode, keep should be set to True')
flags.DEFINE_string('datadir', 'dataset/data.txt', 'set the data root directory')
flags.DEFINE_string('labeldir', 'dataset/label.txt', 'set the label root directory')
flags.DEFINE_string('logdir', 'log_cnn/', 'set the log directory')
flags.DEFINE_string('filename', 'test.txt', 'set the test filename')


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
		testfile = np.zeros((1 , 128, 128, 3))
		with open(filename) as infile:
			for line in infile:
				graph = np.fromstring(line, dtype = int, sep = ",")
				for i in range(0, int((graph.size / 3))):
					testfile[0][int(i/128)][i%128][0] = graph[i]
					testfile[0][int(i/128)][i%128][1] = graph[i + 16384]
					testfile[0][int(i/128)][i%128][2] = graph[i + 16384 * 2]

		#predict
		predictions = loaded_model.predict(testfile)
		print('The recommended price is ' + str(predictions[0][0]))


	if mode == 'train':
		with open(datadir) as infile:
			dataset = np.zeros((22384 , 128, 128, 3))
			a = 0
			for line in infile:
				graph = np.fromstring(line, dtype = int, sep = ",")
				for i in range(0, int((graph.size / 3))):
					dataset[a][int(i/128)][i%128][0] = graph[i]
					dataset[a][int(i/128)][i%128][1] = graph[i + 16384]
					dataset[a][int(i/128)][i%128][2] = graph[i + 16384 * 2]
				a = a + 1
	
		f = open(labeldir, 'r')
		label_list = f.read().splitlines()
		f.close()
		label = np.asarray(label_list)
		label = label.astype(np.float)

		adam = optimizers.Adam(lr = 0.1, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
		loaded_model.compile(loss = 'mean_squared_logarithmic_error', optimizer = adam, metrics = ['accuracy'])
		loaded_model.fit(dataset, label, epochs = num_epochs, batch_size = batch_size, callbacks=[TensorBoard(log_dir='tensorboard/')])

		scores = loaded_model.evaluate(dataset, label)
		print("Validation loss: %.2f" %scores[0])
		print(r2_score(label.astype(np.float64), loaded_model.predict(dataset).astype(np.float64)))
		print("%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))

		#save model
		model_json = loaded_model.to_json()
		with open(path.join(logdir ,"model.json"), "w") as json_file:
		    json_file.write(model_json)
		loaded_model.save_weights(path.join(logdir ,"model.h5"))
		print("Saved model to disk")	


if keep == False:
	#readfile and convert to numpy array
	with open(datadir) as infile:
		dataset = np.zeros((22384 , 128, 128, 3))
		a = 0
		for line in infile:
			graph = np.fromstring(line, dtype = int, sep = ",")
			for i in range(0, int((graph.size / 3))):
				dataset[a][int(i/128)][i%128][0] = graph[i]
				dataset[a][int(i/128)][i%128][1] = graph[i + 16384]
				dataset[a][int(i/128)][i%128][2] = graph[i + 16384 * 2]
			a = a + 1
			print(a)
	
	f = open(labeldir, 'r')
	label_list = f.read().splitlines()
	f.close()
	label = np.asarray(label_list)
	label = label.astype(np.float)

	model = Sequential()
	model.add(BatchNormalization())
	model.add(Conv2D(4, kernel_size = (4, 4), strides = (1, 1), activation = 'relu', input_shape = (128, 128, 3), data_format = 'channels_last'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(8, kernel_size = (4, 4), strides = (1, 1), activation = 'relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(BatchNormalization())
	model.add(Dense(3000, activation='relu'))
	model.add(Dense(2000, activation='relu'))
	model.add(Dense(1000, activation='relu'))
	model.add(Dense(500, activation='relu'))
	model.add(Dense(250, activation='relu'))
	model.add(Dense(125, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(25, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(1, activation='relu'))
	adam = optimizers.Adam(lr = 0.1, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
	model.compile(loss = 'mean_squared_logarithmic_error', optimizer = adam, metrics = ['accuracy'])
	model.fit(dataset, label, epochs = num_epochs, batch_size = batch_size, callbacks=[TensorBoard(log_dir='tensorboard/')])

	scores = model.evaluate(dataset, label)
	print("Validation loss: %.2f" %scores[0])
	print(model.summary())
	print(r2_score(label.astype(np.float64), model.predict(dataset).astype(np.float64)))
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	#save model
	model_json = model.to_json()
	with open(path.join(logdir ,"model.json"), "w") as json_file:
	    json_file.write(model_json)
	model.save_weights(path.join(logdir ,"model.h5"))
	print("Saved model to disk")
	