# -*- coding: utf-8 -*-
"""
============
 NDR - LSTM
============
"""
import os
import time
import keras
import keras.layers as L
import numpy as np
import scipy.io as sio
from keras.models import Sequential,Model,load_model
from keras.layers import LSTM,Dense,Dropout
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import seaborn as sns
from CellRNN.___prepareSpectral___ import DataLoad_mat		# 97.58% ~ 97.75%
# from CellRNN.___prepareSpectral___ import DataLoad
from CellRNN.module.R import RocAucMetricCallback
from CellRNN.module.F import kappa_cal, recall_AA
from CellRNN.ConfigSetting import *
import matplotlib.pyplot as plt
import spectral
import cv2
print(tf.__version__)

n_input = 218  # HSI data input
n_steps1 = 9  # timesteps - region1
n_steps_ = 25  # timesteps - region_ (Medium Region)
n_steps2 = 121  # timesteps - region2(Global Region)
n_steps_other = 49  # TR、TL、BR、BL
n_hidden = 128  # hidden layer num of features
n_classes = 16  # IP total classes (0-15 digits) //共十六大类祛除了背景
Epoch = 50
indian_Width = 145
indian_Height = 145

indian_label = sio.loadmat(path_IP_label)['indian_pines_gt']

timestamp = str(int(time.time()))
# filename = "best_weights_"+timestamp+".h5"
filename = "best_weights_9775.h5"
out_dir = os.path.abspath(os.path.join(os.path.curdir, "weights"))
filepath = os.path.abspath(os.path.join(out_dir, filename))


def UNION_net(drop_rate=1.0):
	input_Medium = L.Input((n_steps_, n_input), name="MediumInput")
	lstm_out1 = LSTM(n_hidden, return_sequences=False, input_shape=(n_steps_, n_input))(input_Medium)

	input_Right = L.Input((n_steps_other, n_input), name="TopRight")
	lstm_out2 = LSTM(n_hidden, return_sequences=False, input_shape=(n_steps_other, n_input))(input_Right)

	input_Left = L.Input((n_steps_other, n_input), name="TopLeft")
	lstm_out3 = LSTM(n_hidden, return_sequences=False, input_shape=(n_steps_other, n_input))(input_Left)

	input_Up = L.Input((n_steps_other, n_input), name="BottomRight")
	lstm_out4 = LSTM(n_hidden, return_sequences=False, input_shape=(n_steps_other, n_input))(input_Up)

	input_Bottom = L.Input((n_steps_other, n_input), name="BottomLeft")
	lstm_out5 = LSTM(n_hidden, return_sequences=False, input_shape=(n_steps_other, n_input))(input_Bottom)

	input_Center = L.Input((n_steps1, n_input), name="CenterInput")
	lstm_out_center = LSTM(n_hidden, return_sequences=False, input_shape=(n_steps1, n_input))(input_Center)

	# Combine all branches
	merge0 = L.concatenate([lstm_out1, lstm_out2, lstm_out3, lstm_out4, lstm_out5, lstm_out_center], axis=-1)
	merge0_1 = Dropout(drop_rate)(merge0)
	# merge1 = Dense(128, activation='relu')(merge0_1)
	merge1 = Dense(256, activation='relu')(merge0_1)
	# merge1 = L.BatchNormalization(axis=-1, name='BatchNorm1')(merge1)
	merge2 = Dense(64, activation='relu')(merge1)
	logits = Dense(n_classes, activation='softmax')(merge2)
	new_model = Model([input_Medium, input_Right, input_Left, input_Up, input_Bottom, input_Center], logits)
	sgd = keras.optimizers.SGD(lr=0.001, momentum=0.98)
	gradient_opt = keras.optimizers.RMSprop(lr=0.001)	 # 采用这个优化器比SGD要效率高不少!
	new_model.compile(optimizer=gradient_opt, loss='categorical_crossentropy', metrics=['acc'])
	return new_model


def train(train_loader):
	# trainLoader, testLoader = DataLoad(16)
	model = UNION_net(drop_rate=0.5)
	model.summary()

	# checkpoint
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, cooldown=4)
	early_stop = EarlyStopping(monitor='roc_auc', patience=5, mode='max', min_delta=0.001, verbose=2)
	tbCallBack = TensorBoard(log_dir="./model")
	callback_list = [checkpoint, reduce_lr, tbCallBack]
	model.fit(train_loader['data'], train_loader['target'], batch_size=16, epochs=Epoch, verbose=2, validation_split=0.1, callbacks=callback_list)


"""
Test the model
"""
def test(test_loader):
	# model = load_model('best_weights.h5')
	model = UNION_net()
	model.load_weights(filepath)
	test_data = test_loader['data']
	test_label = test_loader['target']
	score = model.evaluate(test_data, test_label, batch_size=16)
	print(score)
	prediction = model.predict(test_data, batch_size=16)
	test_prediction = np.argmax(prediction, 1)
	# === 分类图：Classification Map ===
	new_show = np.zeros((indian_Width, indian_Height))
	print(np.unique(new_show))
	for i in range(indian_Width):
		for j in range(indian_Height):
			if indian_label[i][j] != 0:
				new_show[i][j] = indian_label[i][j]
	print(np.unique(new_show))
	for n in range(len(pixel_point)):
		a, b = pixel_point[n]
		new_show[a][b] = test_prediction[n]+1
	g_pic = spectral.imshow(classes=indian_label, figsize=(9,9))
	cv2.imshow('1', new_show)
	t_pic = spectral.imshow(classes=new_show.astype(int), figsize=(9,9), title='Indian Pines')
	cv2.imshow('2', new_show)
	# === 混淆矩阵：真实值与预测值的对比 ===
	con_mat = confusion_matrix(np.argmax(test_label, 1), np.argmax(prediction, 1))
	con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
	con_mat_norm = np.around(con_mat_norm, decimals=2)
	kappaByHand = kappa_cal(con_mat)
	AverageAccuracy = recall_AA(con_mat)
	print("Average Accuracy:", AverageAccuracy)
	print("Kappa Coefficient: ", kappaByHand)
	# === plot ===
	plt.figure(figsize=(8, 8))
	sns.heatmap(con_mat_norm, annot=True, cmap='Blues')
	plt.ylim(0, n_classes)
	plt.xlabel('Predicted labels')
	plt.ylabel('True labels')
	plt.title('Indian Pine: Epoch 20')
	# plt.savefig('indian_pine_'+timestamp+'.png')
	plt.show()
	cv2.imshow('3', new_show)
	cv2.waitKey(0)

	# print("Testing Accuracy:", sess.run(accuracy, feed_dict={x1: test_data[0], x2: test_data[1], y: test_label}))
	# print("Confusion Matrix:", sess.run(calc_confusion, feed_dict={x1: test_data[0], x2: test_data[1], y: test_label}))
	with tf.Session() as sess:
		A = tf.confusion_matrix(tf.argmax(test_label, 1), tf.argmax(prediction, 1))
		print("Confusion Matrix:", sess.run(A))


if __name__ == '__main__':
	trainLoader, testLoader, pixel_point = DataLoad_mat()
	# trainLoader, testLoader, pixel_point = DataLoad(16)
	# train(trainLoader)	# 如果不愿意训练，可以将这行注掉
	test(testLoader)