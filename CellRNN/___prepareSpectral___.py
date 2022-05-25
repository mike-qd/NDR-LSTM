# -*- coding: utf-8 -*-
import os
import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from CellRNN.ConfigSetting import *
"""
每个像素点做成多张图
例如：将3x3数据立方体分成9个长条作为RNN的输入
一个样本点包含多个数据立方体：（9，103），（25，103）
"""
pavia_mat = sio.loadmat(path_PU_mat)['paviaU']
india_mat = sio.loadmat(path_IP_mat)['indian_pines']
pavia_label = sio.loadmat(path_PU_label)['paviaU_gt']
india_label = sio.loadmat(path_IP_label)['indian_pines_gt']
# mat = pavia_mat
# label = pavia_label
mat = india_mat
label = india_label
#

# Height = pavia_Height
# Width = pavia_Width
Height = india_Height
Width = india_Width
pca = PCA(n_components='mle')
# pca = PCA(n_components=100)
ratio = 0.1
np.random.seed(1)


def sample_wise_standardization(data):
    import math
    _mean = np.mean(data)
    _std = np.std(data)
    npixel = np.size(data) * 1.0
    min_stddev = 1.0 / math.sqrt(npixel)
    return (data - _mean) / max(_std, min_stddev)

re_mat = mat.reshape(Width*Height, mat.shape[2])
input_mat = pca.fit_transform(re_mat)
# print(input_mat.shape)
data_ = input_mat.reshape(mat.shape[0], mat.shape[1], input_mat.shape[1])
data_ = sample_wise_standardization(data_)
print(data_.shape)


def class_seperation(data, ground_truth, region_size):
    m, n = label.shape
    r = region_size['center']   # r = 1
    r_ = region_size['medium']  # r = 2
    r1 = region_size['global']-1  # r1 = 4
    r2 = region_size['global']  # r2 = 5
    r3 = region_size['global']+1
    # print(data.shape)
    # TrnData, TestData, TrnLabel, TestLabel = np.array([]), np.array([]), np.array([]), np.array([])
    labels = []
    site_feature = []
    site_center = []
    site_medium = []
    site_right = []
    site_left = []
    site_up = []
    site_bottom = []
    pixel_point = []
    for i in range(m):
        for j in range(n):
            # print(j)
            if label[i][j] != 0:
                labels.append(label[i][j]-1)
                cube = []
                medium = []
                X_Global = []
                xur = []
                xul = []
                xbr = []
                xbl = []
                # Center
                for idx in range(i-r, i+r+1):
                    for idy in range(j-r, j+r+1):
                        idx = (idx+Width) % Width
                        idy = (idy+Height) % Height
                        cube.append(data[idx][idy][:])
                # Medium
                for idx in range(i - r_, i + r_ + 1):
                    for idy in range(j - r_, j + r_ + 1):
                        idx = (idx + Width) % Width
                        idy = (idy + Height) % Height
                        medium.append(data[idx][idy][:])
                # Top Left
                for idx in range(i-r2, i+r+1):
                    for idy in range(j-r2, j+r+1):
                        idx = (idx+Width) % Width
                        idy = (idy+Height) % Height
                        xur.append(data[idx][idy][:])
                # Top Right
                for idx in range(i-r2, i+r+1):
                    for idy in range(j-r, j+r2+1):
                        idx = (idx+Width) % Width
                        idy = (idy+Height) % Height
                        xul.append(data[idx][idy][:])
                # Bottom Left
                for idx in range(i-r, i+r2+1):
                    for idy in range(j-r2, j+r+1):
                        idx = (idx+Width) % Width
                        idy = (idy+Height) % Height
                        xbr.append(data[idx][idy][:])
                # Bottom Right
                for idx in range(i-r, i+r2+1):
                    for idy in range(j-r, j+r2+1):
                        idx = (idx+Width) % Width
                        idy = (idy+Height) % Height
                        xbl.append(data[idx][idy][:])
                site_center.append(np.asarray(cube))
                # site_feature.append(np.asarray(X_Global))
                site_medium.append(np.asarray(medium))
                site_right.append(np.asarray(xur))
                site_left.append(np.asarray(xul))
                site_up.append(np.asarray(xbr))
                site_bottom.append(np.asarray(xbl))
                pixel_point.append([i, j])

    site_medium = np.array(site_medium)
    site_center = np.array(site_center)
    site_right = np.array(site_right)
    site_left = np.array(site_left)
    site_up = np.array(site_up)
    site_bottom = np.array(site_bottom)
    pixel_point = np.array(pixel_point)
    print('site medium shape: ', site_medium.shape)
    print('site center shape: ', site_center.shape)
    print('site right shape: ', site_right.shape)
    print('site left shape: ', site_left.shape)
    print('site up shape: ', site_up.shape)
    print('site bottom shape: ', site_bottom.shape)
    # Data = sample_wise_standardization(Data)
    Label = np.array(labels)
    print(Label.shape)
    Label = to_categorical(Label)
    total_i_class = len(Label)
    print(total_i_class)
    arr_index = np.arange(total_i_class)
    np.random.shuffle(arr_index)
    sep_point = int(total_i_class * ratio)
    trn_index = arr_index[:sep_point]
    tst_index = arr_index[sep_point:]
    # TrnData, TestData, TrnLabel, TestLabel = train_test_split(Data, Label, test_size=0.9, random_state=6)
    TrnMedium = site_medium[trn_index]
    TestMedium = site_medium[tst_index]
    TrnCenter = site_center[trn_index]
    TestCenter = site_center[tst_index]
    TrnRight = site_right[trn_index]
    TestRight = site_right[tst_index]
    TrnLeft = site_left[trn_index]
    TestLeft = site_left[tst_index]
    TrnUp = site_up[trn_index]
    TestUp = site_up[tst_index]
    TrnBottom = site_bottom[trn_index]
    TestBottom = site_bottom[tst_index]
    print('TrnMedium shape: ', TrnMedium.shape)
    print('TrnCenter shape: ', TrnCenter.shape)
    print('TrnRight shape: ', TrnRight.shape)
    print('TrnLeft shape: ', TrnLeft.shape)
    print('TrnUp shape: ', TrnUp.shape)
    print('TrnBottom shape: ', TrnBottom.shape)
    TrnLabel = Label[trn_index]
    TestLabel = Label[tst_index]
    return TrnMedium, TestMedium, TrnCenter, TestCenter, TrnRight, TestRight, TrnLeft, TestLeft, TrnUp, TestUp, TrnBottom, TestBottom, TrnLabel, TestLabel, pixel_point[tst_index]


def data_format(Data_or_Label, Batch):
    """
    :param Data_or_Label: (total, region, seq_len)
    :param Batch: 16 or 64
    :return: (Epoch, Batch, region, channel)
    """
    sample_nums = int(len(Data_or_Label))
    S = Data_or_Label.shape
    Epoch = sample_nums // Batch
    Data = Data_or_Label[:Epoch*Batch]
    if len(S) == 3: # Data
        Temp = Data.reshape(Epoch, Batch, S[1], S[2])  # Temp = (Epoch, Batch, features, channel)
    else:   # Label
        Temp = Data.reshape(Epoch, Batch, S[1])
    Input = Temp  # (Epoch, Batch, region, channel) ## region≈seq_len, channel≈features

    return Input


def generate_loader(TrainData_List, TrainLabel):
    Epoch = len(TrainLabel)
    dataLoader = [([TrainData_List[0][i], TrainData_List[1][i], TrainData_List[2][i], TrainData_List[3][i], TrainData_List[4][i], TrainData_List[5][i]], TrainLabel[i]) for i in range(Epoch)]

    return dataLoader


def DataLoad(Batch):
    trainData_medium, testMedium, trainData_center, testCenter, trainData_right, testRight, trainData_left, testLeft,\
    trainData_up, testUp, trainData_bottom, testBottom, trainLabel, test_Y, pixel_point = \
        class_seperation(data_, label, {'center':1, 'medium':2, 'global':5})
    s1 = len(trainLabel) // Batch
    s2 = len(test_Y) // Batch
    print(trainData_medium.shape)
    print(trainData_center.shape)
    print(trainData_right.shape)
    print(trainData_left.shape)
    print(trainData_up.shape)
    print(trainData_bottom.shape)
    print(trainLabel.shape)
    Medium_B = data_format(trainData_medium, Batch)
    Center_B = data_format(trainData_center, Batch)
    Right_B = data_format(trainData_right, Batch)
    Left_B = data_format(trainData_left, Batch)
    Up_B = data_format(trainData_up, Batch)
    Bottom_B = data_format(trainData_bottom, Batch)
    Label_B = data_format(trainLabel, Batch)
    print(Medium_B.shape)
    print(Center_B.shape)
    print(Right_B.shape)
    print(Left_B.shape)
    print(Up_B.shape)
    print(Bottom_B.shape)
    print(Label_B.shape)

    return {'data':[trainData_medium, trainData_right, trainData_left, trainData_up, trainData_bottom, trainData_center], 'target': trainLabel},\
           {'data':[testMedium, testRight, testLeft, testUp, testBottom, testCenter], 'target':test_Y}, pixel_point
    # return train_ds, test_ds, s1, s2


def DataLoad_mat():
    file_train = 'train_data_H_new.mat'
    file_test = 'test_data_H_new.mat'
    savepoint_dir = os.path.abspath(os.path.join(os.path.curdir, "construct\\indian_dataset"))
    # savepoint_dir = os.path.abspath(os.path.join(os.path.curdir, "construct\\indian_dataset_1632665229"))   # 正解

    mdata = sio.loadmat(os.path.join(savepoint_dir, file_train))
    mdata_test = sio.loadmat(os.path.join(savepoint_dir, file_test))
    trainData_medium, trainData_right, trainData_left, trainData_up, trainData_bottom, trainData_center = mdata['data'], mdata['XR'], mdata['XL'], mdata['XU'], mdata['XB'], mdata['XC']
    trainLabel = mdata['label']
    testMedium, testRight, testLeft, testUp, testBottom, testCenter = mdata_test['data'], mdata_test['XR'], mdata_test['XL'], mdata_test['XU'], mdata_test['XB'], mdata_test['XC']
    test_Y = mdata_test['label']
    pixel_point = mdata_test['point']
    print(trainData_medium.shape)
    print(trainData_center.shape)
    print(trainData_right.shape)
    print(trainData_left.shape)
    print(trainData_up.shape)
    print(trainData_bottom.shape)
    print(trainLabel.shape)

    # trainLoader = generate_loader([Medium_B, Center_B, Right_B, Left_B, Up_B, Bottom_B], Label_B)
    return {'data':[trainData_medium, trainData_right, trainData_left, trainData_up, trainData_bottom, trainData_center], 'target': trainLabel},\
           {'data':[testMedium, testRight, testLeft, testUp, testBottom, testCenter], 'target':test_Y}, pixel_point

print("Finish Data!")

