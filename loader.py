from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import random


def loadDataLabel(dir_name, shuffle=False):
    assert os.path.isdir(dir_name), "dir_name is not dir"
    dir = os.listdir(dir_name)
    dir.sort()
    len_dir = len(dir)
    datas = []
    labels = []
    righthand_position = np.zeros((3))
    righthand_rotation = np.zeros((3))
    lefthand_position = np.zeros((3))
    lefthand_rotation = np.zeros((3))
    rightelbow_position = np.zeros((3))
    rightelbow_rotation = np.zeros((3))
    leftelbow_position = np.zeros((3))
    leftelbow_rotation = np.zeros((3))
    camera_position = np.zeros((3))
    camera_rotation = np.zeros((3))
    for i in range(len_dir):
        if os.path.isdir(dir_name + '/' + dir[i]):
            continue
        f = open(dir_name + '/' + dir[i], "r")
        lines = f.readlines()
        count = 0
        for line in lines:
            if count == 0:
                pass
            elif count == 1:
                words = line.split(' ')
                if words[0] == "RH":
                    righthand_position[0] = float(words[1])
                    righthand_position[1] = float(words[2])
                    righthand_position[2] = float(words[3])
                    righthand_rotation[0] = float(words[4]) / 360.
                    righthand_rotation[1] = float(words[5]) / 360.
                    righthand_rotation[2] = float(words[6]) / 360.
                else:
                    count = (count - 1) % 6
            elif count == 2:
                words = line.split(' ')
                lefthand_position[0] = float(words[1]) * -1
                lefthand_position[1] = float(words[2])
                lefthand_position[2] = float(words[3])
                lefthand_rotation[0] = float(words[4]) / 360.
                lefthand_rotation[1] = 1 - float(words[5]) / 360.
                lefthand_rotation[2] = 1 - float(words[6]) / 360.
            elif count == 3:
                words = line.split(' ')
                rightelbow_position[0] = float(words[1])
                rightelbow_position[1] = float(words[2])
                rightelbow_position[2] = float(words[3])
                rightelbow_rotation[0] = float(words[4]) / 360.
                rightelbow_rotation[1] = float(words[5]) / 360.
                rightelbow_rotation[2] = float(words[6]) / 360.
            elif count == 4:
                words = line.split(' ')
                leftelbow_position[0] = float(words[1]) * -1
                leftelbow_position[1] = float(words[2])
                leftelbow_position[2] = float(words[3])
                leftelbow_rotation[0] = float(words[4]) / 360.
                leftelbow_rotation[1] = 1 - float(words[5]) / 360.
                leftelbow_rotation[2] = 1 - float(words[6]) / 360.
            elif count == 5:
                words = line.split(' ')
                camera_position[0] = float(words[1])
                camera_position[1] = float(words[2])
                camera_position[2] = float(words[3])
                camera_rotation[0] = float(words[4]) / 360.
                camera_rotation[1] = float(words[5]) / 360.
                camera_rotation[2] = float(words[6]) / 360.

                data = np.zeros((2, 3))
                label = np.zeros(3)
                for j in range(3):
                    data[0][j] = righthand_position[j]
                    data[1][j] = righthand_rotation[j]
                    label[j] = rightelbow_position[j]
                datas.append(np.reshape(data, -1))
                labels.append(np.reshape(label, -1))

                data_l = np.zeros((2, 3))
                label_l = np.zeros(3)
                for j in range(3):
                    data_l[0][j] = lefthand_position[j]
                    data_l[1][j] = lefthand_rotation[j]
                    label_l[j] = leftelbow_position[j]
                datas.append(np.reshape(data_l, -1))
                labels.append(np.reshape(label_l, -1))

            count = (count + 1) % 6
        f.close()

    if shuffle:
        print("Shuffling...")
        index = list(range(len(labels)))
        random.shuffle(index)
        xx = []
        yy = []
        for i in range(len(labels)):
            xx.append(datas[index[i]])
            yy.append(labels[index[i]])
        datas = xx
        labels = yy
    return np.asarray(datas, dtype=np.float32), np.asarray(labels, dtype=np.float32)



