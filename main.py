import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# X ; Y; Z; Pressure; GripAngle; Timestamp; Test ID
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from Utils import tableCreaterArticle, listPeopleArticle

path_control = [
    'data_motion/hw_dataset/control/C_0001.txt',
    'data_motion/hw_dataset/control/C_0002.txt',
    'data_motion/hw_dataset/control/C_0003.txt',
    'data_motion/hw_dataset/control/C_0004.txt',
    'data_motion/hw_dataset/control/C_0005.txt',
    'data_motion/hw_dataset/control/C_0006.txt',
    'data_motion/hw_dataset/control/C_0007.txt',
    'data_motion/hw_dataset/control/C_0008.txt',
    'data_motion/hw_dataset/control/C_0009.txt',
    'data_motion/hw_dataset/control/C_0010.txt',
    'data_motion/hw_dataset/control/C_0011.txt',
    'data_motion/hw_dataset/control/C_0012.txt',
    'data_motion/hw_dataset/control/C_0013.txt',
    'data_motion/hw_dataset/control/C_0014.txt',
    'data_motion/hw_dataset/control/C_0015.txt'
]

path_parkinson = [
    'data_motion/hw_dataset/parkinson/P_02100001.txt',
    'data_motion/hw_dataset/parkinson/P_02100002.txt',
    'data_motion/hw_dataset/parkinson/P_05060003.txt',
    'data_motion/hw_dataset/parkinson/P_05060004.txt',
    'data_motion/hw_dataset/parkinson/P_09100001.txt',
    'data_motion/hw_dataset/parkinson/P_09100003.txt',
    'data_motion/hw_dataset/parkinson/P_09100005.txt',
    'data_motion/hw_dataset/parkinson/P_11120003.txt',
    'data_motion/hw_dataset/parkinson/P_11120004.txt',
    'data_motion/hw_dataset/parkinson/P_11120005.txt',
    'data_motion/hw_dataset/parkinson/P_12060001.txt',
    'data_motion/hw_dataset/parkinson/P_12060002.txt',
    'data_motion/hw_dataset/parkinson/P_16100003.txt',
    'data_motion/hw_dataset/parkinson/P_16100004.txt',
    'data_motion/hw_dataset/parkinson/P_23100002.txt',
    'data_motion/hw_dataset/parkinson/P_23100003.txt',
    'data_motion/hw_dataset/parkinson/P_26060001.txt',
    'data_motion/hw_dataset/parkinson/P_26060002.txt',
    'data_motion/hw_dataset/parkinson/P_26060003.txt',
    'data_motion/hw_dataset/parkinson/P_26060006.txt',
    'data_motion/hw_dataset/parkinson/P_26060007.txt',
    'data_motion/hw_dataset/parkinson/P_27110001.txt',
    'data_motion/hw_dataset/parkinson/P_27110003.txt',
    'data_motion/hw_dataset/parkinson/P_30100001.txt',
    'data_motion/hw_dataset/parkinson/P_30100002.txt',
    'data_motion/new_dataset/parkinson/H_P000-0001.txt',
    'data_motion/new_dataset/parkinson/H_P000-0002.txt',
    'data_motion/new_dataset/parkinson/H_P000-0003.txt',
    'data_motion/new_dataset/parkinson/H_P000-0004.txt',
    'data_motion/new_dataset/parkinson/H_P000-0007.txt',
    'data_motion/new_dataset/parkinson/H_P000-0008.txt',
    'data_motion/new_dataset/parkinson/H_p000-0010.txt',
    'data_motion/new_dataset/parkinson/H_P000-0011.txt',
    'data_motion/new_dataset/parkinson/H_P000-0012.txt',
    'data_motion/new_dataset/parkinson/H_P000-0013.txt',
    'data_motion/new_dataset/parkinson/H_P000-0014.txt',
    'data_motion/new_dataset/parkinson/H_P000-0015.txt',
    'data_motion/new_dataset/parkinson/H_P000-0016.txt',
    'data_motion/new_dataset/parkinson/H_p000-0017.txt',
    'data_motion/new_dataset/parkinson/H_p000-0018.txt',
    'data_motion/new_dataset/parkinson/H_P000-0019.txt',
    'data_motion/new_dataset/parkinson/H_P000-0020.txt',
    'data_motion/new_dataset/parkinson/H_P000-0021.txt',
    'data_motion/new_dataset/parkinson/H_P000-0022.txt',
    'data_motion/new_dataset/parkinson/H_P000-0023.txt',
    'data_motion/new_dataset/parkinson/H_P000-0024.txt',
    'data_motion/new_dataset/parkinson/H_P000-0025.txt',
    'data_motion/new_dataset/parkinson/H_p000-0028.txt',
    'data_motion/new_dataset/parkinson/H_P000-0029.txt',
    'data_motion/new_dataset/parkinson/H_P000-0030.txt',
    'data_motion/new_dataset/parkinson/H_P000-0031.txt',
    'data_motion/new_dataset/parkinson/H_P000-0032.txt',
    'data_motion/new_dataset/parkinson/H_P000-0033.txt',
    'data_motion/new_dataset/parkinson/H_P000-0034.txt',
    'data_motion/new_dataset/parkinson/H_P000-0035.txt',
    'data_motion/new_dataset/parkinson/H_P000-0036.txt',
    'data_motion/new_dataset/parkinson/H_P000-0037.txt',
    'data_motion/new_dataset/parkinson/H_P000-0039.txt',
    'data_motion/new_dataset/parkinson/H_P000-0040.txt',
    'data_motion/new_dataset/parkinson/H_p000-0041.txt',
    'data_motion/new_dataset/parkinson/H_p000-0042.txt',
    'data_motion/new_dataset/parkinson/H_p000-0043.txt'
]

Fs = 133
listControl_0 = listPeopleArticle(path_control, Fs, 0)
listParkinson_0 = listPeopleArticle(path_parkinson, Fs, 0)
listControl_1 = listPeopleArticle(path_control, Fs, 1)
listParkinson_1 = listPeopleArticle(path_parkinson, Fs, 1)
listControl_2 = listPeopleArticle(path_control, Fs, 2)
listParkinson_2 = listPeopleArticle(path_parkinson, Fs, 2)
"""
listPerson = pd.concat([listControl, listParkinson])
listDiagnoses = np.concatenate([controlDiagnoses, parkinsonDiagnoses])

xTrain, xTest, yTrain, yTest = train_test_split(listPerson, listDiagnoses, test_size=0.8)
knnc = KNeighborsClassifier().fit(xTrain, yTrain)
cartc = DecisionTreeClassifier().fit(xTrain, yTrain)
rfc = RandomForestClassifier().fit(xTrain, yTrain)
modelsc = [knnc, cartc, rfc]

for model in modelsc:
    name = model.__class__.__name__
    predict = model.predict(xTest)
    print(name + ": ")
    print("ACC-->", accuracy_score(yTest, predict))"""
