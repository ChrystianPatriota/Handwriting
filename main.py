import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# X ; Y; Z; Pressure; GripAngle; Timestamp; Test ID
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from getThings import tableCreater, listPeople, tableCreaterArticle, listPeopleArticle

path_control = [
    '2/hw_dataset/control/C_0001.txt',
    '2/hw_dataset/control/C_0002.txt',
    '2/hw_dataset/control/C_0003.txt',
    '2/hw_dataset/control/C_0004.txt',
    '2/hw_dataset/control/C_0005.txt',
    '2/hw_dataset/control/C_0006.txt',
    '2/hw_dataset/control/C_0007.txt',
    '2/hw_dataset/control/C_0008.txt',
    '2/hw_dataset/control/C_0009.txt',
    '2/hw_dataset/control/C_0010.txt',
    '2/hw_dataset/control/C_0011.txt',
    '2/hw_dataset/control/C_0012.txt',
    '2/hw_dataset/control/C_0013.txt',
    '2/hw_dataset/control/C_0014.txt',
    '2/hw_dataset/control/C_0015.txt'
]

path_parkinson = [
    '2/hw_dataset/parkinson/P_02100001.txt',
    '2/hw_dataset/parkinson/P_02100002.txt',
    '2/hw_dataset/parkinson/P_05060003.txt',
    '2/hw_dataset/parkinson/P_05060004.txt',
    '2/hw_dataset/parkinson/P_09100001.txt',
    '2/hw_dataset/parkinson/P_09100003.txt',
    '2/hw_dataset/parkinson/P_09100005.txt',
    '2/hw_dataset/parkinson/P_11120003.txt',
    '2/hw_dataset/parkinson/P_11120004.txt',
    '2/hw_dataset/parkinson/P_11120005.txt',
    '2/hw_dataset/parkinson/P_12060001.txt',
    '2/hw_dataset/parkinson/P_12060002.txt',
    '2/hw_dataset/parkinson/P_16100003.txt',
    '2/hw_dataset/parkinson/P_16100004.txt',
    '2/hw_dataset/parkinson/P_23100002.txt',
    '2/hw_dataset/parkinson/P_23100003.txt',
    '2/hw_dataset/parkinson/P_26060001.txt',
    '2/hw_dataset/parkinson/P_26060002.txt',
    '2/hw_dataset/parkinson/P_26060003.txt',
    '2/hw_dataset/parkinson/P_26060006.txt',
    '2/hw_dataset/parkinson/P_26060007.txt',
    '2/hw_dataset/parkinson/P_27110001.txt',
    '2/hw_dataset/parkinson/P_27110003.txt',
    '2/hw_dataset/parkinson/P_30100001.txt',
    '2/hw_dataset/parkinson/P_30100002.txt',
    '2/new_dataset/parkinson/H_P000-0001.txt',
    '2/new_dataset/parkinson/H_P000-0002.txt',
    '2/new_dataset/parkinson/H_P000-0003.txt',
    '2/new_dataset/parkinson/H_P000-0004.txt',
    '2/new_dataset/parkinson/H_P000-0007.txt',
    '2/new_dataset/parkinson/H_P000-0008.txt',
    '2/new_dataset/parkinson/H_p000-0010.txt',
    '2/new_dataset/parkinson/H_P000-0011.txt',
    '2/new_dataset/parkinson/H_P000-0012.txt',
    '2/new_dataset/parkinson/H_P000-0013.txt',
    '2/new_dataset/parkinson/H_P000-0014.txt',
    '2/new_dataset/parkinson/H_P000-0015.txt',
    '2/new_dataset/parkinson/H_P000-0016.txt',
    '2/new_dataset/parkinson/H_p000-0017.txt',
    '2/new_dataset/parkinson/H_p000-0018.txt',
    '2/new_dataset/parkinson/H_P000-0019.txt',
    '2/new_dataset/parkinson/H_P000-0020.txt',
    '2/new_dataset/parkinson/H_P000-0021.txt',
    '2/new_dataset/parkinson/H_P000-0022.txt',
    '2/new_dataset/parkinson/H_P000-0023.txt',
    '2/new_dataset/parkinson/H_P000-0024.txt',
    '2/new_dataset/parkinson/H_P000-0025.txt',
    '2/new_dataset/parkinson/H_p000-0028.txt',
    '2/new_dataset/parkinson/H_P000-0029.txt',
    '2/new_dataset/parkinson/H_P000-0030.txt',
    '2/new_dataset/parkinson/H_P000-0031.txt',
    '2/new_dataset/parkinson/H_P000-0032.txt',
    '2/new_dataset/parkinson/H_P000-0033.txt',
    '2/new_dataset/parkinson/H_P000-0034.txt',
    '2/new_dataset/parkinson/H_P000-0035.txt',
    '2/new_dataset/parkinson/H_P000-0036.txt',
    '2/new_dataset/parkinson/H_P000-0037.txt',
    '2/new_dataset/parkinson/H_P000-0039.txt',
    '2/new_dataset/parkinson/H_P000-0040.txt',
    '2/new_dataset/parkinson/H_p000-0041.txt',
    '2/new_dataset/parkinson/H_p000-0042.txt',
    '2/new_dataset/parkinson/H_p000-0043.txt'
]
testID = 0
listControl = listPeopleArticle(path_control, testID)
print(listControl[0])
listParkinson = listPeopleArticle(path_parkinson, testID)
listControl, controlDiagnoses = tableCreaterArticle(listControl, 0, testID)
listParkinson, parkinsonDiagnoses = tableCreaterArticle(listParkinson, 1, testID)

listPerson = pd.concat([listControl, listParkinson])
listDiagnoses = np.concatenate([controlDiagnoses, parkinsonDiagnoses])
print(listPerson.to_string())

xTrain, xTest, yTrain, yTest = train_test_split(listPerson, listDiagnoses, test_size=0.5)
knnc = KNeighborsClassifier().fit(xTrain, yTrain)
cartc = DecisionTreeClassifier().fit(xTrain, yTrain)
rfc = RandomForestClassifier().fit(xTrain, yTrain)
modelsc = [knnc, cartc, rfc]

for model in modelsc:
    name = model.__class__.__name__
    predict = model.predict(xTest)
    print(name + ": ")
    print("ACC-->", accuracy_score(yTest, predict))
