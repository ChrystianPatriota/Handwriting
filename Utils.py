import numpy as np
import pandas as pd


def doAllArticle(df, T, testID):
    modulePosition(df, testID)
    moduleVelocityArticle(df, T, testID)


def doAll(df, T, testID):
    modulePosition(df, testID)
    moduleVelocity(df, T, testID)
    moduleAcceleration(df, T, testID)


def modulePosition(df, testID):
    df_testID = df[df.TestID == testID]
    module = np.full(df.shape[0], np.NaN, dtype=float)
    x = df_testID['x'].copy()
    y = df_testID['y'].copy()
    for i in df_testID.index:
        if i == df_testID.index[0]:
            module[i] = 0
        else:
            module[i] = ((x[i - 1] - x[i]) ** 2 + (y[i - 1] - y[i]) ** 2) ** (1 / 2)
    df.insert(3, 'position_' + str(testID), module)
    return df


def moduleVelocityArticle(df, T, testID):
    df_testID = df[df.TestID == testID]
    module = np.full(df.shape[0], np.NaN, dtype=float)
    for i in df_testID.index:
        module[i] = df["position_" + str(testID)][i] / T

    df["velocity_" + str(testID)] = module
    return df


"""
def moduleVelocity(df, T, testID):
    df_testID = df[df.TestID == testID]
    fulfill = np.full(df.shape[0], np.NaN, dtype=float)
    x = fulfill.copy()
    y = fulfill.copy()
    module = fulfill.copy()
    for i in df_testID.index:
        if i < df_testID.shape[0] - 1:
            x[i] = (df_testID["x"][i + 1] - df_testID["x"][i]) / T
            y[i] = (df_testID["y"][i + 1] - df_testID["y"][i]) / T
        else:
            x[i] = 0
            y[i] = 0

        module[i] = ((x[i]) ** 2 + (y[i]) ** 2) ** (1 / 2)
    df["velocity_x_" + str(testID)] = x
    df["velocity_y_" + str(testID)] = y
    df["velocity_" + str(testID)] = module
    return df


def moduleAcceleration(df, T, testID):
    df_testID = df[df.TestID == testID]
    fulfill = np.full(df.shape[0], np.NaN, dtype=float)
    x = fulfill.copy()
    y = fulfill.copy()
    module = fulfill.copy()
    for i in df_testID.index:
        if i < df_testID.shape[0] - 2:
            x[i] = (df_testID["x"][i + 2] - 2 * df_testID["x"][i + 1] + df_testID["x"][i]) / T ** 2
            y[i] = (df_testID["y"][i + 2] - 2 * df_testID["y"][i + 1] + df_testID["y"][i]) / T ** 2
        elif i == df_testID.shape[0] - 2:
            x[i] = (df_testID["x"][i] - 2 * df_testID["x"][i + 1] + df_testID["x"][i]) / T ** 2
            y[i] = (df_testID["y"][i] - 2 * df_testID["y"][i + 1] + df_testID["y"][i]) / T ** 2
        else:
            x[i] = 0
            y[i] = 0
        module[i] = ((x[i]) ** 2 + (y[i]) ** 2) ** (1 / 2)
        df["acceleration_x_" + str(testID)] = x
        df["acceleration_y_" + str(testID)] = y
        df["acceleration_" + str(testID)] = module
    return df
"""


def weightedAverage(list1, list2):
    multi = np.dot(list1.dropna(), list2[list1.dropna().index])
    return multi / list1.sum()


def tableCreaterArticle(listPerson, diagnoses, testID):
    table = np.empty((len(listPerson), 3))
    for i in range(len(listPerson)):
        table[i][0] = weightedAverage(listPerson[i]['position_' + str(testID)],
                                      listPerson[i]['velocity_' + str(testID)])
        table[i][1] = weightedAverage(listPerson[i]['position_' + str(testID)], listPerson[i]['Pressure'])
        table[i][2] = table[i][0] * table[i][1]
    df = pd.DataFrame(data=table,
                      columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
    return df, np.full((len(listPerson)), fill_value=diagnoses)


def tableCreater(listPerson, diagnoses, testID):
    table = np.empty((len(listPerson), 3))
    for i in range(len(listPerson)):
        table[i][0] = listPerson[i]['position_' + str(testID)].sum()
        table[i][1] = listPerson[i]['velocity_' + str(testID)].mean()
        table[i][2] = listPerson[i]['acceleration_' + str(testID)].mean()
    df = pd.DataFrame(data=table,
                      columns=['totalDistance', 'meanVelocity', 'meanAcceleration'])
    return df, np.full((len(listPerson)), fill_value=diagnoses)


def listPeopleArticle(paths, testID):
    listPerson = []
    for path in paths:
        person = pd.read_csv(path, sep=';',
                             names=["x", "y", "z", "Pressure", 'GripAngle', 'Timestamp', 'TestID'])

        df_testID = person[person.TestID == testID]
        if not df_testID.empty:
            doAllArticle(person, 1/133, testID)
            listPerson.append(person)

    return listPerson


def listPeople(paths, testID):
    listPerson = []
    for path in paths:
        person = pd.read_csv(path, sep=';',
                             names=["x", "y", "z", "Pressure", 'GripAngle', 'Timestamp', 'TestID'])

        df_testID = person[person.TestID == testID]
        if not df_testID.empty:
            doAll(person, 1/133, testID)
            listPerson.append(person)

    return listPerson
