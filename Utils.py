import numpy as np
import pandas as pd


def listPeopleArticle(paths, Fs, testID):
    listPerson = []

    for path in paths:

        person = pd.read_csv(path, sep=';', names=["x", "y", "z", "Pressure", 'GripAngle', 'Timestamp', 'TestID'])
        df_testID = person[person.TestID == testID].copy()
        if not df_testID.empty:
            doAllArticle(df_testID, Fs, testID)
            listPerson.append(df_testID)

    return listPerson


def doAllArticle(df, Fs, testID):
    modulePosition(df, testID)
    moduleVelocityArticle(df, Fs, testID)


def modulePosition(df, testID):
    module = np.empty(df.shape[0], dtype=float)
    x = df['x'].values.tolist()
    y = df['y'].values.tolist()
    for i in range(df.index.size):
        if i == df.index.size-1:
            module[i] = 0
        else:
            module[i] = ((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2) ** (1 / 2)
    df.insert(3, f'position_{testID}', module)


def moduleVelocityArticle(df, Fs, testID):
    module = np.empty(df.shape[0], dtype=float)
    for i in range(df.index.size):
        module[i] = df[f"position_{testID}"][df.index[0]+i] * Fs

    df[f"velocity_{testID}"] = module


def weightedAverage(list1, list2):
    multi = np.dot(list1, list2)
    return multi / list1.sum()


def tableCreaterArticle(listPerson, diagnoses, testID):
    table = np.empty((len(listPerson), 3))
    for i in range(len(listPerson)):
        table[i][0] = weightedAverage(listPerson[i][f'position_{testID}'], listPerson[i][f'velocity_{testID}'])
        table[i][1] = weightedAverage(listPerson[i][f'position_{testID}'], listPerson[i]['Pressure'])
        table[i][2] = table[i][0] * table[i][1]
    df = pd.DataFrame(data=table, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
    return df, np.full((len(listPerson)), fill_value=diagnoses)


'''
def listPeople(paths, testID):
    listPerson = []
    for path in paths:
        person = pd.read_csv(path, sep=';',
                             names=["x", "y", "z", "Pressure", 'GripAngle', 'Timestamp', 'TestID'])

        df_testID = person[person.TestID == testID]
        if not df_testID.empty:
            doAll(person, 1 / 133, testID)
            listPerson.append(person)

    return listPerson
'''
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

'''
def tableCreater(listPerson, diagnoses, testID):
    table = np.empty((len(listPerson), 3))
    for i in range(len(listPerson)):
        table[i][0] = listPerson[i]['position_' + str(testID)].sum()
        table[i][1] = listPerson[i]['velocity_' + str(testID)].mean()
        table[i][2] = listPerson[i]['acceleration_' + str(testID)].mean()
    df = pd.DataFrame(data=table,
                      columns=['totalDistance', 'meanVelocity', 'meanAcceleration'])
    return df, np.full((len(listPerson)), fill_value=diagnoses)
'''

'''
def doAll(df, T, testID):
    modulePosition(df, testID)
    moduleVelocity(df, T, testID)
    moduleAcceleration(df, T, testID)
'''
