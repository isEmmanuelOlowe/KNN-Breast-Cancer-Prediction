import pandas as pd
import numpy as np
from KNN import KNearestNeighbours


def generateTestTrain(csv):
    """
    Generates training and testing data from the csv file

    :param csv: csv file path
    :return: training and testing dataframe in list
    """
    # reads the csv file
    df = pd.read_csv(csv)
    # deletes id column
    del df["id"]
    del df["Unnamed: 32"]
    # assigins each row a random_number
    df['random_number'] = np.random.randn(len(df))
    # seperates training and testing data from random number value
    train = df[df['random_number'] <= 0.8]
    test = df[df['random_number'] > 0.8]
    # deletes the random number field
    del train['random_number']
    del test['random_number']
    return [train, test]


def run():
    """
    Runs the K Nearest Neighbours model on dataset
    """
    train, test = generateTestTrain("data.csv")
    model = KNearestNeighbours(train)
    model.test(test)


if __name__ == "__main__":
    run()
