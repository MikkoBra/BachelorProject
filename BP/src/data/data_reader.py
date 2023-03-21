import pandas as pd


# Reads the training data into a dataframe and reduces it to relevant features
def read_train_data():
    train_data = pd.read_table("../datasets/train_data.tsv", header=0)
    train_data = train_data[['essay', 'emotion']]
    return train_data


# Reads the test data into a dataframe and reduces it to relevant features
def read_test_data():
    test_data = pd.read_table("../../datasets/test_data.tsv")
    test_data = test_data[['essay']]
    return test_data

