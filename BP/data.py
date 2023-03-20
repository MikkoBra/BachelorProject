import pandas as pd

def read_data():
    train_data = pd.read_table("data/train_data.tsv")
    test_data = pd.read_table("data/test_data.tsv")
