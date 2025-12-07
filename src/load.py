
import pandas as pd

def load_data(path):
    data = pd.read_csv(path, sep='|')
    return data
def load_csv(path):
    data = pd.read_csv(path)
    return data