from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

def dataset_preparation(data_file):
    data = pd.read_excel(data_file)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data['user_id'], _ = pd.factorize(data['user_id'])
    lr = data['linguistic_range'].to_numpy()
    ga = data['grammatical_accuracy'].to_numpy()
    data = data.drop(['linguistic_range', 'grammatical_accuracy'], axis=1, inplace=True)
    return data, lr, ga

def train_clf(data, model):
    return None