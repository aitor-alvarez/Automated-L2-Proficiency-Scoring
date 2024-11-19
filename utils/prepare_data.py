import json
import pandas as pd


def create_temporal_data(file_path):
    #json file is ordered by session id by default
    f = json.load(open(file_path))
    df = pd.DataFrame.from_dict(f)
