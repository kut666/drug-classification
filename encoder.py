from typing import Iterable
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
import pandas as pd

class Encoder:
    def __init__(self, column_list: Iterable):
        self.column_list = column_list

    def encoder_data(self, data: pd.DataFrame, mode: str):
        le = LabelEncoder()
        for column in self.column_list:
            if mode != "test":
                data[column] = le.fit_transform(data[column])
                with open(f'le_{column}.pkl', 'wb') as le_dump_file:
                    pickle.dump(le, le_dump_file)
            else:
                with open(f'le_{column}.pkl', 'rb') as le_dump_file:
                    le = pickle.load(le_dump_file)
                data[column] = le.transform(data[column])
        return np.array(data)

    def encoder_target(self, data: pd.DataFrame, inverse: bool=False):
        if inverse:
            with open('le_target.pkl', 'rb') as le_load_file:
                le = pickle.load(le_load_file)
            data = le.inverse_transform(data) 
        else:
            le = LabelEncoder()
            data = le.fit_transform(data)
            with open('le_target.pkl', 'wb') as le_dump_file:
                pickle.dump(le, le_dump_file)
        return np.array(data)