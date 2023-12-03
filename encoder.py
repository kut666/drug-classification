"""
This module is designed for encoding a dataset.
"""

import pickle
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encoder_target(data: pd.DataFrame, inverse: bool = False):
    if inverse:
        with open("data/le_target.pkl", "rb") as le_load_file:
            le_model = pickle.load(le_load_file)
        data = le_model.inverse_transform(data)
    else:
        le_model = LabelEncoder()
        data = le_model.fit_transform(data)
        # print(le_model.inverse_transform(data))
        with open("data/le_target.pkl", "wb") as le_dump_file:
            pickle.dump(le_model, le_dump_file)
    return np.array(data)


class Encoder:
    """
    This class is designed for encoding a dataset.
    """

    def __init__(self, column_list: Iterable):
        self.column_list = column_list

    def encoder_data(self, data: pd.DataFrame, mode: str):
        le_model = LabelEncoder()
        for column in self.column_list:
            if mode != "test":
                data[column] = le_model.fit_transform(data[column])
                with open(f"data/le_{column}.pkl", "wb") as le_dump_file:
                    pickle.dump(le_model, le_dump_file)
            else:
                with open(f"data/le_{column}.pkl", "rb") as le_dump_file:
                    le_model = pickle.load(le_dump_file)
                data[column] = le_model.transform(data[column])
        return np.array(data)
