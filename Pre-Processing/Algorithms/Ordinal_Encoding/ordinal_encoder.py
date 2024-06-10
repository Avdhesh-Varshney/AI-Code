import numpy as np
import pandas as pd

class OrdinalEncoding:
    def __init__(self):
        self.category_mapping = {}

    # null check function
    def null_check(self, data):
        if data.isnull().values.any():
            raise ValueError("Input data contains null values. Please clean the data before encoding.")
        return False

    # fit function
    def fit(self, data):
        # null check
        if self.null_check(data) == False:
            for column in data.columns:
                unique_values = data[column].unique()
                self.category_mapping[column] = {value: idx for idx, value in enumerate(unique_values)}

    # transform function
    def transform(self, data):
        if self.null_check(data) == False:
            transformed_data = data.copy()
            for column in data.columns:
                transformed_data[column] = data[column].map(self.category_mapping[column])
            return transformed_data

    # fit transform
    def fit_transform(self, data):
        if self.null_check(data) == False:
            self.fit(data)
            return self.transform(data)