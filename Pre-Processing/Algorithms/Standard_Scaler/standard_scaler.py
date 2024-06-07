# import os
# import sys
# current = os.path.dirname(os.path.realpath("standard_scaler.py"))
# parent = os.path.dirname(current)
# sys.path.append(current)
# sys.path.append('/path/to/your/project')

import numpy as np
import pandas as pd

class StandardScaling:
    def __init__(self):
        self.mean = None
        self.std = None


    # data type check and conversion
    def check_type(self, data):
        try:
            if type(data) == pd.core.frame.DataFrame:
                data = data.to_numpy()
            elif type(data) == np.ndarray:
                pass
            return data
        except:
            raise "ValueError:Input given to StandardScaling.fit_transform() should be pandas.Dataframe or numpy.ndarray" 


    # fit function to calculate mean and standard deviation
    def fit(self, data):
        self.mean = data.mean(axis=0)
        self.std = np.sqrt(data.var(axis=0))


    # fit_transform code
    def fit_transform(self, data) -> np.array:
        data = self.check_type(data)
        # data transformation
        try:
            # exception handling for any null values in the data 
            if np.isnan(data).any() == True:
                raise Exception
            self.fit(data)
            return (data - self.mean)/self.std
        except Exception as e:
            print(f"Exception in line number {e.__traceback__.tb_lineno} : {e}")


    # transform code
    def transform(self, data):
        data = self.check_type(data)
        try:
            # exception handling for any null values in the data 
            if np.isnan(data).any() == True:
                raise Exception
            self.fit(data)
            return (data - self.mean)/self.std
        
        except Exception as e:
            print(e)

    
    # return mean and standard deviation for current dataset
    def get_params(self):
        return {
            'mean' : self.mean,
            'standard deviation' : self.std,
        }


    # set custom mean and standard deviation values
    def set_params(self, mean, std):
        self.mean = mean
        self.std = std