import numpy as np
import pandas as pd

class MinMaxScaling:
    def __init__(self):
        self.min = None
        self.max = None

    # data type check and conversion
    def check_type(self, data):
        try:
            if type(data) == pd.core.frame.DataFrame:
                data = data.to_numpy()
            elif type(data) == np.ndarray:
                pass
            return data
        except:
            raise "ValueError:Input given to MinMaxScaling.fit_transform() should be pandas.Dataframe or numpy.ndarray"
        
    # fit function to calculate minimum and maximum values for each column
    def fit(self, data):
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)


    # fit_transform code
    def fit_transform(self, data) -> np.array:
        data = self.check_type(data)
        # data transformation
        try:
            # exception handling for any null values in the data 
            if np.isnan(data).any() == True:
                raise Exception
            self.fit(data)
            return (data - self.min)/(self.max - self.min)
        except Exception as e:
            print(f"Exception in line number {e.__traceback__.tb_lineno} : {e}")
                          

    # transform code
    def transform(self, data):
        data = self.check_type(data)
        try:
            # exception handling for any null values in the data 
            if np.isnan(data).any() == True:
                raise Exception
            if self.min == None or self.max == None:
                raise Exception
            return (data - self.min)/(self.max - self.min)
        
        except Exception as e:
            print(e) 

    
    # return minimum and maximum values for current dataset
    def get_params(self):
        return {
            'minimum' : self.min,
            'maximum' : self.max,
        }

    # set custom minimum and maximum values
    def set_params(self, min, max):
        self.min = min
        self.max = max
