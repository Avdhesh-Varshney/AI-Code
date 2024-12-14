import pandas as pd
import numpy as np

# Custom MinMaxScaler class
class StandardScaling:
    # init function
    def __init__(self):     
        self.data_mean_ = None
        self.data_std_ = None

    # fit function to calculate min and max value of the data
    def fit(self, data):
        # type check
        if not (type(data)==pd.DataFrame or type(data)==np.ndarray):
            raise f"TypeError : parameter should be a Pandas.DataFrame or Numpy.ndarray; {type(data)} found"
        elif type(data)==pd.DataFrame:
            data = data.to_numpy()
        
        self.data_mean_ = np.mean(data, axis=0)
        self.data_std_ = np.sqrt(np.var(data, axis=0))
    
    # transform function
    def transform(self, data):
        if self.data_mean_ is None or self.data_std_ is None:
            raise "Call StandardScaling.fit() first or call StandardScaling.fit_transform() as the required params not found"
        else:
            data_scaled = (data - self.data_mean_) / (self.data_std_)
            return data_scaled

    # fit_tranform function
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    # get_params function
    def get_params(self):
        if self.data_mean_ is None or self.data_std_ is None:
            raise "Params not found! Call StandardScaling.fit() first"
        else:
            return {"Mean" : self.data_mean_,
                    "Standard Deviation" : self.data_std_}
            