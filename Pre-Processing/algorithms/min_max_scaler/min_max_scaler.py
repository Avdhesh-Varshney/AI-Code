import pandas as pd

# Custom MinMaxScaler class
class MinMaxScaling:
    # init function
    def __init__(self, feature_range=(0, 1)):  # feature range can be specified by the user else it takes (0,1)
        self.min = feature_range[0]
        self.max = feature_range[1]
        self.data_min_ = None
        self.data_max_ = None

    # fit function to calculate min and max value of the data
    def fit(self, data):
        # type check
        if not type(data)==pd.DataFrame:
            raise f"TypeError : parameter should be a Pandas.DataFrame; {type(data)} found"
        else:
            self.data_min_ = data.min()
            self.data_max_ = data.max()
    
    # transform function
    def transform(self, data):
        if self.data_max_ is None or self.data_min_ is None:
            raise "Call MinMaxScaling.fit() first or call MinMaxScaling.fit_transform() as the required params not found"
        else:
            data_scaled = (data - self.data_min_) / (self.data_max_ - self.data_min_)
            data_scaled = data_scaled * (self.max - self.min) + self.min
            return data_scaled

    # fit_tranform function
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    # get_params function
    def get_params(self):
        if self.data_max_ is None or self.data_min_ is None:
            raise "Params not found! Call MinMaxScaling.fit() first"
        else:
            return {"Min" : self.data_min_,
                    "Max" : self.data_max_}
            