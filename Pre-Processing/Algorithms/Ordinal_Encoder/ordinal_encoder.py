import pandas as pd

class OrdinalEncoding:
    def __init__(self):
        self.category_mapping = {}
    
    def fit(self, data):
        # Fit the encoder to the data (pandas DataFrame).
        # type check
        if not type(data)==pd.DataFrame:
            raise f"Type of data should be Pandas.DataFrame; {type(data)} found"
        for column in data.columns:
            unique_categories = sorted(set(data[column]))
            self.category_mapping[column] = {category: idx for idx, category in enumerate(unique_categories)}
    
    def transform(self, data):
        # Transform the data (pandas DataFrame) to ordinal integers.
        # checking for empty mapping
        if not self.category_mapping:
            raise "Catrgorical Mapping not found. Call OrdinalExcoding.fit() method or call OrdinalEncoding.fit_transform() method"
        
        data_transformed = data.copy()
        for column in data.columns:
            data_transformed[column] = data[column].map(self.category_mapping[column])
        return data_transformed
    
    def fit_transform(self, data):
        # Fit the encoder and transform the data in one step.
        self.fit(data)
        return self.transform(data)
