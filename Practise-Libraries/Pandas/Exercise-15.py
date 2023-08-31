# Create the DateFrame object from this dictionary and assign it to the df variable. As an index, add a dates from 2020-01-01 as shown below. In response print this DataFrame to the console.

import numpy as np
import pandas as pd

np.random.seed(42)
data_dict = {
    'normal': np.random.normal(loc=0, scale=1, size=1000),
    'uniform': np.random.uniform(low=0, high=1, size=1000),
    'binomial': np.random.binomial(n=1, p=0.2, size=1000)
}

# <---- Write your code here ---->

df = pd.DataFrame(
    data=data_dict,
    index=pd.date_range('2020-01-01', periods=1000)
)

print(df)
