# Create the DatetimeIndex object containing the dates in the format yyyy-mm-dd hh: mm: ss for January 1, 2021 with a time interval of 1h and assign to the variable date_range. In response, print this variable to the console.

import pandas as pd

# <---- Write your code here ---->

# Method 1
date_range = pd.date_range('2021-01-01', periods=24, freq='H')

# Method 2
date_range = pd.date_range('2021-01-01', '2021-01-02', freq='H', closed='left')

print(date_range)
