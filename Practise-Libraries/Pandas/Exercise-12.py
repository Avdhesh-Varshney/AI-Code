# Create the DatetimeIndex object containing the dates in the yyyy-mm-dd format for all Mondays from 2020 and assign it to the date_range variable. Print this variable to the console.

import pandas as pd

# <---- Write your code here ---->

# Method 1
date_range = pd.date_range('2020-01-01', periods=52, freq='W-MON')

# Method 2
date_range = pd.date_range('2020-01-01', '2020-12-31', freq='W-MON')

print(date_range)
