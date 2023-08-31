# Create the DatetimeIndex object containing the yyyy-mm-dd format dates for all days from January 2020 and assign it to the date_range variable. In response, print this variable to the console.

import pandas as pd

# <---- Write your code here ---->

# Method 1
# DatetimeIndex = pd.date_range('2020-01-01', '2020-01-31')

# Method 2
DatetimeIndex = pd.date_range('2020-01-01', periods=31)

print(DatetimeIndex)
