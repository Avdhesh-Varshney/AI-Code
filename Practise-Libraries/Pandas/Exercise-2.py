# create a Series object and assign it to the quotations variable. In response, print quotations variable to the console

import pandas as pd

stocks = {'PLW': 387.00, 'CDR': 339.5, 'TEN': 349.5, '11B': 391.0}

# <---- Write your code here ---->

quotations = pd.Series(data=stocks)
print(quotations)
