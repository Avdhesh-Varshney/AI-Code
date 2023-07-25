# Convert quotations to the list and print it to the console.

import pandas as pd

stocks = {'PLW': 387.00, 'CDR': 339.5, 'TEN': 349.5, '11B': 391.0}
quotations = pd.Series(data=stocks)

# <---- Write your code here ---->

quotations = quotations.tolist()

print(quotations)
