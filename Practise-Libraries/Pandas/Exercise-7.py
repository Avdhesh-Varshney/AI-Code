'''Add two elements to this Series:

key: 'BBT', value: 25.5

key: 'F51', value: 19.2

In response, print quotations Series to the console.'''

import pandas as pd

stocks = {'PLW': 387.00, 'CDR': 339.5, 'TEN': 349.5, '11B': 391.0}
quotations = pd.Series(data=stocks)

# <---- Write your code here ---->

quotations = quotations.append(pd.Series({'BBT': 25.5, 'F51': 19.2}))

print(quotations)
