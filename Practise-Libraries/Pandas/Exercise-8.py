'''Convert the following quotations Series:



stocks = {
    'PLW': 387.00, 
    'CDR': 339.5, 
    'TEN': 349.5, 
    '11B': 391.0, 
    'BBT': 25.5, 
    'F51': 19.2
}
quotations = pd.Series(data=stocks)


to DataFrame. Reset the index and name the columns 'ticker' and 'price' respectively.

In response, print this DataFrame to the console.'''

import pandas as pd

stocks = {
    'PLW': 387.00, 
    'CDR': 339.5, 
    'TEN': 349.5, 
    '11B': 391.0, 
    'BBT': 25.5, 
    'F51': 19.2
}
quotations = pd.Series(data=stocks)

# <---- Write your code here ---->

quotations = pd.DataFrame(quotations).reset_index()

quotations.columns = ['ticker', 'price']

print(quotations)
