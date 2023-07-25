# Convert the quotations to the DataFrame and set the column name to 'price'. In response, print this DataFrame to the console.

import pandas as pd

stocks = {'PLW': 387.00, 'CDR': 339.5, 'TEN': 349.5, '11B': 391.0}
quotations = pd.Series(data=stocks)

# <---- Write your code here ---->

quotations = pd.DataFrame(quotations, columns=['price'])
print(quotations)
