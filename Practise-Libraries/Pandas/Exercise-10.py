# Convert the first column of the companies DataFrame to index. In response, print companies DataFrame to the console.

import pandas as pd

data_dict = {
    'company': ['Amazon', 'Microsoft', 'Facebook'],
    'price': [2375.00, 178.6, 179.2],
    'ticker': ['AMZN.US', 'MSFT.US', 'FB.US']
}

companies = pd.DataFrame(data=data_dict)

# <---- Write your code here ---->

companies = companies.set_index('company')

print(companies)
