'''Create the following DataFrame object and assign it to the companies variable:

     company   price   ticker
0     Amazon  2375.0  AMZN.US
1  Microsoft   178.6  MSFT.US
2   Facebook   179.2    FB.US

In response, print companies DataFrame to the console.'''

import pandas as pd

# <---- Write your code here ---->

companies = {
    'company': ['Amazon', 'Microsoft', 'Facebook'],
    'price' : [2375.0, 178.6, 179.2],
    'ticker' : ['AMZN.US', 'MSFT.US', 'FB.US']
}

companies = pd.DataFrame(companies)

print(companies)
