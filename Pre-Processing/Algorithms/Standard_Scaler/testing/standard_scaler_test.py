# #for any import conflict
import os
import sys
current = os.path.dirname("standard_scaler.py")
parent = os.path.dirname(current)
sys.path.append(current)

# print(current)
import numpy as np
import pandas as pd

from Standard_Scaler.standard_scaler import StandardScaling

data = np.random.uniform(0,10000, size=(100000, 5))
data = pd.DataFrame(data, columns=["1", "2", "3", "4", "5"])

scaler = StandardScaling()

scaled_data = scaler.fit_transform(data)

print(scaled_data)
print(scaler.get_params())