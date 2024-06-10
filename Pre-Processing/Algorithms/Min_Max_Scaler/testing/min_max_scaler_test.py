# for any import conflict
import os
import sys
current = os.path.dirname("min_max_scaler.py")
parent = os.path.dirname(current)
sys.path.append(current)

import numpy as np
import pandas as pd

from Min_Max_Scaler.min_max_scaler import MinMaxScaling

data = np.random.uniform(0,10000, size=(100000, 5))
data = pd.DataFrame(data, columns=["1", "2", "3", "4", "5"])

print(f"Original Data : \n{data}")

scaler = MinMaxScaling()

scaler.fit(data)

print(f"Scaled Data : \n{scaler.fit_transform(data)}")
print(f"Associated Parameters : \n{scaler.get_params()}")