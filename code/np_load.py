import numpy as np
import pandas as pd

a = np.loadtxt("np_numbers.csv",delimiter = ",")
print(a)

df = pd.read_csv('np_numbers.csv')
print(df.to_string())

