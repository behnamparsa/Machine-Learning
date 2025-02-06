import numpy as np
import pandas as pd
print("pd version: ", pd.__version__)

array = np.arange(5)
np_array = array
print("np array: \n",np_array)
pd_series = pd.Series(array,name = "Test Array")
print("pd Series: \n", pd_series )
print()
series = pd.Series(np.arange(5))
print(series.values)
print(series.values.mean())

series.index = [10, 20, 30, 40, 50]
print(series.index)
print(series)
series.name = 'special name'
print(series)
print(series.dtype)
#------------------------ pandas data type-------------------
series = series.astype("float")
print(series.dtype)


print()

import numpy as np
import pandas as pd

ages = np.array([25, 12, 15, 64, 35, 80, 45, 10, 22, 55])
ages_series = pd.Series(ages)
ages_float = ages_series.astype("float")
print(ages_float.dtype)

oil = pd.read_csv("C:\GitHub\Machine-Learning\PythonCourse_DataAnalysis\project_data\oil.csv").dropna()
print(oil.head())
oil_array = np.array(oil["dcoilwtico"].iloc[1000:1100])

print(oil_array)

oil_series = pd.Series(oil_array, name = 'oil_prices')
print(oil_series)

print(oil_series.mean())
print(oil_series.name)
print(oil_series.size)
print(oil_series.astype("int").values.mean())


my_series = pd.Series(range(5))

print(my_series)

print(my_series[3])

print(my_series[1:3])
print(my_series[1::2]) # one slice to the end and grap every other piece there. 

#------------ iloc method ------------------
#last index number in iloc method is not inclusive
print()
print(my_series)
print("select one value: ",my_series[2])        # only value
print("Slicing: \n",my_series[2:4]) # includes the index
print("filter to a value: ",my_series.iloc[2])
print("Slicing: \n",my_series.iloc[2:4])
print("Filter a list:\n", my_series.iloc[[1,3]])
#--------------- lok method --------------------
# last column label in loc method is inclusive
print("loc method")
print(my_series.loc[2])
print("Slicing: \n",my_series.loc[2:4]) #5th row is inclusive here while it was not inclusive in iloc method.

# duplicate index values is possible but can cause confusion for users

import pandas as pd

# Provided series of ages with duplicate index values
ages = pd.Series(
    [25, 30, 35, 40, 45, 25, 30], 
    index=['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Alice', 'Bob']
)
print(ages.index)

for index in ages.index:
        print(ages[index])