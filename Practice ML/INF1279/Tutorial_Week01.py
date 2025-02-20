import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# Reading a CSV
df = pd.read_csv('C:\GitHub\Machine-Learning\Practice ML\INF1279\possum.csv')
print(df)

print(df.loc[[2,1],:])
print()
print(df.iloc[[2,1],:])

print(df.loc[[2,3], ['sex', 'age']])
print()
print(df.iloc[[2,3], [3,4]])
#print(df.loc[[2,3],:]) #
print(df.loc[2]) # Index 2
print()
print(df.iloc[2]) # Index 2
print(df.set_index('age').loc[2,:])
print(df.set_index('age').iloc[[0,1,2,3],:])

new_df = pd.DataFrame([{'age':22,'sex':'m'},{'age':23,'sex':'m'},{'age':24,'sex':'m'}])
print(new_df)

print(df['site'].unique())
print()
print(df.groupby('site')['age'].mean())


import numpy as np
import timeit
import time
a = np.random.rand(2, 5)
print(a)

b = np.random.rand(5)

print(b)

#%%timeit
x = np.dot(a, b)
print(x)

# %%

for index, row in df.iterrows():
    print(df.loc[index,'age'])
    df.loc[index, 'age'] += 1
    print(df.loc[index,'age'])
    print(index)
    print()
    print(row)
    exit()
    