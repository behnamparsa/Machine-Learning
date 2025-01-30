import math
import pandas as pd
import pickle
import random
import numpy as np

df = pd.read_pickle('titanic.pickle')
features = df.drop('alive', axis=1)
target = df['alive']

X = []

for i in range(10):
    x_draw = random.randint(0,1)
    X.append(x_draw)
    
print(X)

def entropy(X,bins):
    binned_dist = np.histogram(X, bins)[0]
    #print(binned_dist)
    probs = binned_dist/np.sum(binned_dist)
    #print(probs)
    probs = probs[probs>0]
    entropy = -np.sum(probs*np.log2(probs)) 
    return entropy
    
    
print(entropy(X,2))
    
    
    
    