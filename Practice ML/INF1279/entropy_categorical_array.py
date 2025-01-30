import math
import pandas as pd
import pickle
import random
import numpy as np

df = pd.read_pickle('titanic.pickle')
features = df.drop('alive', axis=1)
target = pd.Series(df['alive'])


def entropy_score(y):
    """Compute the entropy of a list of classes.

    Parameters
    ----------
    y : pd.Series
        A pandas Series containing the classes.

    Returns
    -------
    float
        The entropy of the classes.
    """
    class_distribution = y.value_counts(normalize=True).tolist()
    entropy = 0
    for p in class_distribution:
        entropy -= p * math.log(p,2)
    return entropy

def main():
    print(entropy_score(target))
    
if __name__ == '__main__':
    main()
    

    
    
    
    