import pandas as pd
import numpy as np


def main():


    data = {
        'one':[1,2,3],
        'two':[0.1,0.2,0.3],
        'three':[-10,20,-5],

    }

    df = pd.DataFrame(data)
    print(df)

    df *= 0.5
    df = np.sin(df)
    df.iloc[:,1] = 5
    df.iloc[:,2] = [1,2,3]
    df.iloc[1,:] = 0
    print(df)
main()
    
