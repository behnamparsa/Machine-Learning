import numpy as np
import pandas as pd




def main():

    df = pd.read_csv('exercises.csv',sep = '\s*,\s*',index_col=0)
    print(df)
    print(df.columns)

    print(df.head(3))
    print(df.tail(3))





main()
