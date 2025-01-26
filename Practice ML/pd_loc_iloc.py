import numpy as np
import pandas as pd




def main():

    #df = pd.read_csv('exercises.csv',sep = '\s*,\s*',index_col=0)
    df = pd.read_csv('exercises.csv',sep = ',',index_col=0)

    df.columns = df.columns.str.strip()
    print(df)
    print()

    print(df.loc['Wed':'Fri'])
    print(df.loc['Wed'])
    print(df.loc['Mon':'Fri':2])
    print()
    print(df.loc['Thu':'Fri','Pullups':'Pushups'])

    print()
    print(df.loc['Thu':'Fri',['Pullups','Crunches']])
    print(df.loc[:'Fri',['Pullups','Crunches']])
    print()
    print(df.loc[:,:])                      # gives the entire table
    print()
    print(df.iloc[3:6,:2])
    






main()
