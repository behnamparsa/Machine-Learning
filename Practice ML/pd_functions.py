import pandas as pd


def main():

    df = pd.DataFrame([[1,1,1],[2,2,2],[3,3,3]],
         columns = ['cow','goat','sheep'],
         index = ['grass','grain','hay']
         )
    print(df)


    for func in ['min','max','std','var','mean','sum']:
        f = getattr(df,func)
        print(func.title())
        print(f(axis=0))
        #print(f)
        print()
    

main()