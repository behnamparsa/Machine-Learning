import pandas as pd

retail_df = pd.read_csv(r"C:\GitHub\Machine-Learning\PythonCourse_DataAnalysis\project_data\retail.csv")
print(retail_df.head())
transactions = pd.read_csv(r".\PythonCourse_DataAnalysis\project_data\transactions.csv")
print(transactions.head())
print(transactions.shape)
print(transactions.info())
print(transactions.columns)
print(transactions.dtypes)
print(transactions.describe(include = "all").round())

oil = pd.read_csv(r".\PythonCourse_DataAnalysis\project_data\oil.csv")
print(oil.info())

print(transactions.isna().sum())
print(transactions.info())
print(transactions.describe(include = "all"))
print(retail_df["family"].head())
print()
print(retail_df.family.head())

oil.columns = ['date','price']
oil["euro_price"] = oil["price"] * 1.1
print(oil[["price","date"]].head())
print()
print(retail_df.columns)
print(retail_df.iloc[:5,1:4])
print(retail_df.loc[:5,"date"])
print()
print(oil.iloc[:5,[-1]])
print(oil.loc[:5,"date":"euro_price"])
print()
print(transactions.loc[1:,"store_nbr":"transactions"])
print()
print(transactions["store_nbr"].nunique())
print(transactions["transactions"].sum()/1000000)
oil = pd.read_csv(r".\PythonCourse_DataAnalysis\project_data\oil.csv")
#oil = oil.append(oil.iloc[-1]).reset_index(drop = True)
print(oil.iloc[-1:])
print(pd.concat([oil,oil.iloc[-5:]]))
#print(oil.tail())

