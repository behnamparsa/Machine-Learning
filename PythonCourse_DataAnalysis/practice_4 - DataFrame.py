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


print()

print("concat:")
oil_concat = pd.concat([oil.iloc[-10:],oil.iloc[-5:]]).reset_index(drop=True)
print(oil_concat)
print(oil_concat.nunique(dropna = False))
print(oil_concat.duplicated(subset = "date").sum())
print()
oil_concat.drop_duplicates(inplace= True)
print()
print(oil_concat)

#print(pd.concat([oil.iloc[-2:],oil.iloc[-1:]]))
#print(oil.tail())
print(transactions.head())
transactions.drop(0,axis= 0, inplace= True)
print(transactions)
new_tran = transactions.drop("date",axis = 1)
print(new_tran.head())
new_tran.drop_duplicates( subset= "store_nbr", keep = "last",inplace= True)
print(new_tran)

oil = pd.read_csv(r".\PythonCourse_DataAnalysis\project_data\oil.csv")
print(oil.info())
print(oil.isna().sum())
print("org mean: ",oil["dcoilwtico"].mean())
oil_zero = oil.fillna({"dcoilwtico": 0})
print("with n/a to zero: ",oil_zero["dcoilwtico"].mean())
oil_mean = oil.fillna({"dcoilwtico": oil["dcoilwtico"].mean()})
print("oil_mean: ", oil_mean["dcoilwtico"].mean())
print()
transactions = pd.read_csv(r".\PythonCourse_DataAnalysis\project_data\transactions.csv")
print(transactions.head())
result = transactions.groupby('store_nbr')['transactions'].sum()
print(result.loc[transactions.groupby('store_nbr')['transactions'].sum() > 4384444])
#print(result.sort_values("transactions"))

mask = ((transactions["transactions"] > 2000) & (transactions['store_nbr'] == 25))
print(transactions.loc[mask,"transactions"].count() / transactions.loc[(transactions['store_nbr'] == 25),"transactions"].count())
print(transactions.query("store_nbr in [25, 31] and date.str[6] in ['5', '6'] and transactions < 2000").loc[:,"transactions"].sum())
print()
transactions = pd.read_csv(r".\PythonCourse_DataAnalysis\project_data\transactions.csv")

print(transactions.sort_values("transactions", ascending=False).iloc[:5])

print(transactions.sort_values(["date","transactions"], ascending= [True,False]))
print()
print(transactions.sort_index(axis= 1 , ascending= False))
print()
print(transactions["store_nbr"].value_counts())
print(transactions.sort_values("store_nbr"))
print()
retail_df = pd.read_csv(r"C:\GitHub\Machine-Learning\PythonCourse_DataAnalysis\project_data\retail.csv")
sample_df = retail_df.sample(5,random_state = 85)
sample_df = sample_df.assign(
        onpromotion_flag = sample_df["onpromotion"]>0,
        family_abbrev = sample_df["family"].str[:3],
        onpromotion_ratio = sample_df["sales"] / sample_df["onpromotion"],
        sales_onprom_target = lambda x: x["onpromotion_ratio"]>100
).query("sales_onprom_target == True")
print(sample_df)
transactions["store_nbr"].column = "store_number"
transactions.rename(columns={"store_nbr":"store_number","transactions":"transaction_count"},inplace=True)
print(transactions)

