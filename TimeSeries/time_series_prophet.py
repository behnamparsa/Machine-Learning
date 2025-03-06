import warnings
warnings.simplefilter('ignore')

import pandas as pd

from prophet import Prophet

#visualization
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("C:\GitHub\Machine-Learning\data\dataset.csv")

#print(df.head())
#print(df.describe())

print(df.dtypes)

df['Year'] = df['Time Date'].apply(lambda x: str(x)[-4:])
df['Month'] = df['Time Date'].apply(lambda x: str(x)[-6:-4])
df['Day'] = df['Time Date'].apply(lambda x: str(x)[:-6])
df['ds'] = pd.DatetimeIndex(df['Year']+'-'+df['Month']+'-'+df['Day'])

#print(df.head())
#print(df.dtypes)

df.drop(["Time Date","Product","Store","Year","Month","Day"], axis=1, inplace=True)
df.columns = ['y','ds']

print(df.head())
print(df.describe())

#Train Model
m = Prophet(interval_width=0.95, daily_seasonality=True)
model = m.fit(df)


#forecast
future = m.make_future_dataframe(periods=100, freq='D')
forecast = m.predict(future)
print(forecast.head())
print(forecast.tail())

print(forecast[["ds","trend","yhat"]])

plot1 = m.plot(forecast)
plot2 = m.plot_components(forecast)
plt.show()