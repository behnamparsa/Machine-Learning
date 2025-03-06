import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#%matplotlib inline
#%config InlineBackend.figure_formatj = 'retina'

sns.set(style='whitegrid', palette='muted', font_scale= 1.2)

HAPPY_COLORS_PALETTE = ['#01BEFE', "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

tqdm.pandas()

pl.seed_everything(42)

# Laod Data


# Mouting Google Drive

# Dataset Source: https://www.cryptodatadownload.com/data/binance/

data_path = "C:\GitHub\Machine-Learning\data/Binance_BTCUSDT_minute.csv"
df = (pd.read_csv(data_path, parse_dates = ["date"], sep=',', index_col=False, skiprows=1))
df = df.sort_values(by="date").reset_index(drop=True)
df = df.loc[(df["date"]>= '2020-09-11 20:40:00')].reset_index(drop=True)
#df_1 = df.loc[(df["unix"] == '1599856800000')]
#print(df_1)


#print(df.head())
#print(df.shape)
df["prev_close"] = df.shift(1)["close"]
#print(df.head())

# Adding another column that will be a difference from previous close
# this will have the change in close price data per minute
df["close_change"] = df.apply(
    lambda row: 0 if np.isnan(row.prev_close) else row.close - row.prev_close,
    axis=1
)

#df["close_change"] = (df["close"] - df["prev_close"]).apply(lambda x: 0 if np.isnan(x) else x)

print(df.head())

rows = []

for _, row in tqdm(df.iterrows(), total = df.shape[0]):
    row_data = dict(
        day_of_week = row.date.dayofweek,
        day_of_month = row.date.day,
        week_of_year = row.date.week,
        month        = row.date.month,
        open = row.open,
        hight = row.high,
        close_change = row.close_change,
        close = row.close
    )

rows.append(row_data)

feature_df = pd.DataFrame(rows)

print(feature_df.head())
print(feature_df.shape)






