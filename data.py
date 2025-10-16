import pandas as pd
import numpy as np
import os
from sklearn import linear_model

df = pd.read_csv("./data/all_stocks_5yr.csv")
nan_indices = df[df.isna().any(axis=1)].index

df_clean = df.dropna()
print(df_clean["date"].dtype)

obj = df_clean.iloc[0]["date"]
print(type(obj))
print(obj)

print(df_clean.shape)
print(df.columns)
print(df.head(10))

def clean_data(df_clean):
    prev = 0
    d1 = pd.Timestamp(df_clean.iloc[0]["date"])
    for i in range(1, len(df_clean)):
        d1 = pd.Timestamp(df_clean.iloc[i-1]["date"])
        d2 = pd.Timestamp(df_clean.iloc[i]["date"])

        if(d2 == d1 + pd.Timedelta(days=1)): continue
        if (i - prev > 200): 
            return df_clean.iloc[prev:i]

        prev = i
        
    return df_clean.iloc[prev:]

df_proper = clean_data(df_clean)
print(df_proper.shape)
print(df_proper)




