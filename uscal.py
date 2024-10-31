import pandas as pd

df = pd.read_csv("/Users/sophieli/stockdata/yahoo/pricevolume/AAPL.csv")

def rmtime(s): return s[:10]
    
print(df.head())
print(df['Date'])
df['cal_dates'] = df['Date'].apply(rmtime)
print(df['cal_dates'])

df['cal_dates'].to_csv('uscal.txt', index=False)



