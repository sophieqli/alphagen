#        AAPL, IBM, MSFT...
#20100101 0.01, 0.02, ...
#20100102 -0.01, -0.015

import pandas as pd

symbs = ['AAPL', 'AAP']
all_rets = pd.read_csv("/Users/sophieli/alphagen/uscal.csv")
print(all_rets)

for s in symbs: 

    df = pd.read_csv(f"/Users/sophieli/stockdata/yahoo/pricevolume/{s}.csv")                             
    print(df)
    df['rets'] = 0
    print(df)
    for i in range(1, len(df.index)): 
        df['rets'][i] = (df['Adj Close'][i] - df['Adj Close'][i-1])/df['Adj Close'][i-1]
    print(df)
    print("done with ", s)
       






'''
def rmtime(s): return s[:10]                                                                         

print(df.head())                                                                                     
print(df['Date'])                                                                                    
df['cal_dates'] = df['Date'].apply(rmtime)                                                           
print(df['cal_dates'])                                                                               

df.drop(columns = ['Date'], inplace = True)
print(df.columns)

df = df[['cal_dates', ' Price', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']]
'''
