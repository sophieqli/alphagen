import os
import pandas as pd

username = os.getenv("USER") or os.getenv("LOGNAME")
filename = f"/Users/{username}/stockdata/universe/SP500.csv"
x = pd.read_csv(filename)

ret_df = pd.DataFrame()

for symbol in x['Symbol']:
    print(symbol)
    #loading symbol
    filename = f"/Users/{username}/stockdata/yahoo/pricevolume/{symbol}.csv"
    if os.path.isfile(filename):
        data = pd.read_csv(filename)
        data.set_index('Date', inplace=True)
        #create the return sequence
        #print( data.head() )

        #TODO: compute the return t = close_t / close_t-1 - 1 put the return here 
        ret = data['Close'].rename(symbol)

        ret_df = pd.concat( [ret_df, ret ], axis=1 )
    else:
        print(f"{filename} does not exist.")


ret_df = ret_df.sort_index(axis=1)
print( ret_df )
