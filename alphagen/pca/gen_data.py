import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
        close = data['Close'].rename(symbol)
        ret = close / close.shift(1) - 1

        ret_df = pd.concat( [ret_df, ret ], axis=1 )
    else:
        print(f"{filename} does not exist.")


ret_df = ret_df.sort_index(axis=1)
print( ret_df )
ret_df = ret_df.fillna(0)
ret_df = ret_df.loc[:,ret_df.sum() != 0]


# Standardize the data
scaler = StandardScaler()
ret_df_scaled = scaler.fit_transform(ret_df)

print( np.var( ret_df_scaled, axis=0 ) )

pca = PCA(n_components=2)  # Number of components to keep
principal_components = pca.fit_transform(ret_df_scaled)

# Create a DataFrame for the principal components
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

import pdb; pdb.set_trace()
