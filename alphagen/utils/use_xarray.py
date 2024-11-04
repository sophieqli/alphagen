import numpy as np
import pandas as pd
import xarray as xr

# Define the dimensions
symbols = ['AAPL', 'GOOGL', 'MSFT']
dates = pd.date_range('2023-01-01', periods=3)  # 3 days
times = pd.date_range('09:30', '16:00', freq='5min').time  # 8 hours
features = ['open', 'high', 'low', 'close']

# Create random data
data = np.random.rand(len(dates), len(times), len(symbols), len(features))

# Create the xarray DataArray
xarr = xr.DataArray(
    data,
    dims=('date', 'time', 'symbol', 'feature'),
    coords={
        'date': dates,
        'time': times,
        'symbol': symbols,
        'feature': features
    }
)

print(xarr)


import pdb; pdb.set_trace()
#print( xarr.sel(symbol='AAPL').isel(date=pd.Timestamp('20230101') ) )
print( xarr.sel(symbol='AAPL').sel(date='20230101' ) )

