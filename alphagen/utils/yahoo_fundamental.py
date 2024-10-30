import yfinance as yf

# Define the stock symbol
symbol = "AAPL"

# Create a Ticker object
stock = yf.Ticker(symbol)

# Fetch historical financial data
financials = stock.financials
quarterly_financials = stock.quarterly_financials

# Print the annual financials
print("\nAnnual Financials:")
print(financials)

# Print the quarterly financials
print("\nQuarterly Financials:")
print(quarterly_financials)

# If you want to calculate ROE manually (Net Income / Shareholder's Equity)
# Fetch balance sheet for shareholder's equity
balance_sheet = stock.balance_sheet

# Calculate ROE for the latest financial year
if 'Net Income' in financials.index and 'Shareholder"s Equity' in balance_sheet.index:
    net_income = financials.loc['Net Income'].iloc[0]
    shareholder_equity = balance_sheet.loc['Total Equity Net Minority Interest'].iloc[0]
    
    roe = net_income / shareholder_equity if shareholder_equity != 0 else None
    print(f"\nLatest ROE for {symbol}: {roe:.2%}" if roe else "ROE data not available.")
else:
    print("Unable to calculate ROE: missing data.")

