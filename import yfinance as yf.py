import yfinance as yf

# Specify the stock ticker (e.g., 'AAPL' for Apple)
stock = yf.Ticker("GOOG")

# Download historical data for a specified period
data = stock.history(start="2010-01-01", end="2024-10-21")  # Last 5 years

# Save to CSV
data.to_csv("GOOGL2.csv")

print(data.head())  # View the first few rows
