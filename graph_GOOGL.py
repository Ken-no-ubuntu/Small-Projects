import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data into a DataFrame
df = pd.read_csv("E:\pythntest\EU_law\GOOGL2.csv", parse_dates=["Date"], index_col="Date")

# Plot the closing price as a time series
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Close"], label="Close Price", color="blue")

# Add a vertical line for June 2017
plt.axvline(pd.Timestamp("2017-06-01"), color='red', linestyle='--', label="June 2017 Drop")

# Add labels and title
plt.title("GOOG Stock Price Over Time", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Stock Price (USD)", fontsize=12)
plt.legend()

# Show the plot
plt.show()
