# E. Culurciello
# May 2023


import yfinance as yf


msft = yf.Ticker("MSFT")
# get historical market data
hist = msft.history(period="1y")
print(hist["Close"])
print(hist["Close"][0:10])