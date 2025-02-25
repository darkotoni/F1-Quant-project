

# Import libraries
import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt

# Define ticker symbols and date range
ticker = '^DJI'  # Dow Jones Industrial Average
ticker2 = 'RACE' # Ferrari

# Download stock data from Yahoo Finance
data = yf.download(ticker, start='2024-01-01')
data2 = yf.download(ticker2, start='2024-01-01')

# Ensure data downloaded correctly
if data.empty or data2.empty:
    raise ValueError("Data not downloaded properly. Check ticker symbols or date range.")

# Compute log returns for better statistical properties
data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
data2['Log_Return'] = np.log(data2['Close'] / data2['Close'].shift(1))

# Drop missing values after return calculation
data = data.dropna()
data2 = data2.dropna()

# Normalize returns (Fixes increasing trend in volatility)
data['Log_Return'] = (data['Log_Return'] - data['Log_Return'].mean()) / data['Log_Return'].std()
data2['Log_Return'] = (data2['Log_Return'] - data2['Log_Return'].mean()) / data2['Log_Return'].std()

# Calculate return spread (DJI - RACE) and standardize
returns2 = data['Log_Return'] - data2['Log_Return']
returns2 = (returns2 - returns2.mean()) / returns2.std()

# Fit GARCH(1,1) model on DJI returns
model = arch_model(data['Log_Return'], vol='Garch', p=1, q=1)
garch_fit = model.fit(disp='off')
fitted_volatility = garch_fit.conditional_volatility

# Fit GARCH(1,1) model on DJI - RACE return spread
model2 = arch_model(returns2, vol='Garch', p=1, q=1)
garch_fit2 = model2.fit(disp='off')
fitted_volatility2 = garch_fit2.conditional_volatility

# Forecast the next 5 business days
forecast_days = 5
forecast = garch_fit.forecast(horizon=forecast_days)
forecasted_volatility = np.sqrt(forecast.variance.iloc[-1].values)

# Create forecast date index
forecast_index = pd.date_range(start=data.index[-1], periods=forecast_days + 1, freq='B')[1:]

# Store forecasted volatilities in a DataFrame
forecasted_vol_df = pd.DataFrame(forecasted_volatility, index=forecast_index, columns=['Forecasted Volatility'])

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data.index, fitted_volatility, label='Fitted Volatility (GARCH) - DJI', color='orange')
plt.plot(data.index, fitted_volatility2, label='Fitted Volatility (GARCH) - DJI - RACE', color='blue')
plt.plot(forecasted_vol_df.index, forecasted_volatility, label='Forecasted Volatility (Next 5 Days)', color='purple', linestyle='--')

# Improve aesthetics
plt.title(f'{ticker} Returns and Volatility (Fitted and Forecasted)')
plt.xlabel('Date')
plt.ylabel('Volatility (Standardized)')
plt.legend()
plt.grid()

# Print volatility values for reference
print(f"Today's DJI volatility: {fitted_volatility.iloc[-1]:.4f}")
print(f"Today's DJI-RACE volatility: {fitted_volatility2.iloc[-1]:.4f}")
print(f"5-day forecasted volatility: {forecasted_volatility[-1]:.4f}")

# Show the plot
plt.show()