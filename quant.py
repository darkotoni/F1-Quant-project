import requests
import pandas as pd
import yfinance as yf

# Function to get race results from Ergast API
def get_race_results(season_start=2020, season_end=2023):
    results = []
    for year in range(season_start, season_end + 1):
        url = f"http://ergast.com/api/f1/{year}/constructorStandings.json"
        response = requests.get(url).json()
        standings = response['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings']
        
        for team in standings:
            results.append({
                'season': year,
                'team': team['Constructor']['name'],
                'position': team['position'],
                'points': team['points'],
                'wins': team['wins']
            })
    return pd.DataFrame(results)

# Function to get stock price movements of sponsors
def get_stock_prices(tickers, start_date, end_date):
    stock_data = yf.download(tickers, start=start_date, end=end_date)
    print(stock_data.columns)  # Debugging: Check available columns
    
    # Ensure we're correctly selecting 'Close' prices
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data = stock_data["Close"]  # Extract only Close prices
    
    return stock_data.pct_change(fill_method=None).dropna()

# Fetch race results
df_race_results = get_race_results()

# Example sponsors (to be expanded)
sponsor_tickers = ['RACE', 'TCS.NS', 'ORCL']  # Ferrari, Tata Consultancy, Oracle
start_date = '2020-01-01'
end_date = '2023-12-31'
df_stock_prices = get_stock_prices(sponsor_tickers, start_date, end_date)

# Save to CSV
df_race_results.to_csv("f1_race_results.csv", index=False)
df_stock_prices.to_csv("sponsor_stock_prices.csv")

print("Data extraction complete. Files saved!")