import requests
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

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
                'position': int(team['position']),  # Convert to integer for easier filtering
                'points': float(team['points']),    # Convert to float for plotting
                'wins': int(team['wins'])           # Convert to integer for analysis
            })
    return pd.DataFrame(results)

# Function to get stock price movements of sponsors
def get_stock_prices(tickers, start_date, end_date):
    stock_data = yf.download(tickers, start=start_date, end=end_date)
    
    # Ensure we're correctly selecting 'Close' prices
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data = stock_data["Close"]  # Extract only Close prices
    
    # Calculate daily returns
    stock_returns = stock_data.pct_change(fill_method=None).dropna()
    
    return stock_data, stock_returns

# Define the season end dates (approximated as December 15th of each year)
def get_season_end_dates(years):
    return [f"{year}-12-15" for year in years]

# Main execution
if __name__ == "__main__":
    # Fetch race results
    season_start = 2020
    season_end = 2023
    df_race_results = get_race_results(season_start, season_end)
    
    # Get Ferrari results specifically
    ferrari_results = df_race_results[df_race_results['team'] == 'Ferrari'].copy()
    
    # Get season end dates
    seasons = list(range(season_start, season_end + 1))
    season_end_dates = get_season_end_dates(seasons)
    
    # Convert to datetime for plotting
    season_end_dates = [datetime.strptime(date, "%Y-%m-%d") for date in season_end_dates]
    
    # Get Ferrari stock data
    start_date = f"{season_start}-01-01"
    end_date = f"{season_end}-12-31"
    ferrari_stock, ferrari_returns = get_stock_prices('RACE', start_date, end_date)
    
    # Create the figure with two subplots (stock price and volatility)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot Ferrari stock price
    ax1.plot(ferrari_stock.index, ferrari_stock, label='Ferrari (RACE) Stock Price', color='#FF2800')
    
    # Add vertical lines for season ends and annotate with Ferrari's position and points
    for i, end_date in enumerate(season_end_dates):
        season_year = seasons[i]
        season_data = ferrari_results[ferrari_results['season'] == season_year].iloc[0]
        
        # Try to find the closest trading day if the season end date falls on a weekend/holiday
        closest_date = min(ferrari_stock.index, key=lambda x: abs((x - end_date).total_seconds()))
        price_at_season_end = ferrari_stock.loc[closest_date]
        
        # Add vertical line
        ax1.axvline(x=closest_date, color='gray', linestyle='--', alpha=0.7)
        
        # Add annotation with position and points
        position_text = f"{season_year} Season\nPosition: {season_data['position']}\nPoints: {season_data['points']}"
        y_pos = price_at_season_end + (ferrari_stock.max() - ferrari_stock.min()) * 0.05
        ax1.annotate(position_text, xy=(closest_date, price_at_season_end), 
                     xytext=(10, 30), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='gray'),
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
    
    # Calculate 30-day rolling volatility for the second subplot
    ferrari_returns['30d_vol'] = ferrari_returns.rolling(window=30).std() * np.sqrt(252)  # Annualized
    
    # Plot volatility in second subplot
    ax2.plot(ferrari_returns.index, ferrari_returns['30d_vol'], color='darkred', label='30-Day Rolling Volatility')
    ax2.fill_between(ferrari_returns.index, 0, ferrari_returns['30d_vol'], color='darkred', alpha=0.2)
    
    # Add grid and legends
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax1.legend()
    ax2.legend()
    
    # Add titles and labels
    ax1.set_title('Ferrari (RACE) Stock Price with F1 Season Results', fontsize=16)
    ax1.set_ylabel('Stock Price ($)', fontsize=12)
    ax2.set_ylabel('Volatility', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('ferrari_stock_f1_performance.png', dpi=300)
    plt.show()
    
    # Save the data to CSV for further analysis
    df_race_results.to_csv("f1_race_results.csv", index=False)
    ferrari_stock.to_csv("ferrari_stock_prices.csv")
    
    print("Analysis complete! Data and visualization saved.")
