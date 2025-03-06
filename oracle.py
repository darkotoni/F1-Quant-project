import requests
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

# Function to get race results from Ergast API
def get_race_results(season_start=2020, season_end=2024):
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
def get_stock_prices(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Ensure we're correctly selecting 'Close' prices
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data = stock_data["Close"]  # Extract only Close prices
    elif 'Close' in stock_data.columns:
        stock_data = stock_data['Close']
    
    # Calculate daily returns
    stock_returns = stock_data.pct_change().dropna()
    
    return stock_data, stock_returns

# Define the season end dates (approximated as December 15th of each year)
def get_season_end_dates(years):
    return [f"{year}-12-15" for year in years]

# Define dates when Max Verstappen won championships
def get_max_championship_dates():
    return [
        "2021-12-12",  # 2021 Abu Dhabi Grand Prix
        "2022-10-09",  # 2022 Japanese Grand Prix
        "2023-10-07",  # 2023 Qatar Grand Prix Sprint Race
        "2024-11-24"   # 2024 Las Vegas Grand Prix
    ]

# Main execution
if __name__ == "__main__":
    # Fetch race results
    season_start = 2020
    season_end = 2024
    df_race_results = get_race_results(season_start, season_end)
    
    # Get Red Bull results specifically
    # Note: Red Bull might be listed as "Red Bull" or "Red Bull Racing" in the API
    redbull_results = df_race_results[df_race_results['team'].str.contains('Red Bull')].copy()
    
    # Get season end dates
    seasons = list(range(season_start, season_end + 1))
    season_end_dates = get_season_end_dates(seasons)
    
    # Get Max Verstappen championship dates
    max_championship_dates = get_max_championship_dates()
    
    # Convert to datetime for plotting
    season_end_dates = [datetime.strptime(date, "%Y-%m-%d") for date in season_end_dates]
    max_championship_dates = [datetime.strptime(date, "%Y-%m-%d") for date in max_championship_dates]
    
    # Filter championship dates to only include those within our date range
    max_championship_dates = [date for date in max_championship_dates 
                             if date.year >= season_start and date.year <= season_end]
    
    # Get Oracle stock data (major Red Bull sponsor)
    start_date = f"{season_start}-01-01"
    end_date = f"{season_end}-12-31"
    oracle_stock, oracle_returns = get_stock_prices('ORCL', start_date, end_date)
    
    # Create the figure with two subplots (stock price and volatility)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot Oracle stock price
    ax1.plot(oracle_stock.index, oracle_stock, label='Oracle (ORCL) Stock Price', color='#0600EF')  # Using a blue similar to Red Bull Racing
    
    # Add vertical lines for season ends and annotate with Red Bull's position and points
    for i, end_date in enumerate(season_end_dates):
        season_year = seasons[i]
        season_data = redbull_results[redbull_results['season'] == season_year].iloc[0]
        
        # Try to find the closest trading day if the season end date falls on a weekend/holiday
        closest_date = min(oracle_stock.index, key=lambda x: abs((x - end_date).total_seconds()))
        price_at_season_end = oracle_stock.loc[closest_date]
        
        # Add vertical line
        ax1.axvline(x=closest_date, color='gray', linestyle='--', alpha=0.7)
        
        # Add annotation with position and points
        position_text = f"{season_year} Season\nPosition: {season_data['position']}\nPoints: {season_data['points']}\nWins: {season_data['wins']}"
        ax1.annotate(position_text, xy=(closest_date, price_at_season_end), 
                     xytext=(10, 30), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='gray'),
                     bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.7))
                     
    # Add vertical lines for Max Verstappen's championship wins
    for champ_date in max_championship_dates:
        # Try to find the closest trading day
        try:
            closest_date = min(oracle_stock.index, key=lambda x: abs((x - champ_date).total_seconds()))
            price_at_champ = oracle_stock.loc[closest_date]
            
            # Add prominent vertical line for championship
            ax1.axvline(x=closest_date, color='red', linestyle='-', linewidth=2, alpha=0.8)
            
            # Add special annotation for Max's championship
            champ_text = f"Max Verstappen\nWorld Champion\n{champ_date.strftime('%b %d, %Y')}"
            ax1.annotate(champ_text, xy=(closest_date, price_at_champ), 
                         xytext=(15, -40), textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='red', linewidth=2),
                         bbox=dict(boxstyle='round,pad=0.5', fc='gold', alpha=0.9),
                         fontweight='bold')
        except (ValueError, KeyError):
            print(f"Warning: Could not find stock data near {champ_date.strftime('%Y-%m-%d')}")
    
    # Calculate 5-day rolling volatility for the second subplot
    # Make sure we're working with a Series for the volatility calculation
    if isinstance(oracle_returns, pd.DataFrame):
        returns_series = oracle_returns.iloc[:, 0]  # Take first column if it's a DataFrame
    else:
        returns_series = oracle_returns
        
    # Calculate volatility as rolling standard deviation
    volatility = returns_series.rolling(window=5).std() * np.sqrt(252)  # Annualized
    
    # Plot volatility in second subplot
    ax2.plot(volatility.index, volatility.values, color='darkblue', label='5-Day Rolling Volatility')
    ax2.fill_between(volatility.index, 0, volatility.values, color='darkblue', alpha=0.2)
    
    # Add grid and legends
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax1.legend()
    ax2.legend()
    
    # Add titles and labels
    ax1.set_title('Oracle (ORCL) Stock Price with Red Bull F1 Results\nand Max Verstappen Championships', fontsize=16)
    ax1.set_ylabel('Stock Price ($)', fontsize=12)
    ax2.set_ylabel('Volatility', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('redbull_oracle_f1_performance.png', dpi=300)
    plt.show()
    
    # Save the data to CSV for further analysis
    redbull_results.to_csv("redbull_f1_results.csv", index=False)
    oracle_stock.to_csv("oracle_stock_prices.csv")
    
    print("Analysis complete! Data and visualization saved.")
    
    # Additional insight: Compare Oracle's performance during Red Bull partnership
    # Oracle became Red Bull's title sponsor in 2022
    if isinstance(oracle_returns, pd.DataFrame):
        returns_series = oracle_returns.iloc[:, 0]
    else:
        returns_series = oracle_returns
        
    pre_partnership = returns_series['2020-01-01':'2021-12-31']
    during_partnership = returns_series['2022-01-01':]
    
    print("\nOracle Stock Performance Analysis:")
    print(f"Average Daily Return Before Red Bull Partnership (2020-2021): {pre_partnership.mean() * 100:.4f}%")
    print(f"Average Daily Return During Red Bull Partnership (2022-2024): {during_partnership.mean() * 100:.4f}%")
    print(f"Volatility Before Partnership: {pre_partnership.std() * np.sqrt(252) * 100:.4f}%")
    print(f"Volatility During Partnership: {during_partnership.std() * np.sqrt(252) * 100:.4f}%")