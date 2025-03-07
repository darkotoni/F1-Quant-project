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
                'position': int(team['position']),
                'points': float(team['points']),
                'wins': int(team['wins'])
            })
    return pd.DataFrame(results)

# Function to get stock price movements
def get_stock_prices(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Ensure we're correctly selecting 'Close' prices
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data = stock_data["Close"]
    elif 'Close' in stock_data.columns:
        stock_data = stock_data['Close']
    
    # Calculate daily returns
    stock_returns = stock_data.pct_change().dropna()
    
    return stock_data, stock_returns

# Define the season end dates
def get_season_end_dates(years):
    return [f"{year}-12-15" for year in years]

# Main execution
if __name__ == "__main__":
    # Fetch race results
    season_start = 2020
    season_end = 2024  # Include 2024 to capture Hamilton's signing
    df_race_results = get_race_results(season_start, season_end)
    
    # Get Ferrari results specifically
    ferrari_results = df_race_results[df_race_results['team'] == 'Ferrari'].copy()
    
    # Get season end dates
    seasons = list(range(season_start, season_end + 1))
    season_end_dates = get_season_end_dates(seasons)
    
    # Define Lewis Hamilton signing date
    hamilton_signing_date = "2024-02-23"  # February 23, 2024
    
    # Convert to datetime for plotting
    season_end_dates = [datetime.strptime(date, "%Y-%m-%d") for date in season_end_dates]
    hamilton_date = datetime.strptime(hamilton_signing_date, "%Y-%m-%d")
    
    # Get Ferrari stock data
    start_date = f"{season_start}-01-01"
    end_date = f"{season_end}-12-31"
    ferrari_stock, ferrari_returns = get_stock_prices('RACE', start_date, end_date)
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot Ferrari stock price
    ax1.plot(ferrari_stock.index, ferrari_stock, label='Ferrari (RACE) Stock Price', color='#FF2800')
    
    # Add vertical lines for season ends and annotate with Ferrari's position and points
    for i, end_date in enumerate(season_end_dates):
        # Skip future seasons where we don't have complete data
        if end_date > datetime.now():
            continue
            
        season_year = seasons[i]
        season_data = ferrari_results[ferrari_results['season'] == season_year]
        
        # Skip if no data for this season
        if season_data.empty:
            continue
            
        season_data = season_data.iloc[0]
        
        # Try to find the closest trading day
        try:
            closest_date = min(ferrari_stock.index, key=lambda x: abs((x - end_date).total_seconds()))
            price_at_season_end = ferrari_stock.loc[closest_date]
            
            # Add vertical line
            ax1.axvline(x=closest_date, color='gray', linestyle='--', alpha=0.7)
            
            # Add annotation with position and points
            position_text = f"{season_year} Season\nPosition: {season_data['position']}\nPoints: {season_data['points']}\nWins: {season_data['wins']}"
            ax1.annotate(position_text, xy=(closest_date, price_at_season_end), 
                         xytext=(10, 30), textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='gray'),
                         bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.7))
        except (ValueError, KeyError):
            print(f"Warning: Could not find stock data near {end_date.strftime('%Y-%m-%d')}")
    
    # Add vertical line for Lewis Hamilton's signing
    try:
        closest_hamilton_date = min(ferrari_stock.index, key=lambda x: abs((x - hamilton_date).total_seconds()))
        price_at_signing = ferrari_stock.loc[closest_hamilton_date]
        
        # Add prominent vertical line for Hamilton's signing
        ax1.axvline(x=closest_hamilton_date, color='purple', linestyle='-', linewidth=2, alpha=0.8)
        
        # Add special annotation for Hamilton's signing
        hamilton_text = f"Lewis Hamilton\nSigning Announced\nFebruary 23, 2024"
        ax1.annotate(hamilton_text, xy=(closest_hamilton_date, price_at_signing), 
                     xytext=(15, 50), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='purple', linewidth=2),
                     bbox=dict(boxstyle='round,pad=0.5', fc='silver', alpha=0.9),
                     fontweight='bold')
                     
        # Calculate stock performance around Hamilton's announcement
        # Look at 5-day and 30-day windows
        try:
            pre_5d = ferrari_stock.loc[:closest_hamilton_date].iloc[-6:-1]
            post_5d = ferrari_stock.loc[closest_hamilton_date:].iloc[:5]
            
            pre_avg = pre_5d.mean()
            post_avg = post_5d.mean()
            pct_change_5d = ((post_avg - pre_avg) / pre_avg) * 100
            
            # Add this info to the plot
            impact_text = f"5-day impact: {pct_change_5d:.2f}%"
            ax1.annotate(impact_text, xy=(closest_hamilton_date, price_at_signing * 0.95), 
                         xytext=(15, -40), textcoords='offset points',
                         bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
        except:
            print("Could not calculate Hamilton announcement impact")
    except (ValueError, KeyError):
        print(f"Warning: Could not find stock data near Hamilton's signing date")
    
    # Calculate 30-day rolling volatility for the second subplot
    if isinstance(ferrari_returns, pd.DataFrame):
        returns_series = ferrari_returns.iloc[:, 0]
    else:
        returns_series = ferrari_returns
        
    # Calculate volatility as rolling standard deviation
    volatility = returns_series.rolling(window=5).std() * np.sqrt(252)  # Annualized
    
    # Plot volatility in second subplot
    ax2.plot(volatility.index, volatility.values, color='darkred', label='5-Day Rolling Volatility')
    ax2.fill_between(volatility.index, 0, volatility.values, color='darkred', alpha=0.2)
    
    # Add grid and legends
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax1.legend()
    ax2.legend()
    
    # Add titles and labels
    ax1.set_title('Ferrari (RACE) Stock Price with F1 Performance\nand Lewis Hamilton Signing', fontsize=12)
    ax1.set_ylabel('Stock Price ($)', fontsize=12)
    ax2.set_ylabel('Volatility', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('ferrari_hamilton_analysis.png', dpi=300)
    plt.show()
    
    print("Analysis complete! Visualization saved.")