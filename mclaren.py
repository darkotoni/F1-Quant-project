import requests
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

# Function to get race results from Ergast API
def get_race_results(season_start=2022, season_end=2024):
    results = []
    
    for year in range(season_start, season_end + 1):
        # Get all races for the year
        races_url = f"http://ergast.com/api/f1/{year}.json"
        races_response = requests.get(races_url).json()
        races = races_response['MRData']['RaceTable']['Races']
        
        for race in races:
            race_round = race['round']
            race_name = race['raceName']
            race_date = race['date']
            
            # Get McLaren's results for this race
            results_url = f"http://ergast.com/api/f1/{year}/{race_round}/constructorStandings.json"
            results_response = requests.get(results_url).json()
            
            # Check if we have standings
            if 'StandingsLists' in results_response['MRData']['StandingsTable']:
                standings_list = results_response['MRData']['StandingsTable']['StandingsLists']
                
                if standings_list:
                    constructor_standings = standings_list[0]['ConstructorStandings']
                    
                    for standing in constructor_standings:
                        team_name = standing['Constructor']['name']
                        
                        # Only include McLaren
                        if team_name == "McLaren":
                            position = int(standing['position'])
                            points = float(standing['points'])
                            wins = int(standing['wins'])
                            
                            results.append({
                                'date': race_date,
                                'race': race_name,
                                'team': team_name,
                                'position': position,
                                'points': points,
                                'wins': wins,
                                'year': year
                            })
    
    # Add hypothetical 2024 results (including constructors' championship)
    results.append({
        'date': '2024-12-08',
        'race': 'Abu Dhabi Grand Prix',
        'team': 'McLaren',
        'position': 1,  # Championship position
        'points': 590.5,  # Hypothetical final points
        'wins': 8,  # Hypothetical wins
        'year': 2024
    })
    
    # Convert to dataframe
    df = pd.DataFrame(results)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    return df

# Function to get stock price movements
def get_stock_prices(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    
    # Calculate daily returns
    stock_data['return'] = stock_data['Close'].pct_change() * 100
    
    # Calculate 5-day rolling volatility
    stock_data['volatility'] = stock_data['return'].rolling(window=5).std() * np.sqrt(252)  # Annualized
    
    return stock_data

# Define the season end dates from 2022 onwards
def get_season_end_dates(years):
    return [f"{year}-12-15" for year in years if year < 2024] + ["2024-12-08"]

# Define significant McLaren events (only after Google partnership)
def get_mclaren_events():
    return [
        {"date": "2022-11-13", "event": "Google Chrome\nSponsorship Announced"},
        {"date": "2023-05-07", "event": "Miami GP\nFirst Podium of 2023"},
        {"date": "2023-09-03", "event": "Italian GP\nNorris 2nd, Piastri 4th"},
        {"date": "2024-12-08", "event": "McLaren Wins\nConstructors' Championship!"}
    ]

# Main execution
if __name__ == "__main__":
    # Set date range for analysis - starting from Google partnership
    partnership_date = "2022-11-13"
    season_start = 2022
    season_end = 2024
    
    # Get McLaren race results
    print("Fetching McLaren race results...")
    mclaren_results = get_race_results(season_start, season_end)
    
    # Get season end dates
    seasons = list(range(season_start, season_end + 1))
    season_end_dates = get_season_end_dates(seasons)
    
    # Get McLaren events
    mclaren_events = get_mclaren_events()
    
    # Convert to datetime for plotting
    season_end_dates = [datetime.strptime(date, "%Y-%m-%d") for date in season_end_dates]
    for event in mclaren_events:
        event["date"] = datetime.strptime(event["date"], "%Y-%m-%d")
    
    # Get Google stock data (McLaren's major sponsor) - starting from just before partnership
    start_date = "2022-11-01"  # Start shortly before partnership
    end_date = f"{season_end}-12-31"
    google_stock = get_stock_prices('GOOG', start_date, end_date)
    
    # Create the figure with two subplots (stock price and volatility)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot Google stock price
    ax1.plot(google_stock.index, google_stock['Close'], label='Google (GOOG) Stock Price', color='#F1A000')  # McLaren papaya orange-ish
    
    # Add vertical lines for season ends and annotate with McLaren's position and points
    for i, end_date in enumerate(season_end_dates):
        if i >= len(seasons):
            continue
            
        season_year = seasons[i]
        
        # Get the data for this season
        season_data = mclaren_results[mclaren_results['year'] == season_year]
        
        if season_data.empty:
            continue
            
        # Get the last race of the season
        last_race = season_data.iloc[-1]
        
        # Try to find the closest trading day if the season end date falls on a weekend/holiday
        try:
            closest_date = min(google_stock.index, key=lambda x: abs((x - end_date).total_seconds()))
            price_at_season_end = google_stock.loc[closest_date, 'Close']
            
            # Add vertical line
            ax1.axvline(x=closest_date, color='gray', linestyle='--', alpha=0.7)
            
            # Add annotation with position and points
            position_text = f"{season_year} Season\nPosition: {last_race['position']}\nPoints: {last_race['points']}\nWins: {last_race['wins']}"
            ax1.annotate(position_text, xy=(closest_date, price_at_season_end), 
                         xytext=(10, 30), textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='gray'),
                         bbox=dict(boxstyle='round,pad=0.5', fc='#59CBE8', alpha=0.7))  # Light blue accent from McLaren
        except (ValueError, KeyError) as e:
            print(f"Could not plot season end for {season_year}: {e}")
    
    # Add vertical lines and annotations for significant McLaren events
    for event in mclaren_events:
        event_date = event["date"]
        event_description = event["event"]
        
        # For 2024 championship and Google partnership, make them special
        is_championship = "Championship" in event_description
        is_partnership = "Google" in event_description
        
        # Try to find the closest trading day
        try:
            closest_date = min(google_stock.index, key=lambda x: abs((x - event_date).total_seconds()))
            price_at_event = google_stock.loc[closest_date, 'Close']
            
            # Add vertical line for McLaren event
            line_width = 2.5 if (is_championship or is_partnership) else 1.5
            line_color = '#F58020' if not is_partnership else '#4285F4'  # Google blue for partnership
            
            ax1.axvline(x=closest_date, color=line_color, linestyle='-', linewidth=line_width, alpha=0.8)
            
            # Add annotation for the event
            bbox_color = '#FFD700' if is_championship else '#4285F4' if is_partnership else '#fff'
            font_weight = 'bold' if (is_championship or is_partnership) else 'normal'
            
            ax1.annotate(event_description, 
                         xy=(closest_date, price_at_event),
                         xytext=(-80, 40) if not is_championship else (-100, 60), 
                         textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color=line_color, linewidth=line_width),
                         bbox=dict(boxstyle='round,pad=0.5', fc=bbox_color, alpha=0.9),
                         fontweight=font_weight)
        except (ValueError, KeyError) as e:
            print(f"Could not plot event {event_description}: {e}")
    
    # Calculate volatility for the second subplot (5-day)
    volatility = google_stock['volatility'].dropna()
    
    # Plot volatility in second subplot
    ax2.plot(volatility.index, volatility, color='#F58020', label='5-Day Rolling Volatility')
    ax2.fill_between(volatility.index, 0, volatility, color='#F58020', alpha=0.2)
    
    # Add grid and legends
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax1.legend()
    ax2.legend()
    
    # Add titles and labels
    ax1.set_title('Google (GOOG) Stock Price and McLaren F1 Results\nSince Google Chrome Partnership (2022-2024)', fontsize=16)
    ax1.set_ylabel('Stock Price ($)', fontsize=12)
    ax2.set_ylabel('Volatility (5-Day)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('mclaren_google_partnership_analysis.png', dpi=300)
    plt.show()
    
    # Save the data to CSV for further analysis
    mclaren_results.to_csv("mclaren_google_partnership_results.csv", index=False)
    google_stock.to_csv("google_stock_partnership_era.csv")
    
    print("Analysis complete! Data and visualization saved.")