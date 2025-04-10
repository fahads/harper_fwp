# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from io import StringIO

# Load the Airline Passengers dataset - a classic time series dataset
def load_airline_passengers():
    """
    Load the classic Box & Jenkins Airline Passengers dataset (1949-1960)
    Source: https://www.kaggle.com/datasets/rakannimer/air-passengers
    """
    # URL for the dataset (hosted on GitHub for reliability)
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    
    # Download the data
    print("Downloading Airline Passengers dataset...")
    response = requests.get(url)
    
    # Check if download was successful
    if response.status_code == 200:
        # Read the CSV data directly from the response
        data = StringIO(response.text)
        df = pd.read_csv(data)
        
        # Convert the 'Month' column to datetime and set as index
        df['Month'] = pd.to_datetime(df['Month'])
        df = df.set_index('Month')
        
        # Rename the column for clarity
        df.columns = ['Passengers']
        
        print("Dataset loaded successfully!")
        return df
    else:
        print(f"Failed to download data. Status code: {response.status_code}")
        return None

# Alternative method: Load directly from file if downloaded locally
def load_from_local(filepath="airline-passengers.csv"):
    """
    Load the dataset from a local file
    """
    df = pd.read_csv(filepath)
    df['Month'] = pd.to_datetime(df['Month'])
    df = df.set_index('Month')
    df.columns = ['Passengers']
    return df

# Basic time series exploration
def explore_time_series(df):
    """
    Demonstrate basic time series exploration techniques
    """
    # Display basic information about the dataset
    print("\nDataset Info:")
    print(df.info())
    
    # Show first few rows
    print("\nFirst few rows of data:")
    print(df.head())
    
    # Basic statistics
    print("\nDescriptive statistics:")
    print(df.describe())
    
    # Create a basic time series plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Passengers'])
    plt.title('Monthly Airline Passenger Numbers (1949-1960)')
    plt.xlabel('Date')
    plt.ylabel('Number of Passengers (thousands)')
    plt.grid(True)
    plt.show()
    
    # Demonstrate time-based operations
    print("\nYearly average passengers:")
    yearly_avg = df.resample('Y')['Passengers'].mean()
    print(yearly_avg)
    
    # Calculate year-over-year growth
    print("\nYear-over-year growth (%):")
    yearly_totals = df.resample('Y')['Passengers'].sum()
    yoy_growth = yearly_totals.pct_change() * 100
    print(yoy_growth)

# Identifying time series components
def identify_components(df):
    """
    Visualize and identify the components of the time series
    """
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Original time series
    axes[0].plot(df.index, df['Passengers'])
    axes[0].set_title('Original Time Series: Monthly Airline Passengers')
    axes[0].set_ylabel('Passengers (thousands)')
    axes[0].grid(True)
    
    # Trend: 12-month rolling average
    trend = df['Passengers'].rolling(window=12, center=True).mean()
    axes[1].plot(df.index, trend, color='red')
    axes[1].set_title('Trend Component (12-month moving average)')
    axes[1].set_ylabel('Trend')
    axes[1].grid(True)
    
    # Calculate and plot seasonal component
    # Detrended data = original - trend
    detrended = df['Passengers'] - trend
    axes[2].plot(df.index, detrended, color='green')
    axes[2].set_title('Seasonal and Random Components (Original - Trend)')
    axes[2].set_ylabel('Deviation from Trend')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate monthly seasonality
    print("\nMonthly seasonal patterns:")
    # Group by month and calculate average
    monthly_avg = df.groupby(df.index.month)['Passengers'].mean()
    
    plt.figure(figsize=(10, 6))
    monthly_avg.plot(kind='bar')
    plt.title('Average Passengers by Month (Seasonal Pattern)')
    plt.xlabel('Month')
    plt.ylabel('Average Passengers (thousands)')
    plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True, axis='y')
    plt.show()

def student_exercise(df):
    """
    Template for student exercise with step-by-step instructions
    """
    print("STUDENT EXERCISE INSTRUCTIONS")
    print("============================")
    print("1. Calculate and plot quarterly passenger totals")    
    print("2. Find the quarter with the highest average passengers")
    print("3. Calculate the percentage growth from the first year to the last year")
    print("4. Create a 6-month rolling average and plot it against the original data")
    
    # Exercise 1: Calculate quarterly passenger totals
    quarterly_totals = df.resample('Q')['Passengers'].sum()
    plt.figure(figsize=(12,6))
    quarterly_totals.plot(kind='bar')
    plt.title('Quarterly Passenger Totals')
    plt.xlabel('Quarter')
    plt.ylabel('Total Passengers (thousands)')
    plt.grid(True, axis='y')
    plt.show()

    # Exercise 2: Find quarter with highest average
    quarterly_avg = df.resample('Q')['Passengers'].mean()
    highest_quarter = quarterly_avg.idxmax()
    print(f"\nQuarter with the highest avg passengers: {highest_quarter.quarter}Q{highest_quarter.year}")
    print(f"Average passengers: {quarterly_avg.max():.2f} thousand")

    # Exercise 3: Calculate percentage growth from first to last year
    first_year = df.loc['1949'].mean()['Passengers']
    last_year = df.loc['1960'].mean()['Passengers']
    growth = ((last_year / first_year) - 1) * 100
    print(f"\nGrowth from 1949 to 1960: {growth:.2f}%")

    # Exercise 4: Create and plot a 6-month rolling average

    rolling_avg = df['Passengers'].rolling(window=6).mean()
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Passengers'], label="Monthly Passengers", alpha=0.7)
    plt.plot(df.index, rolling_avg, label='6-month moving average', linewidth=2)
    plt.title('Monthly Passengers with 6-month Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Passengers (thousands)')
    plt.legend()
    plt.grid(True)
    plt.show()



# Sample homework assignment
def homework_template(df):
    """
    Template code for homework assignment
    """
    print("HOMEWORK ASSIGNMENT")
    print("==================")
    print("1. Load the airline passenger dataset")
    print(df.head())
    # Part 2
    print("2. Create a new column showing the difference in passengers from the same month in the previous year")
    airline_df = df
    airline_df['Passengers_PrevYear'] = airline_df['Passengers'].shift(12)
    airline_df['YoY_Difference'] = airline_df['Passengers'] - airline_df['Passengers_PrevYear']
    airline_df['YoY_Pct_Change'] = (airline_df['YoY_Difference'] / airline_df['Passengers_PrevYear'] * 100)
    print("\nDataset with year-over-year differences:")
    print(airline_df.tail(15))
    # Part 3
    print("3. Identify the month with the largest year-over-year increase")
    valid_data = airline_df.dropna()
    max_increase_month = valid_data['YoY_Difference'].idxmax()
    max_increase_value = valid_data.loc[max_increase_month, 'YoY_Difference']
    max_increase_pct = valid_data.loc[max_increase_month, 'YoY_Pct_Change']
    print(f"\nMonth with largest year-over-year increase: {max_increase_month.strftime('%B %Y')}")
    print(f"Increase: {max_increase_value:.2f} thousand passengers ({max_increase_pct:.2f}%)")
    # Part 4
    print("4. Create a visualization showing the percentage change in passengers for each year")
    plt.figure(figsize=(14, 7))
    plt.plot(valid_data.index, valid_data['YoY_Pct_Change'], marker='o', linestyle='-', linewidth=2, markersize=4)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.fill_between(valid_data.index, valid_data['YoY_Pct_Change'], 0, 
                    where=(valid_data['YoY_Pct_Change'] >= 0), color='green', alpha=0.3, interpolate=True)
    plt.fill_between(valid_data.index, valid_data['YoY_Pct_Change'], 0, 
                    where=(valid_data['YoY_Pct_Change'] < 0), color='red', alpha=0.3, interpolate=True)
    plt.title('Year-over-Year Percentage Change in Airline Passengers (1950-1960)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Percentage Change (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("5. Write a brief analysis of the patterns you observe in the data")
    # Clear seasonal pattern with peaks during the summmer months
    # Lowest passenger numbers are typically in Februrary
    # 1949 - 1960 -> 300% increase in passengers
    # the compound growth rate is appromixately 12-13%
    # growth accelerated in the mid 1950s, possibly reflecting a post-war economic recovery
    # introduction of commercial jet aircraft in the late 1950s
    # rising middle class with disposable income for leisure travel
    # increasing globalization of business
    # // modern implications
    # shows the early growth phase of commercial aviation
    # we see the same seasonal patterns today but with much higher volumes


# Main demonstration
if __name__ == "__main__":
    # Load the dataset
    airline_df = load_airline_passengers()
    
    if airline_df is not None:
        # Demonstrate basic exploration
        # explore_time_series(airline_df)
        
        # Identify time series components
        # identify_components(airline_df)
        
        # Run student exercise
        # student_exercise(airline_df)
        
        # Show homework template
        homework_template(airline_df)