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

# Sample homework assignment
def homework_template():
    """
    Template code for homework assignment
    """
    print("HOMEWORK ASSIGNMENT")
    print("==================")
    print("1. Load the airline passenger dataset")
    print("2. Create a new column showing the difference in passengers from the same month in the previous year")
    print("3. Identify the month with the largest year-over-year increase")
    print("4. Create a visualization showing the percentage change in passengers for each year")
    print("5. Write a brief analysis of the patterns you observe in the data")

# Main demonstration
if __name__ == "__main__":
    # Load the dataset
    airline_df = load_airline_passengers()
    
    if airline_df is not None:
        # Demonstrate basic exploration
        explore_time_series(airline_df)
        
        # Identify time series components
        # identify_components(airline_df)
        
        # Run student exercise
        # student_exercise(airline_df)
        
        # Show homework template
        # homework_template()