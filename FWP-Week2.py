# Week 2: Exploratory Data Analysis and Visualization
# Using Airline Passengers Dataset (1949-1960)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
import plotly.express as px
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the classic airline passengers dataset
def load_airline_data():
    """
    Loads the classic airline passengers dataset (1949-1960).
    This dataset shows monthly totals of international airline passengers
    and is perfect for demonstrating seasonal patterns and trend.
    """
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    df = pd.read_csv(url)
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
    df.columns = ['Passengers']
    return df

# Hour 1: Time Series Components Analysis
def decompose_series(df):
    """
    Performs and visualizes time series decomposition.
    Shows both additive and multiplicative decomposition for comparison.
    """
    # Create figure with two decomposition methods side by side
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    fig.suptitle('Time Series Decomposition: Additive vs Multiplicative', fontsize=16)
    
    # Perform both decompositions
    decomp_add = seasonal_decompose(df['Passengers'], period=12, model='additive')
    decomp_mult = seasonal_decompose(df['Passengers'], period=12, model='multiplicative')
    
    # Plot additive decomposition
    decomp_add.observed.plot(ax=axes[0,0], title='Observed (Additive)')
    decomp_add.trend.plot(ax=axes[1,0], title='Trend')
    decomp_add.seasonal.plot(ax=axes[2,0], title='Seasonal')
    decomp_add.resid.plot(ax=axes[3,0], title='Residual')
    
    # Plot multiplicative decomposition
    decomp_mult.observed.plot(ax=axes[0,1], title='Observed (Multiplicative)')
    decomp_mult.trend.plot(ax=axes[1,1], title='Trend')
    decomp_mult.seasonal.plot(ax=axes[2,1], title='Seasonal')
    decomp_mult.resid.plot(ax=axes[3,1], title='Residual')
    plt.tight_layout()
    plt.show()
    return decomp_add, decomp_mult

# Hour 2: Advanced Visualization Techniques
def create_visualization_dashboard(df):
    """
    Creates a comprehensive dashboard of time series visualizations.
    Demonstrates various ways to visualize temporal patterns.
    """
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 15))
    
    # 1. Basic Time Series Plot with Trend
    ax1 = plt.subplot(321)
    df['Passengers'].plot(ax=ax1)
    df['Passengers'].rolling(window=12).mean().plot(ax=ax1, color='red')
    ax1.set_title('Passenger Numbers with 12-month Moving Average')
    
    # 2. Seasonal Plot
    ax2 = plt.subplot(322)
    seasonal_plot = df.copy()
    seasonal_plot['Year'] = seasonal_plot.index.year
    seasonal_plot['Month'] = seasonal_plot.index.month
    pivot_table = seasonal_plot.pivot(index='Year', columns='Month', values='Passengers')
    sns.heatmap(pivot_table, cmap='YlOrRd', ax=ax2)
    ax2.set_title('Seasonal Heatmap')
    
    # 3. Box Plot by Month
    ax3 = plt.subplot(323)
    df['Month'] = df.index.month
    sns.boxplot(data=df, x='Month', y='Passengers', ax=ax3)
    ax3.set_title('Monthly Box Plot')
    
    # 4. Lag Plot
    ax4 = plt.subplot(324)
    pd.plotting.lag_plot(df['Passengers'], ax=ax4)
    ax4.set_title('Lag Plot (t vs t-1)')
    
    # 5. Autocorrelation Plot
    ax5 = plt.subplot(325)
    pd.plotting.autocorrelation_plot(df['Passengers'], ax=ax5)
    ax5.set_title('Autocorrelation Plot')
    
    # 6. Year-over-Year Comparison
    ax6 = plt.subplot(326)
    years = df.index.year.unique()
    for year in years[:3]:  # Plot first 3 years
        year_data = df[df.index.year == year]
        year_data['Month'] = year_data.index.month
        ax6.plot(year_data['Month'], year_data['Passengers'], label=str(year))
    ax6.legend()
    ax6.set_title('Year-over-Year Comparison')
    
    plt.tight_layout()
    plt.show()

# Hour 3: Stationarity Analysis and Transformations
def check_stationarity(timeseries):
    """
    Performs comprehensive stationarity analysis including:
    1. Visual tests
    2. Statistical tests (ADF and KPSS)
    3. Rolling statistics
    """
    # Create figure for visual analysis
    fig = plt.figure(figsize=(12, 8))
    
    # Plot rolling statistics
    ax1 = plt.subplot(211)
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()
    
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Statistics')
    plt.show()
    
    # Perform ADF test (Automated Dickey-Fuller)
    # https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test
    adf_test = adfuller(timeseries, autolag='AIC')
    
    # Perform KPSS test
    # https://en.wikipedia.org/wiki/KPSS_test
    kpss_test = kpss(timeseries, regression='ct', nlags="auto")
    
    # Print statistical test results
    print('\nStationarity Tests:')
    print('\nAugmented Dickey-Fuller Test:')
    print(f'ADF Statistic: {adf_test[0]:.4f}')
    print(f'p-value: {adf_test[1]:.4f}')
    
    print('\nKPSS Test:')
    print(f'KPSS Statistic: {kpss_test[0]:.4f}')
    print(f'p-value: {kpss_test[1]:.4f}')

def apply_transformations(df):
    """
    Demonstrates various transformation techniques:
    1. Log transformation
    2. First difference
    3. Seasonal difference
    4. Box-Cox transformation
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original data
    df['Passengers'].plot(ax=axes[0,0], title='Original Series')
    
    # Log transformation
    log_transform = np.log(df['Passengers'])
    log_transform.plot(ax=axes[0,1], title='Log Transformed')
    
    # First difference
    first_diff = df['Passengers'].diff()
    first_diff.plot(ax=axes[1,0], title='First Difference')
    
    # Seasonal difference
    seasonal_diff = df['Passengers'].diff(12)
    seasonal_diff.plot(ax=axes[1,1], title='Seasonal Difference (12 months)')
    
    plt.tight_layout()
    plt.show()
    
    # Box-Cox transformation
    # https://en.wikipedia.org/wiki/Box%E2%80%93Cox_distribution
    data_transformed, lambda_param = stats.boxcox(df['Passengers'])
    print(f'\nBox-Cox Transformation Lambda: {lambda_param:.4f}')
    
    return log_transform, first_diff, seasonal_diff, data_transformed

# Main execution for the notebook
if __name__ == "__main__":
    # Load the data
    airline_data = load_airline_data()
    
    # Perform decomposition
    # decomp_add, decomp_mult = decompose_series(airline_data)
    
    # Create visualization dashboard
    # create_visualization_dashboard(airline_data)
    
    # Check stationarity
    # check_stationarity(airline_data['Passengers'])
    
    # Apply transformations
    log_data, diff_data, seas_diff, boxcox_data = apply_transformations(airline_data)
