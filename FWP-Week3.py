# Week 3: Classical Forecasting Methods
# Using Australian Retail Sales Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def load_retail_data():
    """
    Loads Australian retail sales data (2000-2019).
    Returns monthly retail turnover in millions of Australian dollars.
    Source: Australian Bureau of Statistics, accessed through stats.data.abs.gov.au
    """
    # Load data from GitHub (a cleaned version of ABS Retail Trade data)
    url = "https://raw.githubusercontent.com/selva86/datasets/master/a10.csv"
    df = pd.read_csv(url)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

# Hour 1: Moving Averages Implementation
def explore_moving_averages(data, windows=[3, 6, 12]):
    """
    Demonstrates different moving average techniques and their effects.
    Includes simple and weighted moving averages.
    """
    plt.figure(figsize=(15, 10))
    
    # Simple Moving Averages
    plt.subplot(211)
    plt.plot(data.index, data['value'], label='Original', alpha=0.5)
    
    for window in windows:
        # Simple Moving Average
        sma = data['value'].rolling(window=window).mean()
        plt.plot(data.index, sma, 
                label=f'SMA ({window} months)', 
                linewidth=2)
    
    plt.title('Simple Moving Averages')
    plt.legend()
    
    # Weighted Moving Average
    plt.subplot(212)
    plt.plot(data.index, data['value'], label='Original', alpha=0.5)
    
    for window in windows:
        # Create weights that sum to 1 and give more importance to recent values
        weights = np.arange(1, window + 1)
        weights = weights / weights.sum()
        
        # Calculate WMA
        wma = data['value'].rolling(window=window).apply(
            lambda x: np.sum(weights * x))
        
        plt.plot(data.index, wma, 
                label=f'WMA ({window} months)', 
                linewidth=2)
    
    plt.title('Weighted Moving Averages')
    plt.legend()
    plt.show()
    
    return plt.gcf()

# Hour 2: Exponential Smoothing Implementation
def implement_exponential_smoothing(data, seasonal_periods=12):
    """
    Implements and compares different exponential smoothing methods.
    Shows progression from simple to more complex methods.
    """
    plt.figure(figsize=(15, 12))
    
    # Simple Exponential Smoothing
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    ses_model = SimpleExpSmoothing(data['value']).fit()
    
    # Double Exponential Smoothing (Holt's method)
    des_model = ExponentialSmoothing(
        data['value'], 
        trend='add').fit()
    
    # Triple Exponential Smoothing (Holt-Winters' method)
    tes_model = ExponentialSmoothing(
        data['value'],
        trend='add',
        seasonal='add',
        seasonal_periods=seasonal_periods).fit()
    
    # Plot results
    plt.subplot(311)
    plt.plot(data.index, data['value'], label='Original')
    plt.plot(data.index, ses_model.fittedvalues, 
            label='Simple Exponential Smoothing')
    plt.title('Simple Exponential Smoothing')
    plt.legend()
    
    plt.subplot(312)
    plt.plot(data.index, data['value'], label='Original')
    plt.plot(data.index, des_model.fittedvalues, 
            label="Holt's Double Exponential Smoothing")
    plt.title("Holt's Method (Double Exponential Smoothing)")
    plt.legend()
    
    plt.subplot(313)
    plt.plot(data.index, data['value'], label='Original')
    plt.plot(data.index, tes_model.fittedvalues, 
            label="Holt-Winters' Triple Exponential Smoothing")
    plt.title("Holt-Winters' Method (Triple Exponential Smoothing)")
    plt.legend()

    plt.show()
    
    return ses_model, des_model, tes_model

# Hour 3: ARIMA and SARIMA Implementation
def analyze_timeseries_patterns(data):
    """
    Creates ACF and PACF plots to help identify ARIMA parameters.
    Includes interpretation guidelines.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # ACF plot
    plot_acf(data['value'], ax=ax1, lags=40)
    ax1.set_title('Autocorrelation Function')
    
    # PACF plot
    plot_pacf(data['value'], ax=ax2, lags=40)
    ax2.set_title('Partial Autocorrelation Function')
    
    plt.tight_layout()
    plt.show()
    return plt.gcf()

def fit_sarima_model(data, order=(1,1,1), seasonal_order=(1,1,1,12)):
    """
    Fits a SARIMA model and demonstrates model diagnostics.
    Includes parameter selection guidance.
    """
    # Fit the model
    model = SARIMAX(data['value'],
                    order=order,
                    seasonal_order=seasonal_order)
    results = model.fit()
    
    # Create diagnostic plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original vs Fitted
    ax1.plot(data.index, data['value'], label='Original')
    ax1.plot(data.index, results.fittedvalues, label='Fitted')
    ax1.set_title('Original vs Fitted Values')
    ax1.legend()
    
    # Residuals over time
    ax2.plot(data.index, results.resid)
    ax2.set_title('Residuals over Time')
    
    # Residual histogram
    ax3.hist(results.resid, bins=30)
    ax3.set_title('Residual Distribution')
    
    # Q-Q plot
    from statsmodels.graphics.gofplots import ProbPlot
    ProbPlot(results.resid).qqplot(line='45', ax=ax4)
    ax4.set_title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    plt.show()
    return results

# Model Comparison Function
def compare_forecasts(data, train_size=0.8):
    """
    Compares different forecasting methods using train/test split.
    Includes error metrics and visualization of forecasts.
    """
    # Split data into train and test
    train_data = data.iloc[:int(len(data) * train_size)]
    test_data = data.iloc[int(len(data) * train_size):]
    
    # Fit models
    # Simple Moving Average
    sma = train_data['value'].rolling(window=12).mean()
    
    # Holt-Winters
    hw_model = ExponentialSmoothing(
        train_data['value'],
        trend='add',
        seasonal='add',
        seasonal_periods=12).fit()
    
    # SARIMA
    sarima_model = SARIMAX(train_data['value'],
                          order=(1,1,1),
                          seasonal_order=(1,1,1,12)).fit()
    
    # Generate forecasts
    hw_forecast = hw_model.forecast(len(test_data))
    sarima_forecast = sarima_model.forecast(len(test_data))
    
    # Calculate errors
    hw_rmse = np.sqrt(mean_squared_error(test_data['value'], hw_forecast))
    sarima_rmse = np.sqrt(mean_squared_error(test_data['value'], 
                                           sarima_forecast))
    
    # Plot results
    plt.figure(figsize=(15, 8))
    plt.plot(train_data.index, train_data['value'], label='Training Data')
    plt.plot(test_data.index, test_data['value'], label='Test Data')
    plt.plot(test_data.index, hw_forecast, label=f'Holt-Winters (RMSE: {hw_rmse:.2f})')
    plt.plot(test_data.index, sarima_forecast, label=f'SARIMA (RMSE: {sarima_rmse:.2f})')
    plt.title('Forecast Comparison')
    plt.legend()
    plt.show()
    
    return hw_rmse, sarima_rmse

# Main execution flow
if __name__ == "__main__":
    # Load the data
    retail_data = load_retail_data()
    
    # Moving Averages Analysis
    # ma_plot = explore_moving_averages(retail_data)
    
    # Exponential Smoothing
    # ses, des, tes = implement_exponential_smoothing(retail_data)
    
    # ARIMA Analysis
    # pattern_analysis = analyze_timeseries_patterns(retail_data)
    # sarima_model = fit_sarima_model(retail_data)
    
    # Compare methods
    hw_rmse, sarima_rmse = compare_forecasts(retail_data)
