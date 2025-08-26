import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')

def forecast_demand(data):
    # Prepare data for Prophet
    df = data.copy()
    df['ds'] = pd.to_datetime(df['date'])
    df['y'] = df['sales']

    # Create holiday dataframe
    holidays = pd.DataFrame([
        {'holiday': 'Diwali', 'ds': '2024-10-28', 'lower_window': -7, 'upper_window': 7},
        {'holiday': 'Diwali', 'ds': '2024-10-29', 'lower_window': -7, 'upper_window': 7},
        {'holiday': 'Holi', 'ds': '2024-03-15', 'lower_window': -3, 'upper_window': 3},
        {'holiday': 'Christmas', 'ds': '2024-12-25', 'lower_window': -5, 'upper_window': 5},
        {'holiday': 'New_Year', 'ds': '2024-12-31', 'lower_window': -2, 'upper_window': 2},
        {'holiday': 'New_Year', 'ds': '2025-01-01', 'lower_window': -2, 'upper_window': 2},
        {'holiday': 'Valentine', 'ds': '2024-02-14', 'lower_window': -1, 'upper_window': 1},
        {'holiday': 'Republic_Day', 'ds': '2024-01-26', 'lower_window': -1, 'upper_window': 1}
    ])
    holidays['ds'] = pd.to_datetime(holidays['ds'])

    # Initialize Prophet model
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        holidays=holidays,
        seasonality_mode='multiplicative',
        interval_width=0.8
    )

    # Add regressors
    model.add_regressor('discount')
    df['event_indicator'] = df['event'].apply(lambda x: 1 if x != 'normal' else 0)
    model.add_regressor('event_indicator')

    # Fit the model
    model.fit(df[['ds', 'y', 'discount', 'event_indicator']])

    # Create future dataframe for 14 days
    last_date = df['ds'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=14, freq='D')

    future = pd.DataFrame({
        'ds': future_dates,
        'discount': 0,           # Assume no discount for future
        'event_indicator': 0     # Assume normal days
    })

    # Make predictions
    forecast = model.predict(future)

    # Return forecast with required columns
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    result.columns = ['date', 'forecast', 'lower_bound', 'upper_bound']
    result['forecast'] = result['forecast'].round(0).astype(int)
    result['lower_bound'] = result['lower_bound'].round(0).astype(int)
    result['upper_bound'] = result['upper_bound'].round(0).astype(int)

    # Ensure non-negative forecasts
    result['forecast'] = np.maximum(result['forecast'], 0)
    result['lower_bound'] = np.maximum(result['lower_bound'], 0)
    result['upper_bound'] = np.maximum(result['upper_bound'], 0)

    return result

# Example usage (for testing)
if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    sample_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'sales': np.random.randint(50, 300, 100),
        'discount': np.random.choice([0, 10, 20, 25], 100),
        'event': np.random.choice(['normal', 'festival', 'promotion'], 100)
    })

    forecast_result = forecast_demand(sample_data)
    print(forecast_result)
