import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

def create_sequences(data, lookback=30, forecast_horizon=7):
    X, y = [], []
    for i in range(lookback, len(data) - forecast_horizon + 1):
        X.append(data[i-lookback:i])
        y.append(data[i:i+forecast_horizon])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(7)  # 7 days forecast
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def forecast_sku_demand(sales_data, sku_id):
    # Filter data for specific SKU
    sku_data = sales_data[sales_data['sku_id'] == sku_id].copy()
    sku_data = sku_data.sort_values('date')
    
    # Extract sales values
    sales = sku_data['sales'].values.reshape(-1, 1)
    
    # Normalize data
    scaler = MinMaxScaler()
    sales_scaled = scaler.fit_transform(sales)
    
    # Create sequences
    X, y = create_sequences(sales_scaled.flatten(), lookback=30, forecast_horizon=7)
    
    if len(X) == 0:
        return None, None, None
    
    # Split train/test (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Reshape for LSTM input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Build and train model
    model = build_lstm_model((30, 1))
    
    # Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    # Train model
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=0
    )
    
    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    
    # Inverse transform predictions and actuals
    y_pred_original = scaler.inverse_transform(y_pred)
    y_test_original = scaler.inverse_transform(y_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_original.flatten(), y_pred_original.flatten()))
    mape = mean_absolute_percentage_error(y_test_original.flatten(), y_pred_original.flatten()) * 100
    
    # Create results dataframe
    results = []
    for i in range(len(y_test_original)):
        for day in range(7):
            results.append({
                'sku_id': sku_id,
                'forecast_day': day + 1,
                'actual': max(0, int(y_test_original[i][day])),
                'predicted': max(0, int(y_pred_original[i][day])),
                'sequence_id': i
            })
    
    results_df = pd.DataFrame(results)
    
    return results_df, rmse, mape

def run_lstm_forecast(data):
    all_results = []
    metrics = []
    
    unique_skus = data['sku_id'].unique()
    
    for sku in unique_skus[:5]:  # Limit to first 5 SKUs for demo
        try:
            results_df, rmse, mape = forecast_sku_demand(data, sku)
            if results_df is not None:
                all_results.append(results_df)
                metrics.append({
                    'sku_id': sku,
                    'RMSE': round(rmse, 2),
                    'MAPE': round(mape, 2)
                })
        except Exception as e:
            print(f"Error processing SKU {sku}: {e}")
            continue
    
    # Combine all results
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        metrics_df = pd.DataFrame(metrics)
        return final_results, metrics_df
    else:
        return None, None

# Generate sample data
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    skus = ['SKU001', 'SKU002', 'SKU003', 'SKU004', 'SKU005']

    sample_data = []
    for sku in skus:
        base_demand = np.random.randint(50, 200)
        trend = np.random.uniform(-0.5, 0.5)
        seasonality = np.sin(np.arange(100) * 2 * np.pi / 7) * 20  # Weekly seasonality
        noise = np.random.normal(0, 10, 100)
        
        sales = base_demand + trend * np.arange(100) + seasonality + noise
        sales = np.maximum(sales, 0)  # Ensure non-negative
        
        for i, date in enumerate(dates):
            sample_data.append({
                'date': date,
                'sku_id': sku,
                'sales': int(sales[i])
            })

    sample_df = pd.DataFrame(sample_data)

    # Run LSTM forecast
    results, metrics = run_lstm_forecast(sample_df)

    if results is not None:
        print("LSTM Forecast Results:")
        print(results.head(20))
        print("\nModel Performance Metrics:")
        print(metrics)
        
        # Summary statistics
        print(f"\nOverall RMSE: {metrics['RMSE'].mean():.2f}")
        print(f"Overall MAPE: {metrics['MAPE'].mean():.2f}%")
    else:
        print("No results generated")
