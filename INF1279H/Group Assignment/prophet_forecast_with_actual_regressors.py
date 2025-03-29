
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
import matplotlib.pyplot as plt

# --- Load and Prepare Data ---
df = pd.read_csv('your_data.csv')  # Replace with your actual file
df['ds'] = pd.to_datetime(df['ACC_MONTH'])
df['y'] = df['TOT_PAID']
df = df.sort_values('ds').reset_index(drop=True)

# Step 1: Create 12-month lagged features for selected regressors
lag = 12
regressors = ['EST_COST', 'NBR_OTHER']
for reg in regressors:
    df[f'{reg}_lag{lag}'] = df[reg].shift(lag)

# Drop NA caused by lag
df_model = df.dropna().copy().reset_index(drop=True)

# Step 2: Prepare data for Prophet
train = df_model[:-18]
test = df_model[-18:]

# Step 3: Fit Prophet model using lagged regressors
model = Prophet()
for reg in regressors:
    model.add_regressor(f'{reg}_lag{lag}')
model.fit(train[['ds', 'y'] + [f'{reg}_lag{lag}' for reg in regressors]])

# Step 4: Predict on test set
forecast_test = model.predict(test[['ds'] + [f'{reg}_lag{lag}' for reg in regressors]])
rmse = np.sqrt(mean_squared_error(test['y'], forecast_test['yhat']))
mae = mean_absolute_error(test['y'], forecast_test['yhat'])
nonzero_mask = test['y'] != 0
mape = np.mean(np.abs((test['y'][nonzero_mask] - forecast_test['yhat'][nonzero_mask]) / test['y'][nonzero_mask])) * 100

print("\n Historical Forecast Accuracy (last 18 months):")
print(f"RMSE: {rmse:,.2f}")
print(f"MAE: {mae:,.2f}")
print(f"MAPE: {mape:.2f}%")

# Step 5: Forecast future regressors (only 6 months needed)
future_regressors = {}
forecast_horizon = 6
for reg in regressors:
    df_reg = df_model[['ds', reg]].rename(columns={reg: 'y'})
    reg_model = Prophet()
    reg_model.fit(df_reg)
    future_dates = reg_model.make_future_dataframe(periods=lag + forecast_horizon, freq='MS')
    reg_forecast = reg_model.predict(future_dates)
    future_regressors[reg] = reg_forecast[['ds', 'yhat']].rename(columns={'yhat': f'{reg}_forecast'})

# Step 6: Combine regressor forecasts
future_df = future_regressors[regressors[0]].copy()
for reg in regressors[1:]:
    future_df = future_df.merge(future_regressors[reg], on='ds')

# Step 7: Apply lag 12 to forecasted features
for reg in regressors:
    future_df[f'{reg}_lag{lag}'] = future_df[f'{reg}_forecast'].shift(lag)
future_df = future_df.dropna().reset_index(drop=True)

# Step 8: Forecast should start right after last actual data
last_actual_date = df_model['ds'].max()
future_input = future_df[future_df['ds'] > last_actual_date][['ds'] + [f'{reg}_lag{lag}' for reg in regressors]].head(18)

# Step 9: Forecast TOT_PAID for next 18 months
future_forecast = model.predict(future_input)

# Step 10: Predictions on train set for plotting
forecast_train = model.predict(train[['ds'] + [f'{reg}_lag{lag}' for reg in regressors]])
forecast_all = pd.concat([forecast_train[['ds', 'yhat']], forecast_test[['ds', 'yhat']]], ignore_index=True)
forecast_all = forecast_all.sort_values('ds').reset_index(drop=True)

# Step 11: Plotting
plt.figure(figsize=(14, 7))
plt.plot(df_model['ds'], df_model['y'], label='Actual TOT_PAID', color='black')
plt.plot(forecast_all['ds'], forecast_all['yhat'], label='Model Prediction (Train + Test)', color='green', linestyle='--')
plt.plot(future_forecast['ds'], future_forecast['yhat'], label='Forecast (Next 18 Months)', color='blue', linestyle='--')
plt.fill_between(future_forecast['ds'], future_forecast['yhat_lower'], future_forecast['yhat_upper'], color='blue', alpha=0.2)
plt.axvline(x=test['ds'].min(), color='red', linestyle='dashed', label='Test Period Start')
plt.axvline(x=future_forecast['ds'].min(), color='purple', linestyle='dashed', label='Future Forecast Start')
plt.title("TOT_PAID - Actuals, Model Predictions (Train+Test), and 18-Month Forecast")
plt.xlabel("Date")
plt.ylabel("TOT_PAID")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 12: Print forecasted TOT_PAID
forecast_columns = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
print("\n📅 Forecasted TOT_PAID (Next 18 Months):")
print(future_forecast[forecast_columns].to_string(index=False))
