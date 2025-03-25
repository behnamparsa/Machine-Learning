
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Use the dataset provided by Power BI
# Dataset: dataset

# Convert 'ds' column to datetime
dataset['ds'] = pd.to_datetime(dataset['ds'])

# Filter only records for last run and fiscal years 2024, 2025
filtered_data = dataset[(dataset['The last run'] == 1) & (dataset['Fiscal Year'].isin([2024, 2025]))]

# Set up the plot size and style
plt.figure(figsize=(14, 6))
sns.set(style="whitegrid")

# 1. Line Chart: Prediction vs Actual TOT_PAID with confidence interval
plt.plot(filtered_data['ds'], filtered_data['Prediction_TOT_PAID'], label='Predicted TOT_PAID', color='blue')
plt.fill_between(filtered_data['ds'],
                 filtered_data['Prediction_TOT_PAID_lower'],
                 filtered_data['Prediction_TOT_PAID_upper'],
                 color='blue', alpha=0.2, label='Prediction Range')
plt.plot(filtered_data['ds'], filtered_data['Actual_TOT_PAID'], label='Actual TOT_PAID', color='green', linestyle='--')

plt.title('Prediction vs Actual TOT_PAID (FY 2024 & 2025)')
plt.xlabel('Date')
plt.ylabel('Amount ($)')
plt.legend()
plt.tight_layout()
plt.show()

# 2. KPI-style summary statistics (can be exported to table visual in Power BI)
summary = filtered_data.groupby('Fiscal Year').agg({
    'Prediction_TOT_PAID': 'sum',
    'Actual_TOT_PAID': 'sum'
})
summary['Prediction_Error'] = summary['Actual_TOT_PAID'] - summary['Prediction_TOT_PAID']
print(summary)

# 3. Bar Chart Comparison
summary.plot(kind='bar', figsize=(10, 6))
plt.title('Total Predicted vs Actual TOT_PAID by Fiscal Year')
plt.ylabel('Amount ($)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 4. Forecast vs Actual Budget
plt.figure(figsize=(10, 6))
sns.lineplot(data=filtered_data, x='ds', y='Forecast_BUDGET_AMT', label='Forecast Budget')
sns.lineplot(data=filtered_data, x='ds', y='Actual_BUDGET_AMT', label='Actual Budget')
plt.title('Forecast vs Actual Budget Over Time')
plt.xlabel('Date')
plt.ylabel('Amount ($)')
plt.legend()
plt.tight_layout()
plt.show()

# 5. Murder Rates Over Time
plt.figure(figsize=(10, 6))
sns.lineplot(data=filtered_data, x='ds', y='Forecast_NBR_MURDER', label='Forecast Murders', color='red')
sns.lineplot(data=filtered_data, x='ds', y='Actual_NBR_MURDER', label='Actual Murders', color='orange')
plt.title('Forecast vs Actual Number of Murders')
plt.xlabel('Date')
plt.ylabel('Number of Murders')
plt.legend()
plt.tight_layout()
plt.show()
