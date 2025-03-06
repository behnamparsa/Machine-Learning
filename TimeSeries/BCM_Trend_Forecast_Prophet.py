#Prophet model for time series forecast
from prophet import Prophet



#Data processing
import numpy as np
import pandas as pd

#visualization
import seaborn as sns
import matplotlib.pyplot as plt

#Model performance evaluation
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score


bcm = pd.read_csv("C:\GitHub\Machine-Learning\data\BCM_DATA.csv",index_col=[0],parse_dates=[0])


#color_pal = ['blue', 'red']  # Define a color palette with distinct colors


#bcm[['TOT_PAID','BUDGET_AMT']].plot(style='-',
#          figsize=(10,5),
#          ms=1,
#          color=color_pal,
#          title='ACT_EXP')

#plt.show()

# Date for splitting training and testing dataset
train_end_date = '2024-03-01'

bcm_train = bcm.loc[bcm.index <= train_end_date].copy()
bcm_test = bcm.loc[bcm.index > train_end_date].copy()

print(bcm_train.tail())

# Format data for prophet model using ds and y
bcm_train_prophet = bcm_train["TOT_PAID"].reset_index().rename(columns={'ACC_MONTH':'ds','TOT_PAID':'y'})
bcm_test_prophet = bcm_test["TOT_PAID"].reset_index().rename(columns={'ACC_MONTH':'ds','TOT_PAID':'y'})
#print(bcm_train_prophet.head())

#model fit:
model = Prophet()
model.fit(bcm_train_prophet)

# Predict on test set with model

bcm_test_fcst = model.predict(bcm_test_prophet)
bcm_train_fcst= model.predict(bcm_train_prophet) # fit the model on training dataset
print(bcm_test_fcst.head())
print(bcm_train_fcst.head())


fig, ax = plt.subplots(figsize=(10, 5))
fig = model.plot(bcm_test_fcst, ax=ax)
ax.set_title('Prophet Forecast')


fig = model.plot_components(bcm_test_fcst)


#compare forecast to actuals:

# Plot the forecast with the actuals
f, ax = plt.subplots(figsize=(15, 5))
ax.plot(bcm_test.index, bcm_test['TOT_PAID'], color='r')
fig = model.plot(bcm_test_fcst, ax=ax)



#plot one fiscal year

lower_bound = pd.to_datetime('2024-04-01')  # Ensure correct format
upper_bound = pd.to_datetime('2025-04-01')

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(bcm_test.index, bcm_test['TOT_PAID'], color='r')
fig = model.plot(bcm_test_fcst, ax=ax)
# Set x-axis limits with datetime values
ax.set_xbound(lower=lower_bound, upper=upper_bound)
#ax.set_ylim(0, 60000)
plot = plt.suptitle('Forecast vs Actuals - fiscal year 2024-25')
#plt.show()


#plot the forecast with the actual ternd

# Plot the forecast with the actuals

lower_bound = pd.to_datetime('2015-04-01')  # Ensure correct format
upper_bound = pd.to_datetime('2024-03-01')

f, ax = plt.subplots(figsize=(15, 5))
ax.plot(bcm_train.index, bcm_train['TOT_PAID'], color='r')
fig = model.plot(bcm_train_fcst, ax=ax)
ax.set_xbound(lower=lower_bound, upper=upper_bound)
# Add legend
ax.legend(['Actual TOT_PAID', 'Forecasted TOT_PAID'])
ax.set_title('Forecast vs Actuals - simulation')
#plt.show()


#check the model's accuracy

mean_squared_error_b = np.sqrt(mean_squared_error(y_true=bcm_test['TOT_PAID'],
                   y_pred=bcm_test_fcst['yhat']))

mean_absolute_error=mean_absolute_error(y_true=bcm_test['TOT_PAID'],
                   y_pred=bcm_test_fcst['yhat'])
mean_absolute_percentage_error=mean_absolute_percentage_error(y_true=bcm_test['TOT_PAID'],
                   y_pred=bcm_test_fcst['yhat'])



print("mean_squared_error:", mean_squared_error_b)
print("mean_absolute_error:", mean_absolute_error)
print("mean_absolute_percentage_error:", mean_absolute_percentage_error)



# Create the time range for the forecast
future_baseline = model.make_future_dataframe(periods=18,freq='M',include_history=True)
print(future_baseline.tail())

# Make prediction
forecast_baseline = model.predict(future_baseline)
print(forecast_baseline.tail(24))


# Visualize the forecast
model.plot(forecast_baseline); # Add semi-colon to remove the duplicated chart

# Visualize the forecast components
model.plot_components(forecast_baseline)

#plt.show()
#check the model's accuracy




##################################################################################################################
# Add seasonality
model_season = Prophet(yearly_seasonality=True)

# Fit the model on the training dataset
model_season.fit(bcm_train_prophet)
bcm_test_fcst_season = model_season.predict(bcm_test_prophet)

print(bcm_test_fcst_season.head())
print(bcm_test_fcst.head())
print()



#check the model's accuracy

mean_squared_error_season = np.sqrt(mean_squared_error(y_true=bcm_test['TOT_PAID'],y_pred=bcm_test_fcst_season['yhat']))

print("mean_squared_error:", mean_squared_error_b)
print("mean_squared_error_season:", mean_squared_error_season)


####################################################################################################################

# Format data for prophet model using ds and y & regressor
bcm_train_prophet = bcm_train[["TOT_PAID","BUDGET_AMT","NBR_MURDER"]].reset_index().rename(columns={'ACC_MONTH':'ds'
                                                                                                  ,'TOT_PAID':'y'
                                                                                                  ,'BUDGET_AMT':'bud'
                                                                                                  ,'NBR_MURDER':'case'})
bcm_test_prophet  =  bcm_test[["TOT_PAID","BUDGET_AMT","NBR_MURDER"]].reset_index().rename(columns={'ACC_MONTH':'ds'
                                                                                       ,'TOT_PAID':'y'
                                                                                       ,'BUDGET_AMT':'bud'
                                                                                       ,'NBR_MURDER':'case'})

# Add seasonality 
model_multivariate = Prophet(yearly_seasonality=True)


# Add regressor
model_multivariate.add_regressor('bud', standardize=False)
model_multivariate.add_regressor('case', standardize=False)

# Fit the model on the training dataset
model_multivariate.fit(bcm_train_prophet)
bcm_multi_prdct = model_multivariate.predict(bcm_test_prophet)

#check the model's accuracy

mean_squared_error_multi = np.sqrt(mean_squared_error(y_true=bcm_test['TOT_PAID'],y_pred=bcm_multi_prdct['yhat']))

print("mean_squared_error:", mean_squared_error_b)
print("mean_squared_error_season:", mean_squared_error_season)
print("mean_squared_error_multi:", mean_squared_error_multi)

print(bcm_train.columns)

#https://www.kaggle.com/code/pythonafroz/fb-prophet-with-multiple-regressors