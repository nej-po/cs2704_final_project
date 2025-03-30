import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

crop_filename = 'crops-1980-2025.csv'
pop_filename  = 'population-1980-2025.csv'
crop_columns  = ['Year', 'Type', 'Value']
pop_columns   = ['Date', 'Location', 'Value']
pop_df  = pd.read_csv(pop_filename, names=pop_columns, skiprows=1)
crop_df = pd.read_csv(crop_filename, names=crop_columns, skiprows=1)

# Convert dates to datetimes so we can compare them
pop_df['Year'] = pd.to_datetime(pop_df['Date']).dt.year
crop_df['Year'] = pd.to_datetime(crop_df['Year'], format='%Y').dt.year

# Group by 'Canada', we don't need the provinces.
# Just get the results taken in January, otherwise we'd have 4 measurements per year
canada_pop = pop_df[(pop_df['Location'] == 'Canada') & (pop_df['Date'].str.endswith('-01'))]

# Get the overall yield per year
yearly_yield = crop_df.groupby('Year')['Value'].sum().reset_index()

merged = pd.merge(canada_pop, yearly_yield, on='Year')
merged.rename(columns={'Value_x':'Population'}, inplace=True)
merged.rename(columns={'Value_y':'CropYield'}, inplace=True)

# Get the X and Y values
x = merged[['Population']]
y = merged['CropYield']

# Run the linear regression, get the p-value
x_sm = sm.add_constant(x)
model = sm.OLS(y, x_sm)
results = model.fit()
print(results.summary())

# Plot the linear regression
y_pred = results.predict(x_sm)
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x, y_pred, color='red', label='Regression Line')
plt.title('Crop Yield vs Population')
plt.xlabel('Population')
plt.ylabel('Crop Yield')
plt.legend()
plt.show()

