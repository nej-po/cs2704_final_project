import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import numpy as np

filename = 'homelessness_2019-2023.csv'
column_names = ['Date', 'Type', 'Target', 'Statistics', 'Value']
data = pd.read_csv(filename, names=column_names, skiprows=1)  # Skip the header row

# Convert 'Date' to numeric and 'Value' to numeric
data['Date'] = pd.to_numeric(data['Date'])
data['Value'] = pd.to_numeric(data['Value'])

# Group by 'Date' and sum 'Value'
yearly_totals = data.groupby('Date')['Value'].sum().reset_index()

# Plot the yearly totals
plt.figure(figsize=(10, 6))
plt.plot(yearly_totals['Date'], yearly_totals['Value'], marker='o')
plt.title('Total Homelessness per Year')
plt.xlabel('Year')
plt.ylabel('Total Homelessness')
plt.grid(True)
plt.show()

# Correlation analysis (simple linear correlation)
correlation, p_value = stats.pearsonr(yearly_totals['Date'], yearly_totals['Value'])
print(f"P-value: {p_value}")