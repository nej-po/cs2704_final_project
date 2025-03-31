import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

pop_df  = pd.read_csv('population-1980-2025.csv', names=['Date', 'Location', 'Value'], skiprows=1)
crop_df = pd.read_csv('crops-1980-2025.csv', names=['Year', 'Type', 'Value'], skiprows=1)

# Convert date strings to datetimes so we can compare them
pop_df['Year'] = pd.to_datetime(pop_df['Date']).dt.year
crop_df['Year'] = pd.to_datetime(crop_df['Year'], format='%Y').dt.year

# Group by 'Canada'
# Just get the results taken in January, otherwise we'd have 4 measurements per year
canada_pop = pop_df[(pop_df['Location'] == 'Canada') & (pop_df['Date'].str.endswith('-01'))]


###
# Perform a linear regression
# Display the regression line along with a scatterplot of the indepedent varible.
###
def execute_linear_regression(x, y):
  # Run the linear regression, get the p-value
    x_sm = sm.add_constant(x)
    model = sm.OLS(y, x_sm)
    results = model.fit()
    y_pred = results.predict(x_sm)
    results.x_sm = x_sm
    results.y_pred = y_pred
    return results


###
# Graph a linear regression result set, accepts labels and a title.
###
def graph_linear_regression(x, y, results, title, x_label, y_label):
    # Plot the linear regression
    plt.scatter(x, y, color='blue', label='Data')
    plt.plot(x, results.y_pred, color='red', label='Regression Line')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


###
# Execute a linear regression test for crops nationwide
###
def execute_linear_regression_totals():
    canada_pop = pop_df[(pop_df['Location'] == 'Canada') & (pop_df['Date'].str.endswith('-01'))]

    # Get the overall seeded acres per year
    seeded_acres = crop_df.groupby('Year')['Value'].sum().reset_index()

    merged = pd.merge(canada_pop, seeded_acres, on='Year')
    merged.rename(columns={'Value_x':'Population'}, inplace=True)
    merged.rename(columns={'Value_y':'SeededAcres'}, inplace=True)

    # Get the X and Y values
    x = merged[['Population']]
    y = merged['SeededAcres']
    results = execute_linear_regression(x,y)
    graph_linear_regression(x,y,results, 'Crop Seeded Acres vs Population', 'Population', 'Crop Seeded Acres')


###
# Execute a linear regression test for each individual crop type
###
def execute_linear_regression_individual():   
    crop_types = crop_df.groupby(['Type', 'Year'])[['Value']].sum().reset_index()
    
    # Kinda violating the "don't repeat yourself" principle here, but we'll excuse it this one time... 
    for t in crop_types['Type'].unique():
        filtered_crop = crop_types[crop_types['Type'] == t]
        merged = pd.merge(canada_pop, filtered_crop, on='Year')
        merged.rename(columns={'Value_x':'Population'}, inplace=True)
        merged.rename(columns={'Value_y':'SeededAcres'}, inplace=True)
        x = merged[['Population']]
        y = merged['SeededAcres']
        results = execute_linear_regression(x,y)
        print_summary_linear_regression_result(t, results)
        continue   


###
# Print the stats for a linear regression analysis
###
def print_summary_linear_regression_result(crop_type, results):
    print(f'Crop type [{crop_type}]:')
    print(f'\tP-value: {results.pvalues[f'Population']:.6f}')
    print(f'\tRegresson coefficient: {results.params[f'Population']:.6f}')


###
# Does exactly what it says - print the console menu.
###
def show_menu():
    print('1: Linear Regression Totals')
    print('2: Linear Regression Comparison')
    print('3: Exit')


###
# Main
###
if __name__ == '__main__':
    selection = 0
    while int(selection) != 3:
        show_menu()
        try:
            selection = int(input('::'))
        except:
            break 
        if (selection == 1):
            execute_linear_regression_totals()
        elif (selection == 2):
            execute_linear_regression_individual()
        else:
            exit