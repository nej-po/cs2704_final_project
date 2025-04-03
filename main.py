import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import sys

pop_df  = pd.read_csv('population-1980-2025.csv', names=['Date', 'Location', 'Population'], skiprows=1)
crop_df = pd.read_csv('crops-1980-2025.csv', names=['Year', 'Type', 'Acres'], skiprows=1)

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
def execute_linear_regression(x, y, X_test, Y_test):
    x_sm = sm.add_constant(x)
    model = sm.OLS(y, x_sm)
    results = model.fit()
    X_test_sm = sm.add_constant(X_test)
    y_pred_test = results.predict(X_test_sm) #predict on test data

    results.x_sm = x_sm
    results.y_pred = results.predict(x_sm) #predict on trained data
    results.y_pred_test = y_pred_test 
    return results


###
# Execute a linear regression test for crops nationwide
###
def execute_linear_regression_totals(graph = False):
    seeded_acres = crop_df.groupby('Year')['Acres'].sum().reset_index()
    merged = pd.merge(canada_pop, seeded_acres, on='Year')

    x = merged[['Population']]
    y = merged['Acres']

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    results = execute_linear_regression(X_train, Y_train, X_test, Y_test)
    print_linear_regression_model_stats('Totals', results, Y_test, results.y_pred_test) 
    if (graph):
        graph_linear_regression(X_train, Y_train, results, 'Crop Seeded Acres vs Population', 'Population', 'Crop Seeded Acres', X_test, Y_test) 
    return results


###
# Execute a linear regression test for each individual crop type
###
def execute_linear_regression_individual():   
    crop_types = crop_df.groupby(['Type', 'Year'])[['Acres']].sum().reset_index()
    
    # Kinda violating the "don't repeat yourself" principle here, but we'll excuse it this one time... 
    for t in crop_types['Type'].unique():
        filtered_crop = crop_types[crop_types['Type'] == t]
        merged = pd.merge(canada_pop, filtered_crop, on='Year')
        x = merged[['Population']]
        y = merged['Acres']

        X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.3, random_state=42)
        
        results = execute_linear_regression(X_train, Y_train, X_test, Y_test)
        print_linear_regression_model_stats(t, results, Y_test, results.y_pred_test) #modified to use test
        continue   


###
# Generate a correlation heatmap between population and crop types
###
def generate_correlation_heatmap_totals():
    crop_types = crop_df.groupby(['Type', 'Year'])[['Acres']].sum().reset_index()

    merged_data = canada_pop[['Year', 'Population']]

    for t in crop_types['Type'].unique():
        filtered_crop = crop_types[crop_types['Type'] == t]
        filtered_crop = filtered_crop.rename(columns={'Acres': t})
        merged_data = pd.merge(merged_data, filtered_crop[['Year', t]], on='Year', how='left') #modified to use new column name

    correlation_matrix = merged_data.drop('Year', axis=1).corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap: Population vs. Crop Types')
    plt.tight_layout()
    plt.show()
    

###
# Graph a linear regression result set, accepts labels and a title.
###
def graph_linear_regression(x, y, results, title, x_label, y_label, X_test=None, Y_test=None):
    # Plot the linear regression
    plt.scatter(x, y, color='blue', label='Training Data')
    plt.scatter(X_test, Y_test, color='green', label='Test Data')
    plt.plot(x, results.y_pred, color='red', label='Regr. Line - Training')
    plt.plot(X_test, results.y_pred_test, color='orange', label='Regr. Line - Test')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


###
# Get a prediction based on totals, allows user to input the population to predict
###
def run_prediction_totals():
    results = execute_linear_regression_totals()
    prediction = int(input('Enter population value to predict total acreage: '))
    predicted_acres = predict_totals(results, prediction)
    print(f'\tPredicted acreage with pop({prediction}) : {predicted_acres[0]:.0f}' )
    return 1


###
# Predict based on the incoming model and prediction request
###
def predict_totals(results, prediction):
    numeric_prediction_value = float(prediction)
    new_data_array = np.array([[1.0, numeric_prediction_value]])
    predicted_acres = results.predict(new_data_array)
    return predicted_acres


###
# Print descriptive stats for population
###
def print_descriptive_stats_population():
    print(f'[Population Descriptive Stats]')
    print(f'\tMean: {int(canada_pop['Population'].mean())}')
    print(f'\tMedian: {int(canada_pop['Population'].median())}')
    print(f'\tStandard dev: {int(canada_pop['Population'].std())}')
    print(f'\tVariance: {int(canada_pop['Population'].var())}')


###
# Print the stats for a linear regression analysis
###
def print_linear_regression_model_stats(crop_type, results, Y_test=None, y_pred_test=None):
    print(f'Crop type [{crop_type}] model statistics: ')
    print(f'\tP-value: {results.pvalues[f'Population']:.6f}')
    print(f'\tRegresson coefficient: {results.params[f'Population']:.6f}')
    print(f'\tR-squared (Training): {results.rsquared:.6f}')
    print(f'\tR-squared (Test): {r2_score(Y_test, y_pred_test):.6f}')


###
# Does exactly what it says - print the console menu.
###
def show_menu():
    print('---------------------------------------')
    print('1: Linear Regression - Total Crops')
    print('2: Linear Regression - Individual Crop')
    print('3: Correlation Heatmap')
    print('4: Descriptive Statistics - Population') 
    print('5: Predictions - Total Crops')
    print('6: Exit (Ctrl-C)')
    print('---------------------------------------')


###
# Main
###
if __name__ == '__main__':
    selection = 0
    while True:
        show_menu()
        try:
            selection = int(input('Selection: '))
        except:
            break 
        if (selection == 1):
            execute_linear_regression_totals(True)
        elif (selection == 2):
            execute_linear_regression_individual()
        elif (selection == 3):
            generate_correlation_heatmap_totals()
        elif (selection == 4):
            print_descriptive_stats_population()
        elif (selection == 5):
            run_prediction_totals()
        else:
            break
        input("Input to return to menu... ")  


