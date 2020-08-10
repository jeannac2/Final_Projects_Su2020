#Final Project
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statistics

# Load data sets; check size to make sure they are the same
weight_df = pd.read_csv('Food_Pantry_Weight.csv' , header=0 , index_col='ID')
print(weight_df.head())
print(weight_df.shape)
def read_csv(csv_file, index_col):
    """
    This function creates a DataFrame from a csv file indexed by the specified column name
    param: csv_file: csv file that will be read by pandas
    param: index_col: the name of the column that is the index of the DataFrame
    return: DataFrame
    >>> csv_file = 'Food_Pantry_Weight.csv'
    >>> column_name = 'ID'
    >>> df = read_csv(csv_file, index_col)
    >>> df # doctest: +ELLIPSIS
    {...'ID', 'Weight': 210, 156...}
    """
    weight_df = pd.read_csv(csv_file, header=0, index_col='ID')
    return weight_df

children_df = pd.read_csv('Food_Pantry_Children.csv', header=0, index_col='ID')
print(children_df.head())
print(children_df.shape)
healthy_eat_df = pd.read_csv('Has_eaten_healthy.csv', header=0, index_col='ID')
print(healthy_eat_df.head())
print(healthy_eat_df.shape)

#Join columns, weight and number of children

def merge(weight_df, children_df, left_on, right_index, how):
    """
    This function merges two data sets into one
    :param weight_df: Dataframe of weight that will be read by pandas
    :param children_df: Dataframe of number of children that will be read by pandas
    :param right_index: Index of the two dataframes that pandas will merge by
    :return dataframe: inner_join_pantry
    >>> inner_join_pantry = pd.merge(weight_df, children_df, left_on='ID', right_index=True, how='inner')
    {...'ID',... 'weight_df':210, 156 ...}
    """
    inner_join_pantry = pd.merge(weight_df, children_df, left_on='ID', right_index=True, how='inner')
    return inner_join_pantry

inner_join_pantry = pd.merge(weight_df, children_df, left_on='ID', right_index=True, how='inner')
print(inner_join_pantry.head())
print(inner_join_pantry.shape)
#Join inner_join_pantry and healthy eating
inner_join_pantry_2 = pd.merge(inner_join_pantry, healthy_eat_df, left_on='ID', right_index=True, how='inner')
print(inner_join_pantry_2.head())
print(inner_join_pantry_2.shape)

#Drop NA data, kept having issue of NAs beyond the bottom of the dataframe
inner_join_pantry.dropna()

# Descriptive Statistics
#Mean
mean_weight = inner_join_pantry_2['Weight'].mean()
print(mean_weight)
mean_children = inner_join_pantry_2['LivingWithChildren'].mean()
print(mean_children)
#Healthy eating is boolean so a mean is innapropriate; see mode

#Standard deviation
std_weight = inner_join_pantry_2['Weight'].std()
print(std_weight)
std_children = inner_join_pantry_2['LivingWithChildren'].std()
print(std_children)
#Healthy eating is a boolean term, (yes=1,no=0) therefore std is not appropriate; see mode

#Mode
#Mode of Children
mode_children = statistics.mode(inner_join_pantry_2['LivingWithChildren'])
print(mode_children)
#Mode of healthy eating is a boolean response (1=yes,0=no) to if respondent has eaten healthy since their last visit to the pantry
mode_healthy_eating = statistics.mode(inner_join_pantry_2['NutritionHab_CurrentlyEatHealthyFood'])
print(mode_healthy_eating)

#Look at plot of the data
inner_join_pantry_2.plot(x='LivingWithChildren', y='Weight', style='o')
plt.title('Number of Children vs Weight')
plt.xlabel('Number of Children')
plt.ylabel('Weight')
plt.show()

#Look at plot of the data
inner_join_pantry_2.plot(x='NutritionHab_CurrentlyEatHealthyFood', y='Weight', style='o')
plt.title('Healthy Eating vs Weight')
plt.xlabel('Healthy Eating')
plt.ylabel('Weight')
plt.show()

#Regression, X=LivingWithChildren, Y=Weight
X = inner_join_pantry_2['LivingWithChildren'].values.reshape(-1,1)
Y = inner_join_pantry_2['Weight'].values.reshape(-1,1)
model_reg = LinearRegression()
model_reg.fit(X, Y)
#Looking at the intercept of the regression
print(model_reg.intercept_)
#Looking at the slope
print(model_reg.coef_)

#Regression2, X=Healthy eating, Y=Weight
X = inner_join_pantry_2['NutritionHab_CurrentlyEatHealthyFood'].values.reshape(-1,1)
Y = inner_join_pantry_2['Weight'].values.reshape(-1,1)
model_reg2 = LinearRegression()
model_reg2.fit(inner_join_pantry_2['NutritionHab_CurrentlyEatHealthyFood'], inner_join_pantry_2['Weight'])
model_reg.fit(X, Y)
#Looking at the intercept of the regression
print(model_reg2.intercept_)
#Looking at the slope
print(model_reg2.coef_)


# Define the variable parameters
#weight
avg_weight = 198.4
std_dev_weight = 48.8
#children
avg_kids = 1.3
std_dev_kids = 1.5
#Repititions
num_reps =  151
num_simulations = 100

# Loop through 100 simulations for weight and living with children (number of children in household)
for I in range(100):
    rand_weights = np.random.normal(198.4, 48.8, size=100)
    rand_kids = np.random.normal(1.3, 1.5, size=100)
# Create the Monte carlo path, return the cumulative product of elements along a given axis
    forecasted_values_weights = rand_weights.cumprod()
    forecasted_values_kids = rand_kids.cumprod()

# Plot the Monte Carlo path- I do not know how to do this, I tried a number of values for X but the plot looks weird as is
    plot_1 = plt.plot(forecasted_values_weights)
    plt.show()
    plot_2 = plt.plot(forecasted_values_kids)
    plt.show()

    # Show the simulation data
    plt.show(forecasted_values_weights)
    plt.show(forecasted_values_kids)

#Trying the Modified PERT random distribution by Weible, J. (2020)
#https://github.com/iSchool-590pr/Summer2020_examples/blob/master/week_05_MCsims%26Prob/Probability_Distributions.ipynb
"""Produce random numbers according to the 'Modified PERT' distribution. 

:param low: The lowest value expected as possible.
:param likely: The 'most likely' value, statistically, the mode.
:param high: The highest value expected as possible.
:param confidence: This is typically called 'lambda' in literature 
                    about the Modified PERT distribution. The value
                    4 here matches the standard PERT curve. Higher
                    values indicate higher confidence in the mode.
                    Currently allows values 1-18

Formulas to convert beta to PERT are adapted from a whitepaper 
"Modified Pert Simulation" by Paulo Buchsbaum.
"""
# Check for reasonable confidence levels to allow:
def mod_pert_random(low, likely, high, confidence=4, samples=10000):
    if confidence < 1 or confidence > 18 :
        raise ValueError('confidence value must be in range 1-18.')

    mean = (low + confidence * likely + high) / (confidence + 2)

    a = (mean - low) / (high - low) * (confidence + 2)
    b = ((confidence + 1) * high - low - confidence * likely) / (high - low)

    beta = np.random.beta(a , b , samples)
    beta = beta * (high - low) + low
    return beta