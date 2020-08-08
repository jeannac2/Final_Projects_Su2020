#Final Project
# Import packages
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from math import sqrt

# Load data sets; check size to make sure they are the same
weight_df = pd.read_csv('Food_Pantry_Weight.csv' , header=0 , index_col='ID')
print(weight_df.head())
print(weight_df.shape)
children_df = pd.read_csv('Food_Pantry_Children.csv', header=0, index_col='ID')
print(children_df.head())
print(children_df.shape)
healthy_eat_df = pd.read_csv('Has_eaten_healthy.csv', header=0, index_col='ID')
print(healthy_eat_df.head())
print(healthy_eat_df.shape)

#Join columns, weight and number of children
inner_join_pantry = pd.merge(weight_df, children_df, left_on='ID', right_index=True, how='inner')
'''Doctest
    This function merges two data sets into one
    :param weight_df: Dataframe of weight, indexed by ID
    :param children_df: Dataframe of number of children, indexed by ID
    :return: Merged dataframe 
'''
print(inner_join_pantry.head())
print(inner_join_pantry.shape)
#Drop NA data, kept having issue of NAs beyond the bottom of the dataframe
inner_join_pantry.dropna()

#Join inner_join and healthy eating
inner_join_pantry_2 = pd.merge(inner_join_pantry, healthy_eat_df, left_on='ID', right_index=True, how='inner')
'''Doctest
    This function merges two data sets into one
    :param inner_join_pantry: Dataframe of weight and children, indexed by ID
    :param healthy_eat_df: Dataframe of if eating healthy or not, indexed by ID
    :return: Merged dataframe 
'''

# Descriptive Statistics
#Mean
mean_weight = inner_join_pantry_2.groupby(['ID'])['Weight'].mean()
'''Doctest
    This function calculates the mean of weight
    :param Weight: Column in inner_join_pantry2 dataframe
    :return: int(0:374)
'''
print(mean_weight)
mean_children = inner_join_pantry_2.groupby(['ID'])['Weight'].mean()
print(mean_children)
#STD
std_weight = inner_join_pantry_2.groupby(['ID'])['Weight'].std()
print(std_weight)
std_children = inner_join_pantry_2.groupby(['ID'])['LivingWithChildren'].std()
print(std_children)
#Mode of Children
mode_children = stats.mode(inner_join_pantry_2.groupby(['ID'])['LivingWithChildren'])
#Healthy eating
mean_healthy = inner_join_pantry_2.groupby(['ID'])['NutritionHab_CurrentlyEatHealthyFood'].mean()

#Look at plot of the data
inner_join_pantry_2.plot(x='LivingWithChildren' , y='Weight' , style='o')
'''Doctest
    This function creates a graph, plotting LivingWithChildren and Weight
    :param LivingWithChildren: Column in inner_join_pantry dataframe
    :param Weight: Column in inner_join_pantry dataframe
    :return: plot 
'''
plt.title('Number of Children vs Weight')
plt.xlabel('Number of Children')
plt.ylabel('Weight')
plt.show()

#Look at plot of the data
inner_join_pantry_2.plot(x='healthy_eat_df' , y='Weight' , style='o')
'''Doctest
    This function creates a graph, plotting healthy_eat_df and Weight
    :param LivingWithChildren: Column in inner_join_pantry dataframe
    :param Weight: Column in inner_join_pantry dataframe
    :return: plot 
'''
plt.title('Healthy Eating vs Weight')
plt.xlabel('Healthy Eating')
plt.ylabel('Weight')
plt.show()

#Regressions
model_reg = LinearRegression()
model_reg.fit('LivingWithChildren', 'Weight')
model_reg2 = LinearRegression()
model_reg2.fit('healthy_eat_df', 'Weight')

# Define the variable parameters
#weight
avg_weight = 198.4
std_dev_weight = 48.8
#children
avg_kids = 1.3
std_dev_kids = 1.5
#Healthy eating
avg_eat = .5
std_dev_eat = .24
#Repititions
num_reps =  151
num_simulations = 100

# Loop through 100 simulations
for i in range(100) :
    rand_variables = np.random.normal('LivingWithChildren,' 'healthy_eat_df,' 'Weight')
# Create the Monte carlo path, return the cumulative product of elements along a given axis
    forecasted_values = rand_variables.cumprod()
    '''Doctest
    This function runs the simulation and iterates through a maximum number of cycles of the random values and control values
    :param rand_variables: random values for Living with Children between the statistically least likely value (mode) and the highest value expected
    :return: 100 random values between the statistically least likely value (mode) and the highest value expected
    '''
# Plot the Monte Carlo path
plt.plot(range('LivingWithChildren') , forecasted_values)
plt.plot(range('healthy_eat_df') , forecasted_values)
#The data set I have loaded and reloaded and consistently had in Numeric before bringing over will not convert or become int

# Show the simulation data
plt.show(forecasted_values)

#Trying the Modified PERT random distribution
"""Produce random numbers according to the 'Modified PERT' distribution. 

:param low: The lowest value expected as possible
:param likely: The mode of LivingWithChildren 
:param high: The highest value expected as possible
:param confidence: 0-7

Attempt at Modified Pert Simulation" by Paulo Buchsbaum.
"""
mean = avg_kids
sd = std_dev_kids
# Check for confidence levels
confidence_low = mean-1.960(sd/sqrt(151))
confidence_high =mean-1.960(sd/sqrt(151))
    if confidence_low < 1 or confidence_low > 7:
        raise ValueError('confidence value must be in range 0-7')
    if confidence_high <1 or confidence_high > 7:
        raise ValueError('confidence value must be in range 0-7')

a = (mean - confidence_low) / (confidence_high - confidence_low) * (confidence_low + 2)
b = ((confidence_low + 1) * confidence_high - confidence_low - confidence_low * likely) / (confidence_high - confidence_low)
beta = np.random.beta(a , b , 151)
beta = beta * (confidence_high - confidence_low) + confidence_low
    return beta

