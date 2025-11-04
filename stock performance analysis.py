#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 22:21:00 2024

@author: olyaklincheva
"""

import pandas as pd  
import yfinance as yf  
from statsmodels.formula.api import ols  
import numpy as np   
import matplotlib.pyplot as plt  

# Read the CSV file into a DataFrame
df = pd.read_csv('firms_dates.csv')

# Make sure the dates are string 
df['start'] = pd.to_datetime(df['start'])
df['end'] = pd.to_datetime(df['end'])

# Create a new Dataframe with the required column names
combined_df = pd.DataFrame(columns=['date', 'firm','market', 'rf', 'r_firm', 'r_market', 'r_rf'])

# Initialize lists to store results
firms_list = []
start_dates_list = []
end_dates_list = []
alphas_list = []
betas_list = []
sharpes_list = []
treynors_list = []
annual_returns_list = []

# Function to fetch company stock data from yahoofinance
def fetch_company_stock_data(start_date, end_date):    
    firm_data = yf.download(ticker, start=start_date, end=end_date)
    # Select required columns and rename them and reset the index
    firm_data = firm_data[['Adj Close']].reset_index()
    firm_data.rename(columns={'Date': 'date', 'Adj Close': 'firm'}, inplace=True)    
    return firm_data

# Function to fetch s&p500 index data
def fetch_sp500_data(start_date, end_date):    #Define the name of the function
    sp500_data = yf.download('SPY', start=start_date, end=end_date)
    # Select required columns and rename them and reset the index
    sp500_data = sp500_data[['Adj Close']].reset_index()
    sp500_data.rename(columns={'Date': 'date', 'Adj Close': 'market'}, inplace=True)    
    return sp500_data   

# Function to fetch bond yields data
def fetch_bond_yields(start_date, end_date):   
    bond_yield_data = yf.download('^IRX', start=start_date, end=end_date)   
    bond_yield_data = bond_yield_data[['Adj Close']].reset_index()    
    bond_yield_data.rename(columns={'Date': 'date', 'Adj Close': 'rf'}, inplace=True)     
    return bond_yield_data  

# Function to calculate regression
def calculate_regression(merged_df):    
    merged_df['excess_firm_returns'] = merged_df['r_firm'] - merged_df['r_rf']      
    merged_df['excess_market_returns'] = merged_df['r_market'] - merged_df['r_rf']   
    model=ols(formula="excess_firm_returns~excess_market_returns", data=merged_df)
    return model.fit() 

# Function to calculate Sharpe index
def calculate_sharpe_index(merged_df):    
    r_x_average = merged_df['r_firm'].mean()     
    r_f_average = merged_df['r_rf'].mean()  
    std_r_firm = merged_df['r_firm'].std()  
    return (r_x_average-r_f_average) / std_r_firm

# Function to calculate Treynor index
def calculate_treynor_index(merged_df, beta): 
    r_x_average = merged_df['r_firm'].mean()  
    r_f_average = merged_df['r_rf'].mean() 
    return (r_x_average-r_f_average) / beta

# Function to calculate annual returns
def calculate_annual_returns(merged_df): 
    first_date = merged_df['date'].iloc[0] 
    last_date = merged_df['date'].iloc[-1]  
    # Get the firm prices on the first and last dates
    first_date_price = merged_df.loc[merged_df['date'] == first_date, 'firm'].values[0]
    last_date_price = merged_df.loc[merged_df['date'] == last_date, 'firm'].values[0]
    r_total = (last_date_price / first_date_price) - 1
    # Calculate the number of days between the first and last dates
    days_passed = (last_date - first_date).days
    #Calculate annual returns
    return (1 + r_total) ** (365 / days_passed) - 1

def plot_stock_data(firm_name, merged_df):   
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))  

    # Plot stock prices (First subplot)
    ax[0, 0].plot(merged_df['date'], merged_df['firm'], label='Stock Price', color='green') 
    ax[0, 0].set_title(f'Stock Prices: {firm_name}', fontsize=15)   
    ax[0, 0].set_xlabel('Date', fontsize=14)  
    ax[0, 0].set_ylabel('Price($)', fontsize=14)  
    ax[0, 0].grid(axis='y', alpha=0.4)  
    ax[0, 0].tick_params(axis='x', rotation=45) 

    # Plot stock returns (Second subplot)
    ax[0, 1].plot(merged_df['date'], merged_df['r_firm'], label='Returns (%)', color='green') 
    ax[0, 1].set_title(f'Stock Returns: {firm_name}', fontsize=15)   
    ax[0, 1].set_xlabel('Date', fontsize=14)   
    ax[0, 1].set_ylabel('Returns(%)', fontsize=14)  
    ax[0, 1].grid(axis='y', alpha=0.4)  
    ax[0, 1].tick_params(axis='x', rotation=45)  

    # Plot histogram of stock returns (Third subplot)
    ax[1, 0].hist(merged_df['r_firm'], bins=50, color='green') 
    ax[1, 0].set_title(f'Histogram of Stock Returns: {firm_name}', fontsize=15)  
    ax[1, 0].set_xlabel('Returns (%)', fontsize=14)   
    ax[1, 0].set_ylabel('Frequency', fontsize=14)   
    ax[1, 0].grid(axis='y', alpha=0.4)  

    # Plot regression diagram (Forth subplot)
    ax[1, 1].scatter(merged_df['excess_market_returns'], merged_df['excess_firm_returns'], color='green') 
    ax[1, 1].set_title(f'Returns vs market returns: {firm_name}', fontsize=15)  
    ax[1, 1].set_xlabel('Adjusted market returns', fontsize=14) 
    ax[1, 1].set_ylabel('Adjusted returns', fontsize=14) 
    ax[1, 1].grid(axis='y', alpha=0.4) 
    ax[1, 1].tick_params(axis='x', rotation=45)

    # Calculate regression line
    regression_result = calculate_regression(merged_df) 
    alpha = model_res.params['Intercept']  
    beta = model_res.params['excess_market_returns']  
    x_vals = np.linspace(merged_df['excess_market_returns'].min(), merged_df['excess_market_returns'].max(), 100)
    y_vals = alpha + beta * x_vals
    ax[1, 1].plot(x_vals, y_vals, color='green')

    # Adjust layout and save the figure as a JPG file
    plt.tight_layout() 
    plt.savefig(f'stock_price_and_returns_{firm_name}.jpg') 
    plt.close() 

# Find price for the company stock in date range
for index, row in df.iterrows():
    firm = row['firm']
    ticker = row['ticker']
    start_date = row['start']
    end_date = row['end']

    # Fetch data for the company 1
    firm_df = fetch_company_stock_data(start_date, end_date)

    # Fetch data for s&p500 2.1
    sp500_df = fetch_sp500_data(start_date, end_date)
    # Fetch bond yield data for the last 13 weeks 2.2
    bond_yield_df = fetch_bond_yields(start_date, end_date)

    # Merge company data with S&P 500 data on the 'date' column
    merged_df = pd.merge(firm_df, sp500_df, on='date', how='inner')
    # Merge company stock, S&P 500 index data with bond yields on the 'date' column
    merged_df = pd.merge(merged_df, bond_yield_df, on='date', how='inner')

    # Calculate daily returns for the firm's stock and the market (4.1)
    merged_df['r_firm'] = merged_df['firm'].pct_change()
    # Calculate daily returns for s&p500(4.2)
    merged_df['r_market'] = merged_df['market'].pct_change()
    # Calculate daily returns for yields not in percentage(4.3)
    merged_df['r_rf'] = ((1+ merged_df['rf'] / 100) ** (1/365)) -1
    # Remove all the line with NaN 4.4
    merged_df = merged_df.dropna()

    #Calculate regression, alpha and beta 5.1, 5.2
    #Perform regression analysis
    model_res = calculate_regression(merged_df)
    #alpha and beta
    alpha = model_res.params['Intercept']  
    beta = model_res.params['excess_market_returns'] 

    #Calculate Sharpe index 5.3
    sharpe_ratio = calculate_sharpe_index(merged_df) 

    #Calculate Treynor index 5.4
    treynor_ratio = calculate_treynor_index(merged_df, beta) 

    #Annual returns 5.5
    annual_returns = calculate_annual_returns(merged_df) 

    # Append calculated values for lists from the beginning of the code
    firms_list.append(firm)
    start_dates_list.append(start_date.strftime('%d/%m/%Y'))
    end_dates_list.append(end_date.strftime('%d/%m/%Y'))
    alphas_list.append(alpha)
    betas_list.append(beta)
    sharpes_list.append(sharpe_ratio)
    treynors_list.append(treynor_ratio)
    annual_returns_list.append(annual_returns)


    # Plot stock data and save the figures
    plot_stock_data(firm, merged_df)

# Create the results DataFrame
result_df = pd.DataFrame({
    'Firm': firms_list,
    'Start Date': start_dates_list,
    'End Date': end_dates_list,
    'Alpha': alphas_list,
    'Beta': betas_list,
    'Sharpe': sharpes_list,
    'Treynor': treynors_list,
    'Annual Returns': annual_returns_list
})

print(result_df.head()) #To print the first few rows of the dataframe to check

# Save the results to a CSV file
result_df.to_csv('results.csv', index=False)

print("Combined data saved to combined_data.csv")