# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 10:18:20 2022

@author: Manikandan
"""

import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pickle


# Establishing the connection with MySQL
try:
    conn = mysql.connector.connect(host='127.0.0.1',
                                         database='project',
                                         user='root',
                                         password='Gayu@geetha1')
    
    #Creating a cursor object using the cursor() method

    sql_select_Query = "select * from creditanalysis_data"
    cur = conn.cursor()
    
    #Executing an MYSQL function using the execute() method
    
    cur.execute(sql_select_Query)
    # get all records
    records = cur.fetchall()
    print("Total number of rows in table: ", cur.rowcount)

except mysql.connector.Error as e:
    print("Error reading data from MySQL table", e)

finally:
    
    #Closing the connection
    
    if conn.is_connected():
        conn.close()
        cur.close()
        print("MySQL connection is closed")

# While retrieving the records, null values have already been dropped from MySQL database 

df = pd.DataFrame(records, columns = ['Unnamed: 0', 'master_order_id', 'master_order_status', 'created',
       'order_id', 'order_status', 'ordereditem_quantity', 'prod_names',
       'ordereditem_unit_price_net', 'ordereditem_product_id', 'value',
       'group', 'dist_names', 'retailer_names', 'bill_amount'])

df.drop(['Unnamed: 0'] ,axis = 1, inplace = True )
df = df[df['master_order_status'].str.contains('cancelled') == False]
df.isna().sum()

# Convert to show date only
df['created'] = pd.to_datetime(df['created'])

from datetime import datetime
df['created'] = df['created'].dt.date

# Create date variable that records recency
import datetime
snapshot_date = max(df.created)+ datetime.timedelta(days = 1)


'''RFM Customer Segmentation
RFM segmentation starts from here.
Create a RFM table'''

def obtain_rfm_scores(df):
    # Aggregate data by each customer
   
    rfm_table = df.groupby(['retailer_names']).agg({'created': lambda x: (snapshot_date - x.max()).days,
                                                        'order_id':'count',
                                                        'value': 'sum'})
    
    index = rfm_table.index
    rfm_table['Customer'] = index
    rfm_table.reset_index(drop = True, inplace = True)
    
    # Rename columns
    rfm_table.rename(columns = {'created':'Recency',
                                'order_id':'Frequency',
                                'value':'Monetary_value'},
                     inplace = True)
    
    rfm_table = rfm_table.reindex(['Customer', 'Recency', 'Frequency', 'Monetary_value'], axis = 1)


    '''The lowest recency, highest frequency and monetary amounts are our best customers.'''

    # To check the skewness of RFM

    R = rfm_table['Recency']
    sns.distplot(R)

    # frequency distribution plot taking values frequency less than 1000
    F = rfm_table.query('Frequency < 1000')['Frequency']  
    sns.distplot(F)

    # monetary_value distribution plot taking values less than 10000
    M = rfm_table.query('Monetary_value < 10000')['Monetary_value']
    sns.distplot(M)

    '''Split the metrics,

    The easiest way to split metrics into segments is by using quartiles.

    This gives us a starting point for the detailed analysis.
    4 segments are easy to understand and explain.'''

    quantiles = rfm_table.quantile(q = [0.25, 0.5, 0.75])
    quantiles = quantiles.to_dict()

    #Create a segmented RFM table
    segmented_customers = rfm_table

    # defining function for calculating Recency score

    def R_score(x,p,d):
        if x<=d[p][0.25]:
            return 1
        elif x<=d[p][0.5]:
            return 2
        elif x<=d[p][0.75]:
            return 3
        else:
            return 4
        
    # defining function for calculating frequency and monetary value score

    def FM_score(x,p,d):
        if x<=d[p][0.25]:
            return 1
        elif x<=d[p][0.5]:
            return 2
        elif x<=d[p][0.75]:
            return 3
        else:
            return 4
        
    # Add segment numbers to the newly created segmented RFM table
        
    segmented_customers['R_quantile'] = segmented_customers['Recency'].apply(R_score,
                                                               args = ('Recency',quantiles))

    segmented_customers['F_quantile'] = segmented_customers['Frequency'].apply(FM_score, 
                                                                 args = ('Frequency',quantiles))

    segmented_customers['M_quantile'] = segmented_customers['Monetary_value'].apply(FM_score, 
                                                                    args = ('Monetary_value',quantiles))

    # Adding new columns to sum the RFM group & score: 10 is the highest score.

    segmented_customers['RFM_score'] = segmented_customers.R_quantile.map(str)+segmented_customers.F_quantile.map(str)+segmented_customers.M_quantile.map(str)
    segmented_customers['Credit_scores'] = segmented_customers['R_quantile'] + segmented_customers['F_quantile'] + segmented_customers['M_quantile']

    # Assigning the scores acording to their loyalty level

    customer_loyalty_level = ['Bronze', 'Silver', 'Gold', 'Platinum']
    credit_loyalty = pd.qcut(segmented_customers['Credit_scores'], q=4, labels = customer_loyalty_level)
    segmented_customers['Loyalty_level'] = credit_loyalty.values
    
    # KMeans clustering
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    rfm_normalized = scaler.fit_transform(rfm_table.iloc[:,1:4])

    print(rfm_normalized.mean(axis = 0).round(2))
    print(rfm_normalized.std(axis = 0).round(2))

    from sklearn.cluster import KMeans

    sse = {}

    for k in range(1, 11):
        kmeans = KMeans(n_clusters = k, random_state = 42)
        kmeans.fit(rfm_normalized)
        sse[k] = kmeans.inertia_
        
    # plotting the elbow curve
    plt.title('The Elbow Curve')
    plt.xlabel('K')
    plt.ylabel('SSE')
    sns.pointplot(x = list(sse.keys()), y = list(sse.values()))
    plt.show()

    model = KMeans(n_clusters = 3, random_state = 42)
    model.fit(rfm_normalized)

    model.labels_.shape

    rfm_table['cluster'] = model.labels_
    rfm_table.groupby('cluster').agg({'Recency':'mean', 
                                      'Frequency':'mean',
                                      'Monetary_value':['mean','count']}).round(2)

    # scatter plot recency vs frequency
    plt.figure(figsize = (7,7))
    colors = ['red', 'green', 'blue']
    rfm_table['color'] = rfm_table['cluster'].map(lambda p: colors[p])
    ax = rfm_table.plot(
        kind = 'scatter',
        x = 'Recency',
        y = 'Frequency',
        figsize = (10,8),
        c = rfm_table['color'])

    # sorting out excellent and good scored customers
    
    rfm_table.drop(rfm_table.iloc[:, 1:8], inplace = True, axis = 1)
    customer = input('Enter Customer: \n')    
    rfm_table = rfm_table.loc[rfm_table['Customer'] == customer]

    return rfm_table

credit_scores = obtain_rfm_scores(df)

# pickle library for dumping data into file

pickle.dump(df,open('df.pkl','wb'))

customers = df['retailer_names'].unique()
customers = pd.Series(customers)

pickle.dump(customers,open('customers.pkl','wb'))

order_status = df['master_order_status'].unique()
order_status = pd.Series(order_status)

pickle.dump(order_status,open('order_status.pkl','wb'))

city = df['group'].unique()
city = pd.Series(city)

pickle.dump(city,open('city.pkl','wb'))

