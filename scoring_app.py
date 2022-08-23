# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 14:10:10 2022

@author: Manikandan
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title= "Credit Scores Webpage", layout= "centered", page_icon=":bar_chart:")

def main():
    st.title("WELCOME TO SCORING MECHANISM!")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">The Scoring Mechanism App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # header
    st.header("The Credit Score Analysis of Pharma Retailers")
    
    # sub header
    st.subheader("To know the Credit Score of your Customer: ")
    
    df = pickle.load(open('df.pkl','rb'))
    customers = pickle.load(open('customers.pkl','rb'))
    order_status = pickle.load(open('order_status.pkl','rb'))
    city = pickle.load(open('city.pkl','rb'))

    # Create date variable that records recency
    import datetime
    snapshot_date = max(df.created)+ datetime.timedelta(days = 1)
    
    # Name of Customer as input
    Customer = st.selectbox("Select One Customer: ", (customers.values))

    # print the selected zone
    st.write("Selected Customer:", Customer)
    
    # select box
    Order_Status  = st.selectbox("Select status of your order: ", (order_status.values))
    # print the selected zone
    st.write("Selected status of your order:", Order_Status)

    # select box
    City = st.selectbox("Select City: ", (city.values))
    # print the selected zone
    st.write("Selected City:", City)


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

        rfm_table.drop(rfm_table.iloc[:, 1:8], inplace = True, axis = 1)
        
        rfm_table = rfm_table.loc[rfm_table['Customer'] == Customer]

        return rfm_table
    
    if st.button("Get Credit Scores"):
        credit_scores = obtain_rfm_scores(df)
        st.write("The Credit Scores of Pharma Retailers are:", credit_scores)

    
if __name__=='__main__':
    main()
    
    