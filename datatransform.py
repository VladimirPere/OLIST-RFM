import pandas as pd


# End result should have a table with:
# Rows = unique customers
# Columns = [customer_unique_id, 
#            Recency (days since last order (end at Dec 31 2018)),
#            Frequency (num orders between first order and last),
#            Monetary (total spent between 2016 and 2018)
#            Recency_scaled
#            Frequency_scaled
#            Monetary_scaled

# customer_id, customer_unique_id
cust = pd.read_csv('/home/vladimir/Downloads/Data/OLIST_data/olist_customers_dataset.csv')

# order_id, payment_value -> Aggregate to one row per order_id
payment = pd.read_csv('/home/vladimir/Downloads/Data/OLIST_data/olist_order_payments_dataset.csv')
payment_agg = payment.groupby('order_id')['payment_value'].sum().reset_index()

# order_id, customer_id, order_purchase_timestamp, order_status
orders = pd.read_csv('/home/vladimir/Downloads/Data/OLIST_data/olist_orders_dataset.csv')
orders = orders[orders['order_status'].isin(['delivered'])] # Shipped orders not used so only known business is used, but may be added later

