import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders_merged = orders.merge(cust, on='customer_id', how='left')
reference_date = pd.Timestamp('2018-12-31')

recency = (orders_merged
           .groupby('customer_unique_id')['order_purchase_timestamp']
           .max()                              # most recent order date per customer
           .reset_index()
           .rename(columns={'order_purchase_timestamp': 'last_order_date'}))

recency['Recency'] = (reference_date - recency['last_order_date']).dt.days

frequency = (orders_merged
             .groupby('customer_unique_id')['order_id']
             .count()
             .reset_index()
             .rename(columns={'order_id': 'Frequency'}))

monetary = (orders_merged
            .merge(payment_agg, on='order_id', how='left')
            .groupby('customer_unique_id')['payment_value']
            .sum()
            .reset_index()
            .rename(columns={'payment_value': 'Monetary'}))


rfm = recency.merge(frequency, on='customer_unique_id', how='left').merge(monetary, on='customer_unique_id', how='left')
rfm = rfm.dropna()

# drop zero monetary
rfm = rfm[rfm['Monetary'] > 0]

# drop the top 1% of monetary values to remove extreme outliers
Q99 = rfm['Monetary'].quantile(0.99)
rfm = rfm[rfm['Monetary'] <= Q99]

# drop the extreme frequency outlier of 15 orders
rfm = rfm[rfm['Frequency'] < 15]

# scale monetary using log transform
rfm['Monetary_log'] = np.log1p(rfm['Monetary'])

#dropping unnecessary columns for modeling
rfm = rfm.drop(columns=['last_order_date'])

# Skew check and final processes
if __name__ == "__main__":
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ['Recency', 'Frequency', 'Monetary']):
        sns.histplot(rfm[col], ax=ax, bins=50)
        ax.set_title(col)
    plt.tight_layout()
    plt.show()
    print(f'Final RFM table: {rfm.shape[0]} customers, {rfm.shape[1]} columns')
    print(rfm.columns.tolist())
    print(rfm.head())



rfm.to_csv('/home/vladimir/Coding/Projects/OLIST-RFM/rfm.csv', index=False)