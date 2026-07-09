import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

rfm = pd.read_csv('/home/vladimir/Coding/Projects/OLIST-RFM/rfm.csv')
rfm_features = rfm[['Recency', 'Frequency', 'Monetary_log']]

pipeline = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(transformers=[
        ('scaler', StandardScaler(), ['Recency', 'Frequency', 'Monetary_log'])
    ])),
    ('kmeans', KMeans(n_clusters=4, init='k-means++', random_state=9))
])

rfm['Cluster'] = pipeline.fit_predict(rfm_features)

rfm_transformed = pipeline.named_steps['preprocessor'].transform(rfm_features)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(rfm_transformed)

# create a plotting DataFrame
plot_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
plot_df['Cluster'] = rfm['Cluster'].values

cluster_names = {
    0: "Recent High Spenders",
    1: "Recent Low Spenders",
    2: "Lost Customers",
    3: "Loyal High Spenders",
}

rfm['Cluster_name'] = rfm['Cluster'].map(cluster_names)
plot_df['Cluster_name'] = rfm['Cluster_name'].values


sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue='Cluster_name', palette='tab10')
plt.title('K-Means Clusters (PCA Projection)')
plt.show()

print(pca.explained_variance_ratio_)
print(f'Total variance explained: {sum(pca.explained_variance_ratio_):.1%}')
print(pd.DataFrame(
    pca.components_,
    columns=['Recency', 'Frequency', 'Monetary_log'],
    index=['PC1', 'PC2']
).round(3))

cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
cluster_summary['Count'] = rfm.groupby('Cluster')['Cluster'].count()
cluster_summary['Segment'] = cluster_summary.index.map(cluster_names)
print(cluster_summary)

rfm.to_csv('/home/vladimir/Coding/Projects/OLIST-RFM/rfm_clustered.csv', index=False)