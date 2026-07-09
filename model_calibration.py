import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score

rfm = pd.read_csv('/home/vladimir/Coding/Projects/OLIST-RFM/rfm.csv')
rfm_features = rfm[['Recency', 'Frequency', 'Monetary_log']]
wcss = []
sil_scores = []


for k in range(2, 11):
    # Use 'k-means++' to avoid the random initialization trap
    pipeline = Pipeline(steps=[
        ('preprocessor', ColumnTransformer(transformers=[
            ('scaler', StandardScaler(), ['Recency', 'Frequency', 'Monetary_log'])
        ])),
        ('kmeans', KMeans(n_clusters=k, init='k-means++', random_state=9))
    ])

    pipeline.fit(rfm_features)
    
    # .inertia_ extracts the calculated WCSS score for that specific k
    wcss.append(pipeline.named_steps['kmeans'].inertia_)

    # silhouette scores for each k
    rfm_transformed = pipeline.named_steps['preprocessor'].transform(rfm_features)
    sil_scores.append(silhouette_score(rfm_transformed, pipeline.named_steps['kmeans'].labels_))


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(2, 11), wcss, marker='o', linestyle='--', color='b')
axes[0].set_title('Elbow Method')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('WCSS (Inertia)')
axes[0].set_xticks(range(2, 11))
axes[0].grid(True)

axes[1].plot(range(2, 11), sil_scores, marker='o', linestyle='--', color='g')
axes[1].set_title('Silhouette Scores')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_xticks(range(2, 11))
axes[1].grid(True)

plt.tight_layout()
plt.show()