
'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Load the dataset from a URL
# Get information about the dataset
url = 'C:/Users/user/Downloads/Mall_Customers.csv'
df = pd.read_csv(url)
# Display the first few rows of the dataframe
print("Dataset Head:")
print(df.head())
# Get information about the dataseturl = 'C:\Users\user\Downloads\Mall_Customers.csv'
print("\nDataset Info:")
df.info()

features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Convert the scaled array back to a DataFrame for clarity
X_scaled_df = pd.DataFrame(X_scaled, columns=features)
print("\nScaled Data Head:")
print(X_scaled_df.head())


# Calculate inertia for a range of k values
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.xticks(k_range)
plt.grid(True)
plt.show()


# Apply K-Means with the optimal k
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
# Add the cluster labels to our original dataframe
df['Cluster'] = clusters
# Apply PCA to reduce dimensions to 2 for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
# Create a new DataFrame with the PCA results and cluster labels
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Cluster'] = clusters
print("\nPCA Data with Cluster Labels:")
print(pca_df.head())


# Visualize the clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=100, alpha=0.8)

plt.title('Customer Segments Visualized with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Customer Cluster')
plt.grid(True)
plt.show()


cluster_analysis = df.groupby('Cluster')[features].mean()

print("\nCluster Analysis (Mean Values):")
print(cluster_analysis)'''



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Step 1: Load the wine dataset
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Elbow method to find optimal k
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k (Wine Dataset)")
plt.show()


