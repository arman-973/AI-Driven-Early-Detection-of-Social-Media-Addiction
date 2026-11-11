# Social Media Addiction Clustering Analysis - README

## 📋 Overview

This README explains the K-Means clustering implementation in [exe3.ipynb](Experimental/exe3.ipynb), which analyzes social media usage patterns to classify users into three addiction risk categories: **Safe**, **At Risk**, and **Addicted**.

---

## 🎯 What is Clustering?

**Clustering** is an unsupervised machine learning technique that groups similar data points together based on their features. Unlike supervised learning (where we have labels), clustering discovers patterns in unlabeled data.

### Key Concepts:
- **Unsupervised Learning**: No predefined labels; the algorithm finds patterns automatically
- **Similarity**: Data points in the same cluster are more similar to each other than to points in other clusters
- **Centroids**: The center point of each cluster, representing the "average" member

---

## 🔍 What is K-Means Clustering?

**K-Means** is one of the most popular clustering algorithms. It partitions data into K distinct clusters based on feature similarity.

### How K-Means Works (General Algorithm):

1. **Choose K**: Decide the number of clusters you want
2. **Initialize**: Randomly place K centroids in the feature space
3. **Assignment**: Assign each data point to the nearest centroid
4. **Update**: Recalculate centroids as the mean of all points in each cluster
5. **Repeat**: Steps 3-4 until centroids stop moving (convergence)

### K-Means Objective:
Minimize the **Within-Cluster Sum of Squares (WCSS)** - the total squared distance between each point and its cluster centroid.

$$WCSS = \sum_{i=1}^{K} \sum_{x \in C_i} ||x - \mu_i||^2$$

Where:
- $K$ = number of clusters
- $C_i$ = cluster i
- $x$ = data point
- $\mu_i$ = centroid of cluster i

---

## 🚀 K-Means Implementation in exe3.ipynb

### Step 1: Data Loading and Preparation
```python
df = pd.read_csv('Social_Media_Addiction_With_Features_engg.csv')
```

**Dataset**: Contains 500 users with 17 features related to social media usage patterns.

### Step 2: Feature Selection
The analysis uses **17 carefully selected features**:

#### Demographic Features:
- `Age`: User's age

#### Usage Intensity Features:
- `Total_Monthly_Usage`: Total time spent across all platforms
- `Avg_Platform_Usage`: Average time per platform
- `Active_Platforms_Count`: Number of platforms used

#### Platform Diversity Features:
- `Platform_Diversity_Score`: How evenly usage is distributed
- `Dominant_Platform_Percentage`: Concentration on main platform
- `Usage_Concentration`: Measure of usage focus

#### Platform Category Features:
- `Social_Networking_Usage`: Time on social networks
- `Messaging_Usage`: Time on messaging apps
- `Video_Platform_Usage`: Time on video platforms
- `Social_to_Messaging_Ratio`: Balance between social and messaging

#### Consistency Feature:
- `Overall_Consistency`: How regular the usage pattern is

#### Platform-Specific Features:
- `YouTube_Monthly_Total`
- `WhatsApp_Monthly_Total`
- `Instagram_Monthly_Total`
- `Facebook_Monthly_Total`
- `Messenger_Monthly_Total`
- `TikTok_Monthly_Total`

### Step 3: Feature Standardization
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Why Standardize?**
- K-Means uses Euclidean distance, which is sensitive to feature scales
- Features with larger ranges would dominate the clustering
- Standardization transforms all features to mean=0, std=1

**Formula**: $z = \frac{x - \mu}{\sigma}$

### Step 4: Finding Optimal K (Elbow Method)
```python
for k_val in range(2, 8):
    kmeans_temp = KMeans(n_clusters=k_val, random_state=42)
    kmeans_temp.fit(X_scaled)
    inertias.append(kmeans_temp.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans_temp.labels_))
```

**Elbow Method**: Tests different values of K and plots metrics to find the "elbow point" where adding more clusters provides diminishing returns.

**Metrics Evaluated**:
1. **Inertia (WCSS)**: Lower is better (tighter clusters)
2. **Silhouette Score**: Range [-1, 1], higher is better (well-separated clusters)

**Result**: K=3 was selected as optimal

### Step 5: Apply K-Means Clustering
```python
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=300)
cluster_labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = cluster_labels
```

**Parameters**:
- `n_clusters=3`: Create 3 clusters
- `random_state=42`: Ensures reproducible results
- `n_init=10`: Run algorithm 10 times with different initializations, keep best result
- `max_iter=300`: Maximum iterations for convergence

### Step 6: Cluster Quality Evaluation

Three metrics assess clustering quality:

1. **Silhouette Score**: 0.2747
   - Measures how similar points are to their own cluster vs. other clusters
   - Range: [-1, 1], higher is better
   - Formula: $s = \frac{b - a}{\max(a, b)}$
   - Where $a$ = average distance to points in same cluster, $b$ = average distance to points in nearest cluster

2. **Davies-Bouldin Index**: 1.2573
   - Measures average similarity between each cluster and its most similar cluster
   - Lower is better (less overlap between clusters)

3. **Calinski-Harabasz Score**: 217.37
   - Ratio of between-cluster variance to within-cluster variance
   - Higher is better (more separated clusters)

### Step 7: Cluster Interpretation

The algorithm identified **3 distinct usage patterns**:

| Cluster | Users | Avg Total Usage (min/month) | Interpretation |
|---------|-------|------------------------------|----------------|
| 0 | 199 | 4,235 | Low usage |
| 1 | 144 | 6,985 | Medium usage |
| 2 | 157 | 9,631 | High usage |

### Step 8: Target Label Mapping
```python
cluster_mapping = {
    cluster_usage_avg.index[0]: 0,  # Safe
    cluster_usage_avg.index[1]: 1,  # At Risk
    cluster_usage_avg.index[2]: 2   # Addicted
}
df['Target'] = df['Cluster'].map(cluster_mapping)
```

**Clusters are mapped to meaningful labels**:
- **0 (Safe)**: Lowest usage cluster → Healthy social media usage
- **1 (At Risk)**: Medium usage cluster → Potential addiction risk
- **2 (Addicted)**: Highest usage cluster → Addicted behavior pattern

**Final Distribution**:
- Safe: 199 users (39.8%)
- At Risk: 144 users (28.8%)
- Addicted: 157 users (31.4%)

### Step 9: Visualization

The notebook includes multiple visualizations:

1. **Elbow Method Plots**: Show inertia and silhouette scores for k=2 to 7
2. **PCA 2D Projection**: Visualizes clusters in 2D space (explains ~30% of variance)
3. **Box Plots**: Compare Total Usage, Age, and Usage Concentration across clusters
4. **Heatmap**: Shows mean feature values for each cluster
5. **Target Distribution**: Bar chart, pie chart, and box plot of final categories

### Step 10: Model Persistence
```python
pickle.dump(kmeans, open('my_kmeans_model.pkl', 'wb'))
pickle.dump(scaler, open('my_scaler.pkl', 'wb'))
pickle.dump(mapping_data, open('my_cluster_mapping.pkl', 'wb'))
```

**Saved Artifacts**:
1. **my_kmeans_model.pkl**: Trained K-Means model with 3 cluster centroids
2. **my_scaler.pkl**: StandardScaler fitted on training data
3. **my_cluster_mapping.pkl**: Cluster-to-target mapping and feature list

---

## 📊 Output Files

### 1. Social_Media_Addiction_With_Features_And_Target.csv
Enhanced dataset with two new columns:
- **Cluster**: Original cluster assignment (0, 1, 2)
- **Target**: Addiction risk level (0=Safe, 1=At Risk, 2=Addicted)

---

## 🔮 Making Predictions for New Users

To classify a new user, the system:

1. **Loads saved artifacts**: Model, scaler, and mappings
2. **Preprocesses input**: Calculates all 17 required features
3. **Standardizes**: Uses the saved scaler to transform features
4. **Predicts cluster**: Applies K-Means model to find nearest centroid
5. **Maps to target**: Converts cluster ID to risk level (Safe/At Risk/Addicted)

**Example Prediction Flow**:
```
New User Features → Standardize → K-Means Predict → Cluster ID → Map → Target Label
[17 features]    → [z-scores]  → [cluster 0-2]   → [0-2]      → [Safe/At Risk/Addicted]
```

---

## 📈 Key Results Summary

### Cluster Characteristics:

**Cluster 0 (Safe - 199 users)**:
- Average Total Usage: 4,235 min/month (~141 min/day)
- Average Age: 27.6 years
- Average Active Platforms: 3.8
- **Profile**: Moderate, balanced social media users

**Cluster 1 (At Risk - 144 users)**:
- Average Total Usage: 6,985 min/month (~233 min/day)
- Average Age: 27.9 years
- Average Active Platforms: 4.3
- **Profile**: Heavy users showing early warning signs

**Cluster 2 (Addicted - 157 users)**:
- Average Total Usage: 9,631 min/month (~321 min/day)
- Average Age: 28.2 years
- Average Active Platforms: 4.6
- **Profile**: Excessive usage indicating addiction behavior

### Usage Thresholds:
- **Safe**: < ~5,600 min/month
- **At Risk**: ~5,600 - 8,300 min/month
- **Addicted**: > ~8,300 min/month

---

## 🎓 Why K-Means for This Problem?

### Advantages:
1. **Simplicity**: Easy to understand and implement
2. **Efficiency**: Fast computation, scales well to large datasets
3. **Interpretability**: Clear cluster centers represent "typical" users
4. **No Labels Needed**: Discovers patterns without predefined categories

### Limitations:
1. **Assumes spherical clusters**: May miss complex shapes
2. **Sensitive to initialization**: Uses `n_init=10` to mitigate
3. **Requires specifying K**: Uses elbow method to determine
4. **Sensitive to outliers**: Data was pre-processed to handle this

---

## 🛠️ Technical Details

### Dependencies:
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scikit-learn`: K-Means, StandardScaler, PCA, metrics
- `matplotlib` & `seaborn`: Visualization
- `pickle`: Model serialization

### Hyperparameters:
- **n_clusters**: 3 (determined by elbow method)
- **n_init**: 10 (multiple initializations for robustness)
- **max_iter**: 300 (sufficient for convergence)
- **random_state**: 42 (reproducibility)

### Computational Complexity:
- **Time**: O(n × K × i × d)
  - n = number of samples (500)
  - K = number of clusters (3)
  - i = iterations until convergence
  - d = number of features (17)

---

## 📚 Further Reading

### Clustering Concepts:
- [K-Means Algorithm](https://en.wikipedia.org/wiki/K-means_clustering)
- [Silhouette Analysis](https://en.wikipedia.org/wiki/Silhouette_(clustering))
- [Elbow Method](https://en.wikipedia.org/wiki/Elbow_method_(clustering))

### Related Notebooks:
- **input.ipynb**: Predict new user addiction risk using saved model
- **comparison notebooks**: Compare different clustering approaches

---

## 💡 Key Takeaways

1. **K-Means successfully identified 3 distinct user groups** based on social media usage patterns
2. **Clear separation exists** between Safe, At Risk, and Addicted users
3. **Total Monthly Usage is the strongest differentiator** between clusters
4. **The model is production-ready** with saved artifacts for new predictions
5. **Platform diversity and consistency** also contribute to addiction classification

---

## 🔄 Workflow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                     K-Means Clustering Pipeline                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────┐
              │   1. Load Dataset         │
              │   (500 users, 17 features)│
              └───────────┬───────────────┘
                          │
                          ▼
              ┌───────────────────────────┐
              │   2. Feature Selection    │
              │   (17 engineered features)│
              └───────────┬───────────────┘
                          │
                          ▼
              ┌───────────────────────────┐
              │   3. Standardization      │
              │   (mean=0, std=1)         │
              └───────────┬───────────────┘
                          │
                          ▼
              ┌───────────────────────────┐
              │   4. Elbow Method         │
              │   (Find optimal k=3)      │
              └───────────┬───────────────┘
                          │
                          ▼
              ┌───────────────────────────┐
              │   5. K-Means Clustering   │
              │   (3 clusters)            │
              └───────────┬───────────────┘
                          │
                          ▼
              ┌───────────────────────────┐
              │   6. Evaluate Quality     │
              │   (Silhouette, DB, CH)    │
              └───────────┬───────────────┘
                          │
                          ▼
              ┌───────────────────────────┐
              │   7. Map to Targets       │
              │   (Safe/At Risk/Addicted) │
              └───────────┬───────────────┘
                          │
                          ▼
              ┌───────────────────────────┐
              │   8. Visualize Results    │
              │   (PCA, plots, heatmaps)  │
              └───────────┬───────────────┘
                          │
                          ▼
              ┌───────────────────────────┐
              │   9. Save Model           │
              │   (3 .pkl files)          │
              └───────────────────────────┘
```

---

## ❓ FAQ

**Q: Why K=3 specifically?**  
A: The elbow method showed diminishing returns after k=3, and three categories (Safe/At Risk/Addicted) provide actionable insights.

**Q: Can I use a different number of clusters?**  
A: Yes, but k=3 provides the best balance of cluster quality and interpretability for this problem.

**Q: What if a new user has extreme values?**  
A: The standardization step handles this, but predictions may be less accurate for outliers.

**Q: How accurate is the clustering?**  
A: With Silhouette Score of 0.27, clusters are moderately well-separated. This is reasonable for real-world behavioral data.

**Q: Can I retrain with more data?**  
A: Yes, rerun the notebook with updated data, and save new model artifacts.

---

**Created**: January 2026  
**Author**: Social Media Addiction Analysis Project  
**Notebook**: [exe3.ipynb](exe3.ipynb)  
**Purpose**: Educational and research use for understanding social media addiction patterns
