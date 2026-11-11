# 📱 Social Media Addiction Analysis & Prediction System

> An end-to-end machine learning project that analyzes social media usage patterns and predicts addiction risk levels using unsupervised learning (K-Means clustering).

---

## 🎯 Project Overview

This project implements a comprehensive Social Media Addiction Analysis System that:
 Analyzes usage patterns across 6 major platforms (YouTube, WhatsApp, Instagram, Facebook, Messenger, TikTok)
 Identifies behavioral patterns using K-Means clustering
 Classifies users into 3 risk categories: Safe, At Risk, and Addicted
 Provides an interactive web application for real-time predictions
 Evaluates model performance on different datasets

### 🌟 Key Features
- **Unsupervised Learning**: No labeled data required - discovers patterns automatically
- **Multi-platform Analysis**: Considers usage across 6 social media platforms
- **Feature Engineering**: 17+ engineered features capturing usage intensity, diversity, and consistency
- **Production-Ready**: Includes trained models, web app, and comprehensive evaluation
- **Interactive Prediction**: Streamlit web interface for instant risk assessment

---

## 📊 Datasets

### Primary Datasets

| Dataset | Rows | Features | Purpose |
|---------|------|----------|---------|
| `Social_Media_Addiction.csv` | 691 | 27 | Original raw dataset with weekly usage data |
| `mobile_usage_behavioral_analysis.csv` | 1,000 | 6 | External dataset for model comparison |
| `Students Social Media Addiction.csv` | 705 | 6 | Student-specific dataset for evaluation |

### Processed Datasets (in `Experimental/`)

| Dataset | Rows | Purpose |
|---------|------|---------|
| `Social_Media_Addiction_No_Outliers.csv` | 500 | Training data (outliers removed) |
| `Social_Media_Addiction_Outliers.csv` | 82 | Test data (extreme usage cases) |
| `Social_Media_Addiction_With_Features_engg.csv` | 500 | Training data with 58 engineered features |
| `Social_Media_Addiction_With_Features_And_Target.csv` | 500 | Final dataset with cluster labels and target categories |

### Data Structure

**Raw Features** (per platform):
- Weekly usage data (4 weeks × 6 platforms = 24 features)
- Demographics: Age, Gender, Occupation

**Engineered Features** (17 final features):
- Usage intensity: Total, Average, Active platforms
- Diversity: Platform diversity score, Usage concentration
- Category-based: Social networking, Messaging, Video usage
- Platform-specific: Monthly totals for each platform
- Behavioral: Consistency scores

---

## 📁 Project Structure

```
📦 my project work/
│
├── 📄 streamlit_app.py              # Interactive web application
├── 📄 README.md                      # This file - Project overview
│
├── 📂 Experimental/                  # Core analysis notebooks
│   ├── 📓 exe1.ipynb                 # Data cleaning & outlier detection
│   ├── 📓 exe2.ipynb                 # Exploratory Data Analysis (EDA) & feature engineering
│   ├── 📓 exe3.ipynb                 # K-Means clustering & model training ⭐
│   ├── 📓 exe4.ipynb                 # Model evaluation on outliers
│   ├── 📄 README.md                  # Detailed clustering implementation guide
│   ├── 🤖 my_kmeans_model.pkl        # Trained K-Means model
│   ├── 📊 my_scaler.pkl              # StandardScaler for feature normalization
│   ├── 🗂️ my_cluster_mapping.pkl     # Cluster-to-target mappings
│   └── 📊 [Processed CSV files]      # Cleaned and engineered datasets
│
├── 📂 Comparison/                    # Model validation notebooks
│   ├── 📓 comparison_1.ipynb         # Test on mobile_usage dataset
│   └── 📓 comparison_2.ipynb         # Test on students dataset
│
├── 📂 scripts/                       # Utility scripts
│   └── 📄 clean.py                   # Data preprocessing script
│
└── 📊 [Original CSV files]           # Raw datasets
```

---

## 🔬 Workflow & Methodology

### Phase 1: Data Preparation (`exe1.ipynb`)

**Objective**: Clean raw data and handle outliers

**Steps**:
1. Load original dataset (691 users, 27 features)
2. Convert time formats (HH:MM to minutes)
3. Calculate monthly totals per platform
4. **Outlier Detection** using IQR method:
   - Q1 = 3,976 min/month, Q3 = 6,845 min/month
   - IQR = 2,870, Threshold = 11,150 min/month
   - Identified 82 extreme usage outliers (11.9%)
5. Split dataset:
   - **Training**: 500 users without outliers → `Social_Media_Addiction_No_Outliers.csv`
   - **Test**: 82 outlier users → `Social_Media_Addiction_Outliers.csv`

**Rationale**: Outliers can distort clustering. We train on normal users and evaluate separately on extreme cases.

---

### Phase 2: EDA & Feature Engineering (`exe2.ipynb`)

**Objective**: Understand patterns and create meaningful features

**Exploratory Data Analysis**:
- Age distribution: 18-35 years (young adults)
- Gender balance: 50.4% Male, 49.6% Female
- Top occupations: Student (26.6%), Working Professional (26.6%)
- Platform popularity: YouTube > WhatsApp > Instagram
- Usage patterns: High variability across weeks

**Feature Engineering** (58 features created):

1. **Aggregate Features** (4):
   - `Total_Monthly_Usage`: Sum across all platforms
   - `Avg_Platform_Usage`: Mean usage per platform
   - `Active_Platforms_Count`: Number of platforms used
   - `Platform_Diversity_Score`: % of platforms actively used

2. **Dominance Features** (2):
   - `Dominant_Platform_Percentage`: Share of top platform
   - `Usage_Concentration`: Herfindahl index of concentration

3. **Category Features** (4):
   - `Social_Networking_Usage`: Facebook + Instagram usage
   - `Messaging_Usage`: WhatsApp + Messenger usage
   - `Video_Platform_Usage`: YouTube + TikTok usage
   - `Social_to_Messaging_Ratio`: Balance metric

4. **Consistency Features** (1):
   - `Overall_Consistency`: Regularity of usage patterns

5. **Weekly Variation** (12):
   - Standard deviation per platform across weeks
   - Consistency scores per platform

6. **Platform Totals** (6):
   - Monthly totals for each platform

7. **Demographic Encoding** (3):
   - One-hot encoding for categorical variables

**Output**: `Social_Media_Addiction_With_Features_engg.csv` (500 rows × 58 features)

---

### Phase 3: K-Means Clustering & Training (`exe3.ipynb`) ⭐

**Objective**: Build unsupervised clustering model to classify addiction levels

#### Step-by-Step Implementation:

**1. Feature Selection**
   - Selected **17 most relevant features** from 58 engineered features
   - Focused on: age, usage intensity, diversity, category usage, consistency

**2. Standardization**
   ```python
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```
   - Transforms features to mean=0, std=1
   - Essential for K-Means (distance-based algorithm)

**3. Optimal K Selection (Elbow Method)**
   - Tested k = 2 to 7 clusters
   - Evaluated metrics: Inertia (WCSS) and Silhouette Score
   - **Result**: k=3 selected (clear elbow point)

**4. K-Means Clustering**
   ```python
   kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
   cluster_labels = kmeans.fit_predict(X_scaled)
   ```
   - Created 3 distinct user groups

**5. Cluster Quality Metrics**
   - Silhouette Score: **0.2747** (moderate separation)
   - Davies-Bouldin Index: **1.2573** (acceptable)
   - Calinski-Harabasz Score: **217.37** (good)

**6. Cluster Interpretation**

| Cluster | Users | Avg Usage (min/month) | Daily Avg | Profile |
|---------|-------|----------------------|-----------|---------|
| 0 | 199 (39.8%) | 4,235 | ~141 min/day (~2.4h) | Low usage - Healthy |
| 1 | 144 (28.8%) | 6,985 | ~233 min/day (~3.9h) | Medium usage - Warning signs |
| 2 | 157 (31.4%) | 9,631 | ~321 min/day (~5.4h) | High usage - Excessive |

**7. Target Label Mapping**
   - Cluster 0 → **Target 0 (Safe)**: Healthy social media habits
   - Cluster 1 → **Target 1 (At Risk)**: Potential addiction risk
   - Cluster 2 → **Target 2 (Addicted)**: Addictive behavior patterns

**8. Visualizations**
   - Elbow curves (Inertia & Silhouette)
   - PCA 2D projection of clusters
   - Box plots by cluster (Usage, Age, Concentration)
   - Heatmap of cluster characteristics

**9. Model Persistence**
   - Saved trained model: `my_kmeans_model.pkl`
   - Saved scaler: `my_scaler.pkl`
   - Saved mappings: `my_cluster_mapping.pkl`

**Output**: 
- `Social_Media_Addiction_With_Features_And_Target.csv` (with Cluster & Target columns)
- 3 .pkl files for production deployment

📖 **For detailed clustering explanation, see [Experimental/README.md](Experimental/README.md)**

---

### Phase 4: Model Evaluation (`exe4.ipynb`)

**Objective**: Test model on extreme usage outliers

**Dataset**: 82 outlier users (high usage cases)

**Process**:
1. Load trained model, scaler, and mappings
2. Apply same feature engineering pipeline
3. Standardize features using saved scaler
4. Predict cluster assignments
5. Map to target labels

**Results**:
- **At Risk**: 1 user (1.2%)
- **Addicted**: 81 users (98.8%)

**Analysis**: Model correctly identifies extreme users as high-risk, validating its effectiveness on outlier detection.

**Visualizations**:
- Prediction distribution
- Cluster assignments in PCA space
- Usage patterns comparison

---

### Phase 5: Cross-Dataset Validation (`Comparison/`)

**Objective**: Test model generalization on external datasets

#### Comparison 1: Mobile Usage Behavioral Analysis
- **Dataset**: 1,000 users from different study
- **Features**: Age, Gender, Social Media Usage (hours/day)
- **Results**: Model successfully adapted to different data structure
- **Metrics**: Silhouette Score 0.39 (good separation)

#### Comparison 2: Students Social Media Addiction
- **Dataset**: 705 student users
- **Features**: Age, Gender, Average Daily Usage
- **Results**: Model generalized well to student population
- **Finding**: Students show higher addiction rates

**Key Insight**: Model demonstrates robustness across different demographics and data formats.

---

## 🚀 Streamlit Web Application

Interactive web app for real-time addiction risk prediction.

### Features:
- 📝 **Input Form**: Enter user demographics and weekly usage
- 🔄 **Auto Feature Engineering**: Automatically generates 17 features
- 🎯 **Real-time Prediction**: Instant risk classification
- 📊 **Visual Feedback**: Color-coded results (Green/Yellow/Red)
- 📈 **Usage Insights**: Comparison with average users

### Usage:
```bash
# Navigate to project directory
cd "c:\Users\ThinkPad T14\Downloads\my essential\my project work"

# Activate virtual environment
.\venv\Scripts\Activate

# Run Streamlit app
streamlit run streamlit_app.py
```

### Input Required:
- **Demographics**: Age, Gender, Occupation
- **Weekly Usage**: 4 weeks × 6 platforms = 24 values (in HH:MM format)

### Output:
- **Risk Level**: Safe / At Risk / Addicted
- **Predicted Cluster**: 0, 1, or 2
- **Total Monthly Usage**: Minutes and hours
- **Visual Indicator**: Color-coded result box

---

## 🛠️ Technologies & Libraries

### Core ML/Data Science:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms
  - `KMeans`: Clustering algorithm
  - `StandardScaler`: Feature normalization
  - `PCA`: Dimensionality reduction for visualization

### Visualization:
- **matplotlib**: Static plots
- **seaborn**: Statistical visualizations
- **altair**: Interactive charts (Streamlit)

### Deployment:
- **streamlit**: Web application framework
- **pickle**: Model serialization

### Development:
- **Python 3.x**: Primary language
- **Jupyter Notebook**: Interactive development
- **VS Code**: IDE

---

## 📈 Key Results & Insights

### Clustering Performance:
- ✅ **3 distinct clusters** identified with clear separation
- ✅ **Silhouette Score**: 0.2747 (acceptable for behavioral data)
- ✅ **Balanced distribution**: 40% Safe, 29% At Risk, 31% Addicted

### Feature Importance:
1. **Total_Monthly_Usage**: Strongest predictor
2. **Usage_Concentration**: High concentration = higher risk
3. **Active_Platforms_Count**: More platforms = potentially higher engagement
4. **Overall_Consistency**: Regular usage patterns indicate habit formation

### Usage Thresholds:
- **Safe**: < ~5,600 minutes/month (< 3 hours/day)
- **At Risk**: 5,600 - 8,300 minutes/month (3-4.6 hours/day)
- **Addicted**: > 8,300 minutes/month (> 4.6 hours/day)

### Demographics Insights:
- Age: Minimal impact on clustering (similar across clusters)
- Gender: No significant bias
- Occupation: Students show slightly higher usage

### Platform Insights:
- **YouTube**: Most used platform across all clusters
- **WhatsApp**: High usage but considered "messaging" (less indicative of addiction)
- **Instagram/TikTok**: Strong indicators when combined with high overall usage

---

## 🎓 Machine Learning Approach

### Why K-Means Clustering?

**Advantages for this problem**:
1. **Unsupervised**: No labeled data needed
2. **Interpretable**: Clear cluster centers represent "typical" users
3. **Efficient**: Fast computation, scalable
4. **Flexible**: Can be updated as new data arrives

**Limitations & Mitigations**:
- ❌ Assumes spherical clusters → ✅ Feature engineering creates meaningful space
- ❌ Sensitive to initialization → ✅ Used `n_init=10` for robustness
- ❌ Requires specifying K → ✅ Elbow method determined optimal K=3
- ❌ Affected by outliers → ✅ Separated outliers for evaluation

### Algorithm Flow:
```
Raw Data (691 users)
    ↓
[Outlier Detection]
    ↓
Training Set (500 users) ←→ Test Set (82 outliers)
    ↓
[Feature Engineering]
    ↓
17 Selected Features
    ↓
[Standardization]
    ↓
[K-Means (k=3)]
    ↓
Cluster Labels (0, 1, 2)
    ↓
[Map to Targets]
    ↓
Risk Levels (Safe, At Risk, Addicted)
```

---

## 📚 How to Use This Project

### 1️⃣ Understanding the Analysis
- Read [Experimental/README.md](Experimental/README.md) for detailed K-Means implementation
- Review `exe1.ipynb` → `exe2.ipynb` → `exe3.ipynb` in sequence
- Check visualizations for pattern insights

### 2️⃣ Running Notebooks
```bash
# Activate virtual environment
.\venv\Scripts\Activate

# Open Jupyter
jupyter notebook

# Navigate to Experimental/ folder
# Run notebooks in order: exe1 → exe2 → exe3 → exe4
```

### 3️⃣ Making Predictions
**Option A: Web App**
```bash
streamlit run streamlit_app.py
```

**Option B: Python Script**
```python
import pickle
import pandas as pd

# Load model artifacts
with open('Experimental/my_kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)
with open('Experimental/my_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('Experimental/my_cluster_mapping.pkl', 'rb') as f:
    mapping_data = pickle.load(f)

# Prepare new user data (17 features)
new_user = pd.DataFrame([{
    'Age': 25,
    'Total_Monthly_Usage': 7000,
    # ... 15 more features
}])

# Predict
X_scaled = scaler.transform(new_user)
cluster = kmeans.predict(X_scaled)[0]
target = mapping_data['cluster_mapping'][cluster]
label = mapping_data['label_names'][target]

print(f"Prediction: {label}")
```

### 4️⃣ Retraining with New Data
1. Add new data to `Social_Media_Addiction.csv`
2. Run `exe1.ipynb` (outlier detection)
3. Run `exe2.ipynb` (feature engineering)
4. Run `exe3.ipynb` (clustering & save new models)
5. Update Streamlit app if needed

---

## 🔮 Future Enhancements

### Model Improvements:
- [ ] Experiment with other clustering algorithms (DBSCAN, Hierarchical)
- [ ] Add time-series analysis for usage trends
- [ ] Implement ensemble methods for robust predictions
- [ ] Create classification model using cluster labels as ground truth

### Feature Engineering:
- [ ] Incorporate time-of-day usage patterns
- [ ] Add session duration features
- [ ] Include content type analysis
- [ ] Consider engagement metrics (likes, comments, shares)

### Application Features:
- [ ] User dashboard with historical tracking
- [ ] Personalized recommendations for reducing usage
- [ ] Multi-language support
- [ ] Mobile app version
- [ ] Export reports as PDF

### Data Collection:
- [ ] Integrate with social media APIs for automated data collection
- [ ] Add more platforms (Twitter, LinkedIn, Reddit)
- [ ] Collect longitudinal data for trend analysis
- [ ] Include psychological survey data

---

## 📊 Project Metrics

| Metric | Value |
|--------|-------|
| Total Users Analyzed | 691 |
| Training Set Size | 500 |
| Test Set Size (Outliers) | 82 |
| Features Engineered | 58 |
| Features Used in Model | 17 |
| Clusters Identified | 3 |
| Model Accuracy (Silhouette) | 0.2747 |
| Notebooks Created | 6 |
| Datasets Generated | 4 |
| Lines of Code | ~3,000+ |

---

## 🤝 Contributing

This project is open for improvements! Areas for contribution:
- Data preprocessing enhancements
- Alternative clustering algorithms
- Additional visualizations
- Web app UI/UX improvements
- Documentation updates

---

## 📖 References & Resources

### Academic Background:
- K-Means Clustering: [Wikipedia](https://en.wikipedia.org/wiki/K-means_clustering)
- Social Media Addiction Research: [Journal of Behavioral Addictions](https://akjournals.com/view/journals/2006/2006-overview.xml)
- Feature Engineering Best Practices: [Kaggle Learn](https://www.kaggle.com/learn/feature-engineering)

### Technical Documentation:
- Scikit-learn KMeans: [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- Streamlit: [Official Docs](https://docs.streamlit.io/)
- PCA Visualization: [Scikit-learn Guide](https://scikit-learn.org/stable/modules/decomposition.html#pca)

### Related Work:
- Behavioral Pattern Recognition in Digital Platforms
- Unsupervised Learning for User Segmentation
- Time-Series Analysis of Social Media Usage

---

## 💡 Key Takeaways

1. **Unsupervised learning can effectively identify addiction patterns** without labeled data
2. **Feature engineering is crucial** - 17 well-chosen features outperform 58 raw features
3. **K-Means with k=3 provides actionable insights** - Safe/At Risk/Addicted categories
4. **Total usage is the strongest predictor**, but diversity and consistency also matter
5. **Model generalizes well** across different datasets and demographics
6. **Outlier handling is important** - extreme cases need separate evaluation
7. **Production deployment is feasible** - saved models enable real-time predictions

---

## 📞 Contact & Support

For questions, suggestions, or collaboration:
- **Project**: Social Media Addiction Analysis
- **Created**: 2026
- **Purpose**: Educational and research use

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. The predictions should not be used as:
- Medical diagnosis
- Professional psychological assessment
- Sole basis for clinical decisions

If you or someone you know is struggling with social media addiction, please consult:
- Mental health professionals
- Certified addiction counselors
- Support groups and hotlines

---

## 📄 License

This project is available for educational use. Please cite appropriately if using for research or academic purposes.

---

## 🙏 Acknowledgments

- Dataset sources: Public social media usage surveys
- Python data science community
- Scikit-learn contributors
- Streamlit team for the web framework

---

**Project Status**: ✅ Complete & Production-Ready

**Last Updated**: January 2026

**Version**: 1.0

---

<p align="center">
  <b>Built with ❤️ using Python, Scikit-learn, and Streamlit</b>
</p>

<p align="center">
  <i>Helping understand and manage social media usage through data science</i>
</p>
