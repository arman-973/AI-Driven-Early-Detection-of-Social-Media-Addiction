import streamlit as st
import pandas as pd
import numpy as np
import pickle
import altair as alt
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Social Media Addiction Predictor",
    page_icon="📱",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px;
        border-radius: 5px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .safe {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .at-risk {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }
    .addicted {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)

# Load trained model and components
@st.cache_resource
def load_model():
    try:
        model_path = r'Experimental/my_kmeans_model.pkl'
        scaler_path = r'Experimental/my_scaler.pkl'
        mapping_path = r'Experimental/my_cluster_mapping.pkl'
        
        with open(model_path, 'rb') as f:
            kmeans = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(mapping_path, 'rb') as f:
            mapping_data = pickle.load(f)
        
        return kmeans, scaler, mapping_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Title
st.title("📱 Social Media Addiction Prediction System")
st.markdown("---")

# Sidebar for information
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    This app predicts your social media addiction risk level based on:
    - Demographics (Age, Gender, Occupation)
    - Weekly usage patterns across 6 platforms
    - Usage consistency and patterns
    """)
    
    st.header("📊 Risk Categories")
    st.markdown("""
    🟢 **Safe**: Low to moderate usage  
    🟡 **At Risk**: High usage, potential addiction  
    🔴 **Addicted**: Excessive usage, high risk
    """)

# Load model
kmeans, scaler, mapping_data = load_model()

if kmeans is None:
    st.error("Failed to load model. Please ensure model files exist in 'Experimental' folder.")
    st.stop()

# Main input form
st.header("👤 User Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=25, help="Your age in years")
    
with col2:
    gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender")
    
with col3:
    occupation = st.selectbox(
        "Occupation", 
        ["Student", "Employed", "Unemployed", "Businessperson", "Worker", "Driver"],
        help="Your current occupation"
    )

st.markdown("---")
st.header("📱 Social Media Usage Input")

# Define platforms and weeks
platforms = ['Facebook', 'Instagram', 'YouTube', 'TikTok', 'Messenger', 'WhatsApp']
weeks = ['Week 1', 'Week 2', 'Week 3', 'Week 4']

# Toggle between Daily and Weekly input
usage_mode = st.radio(
    "Select Input Mode:",
    ["Daily Usage (total hours for a day in HH:MM format)", "Weekly Usage (total hours for 7 days in HH:MM format)"],
    horizontal=True
)

# Helper function to parse time input
def parse_time_input(time_str):
    """Convert HH:MM, HH;MM, or HH,MM to total minutes"""
    try:
        # Replace semicolon and comma with colon
        time_str = str(time_str).strip().replace(';', ':').replace(',', ':')
        
        # Split by colon
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 2:
                hours = int(parts[0])
                minutes = int(parts[1])
                return hours * 60 + minutes
        
        # If no colon, assume it's just minutes
        return int(float(time_str))
    except:
        return 0

# Dictionary to store usage for one week
one_week_usage = {}

if usage_mode == "Daily Usage (total hours for a day in HH:MM format)":
    st.markdown("*Enter your average daily usage in HH:MM format (e.g., 1:30 for 1 hour 30 minutes per day)*")
    st.info("💡 You can use colon (:), semicolon (;), or comma (,) as separator. Example: 1:30 or 1;30 or 1,30")
    st.subheader("⏱️ Daily Usage (HH:MM per day)")
    
    cols = st.columns(3)
    
    for platform_idx, platform in enumerate(platforms):
        col = cols[platform_idx % 3]
        with col:
            time_input = st.text_input(
                f"{platform}",
                value="0:30",
                key=f"daily_{platform}",
                help=f"Average daily usage for {platform} (e.g., 1:30 for 1h 30min per day)",
                placeholder="HH:MM or HH;MM or HH,MM"
            )
            
            # Parse to get daily minutes (already daily, no division needed)
            daily_minutes = parse_time_input(time_input)
            one_week_usage[platform] = daily_minutes
            
            # Show converted value
            col.caption(f"📊 {daily_minutes:.0f} min/day ({daily_minutes * 7:.0f} min/week)")

else:  # Weekly Usage in HH:MM format
    st.markdown("*Enter total weekly usage (7 days) in HH:MM format (e.g., 5:30 for 5 hours 30 minutes per week)*")
    st.info("💡 You can use colon (:), semicolon (;), or comma (,) as separator. Example: 5:30 or 5;30 or 5,30")
    st.subheader("⏱️ Weekly Usage (HH:MM for 7 days total)")
    
    cols = st.columns(3)
    
    for platform_idx, platform in enumerate(platforms):
        col = cols[platform_idx % 3]
        with col:
            time_input = st.text_input(
                f"{platform}",
                value="0:30",
                key=f"weekly_{platform}",
                help=f"Total weekly usage for {platform} (e.g., 5:30 for 5h 30min per week)",
                placeholder="HH:MM or HH;MM or HH,MM"
            )
            
            # Parse and convert to daily average
            weekly_minutes = parse_time_input(time_input)
            daily_minutes = weekly_minutes / 7  # Convert weekly to daily average
            one_week_usage[platform] = daily_minutes
            
            # Show converted value
            col.caption(f"📊 ≈ {daily_minutes:.0f} min/day ({weekly_minutes:.0f} min/week)")

# Replicate for all 4 weeks (assuming consistent usage)
usage_data = {}
for week in weeks:
    for platform in platforms:
        key = f"{platform} {week}"
        usage_data[key] = one_week_usage[platform]

st.markdown("---")

# Predict button
if st.button("🔮 Predict Addiction Risk", type="primary"):
    
    with st.spinner("Analyzing your usage patterns..."):
        
        # Calculate monthly totals for each platform (daily average * 30 days)
        monthly_totals = {}
        for platform in platforms:
            monthly_total = one_week_usage[platform] * 30  # Convert daily to monthly
            monthly_totals[f"{platform}_Monthly_Total"] = monthly_total
        
        # Calculate aggregate features
        total_monthly_usage = sum(monthly_totals.values())
        avg_platform_usage = total_monthly_usage / len(platforms)
        
        # Calculate active platforms count
        active_platforms_count = sum(1 for usage in monthly_totals.values() if usage > 0)
        
        # Calculate platform diversity score (entropy-based)
        if total_monthly_usage > 0:
            proportions = [usage / total_monthly_usage for usage in monthly_totals.values() if usage > 0]
            platform_diversity_score = -sum(p * np.log(p) for p in proportions if p > 0)
        else:
            platform_diversity_score = 0
        
        # Calculate dominant platform percentage
        if total_monthly_usage > 0:
            dominant_platform_percentage = max(monthly_totals.values()) / total_monthly_usage
        else:
            dominant_platform_percentage = 0
        
        # Calculate usage concentration (1 - diversity)
        usage_concentration = 1 - (platform_diversity_score / np.log(len(platforms)) if platform_diversity_score > 0 else 0)
        
        # Calculate category-based usage
        social_networking_usage = monthly_totals['Facebook_Monthly_Total'] + monthly_totals['Instagram_Monthly_Total'] + monthly_totals['TikTok_Monthly_Total']
        messaging_usage = monthly_totals['Messenger_Monthly_Total'] + monthly_totals['WhatsApp_Monthly_Total']
        video_platform_usage = monthly_totals['YouTube_Monthly_Total']
        
        # Calculate social to messaging ratio
        if messaging_usage > 0:
            social_to_messaging_ratio = social_networking_usage / messaging_usage
        else:
            social_to_messaging_ratio = social_networking_usage
        
        # Calculate overall consistency (assuming consistent usage pattern, set to 0.8 as default)
        overall_consistency = 0.8
        
        # Create feature dictionary matching the exact 18 training features
        all_features = {
            'Age': age,
            'Total_Monthly_Usage': total_monthly_usage,
            'Avg_Platform_Usage': avg_platform_usage,
            'Active_Platforms_Count': active_platforms_count,
            'Platform_Diversity_Score': platform_diversity_score,
            'Dominant_Platform_Percentage': dominant_platform_percentage,
            'Usage_Concentration': usage_concentration,
            'Social_Networking_Usage': social_networking_usage,
            'Messaging_Usage': messaging_usage,
            'Video_Platform_Usage': video_platform_usage,
            'Social_to_Messaging_Ratio': social_to_messaging_ratio,
            'Overall_Consistency': overall_consistency,
            'YouTube_Monthly_Total': monthly_totals['YouTube_Monthly_Total'],
            'WhatsApp_Monthly_Total': monthly_totals['WhatsApp_Monthly_Total'],
            'Instagram_Monthly_Total': monthly_totals['Instagram_Monthly_Total'],
            'Facebook_Monthly_Total': monthly_totals['Facebook_Monthly_Total'],
            'Messenger_Monthly_Total': monthly_totals['Messenger_Monthly_Total'],
            'TikTok_Monthly_Total': monthly_totals['TikTok_Monthly_Total']
        }
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([all_features])
        
        # Reorder columns to match training features
        expected_features = mapping_data['features']
        
        # Select only the expected features in correct order
        X_new = feature_df[expected_features].values
        
        # Scale the features
        X_scaled = scaler.transform(X_new)
        
        # Predict cluster
        cluster_pred = kmeans.predict(X_scaled)[0]
        
        # Map to target class
        target_pred = mapping_data['cluster_mapping'][cluster_pred]
        predicted_class = mapping_data['label_names'][target_pred]
        
        # Display results
        st.markdown("---")
        st.header("🎯 Prediction Results")
        
        # Prediction box with color coding
        if target_pred == 0:
            box_class = "safe"
            emoji = "🟢"
            color = "#28a745"
        elif target_pred == 1:
            box_class = "at-risk"
            emoji = "🟡"
            color = "#ffc107"
        else:
            box_class = "addicted"
            emoji = "🔴"
            color = "#dc3545"
        
        st.markdown(f"""
        <div class="prediction-box {box_class}">
            <h1>{emoji} {predicted_class.upper()}</h1>
            <h3>Your addiction risk level</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Display usage statistics
        st.markdown("---")
        st.header("📊 Your Usage Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Monthly Usage", f"{total_monthly_usage:,.0f} min", 
                     delta=f"{total_monthly_usage/60:.1f} hours")
        
        with col2:
            st.metric("Daily Average", f"{total_monthly_usage/30:.0f} min", 
                     delta=f"{total_monthly_usage/30/60:.1f} hours")
        
        with col3:
            st.metric("Weekly Average", f"{total_monthly_usage/4:.0f} min",
                     delta=f"{total_monthly_usage/4/60:.1f} hours")
        
        with col4:
            st.metric("Usage Consistency", f"{overall_consistency:.2f}",
                     delta="Lower is more variable")
        
        # Platform breakdown
        st.markdown("---")
        st.subheader("📱 Platform Breakdown")
        
        # Create DataFrame for visualization
        platform_df = pd.DataFrame({
            'Platform': platforms,
            'Monthly Usage (min)': [monthly_totals[f"{p}_Monthly_Total"] for p in platforms],
            'Daily Average (min)': [monthly_totals[f"{p}_Monthly_Total"]/30 for p in platforms],
            'Percentage': [monthly_totals[f"{p}_Monthly_Total"]/total_monthly_usage*100 for p in platforms]
        })
        
        # Display as bar chart with platform colors matched to official logos
        chart = alt.Chart(platform_df).mark_bar().encode(
            x=alt.X('Platform:N', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Monthly Usage (min):Q', title='Monthly Usage (min)'),
            color=alt.Color(
                'Platform:N',
                scale=alt.Scale(
                    domain=['Facebook', 'Instagram', 'YouTube', 'TikTok', 'Messenger', 'WhatsApp'],
                    range=['#1877F2', '#E1306C', '#FF0000', '#69C9D0', '#0084FF', '#25D366']
                ),
                legend=None
            ),
            tooltip=[alt.Tooltip('Platform:N'), alt.Tooltip('Monthly Usage (min):Q')]
        ).properties(height=360)
        st.altair_chart(chart, use_container_width=True)
        
        # Display table
        st.dataframe(
            platform_df.style.format({
                'Monthly Usage (min)': '{:.0f}',
                'Daily Average (min)': '{:.1f}',
                'Percentage': '{:.1f}%'
            }),
            use_container_width=True
        )
        
        # Display additional metrics
        st.markdown("---")
        st.subheader("📈 Advanced Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Active Platforms", f"{int(all_features['Active_Platforms_Count'])}/6")
            st.metric("Platform Diversity", f"{all_features['Platform_Diversity_Score']:.2f}")
        
        with col2:
            st.metric("Usage Concentration", f"{all_features['Usage_Concentration']:.2f}")
            st.metric("Dominant Platform %", f"{all_features['Dominant_Platform_Percentage']:.1%}")
        
        with col3:
            st.metric("Social Networking", f"{all_features['Social_Networking_Usage']:.0f} min")
            st.metric("Messaging", f"{all_features['Messaging_Usage']:.0f} min")
        
        # Recommendations
        st.markdown("---")
        st.header("💡 Recommendations")
        
        if target_pred == 0:
            st.success("""
            ✅ **Great job!** Your social media usage is within healthy limits.
            
            **Tips to maintain healthy habits:**
            - Continue monitoring your usage
            - Set time limits for each app
            - Take regular breaks
            - Engage in offline activities
            """)
        
        elif target_pred == 1:
            st.warning("""
            ⚠️ **Warning!** You're at risk of social media addiction.
            
            **Recommendations:**
            - Reduce daily usage by 20-30%
            - Set app time limits
            - Schedule social media-free hours
            - Seek support if needed
            - Practice mindful usage
            """)
        
        else:
            st.error("""
            🚨 **Alert!** Your usage indicates potential addiction.
            
            **Urgent Actions:**
            - Consider professional help
            - Set strict daily limits (max 2-3 hours)
            - Use app blockers
            - Replace social media time with other activities
            - Join support groups
            - Monitor your progress weekly
            """)
        
        # Cluster information
        with st.expander("🔍 Technical Details"):
            st.write(f"**Cluster ID:** {cluster_pred}")
            st.write(f"**Target Value:** {target_pred}")
            st.write(f"**Features Used:** {len(expected_features)} (matching training data)")
            st.write("**Model:** K-Means Clustering")
            st.write("**Training Features:**")
            st.write(", ".join(expected_features))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>📱 Social Media Addiction Prediction System | Built with Streamlit</p>
    <p>⚠️ This is a prediction tool and should not replace professional medical advice.</p>
</div>
""", unsafe_allow_html=True)
