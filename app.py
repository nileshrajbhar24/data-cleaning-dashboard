import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import warnings
warnings.filterwarnings('ignore')

# Initialize session state for data
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None

# Set page configuration
st.set_page_config(
    page_title="Advanced Data Cleaning Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Apply fonts to all elements */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Main header styling */
    .main-header {
        font-size: 2.8rem;
        color: #1a365d;
        text-align: center;
        margin-bottom: 2rem;
        padding: 25px;
        background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%);
        border-radius: 16px;
        border-left: 6px solid #3182ce;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
        font-weight: 700;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sub-header styling */
    .sub-header {
        font-size: 1.9rem;
        color: #2c5282;
        margin-bottom: 1.5rem;
        padding-bottom: 0.8rem;
        border-bottom: 3px solid #3182ce;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.02em;
    }
    
    /* Highlight boxes */
    .highlight {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 22px;
        border-radius: 12px;
        margin-bottom: 22px;
        border-left: 5px solid #4299e1;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Status colors */
    .success {
        color: #2f855a;
        font-weight: 600;
        background-color: rgba(72, 187, 120, 0.12);
        padding: 10px 16px;
        border-radius: 8px;
        border-left: 4px solid #38a169;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
    }
    
    .warning {
        color: #c05621;
        font-weight: 600;
        background-color: rgba(237, 137, 54, 0.12);
        padding: 10px 16px;
        border-radius: 8px;
        border-left: 4px solid #ed8936;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
    }
    
    .danger {
        color: #c53030;
        font-weight: 600;
        background-color: rgba(245, 101, 101, 0.12);
        padding: 10px 16px;
        border-radius: 8px;
        border-left: 4px solid #f56565;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%);
        padding: 22px;
        border-radius: 12px;
        margin: 18px 0;
        border-left: 5px solid #3182ce;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.06);
        font-family: 'Inter', sans-serif;
    }
    
    /* Metric cards */
    .financial-metric {
        background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
        padding: 22px;
        border-radius: 12px;
        margin: 15px 0;
        border-left: 5px solid #4299e1;
        box-shadow: 0 5px 10px rgba(0, 0, 0, 0.07);
        transition: all 0.3s ease;
        border: 1px solid #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    .financial-metric:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.12);
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #3182ce 0%, #2c5282 100%);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.85rem 1.7rem;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        letter-spacing: 0.02em;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #2c5282 0%, #2a4365 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Streamlit specific overrides */
    .stApp {
        background-color: #f8fafc;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.08);
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 25px;
        background: #f7fafc;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #4299e1;
        background: #ebf8ff;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
        font-family: 'Inter', sans-serif;
    }
    
    .dataframe thead th {
        background: linear-gradient(135deg, #3182ce 0%, #2c5282 100%);
        color: white;
        font-weight: 600;
        padding: 14px;
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom badge styling */
    .custom-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0 5px;
        font-family: 'Inter', sans-serif;
    }
    
    .badge-primary {
        background-color: rgba(49, 130, 206, 0.15);
        color: #3182ce;
        border: 1px solid rgba(49, 130, 206, 0.3);
    }
    
    .badge-success {
        background-color: rgba(72, 187, 120, 0.15);
        color: #48bb78;
        border: 1px solid rgba(72, 187, 120, 0.3);
    }
    
    .badge-warning {
        background-color: rgba(237, 137, 54, 0.15);
        color: #ed8936;
        border: 1px solid rgba(237, 137, 54, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# Main app
st.markdown('<h1 class="main-header">🧾 Advanced Data Cleaning & Analysis Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

def clean_duplicate_columns(df):
    """Rename duplicate columns to make them unique"""
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [
            f'{dup}_{i}' if i != 0 else dup for i in range(sum(cols == dup))
        ]
    df.columns = cols
    return df

def handle_file_upload(uploaded_file):
    """Handle file upload with proper error handling for duplicate columns"""
    try:
        if uploaded_file.name.endswith('.csv'):
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            # Clean duplicate column names
            df = clean_duplicate_columns(df)
        else:
            # Read Excel file
            df = pd.read_excel(uploaded_file)
            df = clean_duplicate_columns(df)
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        return None
# =========================================

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your data (CSV/Excel)", type=["csv", "xlsx"])

# Initialize session state for data
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'data_quality_report' not in st.session_state:
    st.session_state.data_quality_report = None

# Function to generate data quality report
def generate_data_quality_report(df):
    report = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values,
        'Total Values': [df.shape[0]] * len(df.columns),
        'Missing Values': df.isnull().sum().values,
        'Missing %': (df.isnull().sum().values / df.shape[0] * 100).round(2),
        'Unique Values': df.nunique().values,
        'Duplicate Rows': [df.duplicated().sum()] * len(df.columns)
    })
    return report

# Function to detect anomalies
def detect_anomalies(df):
    anomalies = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].nunique() > 5:  # Only check columns with sufficient variation
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if not outliers.empty:
                anomalies[col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(df) * 100).round(2),
                    'examples': outliers[col].head(3).tolist()
                }
    
    return anomalies

# Function to generate summary statistics
def generate_summary(df):
    summary = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        summary[col] = {
            'Sum': df[col].sum(),
            'Mean': df[col].mean(),
            'Median': df[col].median(),
            'Std Dev': df[col].std(),
            'Min': df[col].min(),
            'Max': df[col].max()
        }
    
    return summary

# Main content area
if uploaded_file is not None:
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.session_state.original_data = df.copy()
        
        # Display original data
        st.markdown('<h2 class="sub-header">📋 Data Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            missing_values = df.isnull().sum().sum()
            st.metric("Missing Values", missing_values)
        with col4:
            duplicates = df.duplicated().sum()
            st.metric("Duplicate Rows", duplicates)
        
        # Generate and display data quality report
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown("**Data Quality Report**")
        quality_report = generate_data_quality_report(df)
        st.session_state.data_quality_report = quality_report
        st.dataframe(quality_report)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display first few rows of data
        st.markdown("**First 5 Rows of Data**")
        st.dataframe(df.head())
        
        # Data cleaning options
        st.markdown('<h2 class="sub-header">🧹 Data Cleaning Options</h2>', unsafe_allow_html=True)
        
        cleaning_tabs = st.tabs(["Basic Cleaning", "Missing Values", "Advanced Cleaning", "Data Validation"])
        
        with cleaning_tabs[0]:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Basic Data Cleaning Operations**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Remove Duplicate Rows"):
                    initial_rows = df.shape[0]
                    df = df.drop_duplicates()
                    removed_rows = initial_rows - df.shape[0]
                    st.success(f"Removed {removed_rows} duplicate rows")
                
                if st.button("Standardize Column Names"):
                    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
                    st.success("Column names standardized")
            
            with col2:
                remove_cols = st.multiselect("Select columns to remove", df.columns)
                if st.button("Remove Selected Columns") and remove_cols:
                    df = df.drop(columns=remove_cols)
                    st.success(f"Removed columns: {', '.join(remove_cols)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with cleaning_tabs[1]:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Missing Value Handling**")
            
            missing_cols = df.columns[df.isnull().any()].tolist()
            
            if missing_cols:
                selected_missing_col = st.selectbox("Select column to handle missing values", missing_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Remove Rows with Missing Values"):
                        initial_rows = df.shape[0]
                        df = df.dropna(subset=[selected_missing_col])
                        removed_rows = initial_rows - df.shape[0]
                        st.success(f"Removed {removed_rows} rows with missing values in {selected_missing_col}")
                
                with col2:
                    fill_method = st.selectbox(
                        "Fill method",
                        ["Mean", "Median", "Mode", "Forward Fill", "Backward Fill", "Custom Value"]
                    )
                    
                    if st.button(f"Fill with {fill_method}"):
                        if fill_method == "Mean":
                            df[selected_missing_col] = df[selected_missing_col].fillna(df[selected_missing_col].mean())
                        elif fill_method == "Median":
                            df[selected_missing_col] = df[selected_missing_col].fillna(df[selected_missing_col].median())
                        elif fill_method == "Mode":
                            df[selected_missing_col] = df[selected_missing_col].fillna(df[selected_missing_col].mode()[0])
                        elif fill_method == "Forward Fill":
                            df[selected_missing_col] = df[selected_missing_col].ffill()
                        elif fill_method == "Backward Fill":
                            df[selected_missing_col] = df[selected_missing_col].bfill()
                        else:
                            custom_val = st.text_input("Custom value", "0")
                            df[selected_missing_col] = df[selected_missing_col].fillna(custom_val)
                        
                        st.success(f"Filled missing values in {selected_missing_col} using {fill_method}")
            else:
                st.success("No missing values found in the dataset")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with cleaning_tabs[2]:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Advanced Data Cleaning**")
            
            # Detect and handle outliers
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                outlier_col = st.selectbox("Select column for outlier detection", numeric_cols)
                
                if st.button("Detect Outliers"):
                    anomalies = detect_anomalies(df[[outlier_col]])
                    if outlier_col in anomalies:
                        st.warning(f"Found {anomalies[outlier_col]['count']} outliers ({anomalies[outlier_col]['percentage']}%) in {outlier_col}")
                        st.write("Sample outliers:", anomalies[outlier_col]['examples'])
                        
                        if st.button("Cap Outliers (Winsorization)"):
                            Q1 = df[outlier_col].quantile(0.05)
                            Q3 = df[outlier_col].quantile(0.95)
                            df[outlier_col] = np.where(df[outlier_col] < Q1, Q1, df[outlier_col])
                            df[outlier_col] = np.where(df[outlier_col] > Q3, Q3, df[outlier_col])
                            st.success("Outliers capped using winsorization (5th and 95th percentiles)")
                    else:
                        st.success(f"No outliers detected in {outlier_col}")
            
            # Data type conversion
            st.markdown("**Data Type Conversion**")
            convert_col = st.selectbox("Select column to convert", df.columns)
            new_dtype = st.selectbox("Select new data type", ["Auto Detect", "Numeric", "String", "Date", "Category"])
            
            if st.button("Convert Data Type"):
                if new_dtype == "Auto Detect":
                    # Try to convert to numeric first
                    try:
                        df[convert_col] = pd.to_numeric(df[convert_col])
                        st.success(f"Converted {convert_col} to numeric")
                    except:
                        # Try to convert to datetime
                        try:
                            df[convert_col] = pd.to_datetime(df[convert_col])
                            st.success(f"Converted {convert_col} to datetime")
                        except:
                            st.info(f"Could not auto-convert {convert_col}, keeping as string")
                elif new_dtype == "Numeric":
                    df[convert_col] = pd.to_numeric(df[convert_col], errors='coerce')
                    st.success(f"Converted {convert_col} to numeric")
                elif new_dtype == "String":
                    df[convert_col] = df[convert_col].astype(str)
                    st.success(f"Converted {convert_col} to string")
                elif new_dtype == "Date":
                    df[convert_col] = pd.to_datetime(df[convert_col], errors='coerce')
                    st.success(f"Converted {convert_col} to date")
                elif new_dtype == "Category":
                    df[convert_col] = df[convert_col].astype('category')
                    st.success(f"Converted {convert_col} to category")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with cleaning_tabs[3]:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Data Validation Rules**")
            
            # Add validation rules for data
            st.markdown("**Data Validation**")
            
            # Negative values check
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            negative_check_col = st.selectbox("Select column to check for negative values", numeric_cols)
            
            if st.button("Check for Negative Values"):
                negative_count = (df[negative_check_col] < 0).sum()
                if negative_count > 0:
                    st.warning(f"Found {negative_count} negative values in {negative_check_col}")
                else:
                    st.success(f"No negative values found in {negative_check_col}")
            
            # Date validation
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                date_col = st.selectbox("Select date column for validation", date_cols)
                
                if st.button("Check Date Range and Validity"):
                    min_date = df[date_col].min()
                    max_date = df[date_col].max()
                    invalid_dates = df[date_col].isnull().sum()
                    
                    st.write(f"Date range: {min_date} to {max_date}")
                    if invalid_dates > 0:
                        st.warning(f"Found {invalid_dates} invalid or missing dates")
                    else:
                        st.success("All dates are valid")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Store cleaned data
        st.session_state.cleaned_data = df.copy()
        
        # Show cleaning summary
        st.markdown('<h2 class="sub-header">📊 Cleaning Summary</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="financial-metric">', unsafe_allow_html=True)
            st.markdown("**Before Cleaning**")
            st.write(f"- Rows: {st.session_state.original_data.shape[0]}")
            st.write(f"- Columns: {st.session_state.original_data.shape[1]}")
            st.write(f"- Missing Values: {st.session_state.original_data.isnull().sum().sum()}")
            st.write(f"- Duplicate Rows: {st.session_state.original_data.duplicated().sum()}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="financial-metric">', unsafe_allow_html=True)
            st.markdown("**After Cleaning**")
            st.write(f"- Rows: {st.session_state.cleaned_data.shape[0]}")
            st.write(f"- Columns: {st.session_state.cleaned_data.shape[1]}")
            st.write(f"- Missing Values: {st.session_state.cleaned_data.isnull().sum().sum()}")
            st.write(f"- Duplicate Rows: {st.session_state.cleaned_data.duplicated().sum()}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis Section
        st.markdown('<h2 class="sub-header">💹 Data Analysis</h2>', unsafe_allow_html=True)
        
        if st.session_state.cleaned_data is not None:
            df_clean = st.session_state.cleaned_data
            
            # Generate summary
            summary = generate_summary(df_clean)
            
            # Display metrics
            st.markdown("**Summary Statistics**")
            for col, metrics in summary.items():
                with st.expander(f"Metrics for {col}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sum", f"{metrics['Sum']:,.2f}")
                        st.metric("Mean", f"{metrics['Mean']:,.2f}")
                    with col2:
                        st.metric("Median", f"{metrics['Median']:,.2f}")
                        st.metric("Std Dev", f"{metrics['Std Dev']:,.2f}")
                    with col3:
                        st.metric("Min", f"{metrics['Min']:,.2f}")
                        st.metric("Max", f"{metrics['Max']:,.2f}")
            
            # Data Visualization
            st.markdown('<h2 class="sub-header">📈 Data Visualization</h2>', unsafe_allow_html=True)
            
            viz_tabs = st.tabs(["Distribution", "Trends", "Relationships", "Composition"])
            
            with viz_tabs[0]:
                st.markdown("**Distribution Analysis**")
                numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    dist_col = st.selectbox("Select column for distribution analysis", numeric_cols)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Histogram
                        fig = px.histogram(df_clean, x=dist_col, title=f"Distribution of {dist_col}")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Box plot
                        fig = px.box(df_clean, y=dist_col, title=f"Box Plot of {dist_col}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numeric columns available for distribution analysis")
            
            with viz_tabs[1]:
                st.markdown("**Trend Analysis**")
                date_cols = df_clean.select_dtypes(include=['datetime64']).columns
                numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                
                if len(date_cols) > 0 and len(numeric_cols) > 0:
                    trend_date_col = st.selectbox("Select date column", date_cols)
                    trend_value_col = st.selectbox("Select value column", numeric_cols)
                    
                    # Time series plot
                    trend_df = df_clean.groupby(trend_date_col)[trend_value_col].sum().reset_index()
                    fig = px.line(trend_df, x=trend_date_col, y=trend_value_col, 
                                 title=f"Trend of {trend_value_col} over Time")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need both date and numeric columns for trend analysis")
            
            with viz_tabs[2]:
                st.markdown("**Relationship Analysis**")
                numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    x_col = st.selectbox("Select X axis column", numeric_cols)
                    y_col = st.selectbox("Select Y axis column", numeric_cols)
                    
                    # Scatter plot
                    fig = px.scatter(df_clean, x=x_col, y=y_col, 
                                    title=f"Relationship between {x_col} and {y_col}",
                                    trendline="ols")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Correlation heatmap
                    st.markdown("**Correlation Matrix**")
                    corr_matrix = df_clean[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                                   title="Correlation Heatmap")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least two numeric columns for relationship analysis")
            
            with viz_tabs[3]:
                st.markdown("**Composition Analysis**")
                categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
                numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                
                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    category_col = st.selectbox("Select category column", categorical_cols)
                    value_col = st.selectbox("Select value column for composition", numeric_cols)
                    
                    # Pie chart
                    composition_df = df_clean.groupby(category_col)[value_col].sum().reset_index()
                    fig = px.pie(composition_df, values=value_col, names=category_col, 
                                title=f"Composition of {value_col} by {category_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Bar chart
                    fig = px.bar(composition_df, x=category_col, y=value_col, 
                                title=f"{value_col} by {category_col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need both categorical and numeric columns for composition analysis")
        
        # Download cleaned data
        st.markdown('<h2 class="sub-header">💾 Download Cleaned Data</h2>', unsafe_allow_html=True)
        
        # Convert cleaned dataframe to CSV
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="Download Cleaned CSV",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please make sure you've uploaded a valid CSV or Excel file.")

else:
    # Show instructions when no file is uploaded
    st.markdown("""
    <div class="highlight">
    <h3>Welcome to the Advanced Data Cleaning Dashboard!</h3>
    <p>This application helps you clean, analyze, and visualize data with advanced features.</p>
    
    <h4>How to use:</h4>
    <ol>
        <li>Upload your CSV or Excel file using the file uploader in the sidebar</li>
        <li>Review the data quality report to understand data issues</li>
        <li>Use the cleaning tabs to handle missing values, outliers, and data validation</li>
        <li>Explore the analysis section to understand key metrics</li>
        <li>Visualize your data using various chart types</li>
        <li>Download your cleaned data for further analysis</li>
    </ol>
    
    <h4>Features:</h4>
    <ul>
        <li><strong>Data Quality Assessment:</strong> Comprehensive report on missing values, data types, and duplicates</li>
        <li><strong>Advanced Cleaning:</strong> Handle missing values, outliers, and data type conversion</li>
        <li><strong>Data Validation:</strong> Negative value checks and date validation</li>
        <li><strong>Data Analysis:</strong> Key metrics and statistical summaries</li>
        <li><strong>Data Visualization:</strong> Interactive charts for distribution, trends, relationships, and composition</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Show sample data
    st.markdown('<h3 class="sub-header">Sample Data</h3>', unsafe_allow_html=True)
    
    # Generate sample data
    sample_data = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'Category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 100),
        'Description': ['Transaction ' + str(i) for i in range(100)],
        'Value1': np.random.uniform(0, 5000, 100).round(2),
        'Value2': np.random.uniform(0, 5000, 100).round(2),
        'Type': np.random.choice(['X', 'Y', 'Z'], 100)
    })
    
    # Introduce some missing values and duplicates for demonstration
    sample_data.iloc[5:8, 3] = np.nan
    sample_data.iloc[15:18, 4] = np.nan
    sample_data = pd.concat([sample_data, sample_data.iloc[10:15]], ignore_index=True)
    
    st.dataframe(sample_data.head(10))
    
    st.info("Upload your data to begin the cleaning and analysis process.")

    # ==================== DOWNLOAD SECTION ====================
st.markdown("---")
st.markdown('<h2 class="sub-header">💾 Download Cleaned Data</h2>', unsafe_allow_html=True)

# Check if we have cleaned data available for download
if st.session_state.cleaned_data is not None:
    # Convert the cleaned dataframe to CSV format
    cleaned_csv = st.session_state.cleaned_data.to_csv(index=False)
    
    # Create download button
    st.download_button(
        label="📥 Download Cleaned CSV",
        data=cleaned_csv,
        file_name="cleaned_data.csv",
        mime="text/csv",
        key="unique_download_button_key"
    )
    
    # Show success message
    st.success("✅ Your cleaned data is ready for download!")
    
    # Display file information
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.write("**Dataset Information:**")
    st.write(f"• **Rows:** {st.session_state.cleaned_data.shape[0]}")
    st.write(f"• **Columns:** {st.session_state.cleaned_data.shape[1]}")
    st.write(f"• **Missing Values:** {st.session_state.cleaned_data.isnull().sum().sum()}")
    st.markdown('</div>', unsafe_allow_html=True)
    
else:
    # Show warning if no cleaned data exists
    st.warning("Please clean your data first to enable download.")