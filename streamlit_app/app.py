# basic_app.py - Learn Streamlit fundamentals
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Credit Risk Analyzer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and header
st.title("🏦 Credit Risk Assessment Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Home", "Data Explorer", "Model Demo"])

if page == "Home":
    st.header("Welcome to Credit Risk Assessment")
    
    # Columns layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Key Features")
        st.write("""
        - **Real-time Risk Assessment**
        - **SHAP Explanations**
        - **Interactive Visualizations**
        - **Batch Processing**
        """)
    
    with col2:
        st.subheader("🎯 Model Performance")
        # Create some sample metrics
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Accuracy", "77%", "2%")
        col_b.metric("Precision", "48%", "-1%")
        col_c.metric("Recall", "52%", "3%")

elif page == "Data Explorer":
    st.header("📈 Data Exploration")
    
    # Sample data generation
    @st.cache_data
    def generate_sample_data():
        np.random.seed(42)
        data = {
            'Age': np.random.normal(35, 10, 1000),
            'Income': np.random.normal(50000, 15000, 1000),
            'Credit_Score': np.random.normal(650, 100, 1000),
            'Loan_Amount': np.random.normal(25000, 10000, 1000),
            'Default': np.random.choice([0, 1], 1000, p=[0.78, 0.22])
        }
        return pd.DataFrame(data)
    
    df = generate_sample_data()
    
    # Data display
    st.subheader("Sample Data")
    st.dataframe(df.head(10))
    
    # Interactive plotting
    st.subheader("Interactive Visualizations")
    
    # Feature selection
    feature_x = st.selectbox("Select X-axis", df.columns[:-1])
    feature_y = st.selectbox("Select Y-axis", df.columns[:-1])
    
    # Scatter plot
    fig = px.scatter(df, x=feature_x, y=feature_y, color='Default',
                    title=f"{feature_x} vs {feature_y}")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Model Demo":
    st.header("🤖 Credit Risk Model Demo")
    
    # Input form
    st.subheader("Enter Borrower Information")
    
    with st.form("credit_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 80, 35)
            income = st.number_input("Annual Income ($)", 0, 200000, 50000)
            credit_score = st.slider("Credit Score", 300, 850, 650)
        
        with col2:
            loan_amount = st.number_input("Loan Amount ($)", 0, 100000, 25000)
            employment_years = st.slider("Years of Employment", 0, 40, 5)
            debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
        
        submitted = st.form_submit_button("Calculate Risk Score")
    
    if submitted:
        # Mock prediction (replace with your actual model)
        # Simple risk calculation for demo
        risk_score = (
            (850 - credit_score) * 0.4 + 
            (debt_to_income * 100) * 0.3 + 
            (loan_amount / income) * 0.3
        ) / 100
        
        risk_score = max(0, min(1, risk_score))  # Clamp between 0 and 1
        
        # Results display
        st.subheader("Risk Assessment Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Score", f"{risk_score:.2%}")
        
        with col2:
            risk_level = "Low" if risk_score < 0.3 else "Medium" if risk_score < 0.7 else "High"
            st.metric("Risk Level", risk_level)
        
        with col3:
            approval = "Approved" if risk_score < 0.5 else "Rejected"
            st.metric("Decision", approval)
        
        # Risk gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Default Risk %"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")