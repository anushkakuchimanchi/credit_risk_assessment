# credit_app.py - Main credit scoring application
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Credit Risk Analyzer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load your trained model (you'll need to save it first)
@st.cache_resource
def load_model():
    """Load the trained XGBoost model and preprocessors"""
    try:
        # Replace these with your actual saved model files
        # model = pickle.load(open('models/xgb_model.pkl', 'rb'))
        # scaler = pickle.load(open('models/scaler.pkl', 'rb'))
        # return model, scaler
        
        # For now, return None (you'll implement this after saving your model)
        return None, None
    except:
        return None, None

# Load SHAP explainer
@st.cache_resource
def load_shap_explainer():
    """Load pre-computed SHAP explainer"""
    try:
        # explainer = pickle.load(open('models/shap_explainer.pkl', 'rb'))
        # return explainer
        return None
    except:
        return None

# Main app
def main():
    st.title("🏦 Credit Risk Assessment System")
    st.markdown("Advanced ML-powered credit scoring with explainable AI")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "🏠 Home", 
        "🔍 Individual Assessment", 
        "📊 Batch Processing", 
        "📈 Model Analytics"
    ])
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔍 Individual Assessment":
        show_individual_assessment()
    elif page == "📊 Batch Processing":
        show_batch_processing()
    elif page == "📈 Model Analytics":
        show_model_analytics()

def show_home_page():
    """Home page with overview and instructions"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 System Overview")
        st.write("""
        This Credit Risk Assessment System uses advanced machine learning to evaluate 
        loan default probability. Built with XGBoost and enhanced with SHAP explanations 
        for complete transparency.
        
        **Key Features:**
        - **Real-time Assessment**: Instant credit risk evaluation
        - **Explainable AI**: SHAP-powered decision explanations
        - **Batch Processing**: Evaluate multiple applications at once
        - **Interactive Analytics**: Explore model performance and insights
        """)
        
        st.subheader("📊 Model Performance")
        
        # Performance metrics (replace with your actual results)
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Accuracy", "77%", "2%")
        col_b.metric("Precision", "48%", "-1%")
        col_c.metric("Recall", "52%", "3%")
        col_d.metric("F1-Score", "50%", "1%")
    
    with col2:
        st.subheader("🚀 Quick Start")
        st.info("""
        **Step 1:** Click on 'Individual Assessment' to evaluate a single borrower
        
        **Step 2:** Fill in the borrower information
        
        **Step 3:** Get instant risk score and explanation
        
        **Step 4:** Use insights to make informed decisions
        """)
        
        st.subheader("📋 Requirements")
        st.write("""
        **Required Information:**
        - Personal details (age, income)
        - Credit history information
        - Employment details
        - Loan specifics
        """)

def show_individual_assessment():
    """Individual borrower assessment page"""
    
    st.subheader("🔍 Individual Credit Risk Assessment")
    st.write("Enter borrower information to get instant risk evaluation")
    
    # Input form
    with st.form("borrower_form"):
        st.subheader("📝 Borrower Information")
        
        # Personal Information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Personal Details**")
            age = st.slider("Age", 18, 80, 35)
            annual_income = st.number_input("Annual Income ($)", 0, 500000, 50000)
            employment_length = st.slider("Employment Length (years)", 0, 40, 5)
        
        with col2:
            st.write("**Credit Information**")
            credit_score = st.slider("Credit Score", 300, 850, 650)
            credit_history_length = st.slider("Credit History (years)", 0, 50, 10)
            existing_loans = st.number_input("Number of Existing Loans", 0, 20, 2)
        
        with col3:
            st.write("**Loan Details**")
            loan_amount = st.number_input("Requested Loan Amount ($)", 1000, 100000, 25000)
            loan_purpose = st.selectbox("Loan Purpose", [
                "debt_consolidation", "home_improvement", "major_purchase", 
                "medical", "vacation", "wedding", "other"
            ])
            debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
        
        submitted = st.form_submit_button("🚀 Assess Credit Risk", type="primary")
    
    if submitted:
        # Create input data
        input_data = {
            'age': age,
            'annual_income': annual_income,
            'employment_length': employment_length,
            'credit_score': credit_score,
            'credit_history_length': credit_history_length,
            'existing_loans': existing_loans,
            'loan_amount': loan_amount,
            'debt_to_income': debt_to_income,
            'loan_purpose': loan_purpose
        }
        
        # Make prediction (mock for now - replace with your actual model)
        risk_score, explanation = make_prediction(input_data)
        
        # Display results
        display_results(risk_score, explanation, input_data)

def make_prediction(input_data):
    """Make prediction using the trained model"""
    
    # Mock prediction for demo (replace with your actual model prediction)
    # This simulates what your actual model would return
    
    # Simple risk calculation for demo
    risk_factors = [
        (850 - input_data['credit_score']) / 550,  # Higher score = lower risk
        input_data['debt_to_income'],  # Higher ratio = higher risk
        min(input_data['loan_amount'] / input_data['annual_income'], 2) / 2,  # Higher ratio = higher risk
        max(0, (40 - input_data['age']) / 40),  # Younger = slightly higher risk
        max(0, (10 - input_data['employment_length']) / 10)  # Less employment = higher risk
    ]
    
    risk_score = sum(risk_factors) / len(risk_factors)
    risk_score = max(0, min(1, risk_score))
    
    # Mock SHAP explanation
    explanation = {
        'credit_score': -0.15 if input_data['credit_score'] > 650 else 0.25,
        'debt_to_income': input_data['debt_to_income'] * 0.3,
        'age': -0.05 if input_data['age'] > 30 else 0.1,
        'annual_income': -0.1 if input_data['annual_income'] > 40000 else 0.15,
        'loan_amount': (input_data['loan_amount'] / 50000) * 0.2
    }
    
    return risk_score, explanation

def display_results(risk_score, explanation, input_data):
    """Display prediction results and explanations"""
    
    st.markdown("---")
    st.subheader("📊 Assessment Results")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Default Risk", f"{risk_score:.1%}")
    
    with col2:
        risk_level = "Low" if risk_score < 0.3 else "Medium" if risk_score < 0.7 else "High"
        color = "🟢" if risk_level == "Low" else "🟡" if risk_level == "Medium" else "🔴"
        st.metric("Risk Level", f"{color} {risk_level}")
    
    with col3:
        decision = "Approved" if risk_score < 0.5 else "Review Required" if risk_score < 0.75 else "Declined"
        st.metric("Recommendation", decision)
    
    with col4:
        confidence = min(95, 70 + (1 - abs(risk_score - 0.5)) * 50)
        st.metric("Confidence", f"{confidence:.0f}%")
    
    # Risk gauge
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Default Risk (%)"},
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
                    'value': 80}}))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Decision Factors")
        
        # Create explanation DataFrame
        exp_df = pd.DataFrame([
            {"Factor": k.replace('_', ' ').title(), "Impact": v, "Direction": "↗️ Increases Risk" if v > 0 else "↘️ Decreases Risk"}
            for k, v in explanation.items()
        ]).sort_values('Impact', key=abs, ascending=False)
        
        for _, row in exp_df.head(5).iterrows():
            impact_color = "red" if row['Impact'] > 0 else "green"
            st.markdown(f"**{row['Factor']}**: <span style='color:{impact_color}'>{row['Direction']}</span>", unsafe_allow_html=True)
    
    # Detailed breakdown
    with st.expander("📋 Detailed Assessment Breakdown"):
        st.subheader("Input Summary")
        summary_df = pd.DataFrame([
            {"Attribute": k.replace('_', ' ').title(), "Value": v}
            for k, v in input_data.items()
        ])
        st.dataframe(summary_df, use_container_width=True)

def show_batch_processing():
    """Batch processing page for multiple assessments"""
    
    st.subheader("📊 Batch Credit Assessment")
    st.write("Upload a CSV file to assess multiple borrowers at once")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV file with borrower data",
        type=['csv'],
        help="CSV should contain columns: age, annual_income, credit_score, loan_amount, etc."
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! {len(df)} records found.")
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            if st.button("🚀 Process All Applications"):
                # Mock batch processing
                progress_bar = st.progress(0)
                results = []
                
                for i, row in df.iterrows():
                    # Mock prediction for each row
                    risk_score = np.random.random()  # Replace with actual prediction
                    decision = "Approved" if risk_score < 0.5 else "Review" if risk_score < 0.75 else "Declined"
                    
                    results.append({
                        'Index': i,
                        'Risk_Score': risk_score,
                        'Decision': decision
                    })
                    
                    progress_bar.progress((i + 1) / len(df))
                
                # Display results
                results_df = pd.DataFrame(results)
                combined_df = pd.concat([df, results_df[['Risk_Score', 'Decision']]], axis=1)
                
                st.subheader("📊 Processing Results")
                st.dataframe(combined_df)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    approved = len(results_df[results_df['Decision'] == 'Approved'])
                    st.metric("Approved", approved)
                
                with col2:
                    review = len(results_df[results_df['Decision'] == 'Review'])
                    st.metric("Review Required", review)
                
                with col3:
                    declined = len(results_df[results_df['Decision'] == 'Declined'])
                    st.metric("Declined", declined)
                
                # Download results
                csv = combined_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="credit_assessment_results.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

def show_model_analytics():
    """Model analytics and performance page"""
    
    st.subheader("📈 Model Performance Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Feature Importance", "Model Insights"])
    
    with tab1:
        st.subheader("🎯 Performance Metrics")
        
        # Create mock performance data
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Value': [0.77, 0.48, 0.52, 0.50, 0.74],
            'Benchmark': [0.75, 0.45, 0.55, 0.50, 0.70]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        
        fig = px.bar(metrics_df, x='Metric', y=['Value', 'Benchmark'], 
                    title="Model Performance vs Benchmark", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        confusion_data = np.array([[3926, 747], [637, 690]])
        
        fig = px.imshow(confusion_data, 
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['No Default', 'Default'],
                       y=['No Default', 'Default'],
                       title="Confusion Matrix")
        fig.update_layout(width=500, height=400)
        st.plotly_chart(fig)
    
    with tab2:
        st.subheader("🔍 Feature Importance Analysis")
        
        # Mock feature importance data
        feature_data = {
            'Feature': ['Credit Score', 'Debt-to-Income', 'Annual Income', 'Age', 
                       'Employment Length', 'Loan Amount', 'Credit History'],
            'Importance': [0.35, 0.25, 0.15, 0.10, 0.08, 0.05, 0.02]
        }
        
        feature_df = pd.DataFrame(feature_data).sort_values('Importance', ascending=True)
        
        fig = px.bar(feature_df, x='Importance', y='Feature', orientation='h',
                    title="XGBoost Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("💡 Model Insights & Recommendations")
        
        st.write("""
        **Key Findings:**
        - Credit Score is the most predictive feature (35% importance)
        - Debt-to-Income ratio significantly impacts risk assessment
        - Age and employment length provide moderate predictive power
        
        **Business Recommendations:**
        1. Focus data collection efforts on credit score accuracy
        2. Implement stricter debt-to-income ratio thresholds
        3. Consider additional employment verification for younger applicants
        4. Regular model retraining recommended every 3-6 months
        """)

# Run the app
if __name__ == "__main__":
    main()