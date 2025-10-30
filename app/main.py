import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prediction_helper import predict
import time

# Set the page configuration and title
st.set_page_config(
    page_title="Credit Risk Assessment System", 
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .risk-low {
        border-left-color: #28a745 !important;
        background-color: #d4edda;
    }
    .risk-medium {
        border-left-color: #ffc107 !important;
        background-color: #fff3cd;
    }
    .risk-high {
        border-left-color: #dc3545 !important;
        background-color: #f8d7da;
    }
</style>
""", unsafe_allow_html=True)

# Main title with emoji
st.markdown('<h1 class="main-header">🏦 Credit Risk Assessment System</h1>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Risk Assessment", "Model Information", "Risk Analytics"])

if page == "Risk Assessment":
    st.markdown('<h2 class="sub-header">📊 Loan Application Assessment</h2>', unsafe_allow_html=True)
    
    # Information box
    st.info("💡 Fill in the customer details below to get an instant credit risk assessment")
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["👤 Personal Info", "💰 Loan Details", "📈 Credit History"])
    
    with tab1:

        st.markdown("### 👤 Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input('🎂 Age', min_value=18, step=1, max_value=100, value=35, 
                                help="Customer's age in years")
            income = st.number_input('💵 Annual Income (₹)', min_value=0, value=1200000, step=50000,
                                   help="Customer's annual income in Indian Rupees")
        
        with col2:
            residence_type = st.selectbox('🏠 Residence Type', ['Owned', 'Rented', 'Mortgage'],
                                        help="Type of residence ownership")
            num_open_accounts = st.number_input('📊 Open Loan Accounts', min_value=1, max_value=4, step=1, value=2,
                                              help="Number of currently active loan accounts")
    
    with tab2:
        st.markdown("### 💰 Loan Information")
        col1, col2 = st.columns(2)
        
        with col1:
            loan_amount = st.number_input('💳 Loan Amount (₹)', min_value=0, value=2500000, step=100000,
                                        help="Requested loan amount in Indian Rupees")
            loan_tenure_months = st.number_input('📅 Loan Tenure (months)', min_value=6, step=1, max_value=360, value=36,
                                               help="Loan repayment period in months")
        
        with col2:
            loan_purpose = st.selectbox('🎯 Loan Purpose', ['Education', 'Home', 'Auto', 'Personal'],
                                      help="Primary purpose of the loan")
            loan_type = st.selectbox('🔒 Loan Type', ['Unsecured', 'Secured'],
                                   help="Whether the loan is secured by collateral")
        
        # Calculate and display loan-to-income ratio
        loan_to_income_ratio = loan_amount / income if income > 0 else 0
        st.markdown(f"**📊 Loan-to-Income Ratio: {loan_to_income_ratio:.2f}**")
        
        if loan_to_income_ratio > 5:
            st.warning("⚠️ High loan-to-income ratio detected. This may increase default risk.")
        elif loan_to_income_ratio > 3:
            st.info("ℹ️ Moderate loan-to-income ratio. Consider income verification.")
        else:
            st.success("✅ Healthy loan-to-income ratio.")
    
    with tab3:
        st.markdown("### 📈 Credit History")
        col1, col2 = st.columns(2)
        
        with col1:
            avg_dpd_per_delinquency = st.number_input('📉 Average Days Past Due', min_value=0, value=15, step=1,
                                                    help="Average number of days past due for previous payments")
            delinquency_ratio = st.number_input('📊 Delinquency Ratio (%)', min_value=0, max_value=100, step=1, value=25,
                                              help="Percentage of payments that were delinquent")
        
        with col2:
            credit_utilization_ratio = st.number_input('💳 Credit Utilization Ratio (%)', min_value=0, max_value=100, step=1, value=35,
                                                     help="Percentage of available credit being used")
        
        # Credit health indicators
        st.markdown("#### 🎯 Credit Health Indicators")
        
        health_col1, health_col2, health_col3 = st.columns(3)
        
        with health_col1:
            if avg_dpd_per_delinquency <= 5:
                st.success("✅ Excellent Payment History")
            elif avg_dpd_per_delinquency <= 15:
                st.warning("⚠️ Fair Payment History")
            else:
                st.error("❌ Poor Payment History")
        
        with health_col2:
            if credit_utilization_ratio <= 30:
                st.success("✅ Healthy Credit Usage")
            elif credit_utilization_ratio <= 70:
                st.warning("⚠️ High Credit Usage")
            else:
                st.error("❌ Excessive Credit Usage")
        
        with health_col3:
            if delinquency_ratio <= 10:
                st.success("✅ Low Delinquency")
            elif delinquency_ratio <= 30:
                st.warning("⚠️ Moderate Delinquency")
            else:
                st.error("❌ High Delinquency")


    # Assessment button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button('🔍 Assess Credit Risk', type="primary", use_container_width=True):
            # Add loading animation
            with st.spinner('🤖 Analyzing credit risk...'):
                time.sleep(1)  # Simulate processing time
                
                # Call the predict function
                probability, credit_score, rating = predict(
                    age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                    delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                    residence_type, loan_purpose, loan_type
                )
            
            # Display results in an attractive format
            st.markdown("---")
            st.markdown('<h2 class="sub-header">📊 Risk Assessment Results</h2>', unsafe_allow_html=True)
            
            # Create metrics columns
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(
                    label="🎯 Default Probability",
                    value=f"{probability:.1%}",
                    delta=f"Risk Level: {'High' if probability > 0.15 else 'Medium' if probability > 0.05 else 'Low'}"
                )
            
            with metric_col2:
                st.metric(
                    label="📊 Credit Score",
                    value=f"{credit_score}",
                    delta=f"Range: 300-900"
                )
            
            with metric_col3:
                st.metric(
                    label="⭐ Credit Rating",
                    value=rating,
                    delta=f"{'Approve' if rating in ['Good', 'Excellent'] else 'Review' if rating == 'Average' else 'Decline'}"
                )
            
            # Risk visualization
            st.markdown("### 📈 Risk Visualization")
            
            # Create gauge chart for credit score
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = credit_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Credit Score"},
                delta = {'reference': 650},
                gauge = {
                    'axis': {'range': [None, 900]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [300, 500], 'color': "lightgray"},
                        {'range': [500, 650], 'color': "yellow"},
                        {'range': [650, 750], 'color': "lightgreen"},
                        {'range': [750, 900], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 650
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk breakdown
            st.markdown("### 🔍 Risk Factor Analysis")
            
            # Create risk factor breakdown
            risk_factors = {
                'Payment History': 100 - (avg_dpd_per_delinquency * 2),
                'Credit Utilization': 100 - credit_utilization_ratio,
                'Loan-to-Income': max(0, 100 - (loan_to_income_ratio * 20)),
                'Delinquency History': 100 - delinquency_ratio,
                'Account Diversity': min(100, num_open_accounts * 25)
            }
            
            # Normalize to 0-100 scale
            risk_factors = {k: max(0, min(100, v)) for k, v in risk_factors.items()}
            
            # Create horizontal bar chart
            fig_bar = px.bar(
                x=list(risk_factors.values()),
                y=list(risk_factors.keys()),
                orientation='h',
                title="Risk Factor Scores (Higher is Better)",
                color=list(risk_factors.values()),
                color_continuous_scale="RdYlGn"
            )
            fig_bar.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Recommendation
            st.markdown("### 💡 Recommendation")
            
            if rating == "Excellent":
                st.success("✅ **APPROVE**: Excellent credit profile with minimal risk. Offer premium rates.")
            elif rating == "Good":
                st.success("✅ **APPROVE**: Good credit profile. Standard terms recommended.")
            elif rating == "Average":
                st.warning("⚠️ **REVIEW**: Average credit profile. Consider additional verification or adjusted terms.")
            else:
                st.error("❌ **DECLINE**: High risk profile. Recommend decline or require additional collateral.")

elif page == "Model Information":
    st.markdown('<h2 class="sub-header">🤖 Model Information</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Model Details
        - **Algorithm**: Logistic Regression with Optimization
        - **Training Data**: 37,500 loan applications
        - **Features**: 25+ risk indicators
        - **Validation**: 5-fold cross-validation
        - **Performance**: 91%+ accuracy
        """)
        
        st.markdown("""
        ### 🎯 Key Features
        - Credit utilization ratio
        - Payment history (DPD)
        - Loan-to-income ratio
        - Delinquency patterns
        - Account diversity
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Performance Metrics
        - **Accuracy**: 91.2%
        - **Precision**: 89%
        - **Recall**: 85%
        - **F1-Score**: 87%
        - **AUC-ROC**: 92%
        """)
        
        st.markdown("""
        ### ⚖️ Risk Thresholds
        - **Low Risk**: Score > 750
        - **Medium Risk**: Score 650-750
        - **High Risk**: Score 500-650
        - **Very High Risk**: Score < 500
        """)

elif page == "Risk Analytics":
    st.markdown('<h2 class="sub-header">📊 Risk Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # Sample portfolio data for demonstration
    np.random.seed(42)
    portfolio_data = pd.DataFrame({
        'Score_Range': ['300-500', '500-650', '650-750', '750-900'],
        'Count': [1200, 3500, 8900, 6400],
        'Default_Rate': [0.458, 0.234, 0.087, 0.021],
        'Avg_Loan_Amount': [180000, 320000, 580000, 850000]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Portfolio distribution
        fig_pie = px.pie(
            portfolio_data, 
            values='Count', 
            names='Score_Range',
            title='Portfolio Distribution by Credit Score'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Default rates by score range
        fig_bar = px.bar(
            portfolio_data,
            x='Score_Range',
            y='Default_Rate',
            title='Default Rates by Credit Score Range',
            color='Default_Rate',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Portfolio metrics
    st.markdown("### 📊 Portfolio Metrics")
    
    total_loans = portfolio_data['Count'].sum()
    weighted_default_rate = (portfolio_data['Count'] * portfolio_data['Default_Rate']).sum() / total_loans
    total_exposure = (portfolio_data['Count'] * portfolio_data['Avg_Loan_Amount']).sum()
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Total Loans", f"{total_loans:,}")
    
    with metric_col2:
        st.metric("Portfolio Default Rate", f"{weighted_default_rate:.1%}")
    
    with metric_col3:
        st.metric("Total Exposure", f"₹{total_exposure/1e9:.1f}B")
    
    with metric_col4:
        st.metric("Expected Loss", f"₹{total_exposure * weighted_default_rate / 1e6:.0f}M")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        🏦 Credit Risk Assessment System | Built with Streamlit & Machine Learning<br>
        <small>For demonstration purposes only. Not for actual financial decisions.</small>
    </div>
    """, 
    unsafe_allow_html=True
)
