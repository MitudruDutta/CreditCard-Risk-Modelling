# 📚 Credit Risk Assessment System - Technical Documentation

## 🏗️ System Architecture

### Overview
The Credit Risk Assessment System is built using a modular architecture that separates data processing, model training, and application serving into distinct components.

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
├─────────────────────────────────────────────────────────────┤
│  • customers.csv (Demographics)                             │
│  • loans.csv (Financial Data + Target)                     │
│  • bureau_data.csv (Credit History)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Processing Layer                            │
├─────────────────────────────────────────────────────────────┤
│  • Data Integration & Cleaning                              │
│  • Feature Engineering                                      │
│  • Exploratory Data Analysis                               │
│  • Statistical Analysis                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Model Layer                               │
├─────────────────────────────────────────────────────────────┤
│  • Multiple ML Algorithms                                   │
│  • Hyperparameter Optimization                             │
│  • Cross-Validation                                        │
│  • Model Serialization                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                Application Layer                            │
├─────────────────────────────────────────────────────────────┤
│  • Streamlit Web Interface                                  │
│  • Real-time Predictions                                   │
│  • Interactive Visualizations                              │
│  • Risk Analytics Dashboard                                │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Technical Components

### 1. Data Processing Pipeline

#### Data Integration
```python
# Merge multiple data sources
df = pd.merge(df_customers, df_loans, on='cust_id')
df = pd.merge(df, df_bureau, on='cust_id')
```

#### Feature Engineering
- **Derived Features**: Loan-to-income ratio, debt service ratio
- **Categorical Encoding**: One-hot encoding for categorical variables
- **Scaling**: MinMax scaling for numerical features
- **Missing Value Handling**: Mode imputation for categorical variables

#### Data Quality Checks
- Duplicate detection and removal
- Outlier identification using IQR method
- Missing value analysis and treatment
- Data type validation and conversion

### 2. Machine Learning Pipeline

#### Model Selection
The system implements multiple algorithms for comparison:

1. **Logistic Regression**
   - Linear relationship modeling
   - High interpretability
   - Fast training and prediction
   - Baseline model performance

2. **Random Forest**
   - Non-linear relationship capture
   - Feature importance ranking
   - Robust to outliers
   - Ensemble method benefits

3. **XGBoost**
   - Gradient boosting optimization
   - Superior performance on tabular data
   - Built-in regularization
   - Handles missing values

#### Hyperparameter Optimization
```python
# Optuna-based optimization
def objective(trial):
    params = {
        'C': trial.suggest_float('C', 0.01, 100.0),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
    }
    model = LogisticRegression(**params)
    return cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()
```

#### Class Imbalance Handling
- **SMOTE**: Synthetic Minority Oversampling Technique
- **Random Oversampling**: Simple duplication of minority class
- **Class Weights**: Algorithm-level balancing
- **Threshold Tuning**: Optimal decision boundary selection

### 3. Model Evaluation Framework

#### Performance Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

#### Business Metrics
- **Expected Loss**: P(Default) × Exposure × Loss Given Default
- **Approval Rate**: Percentage of applications approved
- **Portfolio Risk**: Aggregate risk assessment
- **Profit Optimization**: Revenue vs. risk trade-off

### 4. Web Application Architecture

#### Frontend (Streamlit)
```python
# Multi-page application structure
if page == "Risk Assessment":
    # Input forms and prediction interface
elif page == "Model Information":
    # Model details and performance metrics
elif page == "Risk Analytics":
    # Portfolio analytics and visualizations
```

#### Backend (Prediction Service)
```python
# Prediction pipeline
def predict(inputs):
    # 1. Input validation
    # 2. Feature preprocessing
    # 3. Model prediction
    # 4. Score calculation
    # 5. Risk rating assignment
    return probability, score, rating
```

## 📊 Data Schema

### Customer Data Schema
```sql
CREATE TABLE customers (
    cust_id VARCHAR(10) PRIMARY KEY,
    age INTEGER,
    gender VARCHAR(1),
    marital_status VARCHAR(20),
    employment_status VARCHAR(20),
    income INTEGER,
    number_of_dependants INTEGER,
    residence_type VARCHAR(20),
    years_at_current_address INTEGER,
    city VARCHAR(50),
    state VARCHAR(50),
    zipcode INTEGER
);
```

### Loan Data Schema
```sql
CREATE TABLE loans (
    loan_id VARCHAR(10) PRIMARY KEY,
    cust_id VARCHAR(10) FOREIGN KEY,
    loan_purpose VARCHAR(20),
    loan_type VARCHAR(20),
    sanction_amount INTEGER,
    loan_amount INTEGER,
    processing_fee DECIMAL(10,2),
    gst INTEGER,
    net_disbursement INTEGER,
    loan_tenure_months INTEGER,
    principal_outstanding INTEGER,
    bank_balance_at_application INTEGER,
    disbursal_date DATE,
    installment_start_dt DATE,
    default BOOLEAN
);
```

### Bureau Data Schema
```sql
CREATE TABLE bureau_data (
    cust_id VARCHAR(10) FOREIGN KEY,
    number_of_open_accounts INTEGER,
    number_of_closed_accounts INTEGER,
    total_loan_months INTEGER,
    delinquent_months INTEGER,
    total_dpd INTEGER,
    enquiry_count INTEGER,
    credit_utilization_ratio INTEGER
);
```

## 🔍 Feature Engineering Details

### Derived Features
1. **Loan-to-Income Ratio**
   ```python
   loan_to_income = loan_amount / annual_income
   ```
   - Risk Indicator: Higher ratios indicate potential repayment stress
   - Threshold: >5 is high risk, 3-5 is moderate, <3 is low risk

2. **Delinquency Ratio**
   ```python
   delinquency_ratio = delinquent_months / total_loan_months * 100
   ```
   - Payment Behavior: Percentage of time customer was delinquent
   - Threshold: >30% is high risk, 10-30% is moderate, <10% is low risk

3. **Average Days Past Due**
   ```python
   avg_dpd = total_dpd / max(delinquent_months, 1)
   ```
   - Severity Measure: How late payments typically are
   - Threshold: >30 days is concerning, >90 days is severe

### Categorical Encoding
```python
# One-hot encoding for categorical variables
categorical_features = ['residence_type', 'loan_purpose', 'loan_type']
encoded_features = pd.get_dummies(df[categorical_features], prefix=categorical_features)
```

### Feature Scaling
```python
# MinMax scaling for numerical features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[numerical_features])
```

## 🎯 Model Performance Analysis

### Cross-Validation Strategy
```python
# Stratified K-Fold Cross-Validation
cv_scores = cross_val_score(
    model, X_train, y_train, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1'
)
```

### Feature Importance Analysis
```python
# Get feature importance from trained model
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
```

### Model Interpretation
- **SHAP Values**: For individual prediction explanations
- **Permutation Importance**: Feature impact on model performance
- **Partial Dependence Plots**: Feature effect visualization
- **Correlation Analysis**: Feature relationship understanding

## 🚀 Deployment Considerations

### Production Readiness Checklist
- [ ] Model versioning and tracking
- [ ] A/B testing framework
- [ ] Model monitoring and drift detection
- [ ] Automated retraining pipeline
- [ ] API rate limiting and security
- [ ] Logging and error handling
- [ ] Performance monitoring
- [ ] Data validation and quality checks

### Scalability Considerations
- **Horizontal Scaling**: Multiple application instances
- **Caching**: Redis for frequent predictions
- **Database**: PostgreSQL for production data
- **Load Balancing**: Nginx for traffic distribution
- **Containerization**: Docker for consistent deployment

### Security Measures
- **Input Validation**: Prevent injection attacks
- **Rate Limiting**: Prevent abuse
- **Authentication**: User access control
- **Data Encryption**: Sensitive data protection
- **Audit Logging**: Track all predictions

## 📈 Monitoring and Maintenance

### Model Performance Monitoring
```python
# Track key metrics over time
def monitor_model_performance():
    current_metrics = calculate_metrics(y_true, y_pred)
    baseline_metrics = load_baseline_metrics()
    
    if current_metrics['f1'] < baseline_metrics['f1'] * 0.95:
        trigger_retraining_alert()
```

### Data Drift Detection
```python
# Statistical tests for feature drift
def detect_feature_drift(reference_data, current_data):
    for feature in features:
        ks_stat, p_value = ks_2samp(reference_data[feature], current_data[feature])
        if p_value < 0.05:
            log_drift_alert(feature, ks_stat, p_value)
```

### Automated Retraining
```python
# Scheduled model retraining
def retrain_model():
    new_data = load_recent_data()
    if len(new_data) > MIN_TRAINING_SIZE:
        model = train_model(new_data)
        if validate_model(model) > PERFORMANCE_THRESHOLD:
            deploy_model(model)
```

## 🔧 Configuration Management

### Environment Variables
```bash
# Application configuration
STREAMLIT_PORT=8501
MODEL_PATH=artifacts/model_data.joblib
LOG_LEVEL=INFO
CACHE_TTL=3600

# Database configuration (for production)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=credit_risk
DB_USER=app_user
DB_PASSWORD=secure_password
```

### Model Configuration
```yaml
# model_config.yaml
model:
  algorithm: "logistic_regression"
  hyperparameters:
    C: 1.0
    penalty: "l2"
    solver: "liblinear"
  
preprocessing:
  scaling_method: "minmax"
  categorical_encoding: "onehot"
  missing_value_strategy: "mode"

evaluation:
  cv_folds: 5
  scoring_metric: "f1"
  test_size: 0.25
```

## 🐛 Troubleshooting Guide

### Common Issues and Solutions

#### 1. Model Loading Errors
```python
# Error: Model file not found
# Solution: Check file path and permissions
if not os.path.exists(MODEL_PATH):
    logger.error(f"Model file not found: {MODEL_PATH}")
    # Fallback to default model or retrain
```

#### 2. Memory Issues
```python
# Error: Out of memory during training
# Solution: Use batch processing or reduce data size
def train_in_batches(X, y, batch_size=1000):
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        model.partial_fit(batch_X, batch_y)
```

#### 3. Prediction Errors
```python
# Error: Invalid input format
# Solution: Add comprehensive input validation
def validate_input(data):
    required_fields = ['age', 'income', 'loan_amount']
    for field in required_fields:
        if field not in data or data[field] is None:
            raise ValueError(f"Missing required field: {field}")
```

### Performance Optimization

#### 1. Caching Strategies
```python
# Cache frequent predictions
@lru_cache(maxsize=1000)
def cached_predict(input_hash):
    return model.predict(input_data)
```

#### 2. Batch Processing
```python
# Process multiple predictions at once
def batch_predict(input_list):
    input_df = pd.DataFrame(input_list)
    predictions = model.predict_proba(input_df)
    return predictions
```

#### 3. Model Optimization
```python
# Use optimized model formats
import joblib
# Save with compression
joblib.dump(model, 'model.pkl', compress=3)
```

## 📚 Additional Resources

### Documentation Links
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)

### Best Practices
- [ML Model Deployment Best Practices](https://ml-ops.org/)
- [Credit Risk Modeling Guidelines](https://www.bis.org/publ/bcbs128.htm)
- [Model Risk Management](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm)

### Research Papers
- "Credit Risk Assessment Using Machine Learning" (2020)
- "Imbalanced Learning for Credit Default Prediction" (2019)
- "Explainable AI in Financial Services" (2021)

---

*This documentation is maintained alongside the codebase and should be updated with any significant changes to the system architecture or functionality.*