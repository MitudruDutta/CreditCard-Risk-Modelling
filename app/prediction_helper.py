import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging
import os
from typing import Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to the saved model and its components
MODEL_PATH = 'artifacts/model_data.joblib'

# Global variables for model components
model = None
scaler = None
features = None
cols_to_scale = None

def load_model_components():
    """Load model components with error handling."""
    global model, scaler, features, cols_to_scale
    
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        logger.info(f"Loading model from {MODEL_PATH}")
        model_data = joblib.load(MODEL_PATH)
        
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']
        cols_to_scale = model_data['cols_to_scale']
        
        logger.info("Model components loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

# Load model components on import
if not load_model_components():
    logger.warning("Failed to load model components. Predictions may not work correctly.")


def validate_inputs(**kwargs) -> Dict[str, Any]:
    """Validate and sanitize input parameters."""
    validations = {
        'age': (18, 100, "Age must be between 18 and 100"),
        'income': (0, float('inf'), "Income must be positive"),
        'loan_amount': (0, float('inf'), "Loan amount must be positive"),
        'loan_tenure_months': (1, 360, "Loan tenure must be between 1 and 360 months"),
        'avg_dpd_per_delinquency': (0, 365, "Average DPD must be between 0 and 365"),
        'delinquency_ratio': (0, 100, "Delinquency ratio must be between 0 and 100"),
        'credit_utilization_ratio': (0, 100, "Credit utilization must be between 0 and 100"),
        'num_open_accounts': (1, 10, "Number of open accounts must be between 1 and 10")
    }
    
    errors = []
    for param, (min_val, max_val, error_msg) in validations.items():
        if param in kwargs:
            value = kwargs[param]
            if not (min_val <= value <= max_val):
                errors.append(error_msg)
    
    if errors:
        raise ValueError("; ".join(errors))
    
    return kwargs

def prepare_input(age: int, income: float, loan_amount: float, loan_tenure_months: int, 
                 avg_dpd_per_delinquency: float, delinquency_ratio: float, 
                 credit_utilization_ratio: float, num_open_accounts: int, 
                 residence_type: str, loan_purpose: str, loan_type: str) -> pd.DataFrame:
    """
    Prepare input data for model prediction with comprehensive validation.
    
    Args:
        age: Customer age
        income: Annual income
        loan_amount: Requested loan amount
        loan_tenure_months: Loan tenure in months
        avg_dpd_per_delinquency: Average days past due
        delinquency_ratio: Percentage of delinquent payments
        credit_utilization_ratio: Credit utilization percentage
        num_open_accounts: Number of open accounts
        residence_type: Type of residence (Owned/Rented/Mortgage)
        loan_purpose: Purpose of loan (Education/Home/Auto/Personal)
        loan_type: Type of loan (Secured/Unsecured)
    
    Returns:
        pd.DataFrame: Processed input data ready for prediction
    """
    
    # Validate inputs
    validate_inputs(
        age=age, income=income, loan_amount=loan_amount,
        loan_tenure_months=loan_tenure_months, avg_dpd_per_delinquency=avg_dpd_per_delinquency,
        delinquency_ratio=delinquency_ratio, credit_utilization_ratio=credit_utilization_ratio,
        num_open_accounts=num_open_accounts
    )
    
    # Calculate derived features
    loan_to_income = loan_amount / income if income > 0 else 0
    
    # Create input dictionary with proper feature engineering
    input_data = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_to_income,
        'delinquency_ratio': delinquency_ratio,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency,
        
        # One-hot encoded categorical variables
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
        
        # Dummy values for features not used in prediction but required for scaling
        'number_of_dependants': 2,  # Average value
        'years_at_current_address': 10,  # Average value
        'zipcode': 400001,  # Dummy zipcode
        'sanction_amount': loan_amount,  # Use actual loan amount
        'processing_fee': loan_amount * 0.02,  # Estimated 2% processing fee
        'gst': loan_amount * 0.02 * 0.18,  # 18% GST on processing fee
        'net_disbursement': loan_amount * 0.98,  # After deducting fees
        'principal_outstanding': loan_amount * 0.7,  # Estimated outstanding
        'bank_balance_at_application': income * 0.1,  # Estimated 10% of annual income
        'number_of_closed_accounts': 1,  # Average value
        'enquiry_count': 3  # Average value
    }

    try:
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Apply scaling to required columns
        if scaler is not None and cols_to_scale is not None:
            df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        
        # Select only the features expected by the model
        if features is not None:
            df = df[features]
        
        logger.info(f"Input prepared successfully with shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error preparing input: {str(e)}")
        raise


def predict(age: int, income: float, loan_amount: float, loan_tenure_months: int,
           avg_dpd_per_delinquency: float, delinquency_ratio: float, 
           credit_utilization_ratio: float, num_open_accounts: int,
           residence_type: str, loan_purpose: str, loan_type: str) -> Tuple[float, int, str]:
    """
    Predict credit risk for a loan application.
    
    Returns:
        Tuple[float, int, str]: (default_probability, credit_score, rating)
    """
    
    if model is None:
        raise RuntimeError("Model not loaded. Please check model file.")
    
    try:
        # Prepare input data
        input_df = prepare_input(
            age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts, 
            residence_type, loan_purpose, loan_type
        )

        # Calculate prediction
        probability, credit_score, rating = calculate_credit_score(input_df)
        
        logger.info(f"Prediction completed: P={probability:.3f}, Score={credit_score}, Rating={rating}")
        return probability, credit_score, rating
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        # Return default values in case of error
        return 0.5, 500, "Unknown"


def calculate_credit_score(input_df: pd.DataFrame, base_score: int = 300, 
                          scale_length: int = 600) -> Tuple[float, int, str]:
    """
    Calculate credit score and rating from model prediction.
    
    Args:
        input_df: Prepared input DataFrame
        base_score: Minimum credit score (default: 300)
        scale_length: Score range (default: 600, making max score 900)
    
    Returns:
        Tuple[float, int, str]: (default_probability, credit_score, rating)
    """
    
    try:
        # Calculate logit (linear combination)
        x = np.dot(input_df.values, model.coef_.T) + model.intercept_

        # Apply logistic function to get default probability
        default_probability = 1 / (1 + np.exp(-x))
        
        # Calculate non-default probability
        non_default_probability = 1 - default_probability

        # Convert to credit score (300-900 scale)
        credit_score = base_score + non_default_probability.flatten() * scale_length
        
        # Ensure score is within bounds
        credit_score = np.clip(credit_score, base_score, base_score + scale_length)

        # Determine rating category
        rating = get_credit_rating(credit_score[0])

        return float(default_probability.flatten()[0]), int(credit_score[0]), rating
        
    except Exception as e:
        logger.error(f"Error calculating credit score: {str(e)}")
        raise


def get_credit_rating(score: float) -> str:
    """
    Convert credit score to rating category.
    
    Args:
        score: Credit score (300-900)
    
    Returns:
        str: Rating category
    """
    
    if score >= 750:
        return 'Excellent'
    elif score >= 650:
        return 'Good'
    elif score >= 500:
        return 'Average'
    elif score >= 300:
        return 'Poor'
    else:
        return 'Undefined'


def get_risk_factors(age: int, income: float, loan_amount: float, 
                    avg_dpd_per_delinquency: float, delinquency_ratio: float,
                    credit_utilization_ratio: float, num_open_accounts: int) -> Dict[str, float]:
    """
    Calculate individual risk factor scores for interpretability.
    
    Returns:
        Dict[str, float]: Risk factor scores (0-100, higher is better)
    """
    
    loan_to_income = loan_amount / income if income > 0 else 0
    
    risk_factors = {
        'Payment History': max(0, 100 - (avg_dpd_per_delinquency * 3)),
        'Credit Utilization': max(0, 100 - credit_utilization_ratio),
        'Debt-to-Income': max(0, 100 - (loan_to_income * 15)),
        'Delinquency History': max(0, 100 - delinquency_ratio),
        'Account Diversity': min(100, num_open_accounts * 20),
        'Age Factor': min(100, (age - 18) * 2)  # Age stability factor
    }
    
    # Normalize to 0-100 scale
    risk_factors = {k: max(0, min(100, v)) for k, v in risk_factors.items()}
    
    return risk_factors


def get_model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model.
    
    Returns:
        Dict[str, Any]: Model information
    """
    
    if model is None:
        return {"status": "Model not loaded"}
    
    return {
        "status": "Model loaded successfully",
        "model_type": type(model).__name__,
        "features_count": len(features) if features else 0,
        "scalable_features": len(cols_to_scale) if cols_to_scale else 0,
        "model_path": MODEL_PATH
    }