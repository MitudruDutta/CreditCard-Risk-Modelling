#!/usr/bin/env python3
"""
Setup script for Credit Risk Assessment System

This script helps users set up the environment and dependencies
for the credit risk modeling project.
"""

import os
import sys
import subprocess
import platform

def print_header():
    """Print welcome header."""
    print("🏦 Credit Risk Assessment System - Setup")
    print("=" * 50)
    print("This script will help you set up the environment")
    print("for the credit risk modeling project.")
    print("=" * 50)

def check_python_version():
    """Check if Python version is compatible."""
    print("\n🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("💡 This project requires Python 3.8 or higher")
        print("   Please upgrade your Python installation")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible!")
    return True

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating project directories...")
    
    directories = [
        "artifacts",
        "dataset",
        "app",
        "logs"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"   ✅ Created: {directory}/")
        else:
            print(f"   📁 Exists: {directory}/")
    
    return True

def install_dependencies():
    """Install required Python packages."""
    print("\n📦 Installing dependencies...")
    
    try:
        # Check if requirements.txt exists
        if not os.path.exists("requirements.txt"):
            print("❌ requirements.txt not found")
            return False
        
        # Install packages
        print("   Installing packages from requirements.txt...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ All packages installed successfully!")
            return True
        else:
            print(f"   ❌ Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error during installation: {str(e)}")
        return False

def check_jupyter():
    """Check if Jupyter is available and offer to install if not."""
    print("\n📓 Checking Jupyter installation...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "jupyter", "--version"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ Jupyter is available!")
            return True
        else:
            print("   ❌ Jupyter not found")
            return False
            
    except Exception:
        print("   ❌ Jupyter not available")
        return False

def create_sample_data_info():
    """Create information about sample data."""
    print("\n📊 Setting up data information...")
    
    data_info = """# Dataset Information

## Required Files
Place the following CSV files in the dataset/ directory:

1. **customers.csv** - Customer demographic data
   - cust_id, age, gender, marital_status, employment_status
   - income, number_of_dependants, residence_type
   - years_at_current_address, city, state, zipcode

2. **loans.csv** - Loan application data
   - loan_id, cust_id, loan_purpose, loan_type
   - sanction_amount, loan_amount, processing_fee, gst
   - net_disbursement, loan_tenure_months, principal_outstanding
   - bank_balance_at_application, disbursal_date, installment_start_dt
   - default (target variable)

3. **bureau_data.csv** - Credit bureau information
   - cust_id, number_of_open_accounts, number_of_closed_accounts
   - total_loan_months, delinquent_months, total_dpd
   - enquiry_count, credit_utilization_ratio

## Data Format
- All files should be CSV format with headers
- Customer ID should be consistent across all files
- No missing values in key fields
"""
    
    with open("dataset/README.md", "w") as f:
        f.write(data_info)
    
    print("   ✅ Created dataset/README.md with data requirements")
    return True

def print_next_steps():
    """Print next steps for the user."""
    print("\n🎯 Next Steps:")
    print("=" * 30)
    print("1. 📊 Place your data files in the dataset/ directory")
    print("   (See dataset/README.md for format requirements)")
    print()
    print("2. 📓 Run the Jupyter notebook to train the model:")
    print("   jupyter notebook credit_risk_model.ipynb")
    print()
    print("3. 🚀 Launch the web application:")
    print("   python run_app.py")
    print()
    print("4. 🌐 Open your browser to:")
    print("   http://localhost:8501")
    print()
    print("💡 For help, check the README.md file")

def main():
    """Main setup function."""
    print_header()
    
    # Run setup steps
    steps = [
        ("Python Version", check_python_version),
        ("Project Directories", create_directories),
        ("Dependencies", install_dependencies),
        ("Jupyter Notebook", check_jupyter),
        ("Data Information", create_sample_data_info)
    ]
    
    failed_steps = []
    
    for step_name, step_function in steps:
        if not step_function():
            failed_steps.append(step_name)
    
    # Summary
    print("\n" + "=" * 50)
    if failed_steps:
        print("⚠️  Setup completed with some issues:")
        for step in failed_steps:
            print(f"   ❌ {step}")
        print("\nPlease resolve the issues above before proceeding.")
    else:
        print("✅ Setup completed successfully!")
    
    print_next_steps()

if __name__ == "__main__":
    main()