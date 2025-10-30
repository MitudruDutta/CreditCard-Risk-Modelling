#!/usr/bin/env python3
"""
Credit Risk Assessment System - Launcher Script

This script provides a convenient way to launch the Streamlit application
with proper error handling and environment checks.
"""

import os
import sys
import subprocess
import importlib.util

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 
        'scikit-learn', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed!")
    return True

def check_model_file():
    """Check if the model file exists."""
    model_path = "artifacts/model_data.joblib"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("\n💡 Please ensure you have:")
        print("   1. Run the Jupyter notebook to train the model")
        print("   2. The model is saved in the artifacts/ directory")
        return False
    
    print(f"✅ Model file found: {model_path}")
    return True

def check_app_files():
    """Check if application files exist."""
    required_files = [
        "app/main.py",
        "app/prediction_helper.py"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing application files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All application files found!")
    return True

def launch_streamlit():
    """Launch the Streamlit application."""
    try:
        print("🚀 Launching Credit Risk Assessment System...")
        print("📱 The application will open in your default web browser")
        print("🌐 URL: http://localhost:8501")
        print("\n⏹️  Press Ctrl+C to stop the application")
        print("=" * 50)
        
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app/main.py", 
            "--server.port=8501",
            "--server.address=localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching application: {str(e)}")
        return False
    
    return True

def main():
    """Main function to run all checks and launch the app."""
    print("🏦 Credit Risk Assessment System")
    print("=" * 40)
    
    # Run all checks
    checks = [
        ("Dependencies", check_dependencies),
        ("Application Files", check_app_files),
        ("Model File", check_model_file)
    ]
    
    for check_name, check_function in checks:
        print(f"\n🔍 Checking {check_name}...")
        if not check_function():
            print(f"\n❌ {check_name} check failed. Please fix the issues above.")
            sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✅ All checks passed! Ready to launch...")
    print("=" * 50)
    
    # Launch the application
    if not launch_streamlit():
        sys.exit(1)

if __name__ == "__main__":
    main()