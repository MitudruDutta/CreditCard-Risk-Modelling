# Dataset Information

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
