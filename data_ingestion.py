"""
Data Ingestion Module

This module handles the ingestion of UPI transaction data from various sources.
It provides functions for loading, validating, and preprocessing transaction data.
"""

import pandas as pd
import datetime
import os
import json
from typing import Dict, List, Union, Tuple, Optional
import numpy as np


class TransactionData:
    """Class to handle the loading and validation of transaction data."""
    
    def __init__(self):
        """Initialize the TransactionData class."""
        self.data = None
        self.is_valid = False
        self.validation_errors = []
        
        # Define the required columns
        self.required_columns = [
            'transaction_date', 
            'amount', 
            'transaction_type',  # 'credit' or 'debit'
            'description',
            'sender_upi_id',     # Added new required field
            'receiver_upi_id'    # Added new required field
        ]
        
        # Define optional columns
        self.optional_columns = [
            'merchant', 
            'transaction_id', 
            'category',
            'transaction_ref'    # Added new optional field for UPI reference number
        ]
    
    def load_from_csv(self, file_path: str) -> bool:
        """
        Load transaction data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            bool: True if data was loaded successfully, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                self.validation_errors.append(f"File not found: {file_path}")
                return False
                
            # Load data
            self.data = pd.read_csv(file_path)
            
            # Validate the data
            return self.validate_data()
            
        except Exception as e:
            self.validation_errors.append(f"Error loading CSV: {str(e)}")
            return False
    
    def load_from_json(self, json_data: Union[str, Dict, List]) -> bool:
        """
        Load transaction data from a JSON string or object.
        
        Args:
            json_data: JSON string, dictionary, or list of transactions
            
        Returns:
            bool: True if data was loaded successfully, False otherwise
        """
        try:
            # Convert string to JSON if necessary
            if isinstance(json_data, str):
                json_data = json.loads(json_data)
                
            # Convert to DataFrame
            self.data = pd.DataFrame(json_data)
            
            # Validate the data
            return self.validate_data()
            
        except Exception as e:
            self.validation_errors.append(f"Error loading JSON: {str(e)}")
            return False
    
    def validate_data(self) -> bool:
        """
        Validate the loaded transaction data.
        
        Returns:
            bool: True if data is valid, False otherwise
        """
        # Reset validation
        self.is_valid = False
        self.validation_errors = []
        
        # Check if data is loaded
        if self.data is None or len(self.data) == 0:
            self.validation_errors.append("No data loaded or empty dataset")
            return False
        
        # Check for required columns
        missing_columns = [col for col in self.required_columns if col not in self.data.columns]
        if missing_columns:
            self.validation_errors.append(f"Missing required columns: {', '.join(missing_columns)}")
            return False
        
        # Check for data types and basic validation
        validation_checks = [
            self._validate_dates(),
            self._validate_amounts(),
            self._validate_transaction_types(),
            self._validate_upi_ids()  # Added validation for UPI IDs
        ]
        
        # Check if all validations passed
        self.is_valid = all(validation_checks)
        
        return self.is_valid
    
    def _validate_dates(self) -> bool:
        """
        Validate transaction dates.
        
        Returns:
            bool: True if dates are valid, False otherwise
        """
        try:
            # Convert to datetime if it's not already
            if not pd.api.types.is_datetime64_dtype(self.data['transaction_date']):
                try:
                    self.data['transaction_date'] = pd.to_datetime(self.data['transaction_date'])
                except Exception as e:
                    self.validation_errors.append(f"Invalid date format: {str(e)}")
                    return False
            
            # Check for future dates
            today = pd.Timestamp.today()
            future_dates = self.data[self.data['transaction_date'] > today]
            if len(future_dates) > 0:
                self.validation_errors.append(f"Found {len(future_dates)} transactions with future dates")
                # We can still proceed with a warning, so not returning False
            
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Error validating dates: {str(e)}")
            return False
    
    def _validate_amounts(self) -> bool:
        """
        Validate transaction amounts.
        
        Returns:
            bool: True if amounts are valid, False otherwise
        """
        try:
            # Convert to numeric if not already
            if not pd.api.types.is_numeric_dtype(self.data['amount']):
                try:
                    self.data['amount'] = pd.to_numeric(self.data['amount'])
                except Exception as e:
                    self.validation_errors.append(f"Invalid amount format: {str(e)}")
                    return False
            
            # Check for negative or zero amounts
            zero_amounts = self.data[self.data['amount'] == 0]
            if len(zero_amounts) > 0:
                self.validation_errors.append(f"Found {len(zero_amounts)} transactions with zero amount")
            
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Error validating amounts: {str(e)}")
            return False
    
    def _validate_transaction_types(self) -> bool:
        """
        Validate transaction types (credit/debit).
        
        Returns:
            bool: True if transaction types are valid, False otherwise
        """
        valid_types = ['credit', 'debit']
        
        # Convert all to lowercase for case-insensitive comparison
        types = self.data['transaction_type'].str.lower()
        
        # Check for invalid types
        invalid_types = types[~types.isin(valid_types)]
        if len(invalid_types) > 0:
            unique_invalid = invalid_types.unique()
            self.validation_errors.append(
                f"Found {len(invalid_types)} transactions with invalid types: {', '.join(unique_invalid)}"
            )
            return False
        
        # Standardize the format to lowercase
        self.data['transaction_type'] = types
        
        return True
    
    def _validate_upi_ids(self) -> bool:
        """
        Validate UPI IDs format.
        
        Returns:
            bool: True if UPI IDs are valid, False otherwise
        """
        try:
            # Basic pattern check for UPI IDs (username@provider)
            sender_invalid = ~self.data['sender_upi_id'].str.contains('@', regex=False)
            receiver_invalid = ~self.data['receiver_upi_id'].str.contains('@', regex=False)
            
            if sender_invalid.any():
                invalid_count = sender_invalid.sum()
                self.validation_errors.append(f"Found {invalid_count} transactions with invalid sender UPI IDs")
                return False
                
            if receiver_invalid.any():
                invalid_count = receiver_invalid.sum()
                self.validation_errors.append(f"Found {invalid_count} transactions with invalid receiver UPI IDs")
                return False
            
            # Ensure no transaction has same sender and receiver (except for self-transfers)
            same_ids = (self.data['sender_upi_id'] == self.data['receiver_upi_id'])
            
            if same_ids.any():
                # Only flag as warning, not error
                self.validation_errors.append(f"Warning: Found {same_ids.sum()} transactions with same sender and receiver UPI ID")
            
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Error validating UPI IDs: {str(e)}")
            return False
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the transaction data after validation.
        
        Returns:
            DataFrame: Preprocessed transaction data
        """
        if not self.is_valid:
            raise ValueError("Data not valid. Call validate_data() first.")
        
        # Make a copy to avoid modifying the original
        processed_data = self.data.copy()
        
        # Ensure transaction_date is datetime
        processed_data['transaction_date'] = pd.to_datetime(processed_data['transaction_date'])
        
        # Sort by date
        processed_data = processed_data.sort_values('transaction_date')
        
        # Add merchant from description if merchant column doesn't exist
        if 'merchant' not in processed_data.columns:
            # Extract merchant name from description
            processed_data['merchant'] = processed_data['description'].str.extract(r'(?:at|to|from) ([A-Za-z0-9\s]+)', expand=False)
            # If no merchant found, use the first word of description
            processed_data.loc[processed_data['merchant'].isna(), 'merchant'] = processed_data.loc[processed_data['merchant'].isna(), 'description'].str.split().str[0]
        
        # Add transaction_id if it doesn't exist
        if 'transaction_id' not in processed_data.columns:
            processed_data['transaction_id'] = [f"tx_{i}" for i in range(len(processed_data))]
        
        # Add category if it doesn't exist (this would use more sophisticated logic in production)
        if 'category' not in processed_data.columns:
            processed_data['category'] = 'uncategorized'
        
        return processed_data
    
    def get_summary_stats(self) -> Dict:
        """
        Generate summary statistics for the transaction data.
        
        Returns:
            Dict: Summary statistics
        """
        if not self.is_valid or self.data is None:
            return {"error": "No valid data available"}
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_dtype(self.data['transaction_date']):
            self.data['transaction_date'] = pd.to_datetime(self.data['transaction_date'])
        
        # Basic statistics
        credit_txns = self.data[self.data['transaction_type'] == 'credit']
        debit_txns = self.data[self.data['transaction_type'] == 'debit']
        
        date_range = (
            self.data['transaction_date'].min().strftime('%Y-%m-%d'),
            self.data['transaction_date'].max().strftime('%Y-%m-%d')
        )
        
        time_span_days = (self.data['transaction_date'].max() - self.data['transaction_date'].min()).days
        
        summary = {
            "total_transactions": len(self.data),
            "date_range": date_range,
            "time_span_days": time_span_days,
            "total_credit": len(credit_txns),
            "total_debit": len(debit_txns),
            "credit_amount": credit_txns['amount'].sum() if len(credit_txns) > 0 else 0,
            "debit_amount": debit_txns['amount'].sum() if len(debit_txns) > 0 else 0,
            "avg_transaction": self.data['amount'].mean(),
            "max_transaction": self.data['amount'].max(),
            "min_transaction": self.data['amount'].min(),
            "transaction_frequency": len(self.data) / max(time_span_days, 1) if time_span_days > 0 else 0
        }
        
        return summary


def load_sample_data() -> pd.DataFrame:
    """
    Load sample transaction data for testing.
    
    Returns:
        DataFrame: Sample transaction data
    """
    # Create sample data
    today = datetime.datetime.now()
    
    # Generate dates for the last 90 days
    dates = [today - datetime.timedelta(days=i) for i in range(90)]
    
    # Sample merchants with their payment handles
    merchants = [
        {"name": "Swiggy", "upi": "swiggy@ybl"},
        {"name": "Amazon", "upi": "amazon@apl"},
        {"name": "Flipkart", "upi": "flipkart@fbl"},
        {"name": "Zomato", "upi": "zomato@zomato"},
        {"name": "Uber", "upi": "uber@paytm"},
        {"name": "Ola", "upi": "ola@olacabs"},
        {"name": "BigBasket", "upi": "bigbasket@bbasket"},
        {"name": "DMart", "upi": "dmart@hdfcbank"},
        {"name": "Netflix", "upi": "netflix@netbanking"},
        {"name": "Spotify", "upi": "spotify@icici"},
    ]
    
    # Sample income sources
    income_sources = [
        {"name": "Salary", "upi": "acme@icici"},
        {"name": "Freelance", "upi": "upwork@ybl"},
        {"name": "Interest", "upi": "hdfc@hdfcbank"},
        {"name": "Dividend", "upi": "zerodha@okicici"},
        {"name": "Refund", "upi": "refunds@sbi"}
    ]
    
    # User UPI IDs
    user_upi_ids = [
        "user@okaxis",
        "personal@ybl",
        "user@sbi"
    ]
    
    # Transaction description templates
    debit_descriptions = [
        "Payment for {} order #{}", 
        "UPI transaction to {} ref {}",
        "Purchase at {}, UPI reference {}",
        "Bill payment to {}, ref {}",
        "Paid to {}, transaction ID: {}"
    ]
    
    credit_descriptions = [
        "Amount received from {}, ref {}", 
        "UPI credit from {} with reference {}", 
        "Payment received from {}, ID {}",
        "{} transfer, reference {}", 
        "Credited by {} via UPI, ref {}"
    ]
    
    # Generate a transaction ID
    def generate_transaction_id():
        letters = ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 3))
        numbers = ''.join(np.random.choice(list('0123456789'), 9))
        return f"{letters}{numbers}"
    
    # Generate transactions
    transactions = []
    
    # Regular income (salary) on 1st of each month
    for i in range(3):
        month_start = today.replace(day=1) - datetime.timedelta(days=30*i)
        tx_id = generate_transaction_id()
        transactions.append({
            "transaction_date": month_start.strftime("%Y-%m-%d"),
            "merchant": "Salary",
            "amount": 50000,
            "transaction_type": "credit",
            "description": f"Salary credit from Acme Corp, ref {tx_id}",
            "sender_upi_id": income_sources[0]["upi"],
            "receiver_upi_id": user_upi_ids[0],
            "transaction_ref": tx_id
        })
    
    # Regular expenses
    for date in dates:
        # 50% chance of having a transaction on any day
        if date.day % 2 == 0:
            # Food delivery or shopping (small amount)
            merchant = merchants[date.day % len(merchants)]
            tx_id = generate_transaction_id()
            transactions.append({
                "transaction_date": date.strftime("%Y-%m-%d"),
                "merchant": merchant["name"],
                "amount": 100 + (date.day * 10),  # Variable amount
                "transaction_type": "debit",
                "description": np.random.choice(debit_descriptions).format(merchant["name"], tx_id),
                "sender_upi_id": user_upi_ids[date.day % len(user_upi_ids)],
                "receiver_upi_id": merchant["upi"],
                "transaction_ref": tx_id
            })
        
        # Weekly grocery shopping
        if date.weekday() == 5:  # Saturday
            tx_id = generate_transaction_id()
            transactions.append({
                "transaction_date": date.strftime("%Y-%m-%d"),
                "merchant": "BigBasket",
                "amount": 1000 + (date.day * 50),  # Variable amount
                "transaction_type": "debit",
                "description": f"Weekly groceries from BigBasket, ref {tx_id}",
                "sender_upi_id": user_upi_ids[0],
                "receiver_upi_id": merchants[6]["upi"],  # BigBasket
                "transaction_ref": tx_id
            })
        
        # Monthly rent payment
        if date.day == 5:
            tx_id = generate_transaction_id()
            transactions.append({
                "transaction_date": date.strftime("%Y-%m-%d"),
                "merchant": "Rent",
                "amount": 15000,
                "transaction_type": "debit",
                "description": f"Monthly rent payment to Mr. Sharma, ref {tx_id}",
                "sender_upi_id": user_upi_ids[0],
                "receiver_upi_id": "sharma@sbi",
                "transaction_ref": tx_id
            })
    
    # Random extra transactions
    for i in range(30):
        random_date = dates[i * 3 % len(dates)]
        is_credit = i % 5 == 0  # 20% are credits
        
        if is_credit:
            source = income_sources[i % len(income_sources)]
            tx_id = generate_transaction_id()
            transactions.append({
                "transaction_date": random_date.strftime("%Y-%m-%d"),
                "merchant": source["name"],
                "amount": 500 + (i * 100),
                "transaction_type": "credit",
                "description": np.random.choice(credit_descriptions).format(source["name"], tx_id),
                "sender_upi_id": source["upi"],
                "receiver_upi_id": user_upi_ids[i % len(user_upi_ids)],
                "transaction_ref": tx_id
            })
        else:
            merchant = merchants[i % len(merchants)]
            tx_id = generate_transaction_id()
            transactions.append({
                "transaction_date": random_date.strftime("%Y-%m-%d"),
                "merchant": merchant["name"],
                "amount": 200 + (i * 50),
                "transaction_type": "debit",
                "description": np.random.choice(debit_descriptions).format(merchant["name"], tx_id),
                "sender_upi_id": user_upi_ids[i % len(user_upi_ids)],
                "receiver_upi_id": merchant["upi"],
                "transaction_ref": tx_id
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(transactions)
    
    # Sort by date
    df = df.sort_values("transaction_date")
    
    return df


def save_sample_data(file_path: str) -> None:
    """
    Save sample transaction data to a CSV file.
    
    Args:
        file_path: Path to save the CSV file
    """
    df = load_sample_data()
    df.to_csv(file_path, index=False)
    print(f"Sample data saved to {file_path}")


if __name__ == "__main__":
    # Test the module
    sample_data = load_sample_data()
    print(f"Generated {len(sample_data)} sample transactions")
    
    # Save sample data to a file
    save_sample_data("sample_transactions.csv")
    
    # Test loading from CSV
    transaction_data = TransactionData()
    success = transaction_data.load_from_csv("sample_transactions.csv")
    
    if success:
        print("Data validation successful!")
        summary = transaction_data.get_summary_stats()
        print("\nTransaction Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")
    else:
        print("Data validation failed with errors:")
        for error in transaction_data.validation_errors:
            print(f"- {error}") 