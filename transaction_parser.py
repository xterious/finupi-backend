"""
Transaction Parser Module

This module handles the parsing and feature extraction from transaction data.
It provides functions for categorizing transactions, detecting patterns, and building features
for the credit scoring model.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import re
from datetime import datetime, timedelta


class TransactionParser:
    """
    Class to handle the parsing and feature extraction from transaction data.
    """
    
    def __init__(self, data: pd.DataFrame = None):
        """
        Initialize the TransactionParser class.
        
        Args:
            data: Transaction data as a pandas DataFrame (optional)
        """
        self.data = data
        self.processed_data = None
        
        # Define categories for transactions
        self.expense_categories = {
            'food': ['swiggy', 'zomato', 'foodpanda', 'dominos', 'mcdonald', 'restaurant', 'cafe'],
            'transport': ['uber', 'ola', 'rapido', 'metro', 'train', 'bus', 'petrol', 'diesel', 'fuel'],
            'shopping': ['amazon', 'flipkart', 'myntra', 'ajio', 'mall', 'store', 'shop', 'retail'],
            'utilities': ['electricity', 'water', 'gas', 'bill', 'recharge', 'mobile', 'phone', 'internet', 'broadband'],
            'entertainment': ['netflix', 'amazon prime', 'hotstar', 'spotify', 'movie', 'concert', 'event'],
            'housing': ['rent', 'maintenance', 'society', 'apartment', 'flat', 'house'],
            'education': ['tuition', 'school', 'college', 'course', 'books', 'library', 'fee'],
            'healthcare': ['medical', 'hospital', 'doctor', 'medicine', 'pharmacy', 'health', 'clinic'],
            'investments': ['mutual fund', 'stock', 'share', 'invest', 'trading', 'crypto', 'bitcoin'],
            'insurance': ['insurance', 'policy', 'premium', 'life', 'health', 'vehicle'],
            'loan': ['loan', 'emi', 'repayment', 'interest', 'principal'],
            'others': []
        }
        
        self.income_categories = {
            'salary': ['salary', 'stipend', 'wage', 'income', 'pay'],
            'business': ['business', 'profit', 'sales', 'client', 'customer', 'service'],
            'investment': ['dividend', 'interest', 'return', 'profit', 'maturity'],
            'gift': ['gift', 'donation', 'prize', 'cashback', 'reward'],
            'refund': ['refund', 'reimbursement', 'return', 'cashback'],
            'others': []
        }
    
    def set_data(self, data: pd.DataFrame) -> None:
        """
        Set the transaction data.
        
        Args:
            data: Transaction data as a pandas DataFrame
        """
        self.data = data
    
    def parse_transactions(self) -> pd.DataFrame:
        """
        Parse the transaction data and extract features.
        
        Returns:
            DataFrame: Processed transaction data with added features
        """
        if self.data is None:
            raise ValueError("No data available. Use set_data() to set the transaction data.")
        
        # Make a copy to avoid modifying the original
        self.processed_data = self.data.copy()
        
        # Make sure transaction_date is datetime
        if not pd.api.types.is_datetime64_dtype(self.processed_data['transaction_date']):
            self.processed_data['transaction_date'] = pd.to_datetime(self.processed_data['transaction_date'])
        
        # Sort by date
        self.processed_data = self.processed_data.sort_values('transaction_date')
        
        # Process the data
        self._add_day_features()
        self._categorize_transactions()
        self._detect_recurring_transactions()
        self._calculate_running_balance()
        self._detect_roundtrip_payments()
        
        return self.processed_data
    
    def _add_day_features(self) -> None:
        """
        Add day-based features to the transaction data.
        """
        # Extract day features
        self.processed_data['day_of_week'] = self.processed_data['transaction_date'].dt.dayofweek
        self.processed_data['day_of_month'] = self.processed_data['transaction_date'].dt.day
        self.processed_data['month'] = self.processed_data['transaction_date'].dt.month
        self.processed_data['year'] = self.processed_data['transaction_date'].dt.year
        self.processed_data['is_weekend'] = self.processed_data['day_of_week'].isin([5, 6]).astype(int)
        self.processed_data['is_month_start'] = (self.processed_data['day_of_month'] <= 5).astype(int)
        self.processed_data['is_month_end'] = (self.processed_data['day_of_month'] >= 25).astype(int)
    
    def _categorize_transactions(self) -> None:
        """
        Categorize transactions based on description and merchant.
        """
        # Prepare a column for the description in lowercase for better matching
        if 'description' in self.processed_data.columns:
            self.processed_data['description_lower'] = self.processed_data['description'].str.lower()
        else:
            self.processed_data['description_lower'] = ''
        
        # Prepare a column for the merchant in lowercase
        if 'merchant' in self.processed_data.columns:
            self.processed_data['merchant_lower'] = self.processed_data['merchant'].str.lower()
        else:
            self.processed_data['merchant_lower'] = ''
        
        # Combine for searching
        self.processed_data['search_text'] = self.processed_data['description_lower'] + ' ' + self.processed_data['merchant_lower']
        
        # Initialize the category column
        self.processed_data['category'] = 'others'
        
        # Categorize expenses
        for category, keywords in self.expense_categories.items():
            for keyword in keywords:
                mask = self.processed_data['search_text'].str.contains(keyword, regex=False)
                mask &= (self.processed_data['transaction_type'] == 'debit')
                self.processed_data.loc[mask, 'category'] = category
        
        # Categorize income
        for category, keywords in self.income_categories.items():
            for keyword in keywords:
                mask = self.processed_data['search_text'].str.contains(keyword, regex=False)
                mask &= (self.processed_data['transaction_type'] == 'credit')
                self.processed_data.loc[mask, 'category'] = category
        
        # Clean up temporary columns
        self.processed_data = self.processed_data.drop(columns=['description_lower', 'merchant_lower', 'search_text'])
    
    def _detect_recurring_transactions(self) -> None:
        """
        Detect recurring transactions like salary, rent, etc.
        """
        # Group by merchant and transaction type
        grouped = self.processed_data.groupby(['merchant', 'transaction_type'])
        
        # Initialize columns
        self.processed_data['is_recurring'] = 0
        self.processed_data['recurrence_frequency'] = None
        
        for (merchant, tx_type), group in grouped:
            # If there are at least 2 transactions from the same merchant and type
            if len(group) >= 2:
                # Check for monthly patterns (salary, rent)
                if self._is_monthly_recurring(group):
                    self.processed_data.loc[group.index, 'is_recurring'] = 1
                    self.processed_data.loc[group.index, 'recurrence_frequency'] = 'monthly'
                
                # Check for weekly patterns (groceries, etc.)
                elif self._is_weekly_recurring(group):
                    self.processed_data.loc[group.index, 'is_recurring'] = 1
                    self.processed_data.loc[group.index, 'recurrence_frequency'] = 'weekly'
    
    def _is_monthly_recurring(self, group: pd.DataFrame) -> bool:
        """
        Check if a group of transactions follows a monthly pattern.
        
        Args:
            group: Group of transactions from the same merchant and type
            
        Returns:
            bool: True if it follows a monthly pattern
        """
        # Sort by date
        sorted_group = group.sort_values('transaction_date')
        
        # Check if transactions occur around the same day of month
        days = sorted_group['day_of_month'].values
        amounts = sorted_group['amount'].values
        
        # Check if the amounts are similar (within 10% of each other)
        amounts_similar = np.std(amounts) / np.mean(amounts) < 0.1 if len(amounts) > 0 else False
        
        # Check if the days are similar (within 3 days of each other)
        days_similar = np.std(days) < 3 if len(days) > 0 else False
        
        return amounts_similar and days_similar
    
    def _is_weekly_recurring(self, group: pd.DataFrame) -> bool:
        """
        Check if a group of transactions follows a weekly pattern.
        
        Args:
            group: Group of transactions from the same merchant and type
            
        Returns:
            bool: True if it follows a weekly pattern
        """
        # Sort by date
        sorted_group = group.sort_values('transaction_date')
        
        # Check if transactions occur on the same day of week
        days_of_week = sorted_group['day_of_week'].values
        amounts = sorted_group['amount'].values
        
        # Check if the amounts are similar (within 20% of each other)
        amounts_similar = np.std(amounts) / np.mean(amounts) < 0.2 if len(amounts) > 0 else False
        
        # Check if the days of week are similar
        same_day_of_week = len(set(days_of_week)) <= 2 if len(days_of_week) > 0 else False
        
        return amounts_similar and same_day_of_week
    
    def _calculate_running_balance(self) -> None:
        """
        Calculate running balance over time.
        """
        # Sort by date
        self.processed_data = self.processed_data.sort_values('transaction_date')
        
        # Create a copy of amount
        self.processed_data['signed_amount'] = self.processed_data['amount']
        
        # Make debits negative
        mask = self.processed_data['transaction_type'] == 'debit'
        self.processed_data.loc[mask, 'signed_amount'] = -self.processed_data.loc[mask, 'amount']
        
        # Calculate running balance
        self.processed_data['running_balance'] = self.processed_data['signed_amount'].cumsum()
        
        # Calculate daily balance
        daily_balance = self.processed_data.groupby(self.processed_data['transaction_date'].dt.date)['signed_amount'].sum()
        daily_balance_cum = daily_balance.cumsum()
        
        # Map the cumulative daily balance back to the original dataframe
        self.processed_data['daily_balance'] = self.processed_data['transaction_date'].dt.date.map(daily_balance_cum)
    
    def _detect_roundtrip_payments(self) -> None:
        """
        Detect round-trip payments (money sent and received back).
        """
        # Initialize column
        self.processed_data['is_roundtrip'] = 0
        
        # Process each merchant
        grouped_by_merchant = self.processed_data.groupby('merchant')
        
        for merchant, group in grouped_by_merchant:
            # Look for pairs of debit and credit with similar amounts
            debits = group[group['transaction_type'] == 'debit'].copy()
            credits = group[group['transaction_type'] == 'credit'].copy()
            
            # If there are both debits and credits for this merchant
            if len(debits) > 0 and len(credits) > 0:
                for _, debit in debits.iterrows():
                    # Look for matching credits within 7 days of the debit
                    debit_date = debit['transaction_date']
                    debit_amount = debit['amount']
                    
                    # Find credits with similar amount (+/- 1%) within 7 days
                    matching_credits = credits[
                        (credits['transaction_date'] >= debit_date) & 
                        (credits['transaction_date'] <= debit_date + timedelta(days=7)) &
                        (credits['amount'] >= debit_amount * 0.99) &
                        (credits['amount'] <= debit_amount * 1.01)
                    ]
                    
                    if len(matching_credits) > 0:
                        # Mark both transactions as round-trip
                        self.processed_data.loc[debit.name, 'is_roundtrip'] = 1
                        self.processed_data.loc[matching_credits.index[0], 'is_roundtrip'] = 1
    
    def extract_features(self) -> Dict:
        """
        Extract features from the processed data for credit scoring.
        
        Returns:
            Dict: Dictionary of features
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call parse_transactions() first.")
        
        # Get credit and debit transactions
        credits = self.processed_data[self.processed_data['transaction_type'] == 'credit']
        debits = self.processed_data[self.processed_data['transaction_type'] == 'debit']
        
        # Filter out round-trip payments
        valid_credits = credits[credits['is_roundtrip'] == 0]
        valid_debits = debits[debits['is_roundtrip'] == 0]
        
        # Calculate total timespan in days
        first_date = self.processed_data['transaction_date'].min()
        last_date = self.processed_data['transaction_date'].max()
        timespan_days = (last_date - first_date).days if first_date and last_date else 0
        
        # Calculate months of data
        months_of_data = timespan_days / 30 if timespan_days > 0 else 0
        
        # 1. Income Stability Features
        income_features = self._extract_income_features(valid_credits, months_of_data)
        
        # 2. Expense Management Features
        expense_features = self._extract_expense_features(valid_debits, valid_credits, months_of_data)
        
        # 3. Financial Discipline Features
        discipline_features = self._extract_discipline_features(valid_credits, valid_debits)
        
        # 4. Transaction History Features
        history_features = self._extract_history_features(months_of_data)
        
        # Combine all features
        features = {
            **income_features,
            **expense_features,
            **discipline_features,
            **history_features
        }
        
        return features
    
    def _extract_income_features(self, credits: pd.DataFrame, months_of_data: float) -> Dict:
        """
        Extract income-related features from credit transactions.
        
        Args:
            credits: DataFrame of credit transactions
            months_of_data: Number of months of data available
            
        Returns:
            Dict: Income-related features
        """
        features = {}
        
        # Skip if no valid data
        if len(credits) == 0 or months_of_data < 0.5:  # At least 15 days of data
            return {
                'monthly_income': 0,
                'income_frequency': 0,
                'income_regularity': 0,
                'income_sources': 0,
                'income_growth': 0
            }
        
        # 1. Monthly Income
        monthly_income = credits['amount'].sum() / months_of_data if months_of_data > 0 else 0
        features['monthly_income'] = monthly_income
        
        # 2. Income Frequency (transactions per month)
        income_frequency = len(credits) / months_of_data if months_of_data > 0 else 0
        features['income_frequency'] = income_frequency
        
        # 3. Income Regularity
        recurring_income = credits[credits['is_recurring'] == 1]
        recurring_income_pct = len(recurring_income) / len(credits) if len(credits) > 0 else 0
        features['income_regularity'] = recurring_income_pct
        
        # 4. Income Sources Diversity
        income_sources = credits['merchant'].nunique()
        features['income_sources'] = income_sources
        
        # 5. Income Growth
        # Only calculate if we have at least 2 months of data
        if months_of_data >= 2:
            # Split into first and second half
            midpoint = credits['transaction_date'].min() + (credits['transaction_date'].max() - credits['transaction_date'].min()) / 2
            first_half = credits[credits['transaction_date'] <= midpoint]
            second_half = credits[credits['transaction_date'] > midpoint]
            
            first_half_monthly = first_half['amount'].sum() / (months_of_data / 2) if months_of_data > 0 else 0
            second_half_monthly = second_half['amount'].sum() / (months_of_data / 2) if months_of_data > 0 else 0
            
            if first_half_monthly > 0:
                income_growth = (second_half_monthly - first_half_monthly) / first_half_monthly
            else:
                income_growth = 0
            
            features['income_growth'] = income_growth
        else:
            features['income_growth'] = 0
        
        return features
    
    def _extract_expense_features(self, debits: pd.DataFrame, credits: pd.DataFrame, months_of_data: float) -> Dict:
        """
        Extract expense-related features from debit transactions.
        
        Args:
            debits: DataFrame of debit transactions
            credits: DataFrame of credit transactions
            months_of_data: Number of months of data available
            
        Returns:
            Dict: Expense-related features
        """
        features = {}
        
        # Skip if no valid data
        if len(debits) == 0 or months_of_data < 0.5:  # At least 15 days of data
            return {
                'monthly_expense': 0,
                'expense_to_income_ratio': 1,  # Worst case
                'essential_expense_ratio': 0,
                'expense_consistency': 0,
                'large_expense_frequency': 0
            }
        
        # 1. Monthly Expense
        monthly_expense = debits['amount'].sum() / months_of_data if months_of_data > 0 else 0
        features['monthly_expense'] = monthly_expense
        
        # 2. Expense to Income Ratio
        monthly_income = credits['amount'].sum() / months_of_data if months_of_data > 0 and len(credits) > 0 else 0
        if monthly_income > 0:
            expense_ratio = monthly_expense / monthly_income
        else:
            expense_ratio = 1  # Worst case if no income
        features['expense_to_income_ratio'] = min(expense_ratio, 1)  # Cap at 1 for scoring purposes
        
        # 3. Essential vs Discretionary Spending
        essential_categories = ['utilities', 'housing', 'education', 'healthcare', 'transport', 'food']
        essential_expenses = debits[debits['category'].isin(essential_categories)]
        essential_amount = essential_expenses['amount'].sum()
        total_expense = debits['amount'].sum()
        essential_ratio = essential_amount / total_expense if total_expense > 0 else 0
        features['essential_expense_ratio'] = essential_ratio
        
        # 4. Expense Consistency
        # Check if expenses are consistent month-to-month
        if months_of_data >= 2:
            # Group by month and calculate monthly totals
            monthly_expenses = debits.groupby([debits['transaction_date'].dt.year, debits['transaction_date'].dt.month])['amount'].sum()
            # Calculate coefficient of variation (lower is more consistent)
            cv = monthly_expenses.std() / monthly_expenses.mean() if len(monthly_expenses) > 0 and monthly_expenses.mean() > 0 else 1
            expense_consistency = max(0, 1 - cv)  # Higher is better
            features['expense_consistency'] = expense_consistency
        else:
            features['expense_consistency'] = 0
        
        # 5. Large Expense Frequency
        # Define large expense as > 25% of monthly income
        large_expense_threshold = monthly_income * 0.25 if monthly_income > 0 else debits['amount'].median()
        large_expenses = debits[debits['amount'] > large_expense_threshold]
        large_expense_freq = len(large_expenses) / months_of_data if months_of_data > 0 else 0
        features['large_expense_frequency'] = large_expense_freq
        
        return features
    
    def _extract_discipline_features(self, credits: pd.DataFrame, debits: pd.DataFrame) -> Dict:
        """
        Extract features related to financial discipline.
        
        Args:
            credits: DataFrame of credit transactions
            debits: DataFrame of debit transactions
            
        Returns:
            Dict: Discipline-related features
        """
        features = {}
        
        # Skip if no valid data
        if len(self.processed_data) == 0:
            return {
                'savings_ratio': 0,
                'low_balance_frequency': 1,  # Worst case
                'balance_dips': 1,  # Worst case
                'weekend_spending_ratio': 0
            }
        
        # 1. Savings Ratio
        total_credits = credits['amount'].sum()
        total_debits = debits['amount'].sum()
        
        if total_credits > 0:
            savings_ratio = max(0, (total_credits - total_debits) / total_credits)
        else:
            savings_ratio = 0
        
        features['savings_ratio'] = savings_ratio
        
        # 2. Low Balance Frequency
        # Define low balance as negative running balance
        # Get the last balance for each day
        days_with_balance = {}
        for date, group in self.processed_data.groupby(self.processed_data['transaction_date'].dt.date):
            # Get the last running balance for this day
            days_with_balance[date] = group['running_balance'].iloc[-1]
        
        # Convert to Series for easier processing
        balance_series = pd.Series(days_with_balance)
        
        # Count days with negative balance
        low_balance_count = (balance_series < 0).sum()
        low_balance_frequency = low_balance_count / len(balance_series) if len(balance_series) > 0 else 1
        
        features['low_balance_frequency'] = low_balance_frequency
        
        # 3. Balance Dips (e.g., going below zero right before income)
        # Calculate the number of times balance goes negative right before a major credit
        major_credits = credits[credits['amount'] > credits['amount'].median()]
        
        dip_count = 0
        for _, credit in major_credits.iterrows():
            credit_date = credit['transaction_date']
            # Check if balance was negative in the 3 days before this credit
            prev_days = [credit_date - timedelta(days=i) for i in range(1, 4)]
            for day in prev_days:
                day_date = day.date()
                if day_date in days_with_balance and days_with_balance[day_date] < 0:
                    dip_count += 1
                    break
        
        balance_dips = dip_count / len(major_credits) if len(major_credits) > 0 else 1
        features['balance_dips'] = balance_dips
        
        # 4. Weekend Spending Ratio
        weekend_spending = debits[debits['is_weekend'] == 1]['amount'].sum()
        weekday_spending = debits[debits['is_weekend'] == 0]['amount'].sum()
        
        # Normalize by number of days
        weekend_days = (debits['is_weekend'] == 1).sum()
        weekday_days = (debits['is_weekend'] == 0).sum()
        
        if weekend_days > 0 and weekday_days > 0:
            weekend_daily = weekend_spending / weekend_days
            weekday_daily = weekday_spending / weekday_days
            
            if weekday_daily > 0:
                weekend_ratio = weekend_daily / weekday_daily
            else:
                weekend_ratio = 1
        else:
            weekend_ratio = 0
            
        features['weekend_spending_ratio'] = weekend_ratio
        
        return features
    
    def _extract_history_features(self, months_of_data: float) -> Dict:
        """
        Extract features related to transaction history.
        
        Args:
            months_of_data: Number of months of data available
            
        Returns:
            Dict: History-related features
        """
        features = {}
        
        # Skip if no valid data
        if len(self.processed_data) == 0:
            return {
                'transaction_frequency': 0,
                'merchant_diversity': 0,
                'data_months': 0,
                'high_risk_merchant_ratio': 0
            }
        
        # 1. Transaction Frequency
        transaction_count = len(self.processed_data)
        transaction_frequency = transaction_count / (months_of_data * 30) if months_of_data > 0 else 0  # Transactions per day
        features['transaction_frequency'] = transaction_frequency
        
        # 2. Merchant Diversity
        merchant_count = self.processed_data['merchant'].nunique()
        merchant_diversity = min(1, merchant_count / 10)  # Normalize, cap at 1
        features['merchant_diversity'] = merchant_diversity
        
        # 3. Months of Data Available
        features['data_months'] = months_of_data
        
        # 4. High-Risk Merchant Ratio
        # Define high-risk merchants (gambling, frequent cash withdrawals, etc.)
        high_risk_keywords = ['gambling', 'casino', 'bet', 'lottery', 'atm', 'cash withdrawal']
        
        # Check for high-risk merchants in the data
        high_risk_mask = self.processed_data['description'].str.lower().str.contains('|'.join(high_risk_keywords), regex=True)
        high_risk_transactions = self.processed_data[high_risk_mask]
        
        high_risk_ratio = len(high_risk_transactions) / len(self.processed_data)
        features['high_risk_merchant_ratio'] = high_risk_ratio
        
        return features


def analyze_transactions(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze transactions and extract features.
    
    Args:
        data: Transaction data as a pandas DataFrame
        
    Returns:
        Tuple: (Processed data, Features dictionary)
    """
    parser = TransactionParser(data)
    processed_data = parser.parse_transactions()
    features = parser.extract_features()
    
    return processed_data, features


if __name__ == "__main__":
    # Test with sample data
    from data_ingestion import load_sample_data
    
    sample_data = load_sample_data()
    print(f"Loaded {len(sample_data)} sample transactions")
    
    # Analyze transactions
    processed_data, features = analyze_transactions(sample_data)
    
    print("\nExtracted Features:")
    for category, items in {
        "Income Stability": {k: v for k, v in features.items() if k in ['monthly_income', 'income_frequency', 'income_regularity', 'income_sources', 'income_growth']},
        "Expense Management": {k: v for k, v in features.items() if k in ['monthly_expense', 'expense_to_income_ratio', 'essential_expense_ratio', 'expense_consistency', 'large_expense_frequency']},
        "Financial Discipline": {k: v for k, v in features.items() if k in ['savings_ratio', 'low_balance_frequency', 'balance_dips', 'weekend_spending_ratio']},
        "Transaction History": {k: v for k, v in features.items() if k in ['transaction_frequency', 'merchant_diversity', 'data_months', 'high_risk_merchant_ratio']}
    }.items():
        print(f"\n{category}:")
        for key, value in items.items():
            print(f"  {key}: {value:.4f}")
    
    # Check round-trip payments
    roundtrips = processed_data[processed_data['is_roundtrip'] == 1]
    print(f"\nDetected {len(roundtrips)} round-trip payments")
    
    # Check recurring transactions
    recurring = processed_data[processed_data['is_recurring'] == 1]
    print(f"Detected {len(recurring)} recurring transactions") 