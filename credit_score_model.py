import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import random

class FinUPICreditScoreModel:
    """
    FinUPI Credit Score Model for analyzing UPI transactions and generating credit scores.
    
    This model analyzes various parameters from UPI transaction data:
    - Counterparty diversity
    - Transaction amounts
    - Transaction patterns
    - Circular transaction detection
    - Time-based patterns
    """
    
    def __init__(self):
        self.model_weights = {
            'counterparty_diversity': 0.10,       # Further reduced - gig workers often have fewer partners
            'amount_entropy': 0.05,               # Kept low as gig workers often have similar payments 
            'transaction_frequency': 0.30,        # Increased even more - regular income is critical
            'credit_debit_ratio': 0.30,           # Increased for strong cash flow indication
            'circular_transaction_flag': -0.05,   # Reduced penalty for suspicious patterns
            'transaction_time_entropy': 0.05,     # Unchanged
            'transaction_growth': 0.15,           # Maintains importance of growing income
            'average_transaction_size': 0.05,     # Kept same
            'max_transaction_limit': 0.00,        # Removed as not relevant for micro-loans
            'merchant_type_diversity': 0.05       # Unchanged
        }
        
        self.scaler = MinMaxScaler(feature_range=(0, 100))
        
    def preprocess_transactions(self, transactions_data):
        """
        Preprocess raw UPI transaction data into a pandas DataFrame with the necessary structure.
        
        Args:
            transactions_data: List of transaction records or pandas DataFrame
            
        Returns:
            DataFrame with cleaned and preprocessed transactions
        """
        if isinstance(transactions_data, list):
            df = pd.DataFrame(transactions_data)
        else:
            df = transactions_data.copy()
        
        # Map column names if using the gigworker excel format
        if 'Transaction ID' in df.columns and 'Timestamp' in df.columns:
            column_mapping = {
                'Transaction ID': 'id',
                'Timestamp': 'date',
                'Receiver Name': 'merchant',
                'Amount (INR)': 'amount',
                'Sender Name': 'sender'
            }
            df = df.rename(columns=column_mapping)
            
            # Determine transaction type based on sender/receiver UPI IDs
            # If the user is the sender, it's a debit transaction
            if 'Sender UPI ID' in df.columns and 'Receiver UPI ID' in df.columns:
                # Assuming transactions from the user's UPI ID are 'debit'
                # First, find the most common sender UPI ID (likely the user's)
                user_upi = df['Sender UPI ID'].value_counts().idxmax()
                df['type'] = df.apply(
                    lambda row: 'debit' if row['Sender UPI ID'] == user_upi else 'credit', 
                    axis=1
                )
            else:
                # Fallback: randomly assign transaction types for testing
                # In real implementation, this should be determined from the data
                df['type'] = np.random.choice(['credit', 'debit'], size=len(df), p=[0.4, 0.6])
                
            # Add category if not present
            if 'category' not in df.columns:
                # Assign categories based on merchants (just for demo)
                merchants = df['merchant'].unique()
                categories = ["Sales", "Purchases", "Utilities", "Food", "Transport", "Services"]
                merchant_categories = {
                    merchant: random.choice(categories)
                    for merchant in merchants
                }
                df['category'] = df['merchant'].map(merchant_categories)
                
            # Filter out failed transactions if status column exists
            if 'Status' in df.columns:
                df = df[df['Status'] == 'SUCCESS']
            
        # Convert date strings to datetime objects if needed
        if isinstance(df['date'].iloc[0], str):
            df['date'] = pd.to_datetime(df['date'])
            
        # Ensure amount is numeric
        df['amount'] = pd.to_numeric(df['amount'])
        
        # Add a normalized timestamp for time pattern analysis
        df['hour_of_day'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Sort by date
        df = df.sort_values('date')
        
        # Add transaction direction column (1 for incoming, -1 for outgoing)
        df['direction'] = df['type'].apply(lambda x: 1 if x == 'credit' else -1)
        
        return df
        
    def analyze_counterparty_diversity(self, df):
        """
        Analyzes the diversity of counterparties in the transaction data.
        Higher diversity is generally better for credit score.
        
        Returns:
            float: Normalized diversity score (0-100)
        """
        # Count unique merchants/counterparties
        unique_counterparties = df['merchant'].nunique()
        total_transactions = len(df)
        
        # Shannon entropy of merchant distribution
        merchant_counts = df['merchant'].value_counts(normalize=True)
        entropy = -sum(merchant_counts * np.log2(merchant_counts))
        
        # Normalize to 0-100 scale - boosted for gig workers
        normalized_entropy = min(100, entropy * 35)  # Increased scaling factor (was 25)
        
        # Balance between unique count and entropy - boost for gig workers
        # Even with few merchants, give a higher base score
        diversity_score = 0.6 * normalized_entropy + 0.4 * min(100, unique_counterparties * 20)
        
        # Minimum score floor for gig workers
        return max(60, diversity_score)  # Minimum score of 60
    
    def analyze_amount_entropy(self, df):
        """
        Analyzes the diversity and distribution of transaction amounts.
        Higher entropy (more diverse amounts) can indicate healthy financial activity.
        
        Returns:
            float: Normalized amount entropy score (0-100)
        """
        # Create bins for transaction amounts
        amount_bins = pd.cut(df['amount'], bins=10)
        bin_counts = amount_bins.value_counts(normalize=True)
        
        # Calculate entropy if we have enough data
        if len(bin_counts) > 1:
            entropy = -sum(bin_counts * np.log2(bin_counts.clip(lower=1e-10)))
            # Normalize to 0-100 scale
            normalized_entropy = min(100, entropy * 33)  # Scale factor determined empirically
        else:
            normalized_entropy = 50  # Default value if not enough data
            
        return normalized_entropy
    
    def detect_circular_transactions(self, df):
        """
        Detects potential circular transactions which could indicate suspicious activity.
        
        Returns:
            float: Circular transaction flag score (0-100, lower is better)
        """
        # Create a directed graph of transactions
        G = nx.DiGraph()
        
        # Extract sender-receiver pairs (simplified, as we're working with mock data)
        # In real implementation, we would extract actual UPI IDs
        for _, row in df.iterrows():
            if row['type'] == 'debit':
                sender = 'user'
                receiver = row['merchant']
                G.add_edge(sender, receiver, amount=row['amount'])
            else:
                sender = row['merchant']
                receiver = 'user'
                G.add_edge(sender, receiver, amount=row['amount'])
        
        # Look for cycles in the graph
        try:
            cycles = list(nx.simple_cycles(G))
            cycle_score = min(100, len(cycles) * 25)  # Scale factor
        except:
            cycle_score = 0  # No cycles detected
            
        return cycle_score
    
    def analyze_transaction_frequency(self, df):
        """
        Analyzes transaction frequency patterns.
        Regular transaction activity is positive for credit score.
        
        Returns:
            float: Transaction frequency score (0-100)
        """
        # Calculate transactions per day
        date_range = (df['date'].max() - df['date'].min()).days + 1
        if date_range < 1:
            date_range = 1  # Avoid division by zero
            
        transactions_per_day = len(df) / date_range
        
        # Score based on transactions per day - more generous for gig workers
        # Even low frequency gets a decent score
        frequency_score = min(100, 50 + transactions_per_day * 25)  # Base score of 50 + scaling
        
        return frequency_score
    
    def analyze_credit_debit_ratio(self, df):
        """
        Analyzes the ratio between credit (incoming) and debit (outgoing) transactions.
        A healthy balance is good for credit score.
        
        Returns:
            float: Credit-debit ratio score (0-100)
        """
        # Calculate total credit and debit amounts
        credit_amount = df[df['type'] == 'credit']['amount'].sum()
        debit_amount = df[df['type'] == 'debit']['amount'].sum()
        
        # Avoid division by zero
        if debit_amount == 0:
            ratio = 5.0  # Arbitrary high value if no debits
        else:
            ratio = credit_amount / debit_amount
            
        # Balanced ratio - more forgiving for gig workers
        if ratio < 0.8:  # Was 1.0, now more forgiving of higher expenses
            ratio_score = 40 + ratio * 50  # Minimum 40 points, was ratio * 70
        else:
            ratio_score = 100 - min(20, (ratio - 1) * 5)  # Less penalty for high income, was -10
            
        return max(40, ratio_score)  # Minimum score of 40
    
    def analyze_transaction_time_patterns(self, df):
        """
        Analyzes the entropy of transaction times throughout the day and week.
        Regular patterns can indicate stability.
        
        Returns:
            float: Transaction time entropy score (0-100)
        """
        # Calculate entropy of hour distribution
        hour_counts = df['hour_of_day'].value_counts(normalize=True)
        if len(hour_counts) > 1:
            hour_entropy = -sum(hour_counts * np.log2(hour_counts.clip(lower=1e-10)))
        else:
            hour_entropy = 0
            
        # Calculate entropy of day of week distribution
        day_counts = df['day_of_week'].value_counts(normalize=True)
        if len(day_counts) > 1:
            day_entropy = -sum(day_counts * np.log2(day_counts.clip(lower=1e-10)))
        else:
            day_entropy = 0
            
        # Combine the two entropies (normalized)
        time_entropy_score = min(100, (hour_entropy * 10 + day_entropy * 20))
        
        return time_entropy_score
    
    def analyze_transaction_growth(self, df):
        """
        Analyzes the growth trend in transaction volume over time.
        Healthy growth is positive for credit score.
        
        Returns:
            float: Transaction growth score (0-100)
        """
        # Group by week and calculate weekly transaction volume
        df['week'] = df['date'].dt.isocalendar().week
        weekly_volume = df.groupby('week')['amount'].sum()
        
        if len(weekly_volume) < 2:
            return 50  # Not enough data for trend analysis
            
        # Calculate week-over-week growth rate
        growth_rates = weekly_volume.pct_change().dropna()
        
        # Average growth rate
        avg_growth = growth_rates.mean()
        
        # Score based on growth rate
        if avg_growth < -0.1:  # Significant decline
            growth_score = max(0, 50 + avg_growth * 200)
        elif avg_growth < 0.05:  # Slight decline or stable
            growth_score = 50 + avg_growth * 400
        else:  # Growth
            growth_score = min(100, 70 + avg_growth * 150)
            
        return growth_score
    
    def analyze_transaction_size(self, df):
        """
        Analyzes the average and maximum transaction sizes.
        
        Returns:
            tuple: (avg_transaction_score, max_transaction_score)
        """
        avg_amount = df['amount'].mean()
        max_amount = df['amount'].max()
        
        # Score based on average transaction size
        avg_transaction_score = min(100, avg_amount / 100)
        
        # Score based on maximum transaction size
        max_transaction_score = min(100, max_amount / 1000)
        
        return avg_transaction_score, max_transaction_score
    
    def analyze_merchant_type_diversity(self, df):
        """
        Analyzes the diversity of merchant types/categories in transactions.
        Higher diversity is generally better for credit score.
        
        Returns:
            float: Normalized diversity score (0-100)
        """
        # Simplified mocking of merchant categories
        # In real implementation, this would use actual merchant category data
        if 'category' in df.columns:
            df['merchant_category'] = df['category']
        else:
            # Mockup categorization by merchant name
            unique_merchants = df['merchant'].unique()
            categories = ["Retail", "Utility", "Food", "Transport", "Entertainment", "Services"]
            merchant_categories = {
                merchant: random.choice(categories) 
                for merchant in unique_merchants
            }
            df['merchant_category'] = df['merchant'].map(merchant_categories)
        
        # Calculate category diversity
        category_counts = df['merchant_category'].value_counts(normalize=True)
        
        if len(category_counts) > 1:
            category_entropy = -sum(category_counts * np.log2(category_counts))
            diversity_score = min(100, category_entropy * 40)  # Scale factor
        else:
            diversity_score = 30  # Low score for single category
            
        return diversity_score
    
    def separate_transactions_by_type(self, transactions_data):
        """
        Separates transactions into sender and receiver categories.
        
        Args:
            transactions_data: List of transaction records or pandas DataFrame
            
        Returns:
            dict: Dictionary with 'sent', 'received', and 'by_receiver' keys containing DataFrames
        """
        # Preprocess the data
        df = self.preprocess_transactions(transactions_data)
        
        # Separate by transaction type (debit = sent, credit = received)
        sent_transactions = df[df['type'] == 'debit'].copy()
        received_transactions = df[df['type'] == 'credit'].copy()
        
        # Group transactions by receiver (merchant)
        transactions_by_receiver = {}
        for merchant in df['merchant'].unique():
            merchant_df = df[df['merchant'] == merchant].copy()
            transactions_by_receiver[merchant] = merchant_df
            
        return {
            'sent': sent_transactions,
            'received': received_transactions,
            'by_receiver': transactions_by_receiver
        }
    
    def aggregate_transactions_by_receiver(self, transactions_data):
        """
        Aggregates transaction data by receiver (merchant).
        
        Args:
            transactions_data: List of transaction records or pandas DataFrame
            
        Returns:
            DataFrame: Aggregated statistics for each receiver
        """
        # Preprocess the data
        df = self.preprocess_transactions(transactions_data)
        
        # Group by merchant and calculate statistics
        aggregated = df.groupby('merchant').agg({
            'amount': ['count', 'sum', 'mean', 'min', 'max'],
            'type': lambda x: (x == 'credit').mean() * 100,  # Percentage of credit transactions
            'date': [
                lambda x: (x.max() - x.min()).days,  # Date range in days
                lambda x: x.dt.hour.mean(),          # Average hour of transactions
                'first', 'last'                      # First and last transaction dates
            ]
        })
        
        # Flatten the column names
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
        
        # Rename columns for clarity
        aggregated = aggregated.rename(columns={
            'amount_count': 'transaction_count',
            'amount_sum': 'total_amount',
            'amount_mean': 'average_amount',
            'amount_min': 'min_amount',
            'amount_max': 'max_amount',
            'type_<lambda_0>': 'percent_credit',
            'date_<lambda_0>': 'days_active',
            'date_<lambda_1>': 'average_hour',
            'date_first': 'first_transaction',
            'date_last': 'last_transaction'
        })
        
        # Add frequency (transactions per day)
        aggregated['transactions_per_day'] = aggregated['transaction_count'] / aggregated['days_active'].clip(lower=1)
        
        return aggregated
    
    def calculate_credit_score(self, transactions_data):
        """
        Calculates the overall credit score based on all analyzed parameters.
        
        Args:
            transactions_data: List of transaction records or pandas DataFrame
            
        Returns:
            dict: Credit score results including overall score and component scores
        """
        # Preprocess the data
        df = self.preprocess_transactions(transactions_data)
        
        if len(df) == 0:
            return {'score': 50, 'components': {}, 'message': 'No transaction data available'}
            
        # Calculate all component scores
        counterparty_diversity = self.analyze_counterparty_diversity(df)
        amount_entropy = self.analyze_amount_entropy(df)
        circular_transaction_flag = self.detect_circular_transactions(df)
        transaction_frequency = self.analyze_transaction_frequency(df)
        credit_debit_ratio = self.analyze_credit_debit_ratio(df)
        transaction_time_entropy = self.analyze_transaction_time_patterns(df)
        transaction_growth = self.analyze_transaction_growth(df)
        avg_transaction_size, max_transaction_limit = self.analyze_transaction_size(df)
        merchant_type_diversity = self.analyze_merchant_type_diversity(df)
        
        # Compile all component scores
        component_scores = {
            'counterparty_diversity': counterparty_diversity,
            'amount_entropy': amount_entropy,
            'transaction_frequency': transaction_frequency,
            'credit_debit_ratio': credit_debit_ratio,
            'circular_transaction_flag': 100 - circular_transaction_flag,  # Invert so higher is better
            'transaction_time_entropy': transaction_time_entropy,
            'transaction_growth': transaction_growth,
            'average_transaction_size': avg_transaction_size,
            'max_transaction_limit': max_transaction_limit,
            'merchant_type_diversity': merchant_type_diversity
        }
        
        # Calculate weighted score
        weighted_score = sum(
            component_scores[component] * weight 
            for component, weight in self.model_weights.items()
        )
        
        # Determine credit level and loan eligibility
        credit_level = self._determine_credit_level(weighted_score)
        loan_eligibility = self._calculate_loan_eligibility(weighted_score, df)
        
        # Prepare the result
        result = {
            'score': round(weighted_score),
            'level': credit_level,
            'components': component_scores,
            'loan_eligibility': loan_eligibility,
            'last_updated': datetime.now().strftime('%Y-%m-%d')
        }
        
        return result
    
    def _determine_credit_level(self, score):
        """Determines the credit level based on the score."""
        if score >= 80:
            return "Excellent"
        elif score >= 70:
            return "Very Good"
        elif score >= 60:
            return "Good"
        elif score >= 45:  # Lowered from 50
            return "Fair"
        elif score >= 30:  # Lowered from 35
            return "Poor"
        else:
            return "Very Poor"
    
    def _calculate_loan_eligibility(self, score, df):
        """
        Calculates loan eligibility for gig workers with micro loans.
        Loan range: 10,000 to 2,00,000 INR
        Interest rates: 1-3% at onboarding
        """
        # Calculate average monthly credit transactions
        if len(df) > 0 and 'date' in df.columns:
            # Determine date range in months
            date_range = (df['date'].max() - df['date'].min()).days / 30
            if date_range < 0.5:  # Less than 15 days of data
                date_range = 0.5
                
            # Calculate monthly income estimate (from credit transactions)
            monthly_income = df[df['type'] == 'credit']['amount'].sum() / date_range
            
            # For very low transaction volumes, apply a higher minimum threshold
            if len(df) < 10:
                monthly_income = max(monthly_income, 20000)  # Increased minimum assumed income
        else:
            monthly_income = 20000  # Increased default assumption for gig workers
            
        # Calculate monthly surplus (income - expenses)
        if len(df) > 0:
            monthly_expenses = df[df['type'] == 'debit']['amount'].sum() / date_range
            monthly_surplus = max(1000, monthly_income - monthly_expenses)  # Ensure minimum surplus
        else:
            # Default assumption if we don't have expense data
            monthly_surplus = monthly_income * 0.3  # Assume 30% surplus
        
        # Base loan limit calculation - much more generous for gig workers
        if score >= 80:
            loan_limit = min(monthly_income * 8, 200000)  # Increased from 5x to 8x
            interest_rate = 1.0
            duration_days = 60  # Longer duration
        elif score >= 70:
            loan_limit = min(monthly_income * 6, 150000)  # Increased from 4x to 6x
            interest_rate = 1.5
            duration_days = 45
        elif score >= 60:
            loan_limit = min(monthly_income * 5, 120000)  # Increased from 3x to 5x
            interest_rate = 2.0
            duration_days = 45
        elif score >= 45:  # Lowered threshold and increased multiplier
            loan_limit = min(monthly_income * 3, 80000)  # Was 2x, now 3x
            interest_rate = 2.5
            duration_days = 30
        elif score >= 30:
            loan_limit = min(monthly_income * 2, 40000)  # Was 1x, now 2x
            interest_rate = 3.0
            duration_days = 30
        else:
            loan_limit = min(monthly_income * 1, 20000)  # Increased minimum loan
            interest_rate = 3.0
            duration_days = 30
        
        # Ensure minimum loan amount of 15,000 if they qualify
        if score >= 30:
            loan_limit = max(loan_limit, 15000)  # Increased from 10k to 15k
            
        # Ensure loan limit is at least equal to monthly surplus for scores above 45
        if score >= 45:
            loan_limit = max(loan_limit, monthly_surplus * duration_days / 30)
            
        return {
            'max_loan_amount': round(loan_limit, -3),  # Round to nearest thousand
            'interest_rate': interest_rate,
            'max_duration_days': duration_days  # Variable duration based on score
        }
    
    def generate_report(self, transactions_data, output_file=None):
        """
        Generates a comprehensive credit score report including visualizations.
        
        Args:
            transactions_data: List of transaction records or pandas DataFrame
            output_file: If provided, saves the report to this file
            
        Returns:
            dict: Credit score results and report path if saved
        """
        # Calculate credit score
        results = self.calculate_credit_score(transactions_data)
        df = self.preprocess_transactions(transactions_data)
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # Credit score gauge
        plt.subplot(2, 3, 1)
        self._plot_gauge(results['score'], 'Credit Score')
        
        # Component scores
        plt.subplot(2, 3, 2)
        components = {k: v for k, v in results['components'].items() 
                     if k in self.model_weights and self.model_weights[k] >= 0.1}
        self._plot_component_scores(components)
        
        # Transaction volume over time
        plt.subplot(2, 3, 3)
        self._plot_transaction_volume(df)
        
        # Merchant diversity
        plt.subplot(2, 3, 4)
        self._plot_merchant_diversity(df)
        
        # Credit vs Debit ratio
        plt.subplot(2, 3, 5)
        self._plot_credit_debit_ratio(df)
        
        # Transaction heatmap by hour and day
        plt.subplot(2, 3, 6)
        self._plot_transaction_heatmap(df)
        
        plt.tight_layout()
        
        # Save to file if requested
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            results['report_file'] = output_file
            
        plt.close()
        
        # Export to Excel
        if output_file:
            excel_file = output_file.replace('.png', '.xlsx')
            self._export_to_excel(df, results, excel_file)
            results['excel_report'] = excel_file
            
        return results
    
    def _plot_gauge(self, score, title):
        """Plots a gauge chart for the credit score."""
        # Create gauge chart code here
        plt.pie([score, 100-score], colors=['#2ECC71', '#EAECEE'], 
                startangle=90, counterclock=False)
        plt.text(0, 0, f"{int(score)}", ha='center', va='center', fontsize=24, fontweight='bold')
        plt.title(title)
        
    def _plot_component_scores(self, components):
        """Plots a bar chart of component scores."""
        plt.barh(list(components.keys()), list(components.values()), color='#3498DB')
        plt.title('Component Scores')
        plt.xlim(0, 100)
        
    def _plot_transaction_volume(self, df):
        """Plots transaction volume over time."""
        if len(df) == 0:
            plt.text(0.5, 0.5, "No data available", ha='center', va='center')
            plt.title('Transaction Volume')
            return
            
        # Group by day
        daily_volume = df.groupby(df['date'].dt.date)['amount'].sum()
        plt.plot(daily_volume.index, daily_volume.values, marker='o', linestyle='-', color='#9B59B6')
        plt.title('Daily Transaction Volume')
        plt.xticks(rotation=45)
        
    def _plot_merchant_diversity(self, df):
        """Plots a pie chart of merchant diversity."""
        if len(df) == 0:
            plt.text(0.5, 0.5, "No data available", ha='center', va='center')
            plt.title('Merchant Diversity')
            return
            
        merchant_counts = df['merchant'].value_counts()
        plt.pie(merchant_counts, labels=merchant_counts.index, autopct='%1.1f%%')
        plt.title('Merchant Diversity')
        
    def _plot_credit_debit_ratio(self, df):
        """Plots credit vs debit amounts."""
        if len(df) == 0:
            plt.text(0.5, 0.5, "No data available", ha='center', va='center')
            plt.title('Credit vs Debit')
            return
            
        credit = df[df['type'] == 'credit']['amount'].sum()
        debit = df[df['type'] == 'debit']['amount'].sum()
        plt.bar(['Credit', 'Debit'], [credit, debit], color=['#27AE60', '#E74C3C'])
        plt.title('Credit vs Debit Amounts')
        
    def _plot_transaction_heatmap(self, df):
        """Plots a heatmap of transactions by hour and day of week."""
        if len(df) == 0:
            plt.text(0.5, 0.5, "No data available", ha='center', va='center')
            plt.title('Transaction Timing')
            return
            
        # Create pivot table of transaction counts by hour and day
        heatmap_data = pd.pivot_table(
            df, 
            values='amount',
            index='day_of_week', 
            columns='hour_of_day',
            aggfunc='count',
            fill_value=0
        )
        
        sns.heatmap(heatmap_data, cmap='YlGnBu')
        plt.title('Transaction Timing Heatmap')
        
    def _export_to_excel(self, df, results, excel_file):
        """Exports transaction data and results to Excel."""
        with pd.ExcelWriter(excel_file) as writer:
            # Transaction data
            df.to_excel(writer, sheet_name='Transactions', index=False)
            
            # Credit score results
            pd.DataFrame({
                'Metric': ['Credit Score', 'Credit Level', 'Last Updated'],
                'Value': [results['score'], results['level'], results['last_updated']]
            }).to_excel(writer, sheet_name='Credit Score', index=False)
            
            # Component scores
            components_df = pd.DataFrame({
                'Component': list(results['components'].keys()),
                'Score': list(results['components'].values()),
                'Weight': [self.model_weights.get(k, 0) for k in results['components'].keys()]
            })
            components_df['Weighted Score'] = components_df['Score'] * components_df['Weight']
            components_df.to_excel(writer, sheet_name='Component Scores', index=False)
            
            # Loan eligibility
            pd.DataFrame({
                'Metric': ['Maximum Loan Amount', 'Interest Rate', 'Maximum Duration (Days)'],
                'Value': [
                    results['loan_eligibility']['max_loan_amount'],
                    results['loan_eligibility']['interest_rate'],
                    results['loan_eligibility']['max_duration_days']
                ]
            }).to_excel(writer, sheet_name='Loan Eligibility', index=False)
    
    def save_model(self, filename='finupi_credit_model.pkl'):
        """Saves the model to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        return filename
    
    @classmethod
    def load_model(cls, filename='finupi_credit_model.pkl'):
        """Loads the model from a file."""
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model


def process_upi_data(upi_data_file, output_report=None):
    """
    Process UPI transaction data from a file and generate a credit score report.
    
    Args:
        upi_data_file: Path to CSV, JSON, or Excel file containing UPI transaction data
        output_report: Path to save the report graphics (optional)
        
    Returns:
        dict: Credit score results
    """
    # Load the data
    if upi_data_file.endswith('.csv'):
        df = pd.read_csv(upi_data_file)
    elif upi_data_file.endswith('.json'):
        df = pd.read_json(upi_data_file)
    elif upi_data_file.endswith('.xlsx') or upi_data_file.endswith('.xls'):
        df = pd.read_excel(upi_data_file)
    else:
        raise ValueError("Unsupported file format. Please use CSV, JSON, or Excel file.")
    
    # Create model and generate report
    model = FinUPICreditScoreModel()
    
    if output_report:
        results = model.generate_report(df, output_report)
    else:
        results = model.calculate_credit_score(df)
    
    return results


if __name__ == "__main__":
    # Example usage from command line
    import sys
    import json
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        results = process_upi_data(input_file, output_file)
        
        # Print results to console
        print(json.dumps(results, indent=2))
    else:
        print("Usage: python credit_score_model.py <upi_data_file> [output_report_file]") 