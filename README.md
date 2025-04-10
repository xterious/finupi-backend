# FinUPI Credit Score Model

This module provides a comprehensive credit scoring model for analyzing UPI (Unified Payments Interface) transaction data. It's designed to evaluate a user's creditworthiness based on their transaction history.

## Features

- **Transaction Analysis**: Processes UPI transaction data from various formats (CSV, JSON)
- **Multi-factor Credit Scoring**: Evaluates 10+ different parameters:
  - Counterparty diversity
  - Transaction amount entropy
  - Transaction frequency
  - Credit-debit ratio
  - Circular transaction detection
  - Transaction time patterns
  - Transaction growth trends
  - And more
- **Loan Eligibility Assessment**: Calculates maximum loan amount and interest rate
- **Visualization**: Generates comprehensive reports with charts and graphs
- **Data Export**: Exports all analysis data to Excel for further review
- **REST API**: Exposes model functions via a Flask-based API

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### As a Command-Line Tool

The simplest way to use the model is through the command-line interface:

```bash
python run_credit_analysis.py --input sample_transactions.json --output credit_report
```

Options:

- `--input`, `-i`: Path to the transaction data file (CSV or JSON) [required]
- `--output`, `-o`: Output path for report files (PNG and Excel) [optional]
- `--format`, `-f`: Output format for results (json, txt, or both) [default: both]

### Starting the API Server

To run the Flask API server:

```bash
python api.py
```

By default, the server runs on port 5000. You can access the API at `http://localhost:5000/`.

### API Endpoints

The API provides the following endpoints:

- `GET /health`: Health check endpoint
- `POST /api/credit-score`: Calculate credit score from transaction data
- `POST /api/upload-transactions`: Upload and process transaction files
- `POST /api/generate-report`: Generate visual reports from transaction data

### As a Python Module

You can also import and use the model in your own Python code:

```python
from credit_score_model import FinUPICreditScoreModel

# Create model instance
model = FinUPICreditScoreModel()

# Option 1: Load transactions from file
import pandas as pd
transactions = pd.read_json('sample_transactions.json')
results = model.calculate_credit_score(transactions)

# Option 2: Pass transaction data directly
transactions_data = [
    {
        "id": "tx001",
        "date": "2023-07-01 08:30:00",
        "merchant": "Customer1",
        "amount": 450,
        "type": "credit"
    },
    # More transactions...
]
results = model.calculate_credit_score(transactions_data)

# Generate report with visualizations
report_results = model.generate_report(transactions_data, 'credit_report.png')
```

## Example Output

The model generates a comprehensive credit score report including:

1. Overall credit score (0-100)
2. Credit level classification (Excellent, Very Good, Good, Fair, Poor, Very Poor)
3. Component scores for each analyzed factor
4. Loan eligibility details (maximum amount, interest rate, duration)
5. Visualizations of transaction patterns
6. Excel file with detailed data

## Model Parameters

The FinUPI credit score is calculated using the following weights:

- counterparty_diversity: 20%
- amount_entropy: 15%
- transaction_frequency: 15%
- credit_debit_ratio: 15%
- circular_transaction_flag: -10% (penalty)
- transaction_time_entropy: 5%
- transaction_growth: 10%
- average_transaction_size: 10%
- max_transaction_limit: 5%
- merchant_type_diversity: 5%

## Sample Data

The repository includes a `sample_transactions.json` file that can be used to test the model.
