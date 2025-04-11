"""
FinUPI - Main Application

This module integrates all components of the FinUPI credit scoring system.
It provides a simple command-line interface to run the scoring system.
"""
  
import os
import json
import argparse
import pandas as pd
from typing import Dict, Optional

from finupi_model.data_ingestion import TransactionData, load_sample_data, save_sample_data
from finupi_model.transaction_parser import analyze_transactions
from finupi_model.credit_score import calculate_credit_score


def process_transactions(file_path: str, output_path: Optional[str] = None) -> Dict:
    """
    Process transaction data file and generate credit score.
    
    Args:
        file_path: Path to the transaction data file (CSV or JSON)
        output_path: Path to save the output (optional)
        
    Returns:
        Dict: Credit score results
    """
    print(f"Processing transaction data from: {file_path}")
    
    # Create transaction data object
    transaction_data = TransactionData()
    
    # Load and validate data
    if file_path.endswith('.csv'):
        success = transaction_data.load_from_csv(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        success = transaction_data.load_from_json(json_data)
    else:
        raise ValueError("Unsupported file format. Please provide CSV or JSON.")
    
    # Check if data is valid
    if not success:
        print("Error: Data validation failed.")
        for error in transaction_data.validation_errors:
            print(f"- {error}")
        return {"error": "Data validation failed."}
    
    # Data is valid, preprocess it
    preprocessed_data = transaction_data.preprocess_data()
    
    # Get data summary
    summary = transaction_data.get_summary_stats()
    print("\nTransaction Summary:")
    print(f"- Total transactions: {summary.get('total_transactions', 0)}")
    print(f"- Date range: {summary.get('date_range', ['N/A', 'N/A'])[0]} to {summary.get('date_range', ['N/A', 'N/A'])[1]}")
    print(f"- Credits: {summary.get('total_credit', 0)} (₹{summary.get('credit_amount', 0):,.2f})")
    print(f"- Debits: {summary.get('total_debit', 0)} (₹{summary.get('debit_amount', 0):,.2f})")
    
    # Analyze transactions
    print("\nAnalyzing transactions...")
    processed_data, features = analyze_transactions(preprocessed_data)
    
    # Calculate credit score
    print("Calculating credit score...")
    credit_score_result = calculate_credit_score(features)
    
    # Print results
    print("\nCredit Score Results:")
    print(f"Overall Score: {credit_score_result['overall_score']} - {credit_score_result['score_category'].capitalize()}")
    
    print("\nComponent Scores:")
    for component, score in credit_score_result['component_scores'].items():
        print(f"  {component}: {score}")
    
    print("\nLoan Eligibility:")
    loan = credit_score_result['loan_eligibility']
    print(f"  Eligible: {loan['eligible']}")
    print(f"  Maximum Loan: ₹{loan['max_loan_amount']:,}")
    print(f"  Interest Rate: {loan['interest_rate']}%")
    print(f"  Max Duration: {loan['max_duration_months']} months")
    print(f"  Monthly EMI: ₹{loan['monthly_emi']:,}")
    
    print("\nImprovement Recommendations:")
    for rec in credit_score_result['explanations']['improvement_recommendations']:
        print(f"  - {rec}")
    
    # Save output if requested
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_path, 'w') as f:
            json.dump(credit_score_result, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return credit_score_result


def generate_sample_data(file_path: str) -> None:
    """
    Generate sample transaction data for testing.
    
    Args:
        file_path: Path to save the sample data
    """
    save_sample_data(file_path)
    print(f"Sample data saved to: {file_path}")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="FinUPI Credit Scoring System",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process transactions command
    process_parser = subparsers.add_parser(
        "process", 
        help="Process transaction data and generate credit score"
    )
    process_parser.add_argument(
        "input_file", 
        help="Path to transaction data file (.csv or .json)"
    )
    process_parser.add_argument(
        "--output", "-o",
        help="Path to save output results (.json)",
        default=None
    )
    
    # Generate sample data command
    sample_parser = subparsers.add_parser(
        "sample", 
        help="Generate sample transaction data for testing"
    )
    sample_parser.add_argument(
        "output_file", 
        help="Path to save sample data (.csv)"
    )
    
    args = parser.parse_args()
    
    if args.command == "process":
        process_transactions(args.input_file, args.output)
    elif args.command == "sample":
        generate_sample_data(args.output_file)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 