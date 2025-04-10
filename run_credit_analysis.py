#!/usr/bin/env python
"""
FinUPI Credit Score Model Runner

This script provides a command-line interface to run the FinUPI credit score model
on UPI transaction data.

Usage:
    python run_credit_analysis.py --input <transaction_file> [--output <report_file>]
"""

import argparse
import json
import os
import sys
from datetime import datetime
from credit_score_model import FinUPICreditScoreModel, process_upi_data

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate a credit score from UPI transaction data'
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to transaction data file (CSV or JSON)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output path for report files (PNG and Excel)'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['json', 'txt', 'both'],
        default='both',
        help='Output format for results (json, txt, or both)'
    )
    
    return parser.parse_args()

def print_results(results):
    """Print credit score results in a readable format."""
    print("\n" + "="*60)
    print(f"FINUPI CREDIT SCORE REPORT - {datetime.now().strftime('%Y-%m-%d')}")
    print("="*60)
    
    print(f"\nCredit Score: {results['score']}/100 ({results['level']})")
    print(f"Last Updated: {results['last_updated']}")
    print("\nLoan Eligibility:")
    print(f"  Maximum Loan Amount: ₹{results['loan_eligibility']['max_loan_amount']:,}")
    print(f"  Interest Rate: {results['loan_eligibility']['interest_rate']}%")
    print(f"  Maximum Duration: {results['loan_eligibility']['max_duration_days']} days")
    
    print("\nComponent Scores:")
    for component, score in sorted(
        results['components'].items(), 
        key=lambda x: results['components'][x[0]], 
        reverse=True
    ):
        pretty_name = component.replace('_', ' ').title()
        print(f"  {pretty_name}: {score:.1f}/100")
    
    if 'report_file' in results:
        print(f"\nDetailed report saved to: {results['report_file']}")
    if 'excel_report' in results:
        print(f"Excel data saved to: {results['excel_report']}")
    
    print("\n" + "="*60)

def main():
    """Main function to run the credit analysis."""
    args = parse_arguments()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
    
    # Determine output file if specified
    output_file = None
    if args.output:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Add extension if not present
        if not args.output.endswith('.png'):
            output_file = args.output + '.png'
        else:
            output_file = args.output
    
    try:
        # Process the data and generate results
        results = process_upi_data(args.input, output_file)
        
        # Output results in the requested format
        if args.format in ['txt', 'both']:
            print_results(results)
        
        if args.format in ['json', 'both']:
            json_file = output_file.replace('.png', '.json') if output_file else 'credit_score_results.json'
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2)
            if args.format == 'json':
                print(f"Results saved to {json_file}")
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 