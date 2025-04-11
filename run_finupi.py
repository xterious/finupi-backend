from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pymongo import MongoClient
import json
from typing import Dict, List, Tuple

# Import your existing components
from data_ingestion import TransactionData
from transaction_parser import TransactionParser, analyze_transactions
from credit_score import calculate_credit_score

app = Flask(__name__)

# MongoDB Connection
mongo_uri = "mongodb+srv://root:root@ipo-builder.lpq9ub9.mongodb.net/"
client = MongoClient(mongo_uri)
db = client["fin_credit_db"]  # Database name
transactions_collection = db["transactions"]  # Collection for storing transactions
credit_scores_collection = db["credit_scores"]  # Collection for storing credit scores

# Helper function to convert MongoDB data to DataFrame with proper structure
def prepare_transaction_data(transactions):
    """Convert MongoDB transaction data to the expected format for the model"""
    # Map fields from MongoDB to our expected format
    df = pd.DataFrame(transactions)
    
    # Ensure required columns exist
    required_columns = [
        'transaction_date', 'amount', 'transaction_type', 
        'description', 'sender_upi_id', 'receiver_upi_id'
    ]
    
    # Handle different timestamp/date formats
    if 'transaction_date' not in df.columns:
        if 'Timestamp' in df.columns:
            df['transaction_date'] = pd.to_datetime(df['Timestamp'])
        elif 'timestamp' in df.columns:
            df['transaction_date'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            df['transaction_date'] = pd.to_datetime(df['date'])
    else:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    # Handle amount field
    if 'amount' not in df.columns:
        if 'Amount (INR)' in df.columns:
            df['amount'] = df['Amount (INR)'].astype(float)
        elif 'Amount' in df.columns:
            df['amount'] = df['Amount'].astype(float)
        elif 'value' in df.columns:
            df['amount'] = df['value'].astype(float)
    elif 'amount' in df.columns:
        df['amount'] = df['amount'].astype(float)
    
    # Handle UPI IDs
    if 'sender_upi_id' not in df.columns:
        if 'Sender UPI ID' in df.columns:
            df['sender_upi_id'] = df['Sender UPI ID']
        elif 'sender' in df.columns:
            df['sender_upi_id'] = df['sender']
        else:
            df['sender_upi_id'] = 'unknown@upi'
    
    if 'receiver_upi_id' not in df.columns:
        if 'Receiver UPI ID' in df.columns:
            df['receiver_upi_id'] = df['Receiver UPI ID']
        elif 'receiver' in df.columns:
            df['receiver_upi_id'] = df['receiver']
        else:
            df['receiver_upi_id'] = 'unknown@upi'
    
    # Handle different transaction type formats
    if 'transaction_type' not in df.columns:
        if 'Type' in df.columns:
            # Map 'Sent' to 'debit' and 'Received' to 'credit'
            type_mapping = {
                'Sent': 'debit', 
                'Received': 'credit', 
                'SEND': 'debit', 
                'RECEIVE': 'credit',
                'send': 'debit',
                'receive': 'credit'
            }
            df['transaction_type'] = df['Type'].map(lambda x: type_mapping.get(x, 'debit'))
        elif 'type' in df.columns:
            # Map 'RECEIVE' to 'credit' and 'SEND' to 'debit'
            type_mapping = {'RECEIVE': 'credit', 'SEND': 'debit', 'receive': 'credit', 'send': 'debit'}
            df['transaction_type'] = df['type'].map(lambda x: type_mapping.get(x, x.lower() if isinstance(x, str) else 'debit'))
    else:
        # Ensure lowercase consistency
        df['transaction_type'] = df['transaction_type'].str.lower()
    
    # Handle description field
    if 'description' not in df.columns:
        if 'Note' in df.columns:
            df['description'] = df['Note']
        elif 'note' in df.columns:
            df['description'] = df['note']
        elif 'narration' in df.columns:
            df['description'] = df['narration']
        else:
            # Generate description from available data
            df['description'] = df.apply(lambda row: 
                f"Transaction of {row.get('amount', 'unknown amount')} from {row.get('sender_upi_id', 'unknown')} to {row.get('receiver_upi_id', 'unknown')}", 
                axis=1)
    
    # Add merchant field if not present
    if 'merchant' not in df.columns:
        # Extract merchant from description or use receiver_upi_id
        df['merchant'] = df.apply(lambda row: 
            row['receiver_upi_id'].split('@')[0] if row['transaction_type'] == 'debit' 
            else row['sender_upi_id'].split('@')[0], 
            axis=1)
    
    # Add any missing columns with default values
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    
    # Add transaction_ref if not present
    if 'transaction_ref' not in df.columns:
        if 'Reference ID' in df.columns:
            df['transaction_ref'] = df['Reference ID']
        elif 'Status' in df.columns:
            # Use Status + timestamp to create a unique reference
            df['transaction_ref'] = df.apply(
                lambda row: f"TX-{row['Status']}-{pd.to_datetime(row['transaction_date']).strftime('%Y%m%d%H%M%S')}",
                axis=1
            )
        else:
            df['transaction_ref'] = [f"tx_{i}" for i in range(len(df))]
    
    return df

@app.route('/upload_transactions', methods=['POST'])
def upload_transactions():
    data = request.get_json()
    
    if not data or 'user_id' not in data or 'transactions' not in data:
        return jsonify({'error': 'Missing user_id or transactions data'}), 400
        
    try:
        user_id = data['user_id']
        transactions = data['transactions']
        
        # Check if user already exists
        existing_user = transactions_collection.find_one({'user_id': user_id})
        
        if existing_user:
            # Update existing user's transactions
            transactions_collection.update_one(
                {'user_id': user_id},
                {'$set': {'transactions': transactions}}
            )
            return jsonify({
                'message': f'Updated transactions for user {user_id}',
                'transaction_count': len(transactions)
            })
        else:
            # Create new user document
            user_document = {
                'user_id': user_id,
                'transactions': transactions
            }
            transactions_collection.insert_one(user_document)
            return jsonify({
                'message': f'Added new user {user_id} with transactions',
                'transaction_count': len(transactions)
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_credit_score', methods=['POST'])
def get_credit_score():
    data = request.get_json()

    if not data or 'user_id' not in data:
        return jsonify({'error': 'Missing user_id parameter'}), 400

    try:
        user_id = data['user_id']
        
        # Fetch user document from MongoDB
        user_data = transactions_collection.find_one({'user_id': user_id})
        
        if not user_data:
            return jsonify({'error': f'No user found with ID {user_id}. Please upload transactions first.'}), 404
            
        # Access transactions based on schema structure
        # Check both root level 'transactions' and possible nested paths
        transactions = None
        if 'transactions' in user_data and user_data['transactions']:
            transactions = user_data['transactions']
        elif 'data' in user_data and 'transactions' in user_data['data']:
            transactions = user_data['data']['transactions']
        
        if not transactions:
            return jsonify({'error': f'No transactions found for user {user_id}. Please upload transactions first.'}), 404
        
        # Convert transactions array to DataFrame
        df = prepare_transaction_data(transactions)
        
        # Create a TransactionData object and validate
        transaction_data = TransactionData()
        transaction_data.data = df
        if not transaction_data.validate_data():
            return jsonify({
                'error': 'Invalid transaction data',
                'validation_errors': transaction_data.validation_errors
            }), 400
        
        # Preprocess data
        preprocessed_data = transaction_data.preprocess_data()
        
        # Analyze transactions and extract features
        processed_data, features = analyze_transactions(preprocessed_data)
        
        # Calculate credit score
        credit_score_result = calculate_credit_score(features)
        
        # Get last 5 transactions
        last_5 = df.sort_values(by='transaction_date', ascending=False).head(5)
        last_5_dict = last_5.to_dict(orient='records')
        
        # Format datetime objects to strings for JSON serialization
        for tx in last_5_dict:
            if isinstance(tx.get('transaction_date'), pd.Timestamp):
                tx['transaction_date'] = tx['transaction_date'].strftime('%Y-%m-%d')
        
        # Store credit score result
        credit_score_record = {
            'user_id': user_id,
            'credit_score': credit_score_result['overall_score'],
            'score_category': credit_score_result['score_category'],
            'timestamp': datetime.now(),
            'component_scores': credit_score_result['component_scores'],
            'loan_eligibility': credit_score_result['loan_eligibility'],
            'improvement_recommendations': credit_score_result['explanations']['improvement_recommendations']
        }
        
        # Update existing record or create new one if it doesn't exist
        credit_scores_collection.update_one(
            {'user_id': user_id},  # Filter by user_id
            {'$set': credit_score_record},  # Update with new data
            upsert=True  # Create if doesn't exist
        )

        return jsonify({
            'credit_score': credit_score_result['overall_score'],
            'score_category': credit_score_result['score_category'],
            'component_scores': credit_score_result['component_scores'],
            'loan_eligibility': credit_score_result['loan_eligibility'],
            'improvement_recommendations': credit_score_result['explanations']['improvement_recommendations'],
            'last_5_transactions': last_5_dict
        })

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/parse_transactions', methods=['POST'])
def parse_transactions():
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Missing request data'}), 400
    
    transactions = None
    # Check different possible locations of transaction data
    if 'transactions' in data:
        transactions = data['transactions']
    elif 'user_id' in data:
        # Try to get transactions from MongoDB
        user_id = data['user_id']
        user_data = transactions_collection.find_one({'user_id': user_id})
        
        if user_data and 'transactions' in user_data:
            transactions = user_data['transactions']
        elif user_data and 'data' in user_data and 'transactions' in user_data['data']:
            transactions = user_data['data']['transactions']
    
    if not transactions:
        return jsonify({'error': 'No transaction data found'}), 400
        
    try:
        # Convert json to DataFrame
        df = prepare_transaction_data(transactions)
        
        # Create transaction data object and validate
        transaction_data = TransactionData()
        transaction_data.data = df
        if not transaction_data.validate_data():
            return jsonify({
                'error': 'Invalid transaction data',
                'validation_errors': transaction_data.validation_errors
            }), 400
        
        # Preprocess data
        preprocessed_data = transaction_data.preprocess_data()
        
        # Analyze transactions
        processed_data, features = analyze_transactions(preprocessed_data)
        
        # Get summary stats
        summary = transaction_data.get_summary_stats()
        
        # Format features for JSON serialization
        formatted_features = {}
        for k, v in features.items():
            if isinstance(v, (np.integer, np.floating)):
                formatted_features[k] = float(v)
            else:
                formatted_features[k] = v
        
        return jsonify({
            'summary': {
                'total_transactions': summary.get('total_transactions', 0),
                'date_range': summary.get('date_range', ['N/A', 'N/A']),
                'total_credit': summary.get('total_credit', 0),
                'total_debit': summary.get('total_debit', 0),
                'credit_amount': float(summary.get('credit_amount', 0)),
                'debit_amount': float(summary.get('debit_amount', 0)),
                'avg_transaction': float(summary.get('avg_transaction', 0)),
                'transaction_frequency': float(summary.get('transaction_frequency', 0))
            },
            'features': formatted_features
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/user_credit_history/<user_id>', methods=['GET'])
def get_user_credit_history(user_id):
    try:
        # Get user credit score (now we only have one record per user)
        score = credit_scores_collection.find_one(
            {'user_id': user_id},
            {'_id': 0}  # Exclude MongoDB ObjectId
        )
        
        if not score:
            return jsonify({'message': f'No credit score found for user {user_id}'}), 404
        
        # Convert datetime objects to strings for JSON serialization
        if isinstance(score.get('timestamp'), datetime):
            score['timestamp'] = score['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({
            'user_id': user_id,
            'credit_score': score
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':  
    app.run(debug=True)